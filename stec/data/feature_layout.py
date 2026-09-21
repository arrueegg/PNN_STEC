"""How many features the model sees, and where each one sits - computed once.

The transformed input dimension is currently derived **twice**, independently: once in
`model.py` to size the input projection, and once in `collation.py` to lay out the tensor
that fills it. They agree today. Nothing makes them agree tomorrow, and a disagreement
does not raise - it shifts every feature after the disagreement by a constant offset,
which trains to a plausible and wrong model.

`Branch*` models have the same problem in a sharper form. They hardcode::

    self.temporal_split = 10                       # Temporal features
    self.spatial_split = 10 + 4 + 4 + 3            # Temporal + Station + IPP + Direction

Both numbers are right only for the default feature set, and the comment names the blocks
in a different order from the one the collation actually emits (station, *direction*, then
ipp). Disabling any feature silently misaligns them.

This module is the one computation. It produces named blocks, so a consumer asks where the
space-weather features are rather than adding offsets, and the model and the collation are
sized from the same object.

A feature is not one column. The transforms are what make the count non-obvious:

* `year` contributes 1 column, but `doy`, `sod` and `local_time_hours` contribute 3 each -
  sine, cosine and the normalised value - because they are cyclical.
* azimuth and elevation together contribute 3 columns, not 2: they are converted to a
  Cartesian unit vector (up, east, north). Either one alone falls back to 1 column.
* every other scalar contributes 1.
* each enabled coordinate *pair* adds a spherical-harmonic expansion.

For the paper's model that is 10 + 4 + 3 + 4 + 6 + 4x25 = 127, which is exactly the input
width of the published checkpoint.

The layout is **immutable**. `CollateWithSH` currently calls
`feature_registry.set_output_indices(...)` on a registry shared by seven call sites, so
what a consumer sees depends on which collator was constructed last - one script carries
the comment `CollateWithSH(stec_config)  # Sets output_indices in registry`. A frozen
layout cannot have that failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Cyclical quantities are emitted as (sin, cos, norm): a raw normalised value alone would
# put midnight and 23:59 at opposite ends of the range.
CYCLICAL_TEMPORAL = ("doy", "sod", "local_time_hours")

# Azimuth and elevation are replaced by a Cartesian unit vector when both are present.
DIRECTION_PAIR = ("satazi", "satele")
DIRECTION_COMPONENTS = ("e_up", "e_east", "e_north")


class FeatureGroup(str, Enum):
    """Blocks in the order the collation emits them. The order *is* the tensor layout."""

    TEMPORAL = "temporal"
    STATION = "station"
    DIRECTION = "direction"
    IPP = "ipp"
    SWI = "swi"


GROUP_MEMBERS: dict[FeatureGroup, tuple[str, ...]] = {
    FeatureGroup.TEMPORAL: ("year", "doy", "sod", "local_time_hours"),
    # Station is solar-magnetic *first*, IPP is geographic first. The asymmetry is not a
    # mistake to tidy up - it is the order the trained checkpoints were fed, and swapping
    # it produces a tensor of the right width with the wrong columns in it, which trains a
    # plausible and wrong model rather than failing.
    FeatureGroup.STATION: ("sm_lat_sta", "sm_lon_sta", "lat_sta", "lon_sta"),
    FeatureGroup.DIRECTION: DIRECTION_PAIR,
    FeatureGroup.IPP: ("lat_ipp", "lon_ipp", "sm_lat_ipp", "sm_lon_ipp"),
    FeatureGroup.SWI: (
        "Kp_index",
        "R_Sunspot_No",
        "Dst-index,_nT",
        "AE-index,_nT",
        "ap_index,_nT",
        "f107_index",
    ),
}


# Scalar groups in tensor order. SWI is absent because it is emitted after the harmonics,
# not with the other scalars - see FeatureLayout.blocks.
SCALAR_GROUP_ORDER: tuple[FeatureGroup, ...] = (
    FeatureGroup.TEMPORAL,
    FeatureGroup.STATION,
    FeatureGroup.DIRECTION,
    FeatureGroup.IPP,
)


class SHConvention(str, Enum):
    """How many spherical-harmonic terms a degree produces.

    The two conventions are a real difference between the STEC models and the Mao et al.
    VTEC baseline, not a bug: the baseline is a replication and has to match the published
    feature count. Naming them stops the choice being re-derived from a model-type
    substring at each site that needs it.
    """

    SQUARED = "squared"  # STEC models: degree**2
    PLUS_ONE_SQUARED = "plus_one_squared"  # Mao et al. VTEC: (degree + 1)**2

    def legendre_polys(self, degree: int) -> int:
        """The `legendre_polys` argument the encoder is constructed with.

        The encoder emits `legendre_polys**2` terms, so this is the single place the two
        conventions differ - everything downstream follows from it.
        """
        return degree + 1 if self is SHConvention.PLUS_ONE_SQUARED else degree

    def terms(self, degree: int) -> int:
        if degree <= 0:
            return 0
        return self.legendre_polys(degree) ** 2


# An expansion applies to a coordinate pair, and only when both members are enabled.
#
# The order here is the tensor order, and it groups by *coordinate system* rather than by
# location: both geographic expansions, then both magnetic ones. That is what the
# collation emits (sh_sta_geo, sh_ipp_geo, sh_sta_sm, sh_ipp_sm) and therefore what every
# trained checkpoint expects.
SH_COORDINATE_PAIRS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("station_geographic", ("lat_sta", "lon_sta")),
    ("ipp_geographic", ("lat_ipp", "lon_ipp")),
    ("station_magnetic", ("sm_lat_sta", "sm_lon_sta")),
    ("ipp_magnetic", ("sm_lat_ipp", "sm_lon_ipp")),
)


@dataclass(frozen=True)
class FeatureBlock:
    """A named, contiguous run of columns in the model's input tensor."""

    name: str
    group: str
    start: int
    width: int
    columns: tuple[str, ...] = ()

    @property
    def stop(self) -> int:
        return self.start + self.width

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


def _temporal_columns(feature: str) -> tuple[str, ...]:
    if feature in CYCLICAL_TEMPORAL:
        return (f"{feature}_sin", f"{feature}_cos", f"{feature}_norm")
    return (f"{feature}_norm",)


@dataclass(frozen=True)
class FeatureLayout:
    """The complete input layout: which features, expanded how, in what order."""

    enabled: frozenset[str]
    sh_degree: int
    sh_convention: SHConvention

    @property
    def sh_terms_per_location(self) -> int:
        return self.sh_convention.terms(self.sh_degree)

    @property
    def sh_locations(self) -> tuple[str, ...]:
        """Coordinate pairs with a spherical-harmonic expansion.

        `sh_degree` is one number for the whole layout, not a per-pair toggle, so degree 0
        means "no harmonics anywhere" - a legitimate, valid configuration (e.g. an ablation
        that keeps lat/lon as plain scalars but drops their harmonic expansion), not an error.
        Both members of a pair being enabled describes the *scalar* features (they are listed
        separately in `GROUP_MEMBERS` and already produce their own blocks above); it is not
        a promise that some other block will be added for them. Gating on
        `sh_terms_per_location` here is what keeps that promise from being made when there is
        nothing to fill it: without this check, `blocks()` would add a named but zero-width
        `sh_*` block, and `FeatureAssembler` would call its (unbuilt, `None`) encoder on it.
        """
        if self.sh_terms_per_location <= 0:
            return ()
        return tuple(
            name
            for name, (first, second) in SH_COORDINATE_PAIRS
            if first in self.enabled and second in self.enabled
        )

    @property
    def sh_width(self) -> int:
        return len(self.sh_locations) * self.sh_terms_per_location

    def blocks(self) -> tuple[FeatureBlock, ...]:
        """Every named block, in tensor order.

        The order is temporal, station, direction, IPP, **spherical harmonics, then SWI**.
        Space weather comes after the harmonics, not before: the collation appends it last,
        so putting it where its group happens to sit in the enum would produce a tensor of
        the right width holding the wrong columns.
        """
        found: list[FeatureBlock] = []
        cursor = 0

        for group in SCALAR_GROUP_ORDER:
            members = [f for f in GROUP_MEMBERS[group] if f in self.enabled]
            if not members:
                continue

            if group is FeatureGroup.DIRECTION and set(DIRECTION_PAIR) <= self.enabled:
                # Both present: one 3-component unit vector, not two scalars.
                found.append(
                    FeatureBlock(
                        "direction", group.value, cursor, 3, DIRECTION_COMPONENTS
                    )
                )
                cursor += 3
                continue

            for feature in members:
                columns = (
                    _temporal_columns(feature)
                    if group is FeatureGroup.TEMPORAL
                    else (f"{feature}_norm",)
                )
                found.append(
                    FeatureBlock(feature, group.value, cursor, len(columns), columns)
                )
                cursor += len(columns)

        width = self.sh_terms_per_location
        for location in self.sh_locations:
            found.append(
                FeatureBlock(f"sh_{location}", "spherical_harmonics", cursor, width)
            )
            cursor += width

        # Space weather is appended after the harmonics, matching the collation.
        for feature in GROUP_MEMBERS[FeatureGroup.SWI]:
            if feature not in self.enabled:
                continue
            found.append(
                FeatureBlock(
                    feature, FeatureGroup.SWI.value, cursor, 1, (f"{feature}_norm",)
                )
            )
            cursor += 1

        return tuple(found)

    @property
    def total_dim(self) -> int:
        """The number the model's input projection must be built with."""
        blocks = self.blocks()
        return blocks[-1].stop if blocks else 0

    def block(self, name: str) -> FeatureBlock:
        for candidate in self.blocks():
            if candidate.name == name:
                return candidate
        raise KeyError(f"no feature block named {name!r} in this layout")

    def group_slice(self, group: FeatureGroup) -> slice:
        """Where a whole group sits. Replaces the Branch models' hardcoded splits."""
        members = [b for b in self.blocks() if b.group == group.value]
        if not members:
            raise KeyError(f"no features enabled in group {group.value!r}")
        return slice(members[0].start, members[-1].stop)

    def describe(self) -> list[dict]:
        """Row per block, for the generated feature table (manuscript Table 1)."""
        return [
            {
                "block": b.name,
                "group": b.group,
                "start": b.start,
                "width": b.width,
                "columns": ", ".join(b.columns),
            }
            for b in self.blocks()
        ]


def convention_for(target: str, distribution: str) -> SHConvention:
    """Which SH convention a run uses.

    Decided from what the run *is* - its target and its likelihood - rather than from a
    substring of its model type, so a model whose name does not contain "Laplacian" but
    which is nonetheless the VTEC baseline gets the right layout.
    """
    if target == "vtec" or distribution == "laplace":
        return SHConvention.PLUS_ONE_SQUARED
    return SHConvention.SQUARED


def layout_from_feature_control(
    feature_control: dict[str, bool],
    sh_degree: int,
    target: str = "stec",
    distribution: str = "gaussian",
) -> FeatureLayout:
    """Build the one layout that both the model and the collation are sized from.

    `feature_control` is the config block that switches individual inputs on and off.
    """
    return FeatureLayout(
        enabled=frozenset(name for name, on in feature_control.items() if on),
        sh_degree=sh_degree,
        sh_convention=convention_for(target, distribution),
    )
