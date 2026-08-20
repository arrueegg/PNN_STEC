"""Turn raw observation columns into the model's input tensor.

This is the other half of the layout contract: `feature_layout` says which columns exist
and where, and this fills them. Both are driven by the same `FeatureLayout`, so the two can
no longer disagree - which is the defect they replace, where `model.py` sized the input
projection and `collation.py` filled it from independent derivations of the same rule.

Two transforms are why a feature is not one column:

* cyclical quantities become `(sin, cos, norm)`, because a normalised value alone puts
  23:59 and 00:01 at opposite ends of the range;
* azimuth and elevation become a Cartesian unit vector `(up, east, north)`, which is
  continuous across north and encodes the geometry the mapping-function baselines have to
  approximate.

The collator holds no mutable state and mutates nothing it is given. `CollateWithSH` calls
`feature_registry.set_output_indices(...)` on a registry shared by seven call sites, so
what a consumer sees depends on which collator was constructed last.
"""

from __future__ import annotations

import math

import torch

from .feature_layout import (
    CYCLICAL_TEMPORAL,
    DIRECTION_PAIR,
    FeatureGroup,
    FeatureLayout,
    FeatureBlock,
    GROUP_MEMBERS,
)
from .normalization import normalize

TWO_PI = 2 * math.pi


def cyclical_columns(raw: torch.Tensor, feature: str) -> torch.Tensor:
    """`(sin, cos, norm)` for a quantity that wraps.

    The sine and cosine are taken of the *normalised* value scaled to a full turn, which is
    what makes the period the feature's own range rather than 2*pi of raw units.
    """
    norm = normalize(feature, raw).unsqueeze(1)
    return torch.cat([torch.sin(norm * TWO_PI), torch.cos(norm * TWO_PI), norm], dim=1)


def direction_unit_vector(
    azimuth_deg: torch.Tensor, elevation_deg: torch.Tensor
) -> torch.Tensor:
    """`(e_up, e_east, e_north)` from raw degrees - not from normalised values."""
    azimuth = azimuth_deg * math.pi / 180.0
    elevation = elevation_deg * math.pi / 180.0
    return torch.stack(
        [
            torch.sin(elevation),
            torch.cos(elevation) * torch.sin(azimuth),
            torch.cos(elevation) * torch.cos(azimuth),
        ],
        dim=1,
    )


class FeatureAssembler:
    """Builds the model input tensor for a layout, from named raw columns.

    `raw` is a mapping of feature name to a 1-D tensor of that feature's values. Only the
    features the layout enables are read; anything else present is ignored rather than
    silently appended, so a caller cannot widen the tensor by accident.
    """

    def __init__(self, layout: FeatureLayout, sh_encoder=None) -> None:
        self.layout = layout
        self.sh_encoder = sh_encoder
        if layout.sh_width and sh_encoder is None:
            raise ValueError(
                "this layout expects spherical-harmonic columns but no encoder was given"
            )

    def _block_tensor(
        self, block: FeatureBlock, raw: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if (
            block.group == FeatureGroup.TEMPORAL.value
            and block.name in CYCLICAL_TEMPORAL
        ):
            return cyclical_columns(raw[block.name], block.name)
        if block.name == "direction":
            return direction_unit_vector(raw[DIRECTION_PAIR[0]], raw[DIRECTION_PAIR[1]])
        if block.group == "spherical_harmonics":
            location = block.name.removeprefix("sh_")
            pair = SH_PAIR_COLUMNS[location]
            # The encoder takes (longitude, latitude), in that order.
            return self.sh_encoder(torch.stack([raw[pair[1]], raw[pair[0]]], dim=1))
        return normalize(block.name, raw[block.name]).unsqueeze(1)

    def required_columns(self) -> tuple[str, ...]:
        """Raw column names this layout needs, so a caller can check before loading."""
        needed: list[str] = []
        for block in self.layout.blocks():
            if block.name == "direction":
                needed.extend(DIRECTION_PAIR)
            elif block.group == "spherical_harmonics":
                needed.extend(SH_PAIR_COLUMNS[block.name.removeprefix("sh_")])
            else:
                needed.append(block.name)
        return tuple(dict.fromkeys(needed))

    def assemble(self, raw: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [name for name in self.required_columns() if name not in raw]
        if missing:
            raise KeyError(f"missing raw columns for this layout: {missing}")

        pieces = [self._block_tensor(block, raw) for block in self.layout.blocks()]
        assembled = torch.cat(pieces, dim=1)

        # The layout is the contract; if assembly disagrees with it, that is the exact
        # failure this design exists to prevent, so it must not pass silently.
        if assembled.shape[1] != self.layout.total_dim:
            raise RuntimeError(
                f"assembled {assembled.shape[1]} columns but the layout declares "
                f"{self.layout.total_dim}"
            )
        return assembled


# Which raw columns feed each spherical-harmonic expansion, as (latitude, longitude).
SH_PAIR_COLUMNS: dict[str, tuple[str, str]] = {
    "station_geographic": ("lat_sta", "lon_sta"),
    "station_magnetic": ("sm_lat_sta", "sm_lon_sta"),
    "ipp_geographic": ("lat_ipp", "lon_ipp"),
    "ipp_magnetic": ("sm_lat_ipp", "sm_lon_ipp"),
}


def group_members(group: FeatureGroup) -> tuple[str, ...]:
    return GROUP_MEMBERS[group]
