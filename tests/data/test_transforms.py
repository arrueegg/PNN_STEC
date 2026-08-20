"""The transforms that fill the layout, and their agreement with the legacy collation.

`test_assembled_tensor_matches_the_legacy_collation` is the load-bearing one: it is the
other half of Gate A. The layout half proved the rebuilt code computes the same *width*;
this proves it puts the same *values* in the same *order*. A layout that agreed on width
while permuting columns would train a plausible, wrong model.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import pytest
import torch

from stec.data.feature_layout import SHConvention, layout_from_feature_control
from stec.data.transforms import (
    FeatureAssembler,
    cyclical_columns,
    direction_unit_vector,
)
from stec.data.normalization import denormalize, normalize

PAPER_FEATURE_CONTROL = {
    "year": True,
    "doy": True,
    "sod": True,
    "local_time_hours": True,
    "lat_sta": True,
    "lon_sta": True,
    "sm_lat_sta": True,
    "sm_lon_sta": True,
    "satazi": True,
    "satele": True,
    "lat_ipp": True,
    "lon_ipp": True,
    "sm_lat_ipp": True,
    "sm_lon_ipp": True,
    "Kp_index": True,
    "R_Sunspot_No": True,
    "Dst-index,_nT": True,
    "AE-index,_nT": True,
    "ap_index,_nT": True,
    "f107_index": True,
}

RAW_RANGES = {
    "year": (2014, 2024),
    "doy": (1, 366),
    "sod": (0, 86399),
    "local_time_hours": (0, 24),
    "lat_sta": (-80, 80),
    "lon_sta": (-179, 179),
    "sm_lat_sta": (-80, 80),
    "sm_lon_sta": (-179, 179),
    "satazi": (0, 359),
    "satele": (5, 89),
    "lat_ipp": (-80, 80),
    "lon_ipp": (-179, 179),
    "sm_lat_ipp": (-80, 80),
    "sm_lon_ipp": (-179, 179),
    "Kp_index": (0, 90),
    "R_Sunspot_No": (0, 250),
    "Dst-index,_nT": (-300, 50),
    "AE-index,_nT": (0, 2000),
    "ap_index,_nT": (0, 200),
    "f107_index": (65, 400),
}


def raw_batch(rows: int = 64, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    batch = {}
    for name, (low, high) in RAW_RANGES.items():
        span = torch.rand(rows, generator=generator, dtype=torch.float64)
        batch[name] = (low + span * (high - low)).to(torch.float32)
    return batch


def sh_encoder_for(layout):
    from utils.locationencoder.pe import SphericalHarmonics  # noqa: PLC0415

    return SphericalHarmonics(
        legendre_polys=layout.sh_convention.legendre_polys(layout.sh_degree)
    )


# --- unit level -----------------------------------------------------------------------


def test_cyclical_emits_sin_cos_norm_in_that_order():
    raw = torch.tensor([0.0, 43200.0, 86400.0])
    out = cyclical_columns(raw, "sod")
    assert out.shape == (3, 3)
    norm = normalize("sod", raw)
    assert torch.allclose(out[:, 2], norm)
    assert torch.allclose(out[:, 0], torch.sin(norm * 2 * math.pi), atol=1e-6)
    assert torch.allclose(out[:, 1], torch.cos(norm * 2 * math.pi), atol=1e-6)


def test_cyclical_wraps_so_midnight_is_continuous():
    """A normalised value alone puts 23:59 and 00:01 at opposite ends of the range."""
    just_before = cyclical_columns(torch.tensor([86399.0]), "sod")[:, :2]
    just_after = cyclical_columns(torch.tensor([1.0]), "sod")[:, :2]
    # One second either side of midnight is 2*pi/86400 of a turn apart, so the sine and
    # cosine differ by ~1.5e-4. The point is that they are adjacent, not that they match.
    assert torch.allclose(just_before, just_after, atol=1e-3)
    # The normalised column, by contrast, is at opposite ends of its range.
    assert (
        abs(
            float(cyclical_columns(torch.tensor([86399.0]), "sod")[0, 2])
            - float(cyclical_columns(torch.tensor([1.0]), "sod")[0, 2])
        )
        > 0.99
    )


def test_direction_is_a_unit_vector():
    azimuth = torch.tensor([0.0, 90.0, 180.0, 270.0])
    elevation = torch.tensor([10.0, 45.0, 60.0, 89.0])
    vector = direction_unit_vector(azimuth, elevation)
    assert vector.shape == (4, 3)
    assert torch.allclose(vector.norm(dim=1), torch.ones(4), atol=1e-6)


def test_direction_at_zenith_points_straight_up():
    vector = direction_unit_vector(torch.tensor([123.0]), torch.tensor([90.0]))
    assert pytest.approx(float(vector[0, 0]), abs=1e-6) == 1.0
    assert pytest.approx(float(vector[0, 1]), abs=1e-6) == 0.0
    assert pytest.approx(float(vector[0, 2]), abs=1e-6) == 0.0


def test_direction_north_and_east_are_where_they_should_be():
    at_horizon_north = direction_unit_vector(torch.tensor([0.0]), torch.tensor([0.0]))
    assert pytest.approx(float(at_horizon_north[0, 2]), abs=1e-6) == 1.0  # e_north
    at_horizon_east = direction_unit_vector(torch.tensor([90.0]), torch.tensor([0.0]))
    assert pytest.approx(float(at_horizon_east[0, 1]), abs=1e-6) == 1.0  # e_east


def test_denormalize_inverts_normalize():
    for feature in ("doy", "satele", "Dst-index,_nT"):
        low, high = RAW_RANGES[feature]
        raw = torch.linspace(low, high, 16)
        assert torch.allclose(
            denormalize(feature, normalize(feature, raw)), raw, atol=1e-3
        )


def test_denormalised_doy_needs_rounding_not_truncation():
    """DOY 189 comes back as 188.99998; int() silently shifts it to the previous day."""
    recovered = denormalize(
        "doy", normalize("doy", torch.tensor([189.0], dtype=torch.float32))
    )
    value = float(recovered[0])
    assert int(value) in (188, 189)  # may truncate low - that is the defect
    assert round(value) == 189


# --- assembly -------------------------------------------------------------------------


def test_assembly_width_matches_the_layout():
    layout = layout_from_feature_control(PAPER_FEATURE_CONTROL, sh_degree=5)
    assembler = FeatureAssembler(layout, sh_encoder=sh_encoder_for(layout))
    assembled = assembler.assemble(raw_batch())
    assert assembled.shape == (64, 127)


def test_missing_raw_column_is_an_error_not_a_zero():
    layout = layout_from_feature_control(PAPER_FEATURE_CONTROL, sh_degree=5)
    assembler = FeatureAssembler(layout, sh_encoder=sh_encoder_for(layout))
    batch = raw_batch()
    del batch["satele"]
    with pytest.raises(KeyError, match="missing raw columns"):
        assembler.assemble(batch)


def test_extra_raw_columns_are_ignored_not_appended():
    """A caller must not be able to widen the tensor by passing more data."""
    layout = layout_from_feature_control(PAPER_FEATURE_CONTROL, sh_degree=5)
    assembler = FeatureAssembler(layout, sh_encoder=sh_encoder_for(layout))
    batch = raw_batch()
    batch["something_else"] = torch.zeros(64)
    assert assembler.assemble(batch).shape[1] == 127


def test_a_layout_needing_harmonics_refuses_to_build_without_an_encoder():
    layout = layout_from_feature_control(PAPER_FEATURE_CONTROL, sh_degree=5)
    with pytest.raises(ValueError, match="no encoder"):
        FeatureAssembler(layout, sh_encoder=None)


def test_legendre_polys_follows_the_convention():
    assert SHConvention.SQUARED.legendre_polys(5) == 5
    assert SHConvention.PLUS_ONE_SQUARED.legendre_polys(15) == 16
    assert SHConvention.PLUS_ONE_SQUARED.terms(15) == 16**2


# --- Gate A, other half ----------------------------------------------------------------


LEGACY_SRC = "/scratch2/arrueegg/WP4/PNN_STEC/src"


def legacy_available() -> bool:
    """The pre-rebuild tree, which a clean clone will not have."""
    if LEGACY_SRC not in sys.path:
        sys.path.insert(0, LEGACY_SRC)
    return importlib.util.find_spec("data_loader.collation") is not None


@pytest.mark.skipif(
    not legacy_available(), reason="pre-rebuild source tree not available"
)
def test_assembled_tensor_matches_the_legacy_collation():
    """Gate A, values half: same columns, same order, same numbers as the old code.

    Builds the legacy CollateWithSH from an equivalent config and compares its output
    tensor against the rebuilt assembler's, element for element.
    """
    from data_loader.collation import CollateWithSH  # noqa: PLC0415
    from utils.feature_registry import initialize_feature_registry  # noqa: PLC0415

    config = {
        "target": "stec",
        "model": {"model_type": "BayesianResNetSTEC"},
        "data": {"SH_degree": 5, "use_SWI": True},
        "feature_control": dict(PAPER_FEATURE_CONTROL),
    }
    registry = initialize_feature_registry(config)
    legacy = CollateWithSH(config)

    batch = raw_batch(rows=32, seed=3)

    # The legacy collator indexes its input tensor by position, so each feature must be
    # placed at the column its registry assigns it. Stacking in dict-key order instead
    # silently feeds every transform the wrong column.
    if not hasattr(legacy, "input_indices"):
        pytest.skip("legacy collator does not expose its input ordering")
    indices: dict[str, int] = legacy.input_indices
    missing = [name for name in indices if name not in batch]
    if missing:
        pytest.skip(f"fixture does not cover legacy input columns: {missing}")

    rows = next(iter(batch.values())).shape[0]
    raw_matrix = torch.zeros(rows, max(indices.values()) + 1)
    for name, column in indices.items():
        raw_matrix[:, column] = batch[name]
    # Reproduce the legacy concatenation exactly, including that space weather is appended
    # *after* the spherical harmonics rather than with the other scalars.
    sh_sta_geo, sh_ipp_geo, sh_sta_sm, sh_ipp_sm = legacy.compute_sh_embeddings(
        raw_matrix
    )
    legacy_pieces = [
        legacy.transform_temporal(raw_matrix),
        legacy.transform_station(raw_matrix),
        legacy.transform_direction(raw_matrix),
        legacy.transform_ipp(raw_matrix),
        sh_sta_geo,
        sh_ipp_geo,
        sh_sta_sm,
        sh_ipp_sm,
        legacy.transform_swi(raw_matrix),
    ]
    legacy_out = torch.cat([p for p in legacy_pieces if p is not None], dim=1)

    layout = layout_from_feature_control(PAPER_FEATURE_CONTROL, sh_degree=5)
    assembler = FeatureAssembler(layout, sh_encoder=sh_encoder_for(layout))
    rebuilt = assembler.assemble(batch)

    assert rebuilt.shape == legacy_out.shape, (
        f"rebuilt {tuple(rebuilt.shape)} vs legacy {tuple(legacy_out.shape)}"
    )
    difference = (rebuilt - legacy_out).abs()
    if not torch.allclose(rebuilt, legacy_out, atol=1e-6):
        columns = [b.name for b in layout.blocks()]
        worst = int(difference.max(dim=0).values.argmax())
        raise AssertionError(
            f"max difference {float(difference.max()):.3e} at column {worst}; "
            f"differing columns: {(difference.max(dim=0).values > 1e-6).nonzero().flatten().tolist()[:12]} "
            f"(blocks: {columns})"
        )
    assert registry is not None
