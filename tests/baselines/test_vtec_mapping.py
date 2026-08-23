"""Tests for the VTEC + Mapping baseline port (`stec/baselines/vtec_mapping.py`).

Covers the thin-shell mapping factor as exercised through this module's own
entry point (`MappingFunction` itself is already pinned directly in
`test_gim.py`), the linear uncertainty propagation `apply_mapping_function`
performs in the source, and the Laplace scale-vs-std distinction that
`LaplaceStd` exists to make hard to get wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stec.baselines.vtec_mapping import LaplaceStd, map_vtec_to_stec

# --------------------------------------------------------------------------
# Thin-shell mapping factor, as used through map_vtec_to_stec.
# --------------------------------------------------------------------------


def _expected_slm_factor(
    elevation_deg: float, height_km: float = 450.0, earth_radius_km: float = 6371.0
) -> float:
    """Independently-derived SLM factor: quote of `SLM_MF` in
    `src/evaluation/gim_mapper.py`, re-typed here rather than imported."""
    elevation_rad = math.radians(elevation_deg)
    zenith_at_shell = math.asin(
        earth_radius_km
        / (earth_radius_km + height_km)
        * math.sin(math.pi / 2 - elevation_rad)
    )
    return 1.0 / math.cos(zenith_at_shell)


def _expected_mslm_factor(
    elevation_deg: float,
    height_km: float = 506.7,
    earth_radius_km: float = 6371.0,
    alpha: float = 0.9782,
) -> float:
    """Independently-derived MSLM factor: quote of `MSLM_MF` in
    `src/evaluation/gim_mapper.py`, re-typed here rather than imported."""
    elevation_rad = math.radians(elevation_deg)
    zenith_at_shell = math.asin(
        earth_radius_km
        / (earth_radius_km + height_km)
        * math.sin(alpha * (math.pi / 2 - elevation_rad))
    )
    return 1.0 / math.cos(zenith_at_shell)


@pytest.mark.parametrize("mapping_type", ["SLM", "MSLM"])
def test_mapping_factor_is_one_at_zenith(mapping_type: str) -> None:
    result = map_vtec_to_stec(
        vtec=np.array([15.0]),
        elevation_deg=np.array([90.0]),
        mapping_type=mapping_type,
    )
    assert result.mapping_factor[0] == pytest.approx(1.0, abs=1e-9)
    # At the zenith, mapped STEC must equal the input VTEC exactly.
    assert result.stec[0] == pytest.approx(15.0, abs=1e-9)


@pytest.mark.parametrize("mapping_type", ["SLM", "MSLM"])
def test_mapping_factor_increases_monotonically_as_elevation_drops(
    mapping_type: str,
) -> None:
    elevations = np.array([90.0, 60.0, 30.0, 15.0, 5.0])
    result = map_vtec_to_stec(
        vtec=np.full_like(elevations, 10.0),
        elevation_deg=elevations,
        mapping_type=mapping_type,
    )
    # elevations is strictly descending, so the factor must be strictly ascending.
    assert np.all(np.diff(result.mapping_factor) > 0)
    # Every factor away from the zenith must exceed 1 - a grazing line of
    # sight crosses more ionosphere than a straight-up one.
    assert np.all(result.mapping_factor[1:] > 1.0)


def test_slm_mapping_factor_matches_hand_computation_at_low_elevation() -> None:
    elevation_deg = 12.0
    expected = _expected_slm_factor(elevation_deg)
    result = map_vtec_to_stec(
        vtec=np.array([1.0]),
        elevation_deg=np.array([elevation_deg]),
        mapping_type="SLM",
    )
    assert result.mapping_factor[0] == pytest.approx(expected, rel=1e-9)


def test_mslm_mapping_factor_matches_hand_computation_at_low_elevation() -> None:
    elevation_deg = 12.0
    expected = _expected_mslm_factor(elevation_deg)
    result = map_vtec_to_stec(
        vtec=np.array([1.0]),
        elevation_deg=np.array([elevation_deg]),
        mapping_type="MSLM",
    )
    assert result.mapping_factor[0] == pytest.approx(expected, rel=1e-9)


def test_slm_and_mslm_disagree_at_low_elevation() -> None:
    """The convention actually matters: SLM (H=450 km) and MSLM (H=506.7 km,
    alpha=0.9782) must not be silently interchangeable away from the zenith."""
    elevation_deg = np.array([10.0])
    slm = map_vtec_to_stec(
        vtec=np.array([1.0]), elevation_deg=elevation_deg, mapping_type="SLM"
    )
    mslm = map_vtec_to_stec(
        vtec=np.array([1.0]), elevation_deg=elevation_deg, mapping_type="MSLM"
    )
    assert slm.mapping_factor[0] != pytest.approx(mslm.mapping_factor[0], rel=1e-6)


def test_mapping_type_has_no_default() -> None:
    """Production callers of the source always pass mapping_type="MSLM"
    explicitly - MappingFunction's own default ("SLM") is not what produced
    the paper's numbers, so this wrapper must not silently supply either."""
    with pytest.raises(TypeError):
        map_vtec_to_stec(vtec=np.array([1.0]), elevation_deg=np.array([45.0]))  # type: ignore[call-arg]


# --------------------------------------------------------------------------
# Uncertainty propagation: linear in the mapping factor.
# --------------------------------------------------------------------------


def test_stec_std_is_none_when_no_vtec_std_given() -> None:
    result = map_vtec_to_stec(
        vtec=np.array([10.0]), elevation_deg=np.array([45.0]), mapping_type="MSLM"
    )
    assert result.stec_std is None


def test_std_scales_linearly_by_the_mapping_factor() -> None:
    vtec_std = np.array([2.0, 2.0, 2.0])
    elevations = np.array([90.0, 30.0, 10.0])
    result = map_vtec_to_stec(
        vtec=np.full_like(elevations, 10.0),
        elevation_deg=elevations,
        mapping_type="MSLM",
        vtec_std=vtec_std,
    )
    assert isinstance(result.stec_std, LaplaceStd)
    expected_std = vtec_std * result.mapping_factor
    np.testing.assert_allclose(result.stec_std.std, expected_std)


def test_variance_scales_by_the_mapping_factor_squared() -> None:
    vtec_std = np.array([3.0])
    elevation_deg = np.array([15.0])
    result = map_vtec_to_stec(
        vtec=np.array([10.0]),
        elevation_deg=elevation_deg,
        mapping_type="MSLM",
        vtec_std=vtec_std,
    )
    mapping_factor = result.mapping_factor[0]
    expected_variance = (vtec_std[0] ** 2) * (mapping_factor**2)
    assert result.stec_std.variance()[0] == pytest.approx(expected_variance)


# --------------------------------------------------------------------------
# The Laplace point: scale != std, and there is no path to a silent std.
# --------------------------------------------------------------------------


def test_laplace_variance_is_two_times_scale_squared() -> None:
    laplace = LaplaceStd(std=np.array([4.0, 9.0]))
    np.testing.assert_allclose(laplace.variance(), 2.0 * laplace.scale() ** 2)


def test_laplace_scale_is_std_over_sqrt_two() -> None:
    std = np.array([5.0])
    laplace = LaplaceStd(std=std)
    np.testing.assert_allclose(laplace.scale(), std / math.sqrt(2.0))


def test_laplace_std_has_no_bare_std_accessor_besides_the_named_field() -> None:
    """`.variance()` and `.scale()` are the only ways to read a number out -
    both name the distributional assumption. There must be no `.to_std()` or
    similarly-named shortcut that hands back a Gaussian-shaped sigma."""
    instance = LaplaceStd(std=np.array([1.0]))
    public_attributes = {name for name in dir(instance) if not name.startswith("_")}
    assert public_attributes == {"std", "variance", "scale"}


def test_scale_and_std_are_numerically_different_for_nonzero_spread() -> None:
    """Using std where scale is required (or vice versa) must not coincide -
    otherwise the mistake this module guards against would be invisible."""
    laplace = LaplaceStd(std=np.array([1.0]))
    assert laplace.scale()[0] != pytest.approx(laplace.std[0])
