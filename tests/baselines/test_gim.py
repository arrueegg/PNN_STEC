"""Pin the three defects fixed while porting `src/evaluation/gim_mapper.py`.

1. `map_vtec_to_stec` was defined twice; the dead first definition returned a
   list of file paths instead of STEC values. `test_only_one_...` and
   `test_map_vtec_to_stec_returns_numeric_stec_not_a_file_list` pin this.
2. `GIMMapper(gim_path)` used to bind a path string to `shell_height_km`
   because it was the first positional parameter. `test_constructor_*` pin
   that the ported constructor has no positional slot for it at all.
3. Truncating `int(doy)` on a denormalised model-input `doy` silently loaded
   the previous day's IONEX map (DOY 189 -> 188.99998 -> `int()` -> 188).
   `test_doy_*` pin that `date_from_year_doy` rounds instead.
"""

from __future__ import annotations

import inspect
import math
from datetime import datetime

import numpy as np
import pytest

from stec.baselines.gim import (
    DEFAULT_IONEX_ROOT,
    GIMMapper,
    MappingFunction,
    date_from_year_doy,
)
from stec.config import paths

# --------------------------------------------------------------------------
# Defect 3: day-lookup truncation (round(), never int())
# --------------------------------------------------------------------------


def test_doy_just_below_the_integer_resolves_to_the_next_day():
    """This is the exact float that DOY 189 round-trips to through the model's
    normalise/denormalise-in-float32 path; it must not resolve to DOY 188."""
    date = date_from_year_doy(2024, 188.99998)
    assert date.timetuple().tm_yday == 189


def test_doy_just_above_the_integer_stays_on_that_day():
    date = date_from_year_doy(2024, 189.00001)
    assert date.timetuple().tm_yday == 189


def test_an_exact_integer_doy_is_unaffected():
    date = date_from_year_doy(2024, 1)
    assert (date.year, date.timetuple().tm_yday) == (2024, 1)


def test_a_truncating_cast_would_have_given_the_wrong_answer():
    """Documents the bug being fixed: int() on the same input picks DOY 188."""
    assert int(188.99998) == 188
    assert date_from_year_doy(2024, 188.99998).timetuple().tm_yday != int(188.99998)


def test_year_is_rounded_the_same_way_as_doy():
    date = date_from_year_doy(2023.99997, 1)
    assert date.year == 2024


def test_load_for_year_doy_routes_through_the_rounding_helper(monkeypatch):
    """`load_for_year_doy` must not re-derive the date with a truncating cast
    of its own - it should hand the already-rounded date to `load_gim_data`."""
    mapper = GIMMapper()
    captured: dict[str, object] = {}

    def fake_load_gim_data(date, *, ionex_root=None):
        captured["date"] = date
        captured["ionex_root"] = ionex_root

    monkeypatch.setattr(mapper, "load_gim_data", fake_load_gim_data)
    mapper.load_for_year_doy(2024, 188.99998)

    assert captured["date"].timetuple().tm_yday == 189


# --------------------------------------------------------------------------
# Defect 2: constructor cannot silently accept a path as a physical constant
# --------------------------------------------------------------------------


def test_constructor_has_no_positional_slot_after_self():
    """The exact call that shipped the original bug: `GIMMapper(gim_path)`."""
    with pytest.raises(TypeError):
        GIMMapper("/home/space/data/iono/GIM_IONEX")  # type: ignore[misc]


def test_constructor_rejects_a_string_masquerading_as_shell_height():
    with pytest.raises(TypeError):
        GIMMapper(shell_height_km="/home/space/data/iono/GIM_IONEX")


def test_constructor_rejects_a_string_masquerading_as_earth_radius():
    with pytest.raises(TypeError):
        GIMMapper(earth_radius_km="/home/space/data/iono/GIM_IONEX")


def test_constructor_rejects_a_bool_even_though_bool_is_technically_an_int():
    # isinstance(True, numbers.Real) is True in Python, so this is the case the
    # validation has to check for explicitly rather than trusting isinstance alone.
    with pytest.raises(TypeError):
        GIMMapper(shell_height_km=True)


def test_constructor_accepts_valid_keyword_arguments():
    mapper = GIMMapper(
        shell_height_km=450.0,
        earth_radius_km=6371.0,
        mapping_type="MSLM",
        gim_type="CODE",
    )
    assert mapper.shell_height_km == 450.0
    assert mapper.gim_type == "CODE"


def test_constructor_has_no_ionex_root_or_path_parameter():
    """The IONEX root belongs on the loader, keyword-only, never on the
    constructor next to the physical constants."""
    params = inspect.signature(GIMMapper.__init__).parameters
    assert all(name not in params for name in ("gim_path", "ionex_root", "path"))
    # Every parameter besides `self` must be keyword-only.
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY
        for name, p in params.items()
        if name != "self"
    )


def test_ionex_root_defaults_to_the_shared_path_config():
    assert DEFAULT_IONEX_ROOT == paths.GIM_IONEX_ROOT


def test_load_gim_data_ionex_root_is_keyword_only():
    params = inspect.signature(GIMMapper.load_gim_data).parameters
    assert params["ionex_root"].kind is inspect.Parameter.KEYWORD_ONLY


# --------------------------------------------------------------------------
# Defect 1: only the real map_vtec_to_stec survives the port
# --------------------------------------------------------------------------


def test_only_one_map_vtec_to_stec_is_defined():
    source = inspect.getsource(GIMMapper)
    assert source.count("def map_vtec_to_stec") == 1


def _mapper_with_synthetic_constant_gim(vtec_value: float = 20.0) -> GIMMapper:
    """A mapper with hand-built GIM data, bypassing file I/O entirely so the
    numeric tests do not depend on real IONEX files being present."""
    mapper = GIMMapper()
    lat_grid = np.arange(-90.0, 90.1, 5.0)
    lon_grid = np.arange(-180.0, 180.1, 5.0)
    # 13 epochs at a 2-hour interval, matching a real IGS daily file.
    epochs = [datetime(2024, 1, 1, h) for h in range(0, 24, 2)] + [
        datetime(2024, 1, 2, 0)
    ]
    vtec_maps = [np.full((len(lat_grid), len(lon_grid)), vtec_value) for _ in epochs]
    mapper.gim_data = {
        "epochs": epochs,
        "vtec_maps": vtec_maps,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
    }
    return mapper


def test_map_vtec_to_stec_returns_numeric_stec_not_a_file_list():
    mapper = _mapper_with_synthetic_constant_gim(vtec_value=20.0)
    stec = mapper.map_vtec_to_stec(
        sods=np.array([3600.0]),
        ipp_lat=np.array([10.0]),
        ipp_lon=np.array([20.0]),
        elevations=np.array([90.0]),
    )
    assert isinstance(stec, np.ndarray)
    assert stec.dtype.kind == "f"
    # At the zenith the mapping factor is 1, so slant TEC equals the constant VTEC.
    assert np.isclose(stec[0], 20.0, atol=1e-6)


def test_map_vtec_to_stec_raises_without_loaded_data():
    mapper = GIMMapper()
    with pytest.raises(ValueError, match="No GIM data loaded"):
        mapper.map_vtec_to_stec(
            sods=np.array([0.0]),
            ipp_lat=np.array([0.0]),
            ipp_lon=np.array([0.0]),
            elevations=np.array([90.0]),
        )


# --------------------------------------------------------------------------
# Mapping-function maths: unchanged from the source, checked against an
# independently written formula rather than the implementation itself.
# --------------------------------------------------------------------------


def _expected_slm_factor(
    elevation_deg: float, height_km: float = 450.0, earth_radius_km: float = 6371.0
) -> float:
    elevation_rad = math.radians(elevation_deg)
    zenith_at_shell = math.asin(
        earth_radius_km
        / (earth_radius_km + height_km)
        * math.sin(math.pi / 2 - elevation_rad)
    )
    return 1.0 / math.cos(zenith_at_shell)


def test_slm_mapping_factor_is_one_at_the_zenith():
    factor = MappingFunction("SLM").get_mapping_factor(np.radians(np.array([90.0])))[0]
    assert factor == pytest.approx(1.0, abs=1e-9)


def test_slm_mapping_factor_at_low_elevation_matches_hand_computation():
    elevation_deg = 10.0
    expected = _expected_slm_factor(elevation_deg)
    actual = MappingFunction("SLM").get_mapping_factor(
        np.radians(np.array([elevation_deg]))
    )[0]
    assert actual == pytest.approx(expected, rel=1e-9)
    # Sanity check on the physics: a grazing line of sight crosses far more
    # ionosphere than a straight-up one, so the factor must exceed 1.
    assert actual > 1.0


def test_slm_mapping_factor_increases_as_elevation_drops():
    factors = MappingFunction("SLM").get_mapping_factor(
        np.radians(np.array([90.0, 45.0, 10.0, 5.0]))
    )
    assert np.all(np.diff(factors) > 0)


def test_mslm_uses_its_own_shell_height_and_alpha():
    elevation_deg = 10.0
    slm = MappingFunction("SLM").get_mapping_factor(
        np.radians(np.array([elevation_deg]))
    )[0]
    mslm = MappingFunction("MSLM").get_mapping_factor(
        np.radians(np.array([elevation_deg]))
    )[0]
    # Different height (506.7 vs 450 km) and the alpha scaling mean the two
    # models must not coincide away from the zenith.
    assert slm != pytest.approx(mslm, rel=1e-6)


def test_unknown_mapping_type_falls_back_to_slm():
    elevation_deg = 30.0
    default = MappingFunction("bogus").get_mapping_factor(
        np.radians(np.array([elevation_deg]))
    )[0]
    slm = MappingFunction("SLM").get_mapping_factor(
        np.radians(np.array([elevation_deg]))
    )[0]
    assert default == pytest.approx(slm)


# --------------------------------------------------------------------------
# Integration test against real IONEX data, skipped if unavailable.
# --------------------------------------------------------------------------

_SAMPLE_IONEX_FILE = paths.GIM_IONEX_ROOT / "2024" / "igsg0010.24i"


@pytest.mark.skipif(
    not _SAMPLE_IONEX_FILE.exists(), reason="Real IONEX data not present on this host"
)
def test_loading_a_real_ionex_file_and_mapping_a_value():
    mapper = GIMMapper(mapping_type="SLM", gim_type="IGS")
    mapper.load_gim_data(datetime(2024, 1, 1), ionex_root=paths.GIM_IONEX_ROOT)

    assert len(mapper.gim_data["vtec_maps"]) > 0
    assert len(mapper.gim_data["epochs"]) > 0

    stec = mapper.map_vtec_to_stec(
        sods=np.array([0.0, 43200.0]),
        ipp_lat=np.array([45.0, -20.0]),
        ipp_lon=np.array([10.0, 100.0]),
        elevations=np.array([90.0, 30.0]),
    )
    assert stec.shape == (2,)
    assert np.all(np.isfinite(stec))
    # Global VTEC is essentially always within this band; STEC at 90 deg
    # elevation equals VTEC exactly, so this also bounds the raw grid values.
    assert np.all((stec >= 0.0) & (stec < 300.0))
