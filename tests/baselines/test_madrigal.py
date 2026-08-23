"""Pin the defect fixed while porting `src/evaluation/madrigal_loader.py`.

`extract_stec_for_date` joined our observations to Madrigal on exact
equality of rounded integer keys (latitude/longitude at 0.001-degree bins,
second-of-day, elevation and azimuth), with no tolerance and no reporting of
how many rows the join dropped. `test_a_pair_0_001_degrees_apart_...` pins
that `match_exact_key` still drops such a pair (it is ported unmodified,
because it produced the published numbers), and that `match_nearest` with an
explicit tolerance keeps it. `test_zero_tolerance_reproduces_...` pins that
"no tolerance" is one specific value of the new parameter, not a different
code path.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from stec.baselines.madrigal import (
    MadrigalMatchResult,
    find_madrigal_file,
    load_madrigal_table,
    match_exact_key,
    match_nearest,
)
from stec.config import paths

# --------------------------------------------------------------------------
# Shared synthetic fixtures. `_madrigal_frame` mimics what `load_madrigal_table`
# returns; `_obs_frame` mimics one day of our own test-set observations.
# --------------------------------------------------------------------------


def _madrigal_frame(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "station": "ALGO",
        "lat_sta": 45.0,
        "lon_sta": -78.0,
        "sod": 3600,
        "satele": 30.0,
        "satazi": 100.0,
        "los_tec": 12.34,
        "gnss_type": "GPS",
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def _obs_frame(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "station": "ALGO",
        "lat_sta": 45.0,
        "lon_sta": -78.0,
        "sod": 3600,
        "satele": 30.0,
        "satazi": 100.0,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


# --------------------------------------------------------------------------
# The defect: exact-key join has no tolerance, and drops a pair that differs
# by exactly the resolution rounding is supposed to capture.
# --------------------------------------------------------------------------


def test_a_pair_0_001_degrees_apart_straddles_an_exact_key_bin_boundary():
    """Construct a lat value exactly on a 0.001-degree bin boundary so the
    two points round to different integer keys deterministically, rather
    than relying on luck. 10.0005 * 1000 = 10500.5, which rounds (banker's
    rounding, ties to even) to 10500; adding 0.001 moves to 10.0015 * 1000
    = 10501.5, which rounds to 10502 - two bins away despite a 0.001-degree
    true separation."""
    obs = _obs_frame([{"lat_sta": 10.0005}])
    mad = _madrigal_frame([{"lat_sta": 10.0015}])  # 0.001 deg away

    exact_result = match_exact_key(obs, mad)
    assert exact_result.n_matched == 0
    assert np.isnan(exact_result.stec[0])

    tolerant_result = match_nearest(obs, mad, lat_lon_tolerance_deg=0.002)
    assert tolerant_result.n_matched == 1
    assert tolerant_result.stec[0] == pytest.approx(12.34)


def test_exact_key_matches_when_rounded_bins_happen_to_coincide():
    """Sanity check on the fixture above: a pair that rounds to the *same*
    bin does match under the legacy algorithm, so the failure above is really
    about the boundary, not about the join being broken outright."""
    obs = _obs_frame([{"lat_sta": 10.0001}])
    mad = _madrigal_frame([{"lat_sta": 10.0002}])
    result = match_exact_key(obs, mad)
    assert result.n_matched == 1


# --------------------------------------------------------------------------
# Match-rate reporting
# --------------------------------------------------------------------------


def test_match_rate_is_reported_and_correct_on_a_synthetic_frame():
    # Three observations: two share a Madrigal counterpart's key exactly,
    # one is 1 second off in sod and therefore unmatchable under exact keys.
    obs = _obs_frame(
        [
            {"sod": 100, "lat_sta": 1.0},
            {"sod": 200, "lat_sta": 2.0},
            {"sod": 301, "lat_sta": 3.0},  # sod off by 1s from the row below
        ]
    )
    mad = _madrigal_frame(
        [
            {"sod": 100, "lat_sta": 1.0, "los_tec": 10.0},
            {"sod": 200, "lat_sta": 2.0, "los_tec": 20.0},
            {"sod": 300, "lat_sta": 3.0, "los_tec": 30.0},
        ]
    )

    result = match_exact_key(obs, mad)
    assert isinstance(result, MadrigalMatchResult)
    assert result.n_observations == 3
    assert result.n_matched == 2
    assert result.match_rate == pytest.approx(2 / 3)
    assert list(result.matched) == [True, True, False]
    assert result.stec[0] == pytest.approx(10.0)
    assert result.stec[1] == pytest.approx(20.0)
    assert np.isnan(result.stec[2])


def test_match_rate_is_zero_when_nothing_matches():
    obs = _obs_frame([{"sod": 1}, {"sod": 2}])
    mad = _madrigal_frame([{"sod": 999}])
    result = match_exact_key(obs, mad)
    assert result.n_matched == 0
    assert result.match_rate == pytest.approx(0.0)
    assert result.n_observations == 2


def test_match_rate_is_nan_for_an_empty_observation_frame():
    obs = pl.DataFrame(
        schema={
            "station": pl.Utf8,
            "lat_sta": pl.Float64,
            "lon_sta": pl.Float64,
            "sod": pl.Int64,
        }
    )
    mad = _madrigal_frame([{"sod": 1}])
    result = match_exact_key(obs, mad)
    assert result.n_observations == 0
    assert np.isnan(result.match_rate)


# --------------------------------------------------------------------------
# Station-name case normalisation
# --------------------------------------------------------------------------


def test_load_madrigal_table_uppercases_station_names(tmp_path):
    """`gps_site` arrives lowercase from Madrigal; without upper-casing on
    load, a station-identity join against our own (uppercase) test set would
    fail outright."""
    import h5py

    h5path = tmp_path / "los_20240101_IGS.h5"
    dtype = np.dtype(
        [
            ("gps_site", "S4"),
            ("sat_id", "<i8"),
            ("gnss_type", "S8"),
            ("gdlatr", "<f8"),
            ("gdlonr", "<f8"),
            ("los_tec", "<f8"),
            ("dlos_tec", "<f8"),
            ("tec", "<f8"),
            ("azm", "<f8"),
            ("elm", "<f8"),
            ("gdlat", "<f8"),
            ("glon", "<f8"),
            ("rec_bias", "<f8"),
            ("sod", "<u4"),
        ]
    )
    row = np.array(
        [
            (
                b"algo",
                1,
                b"GPS     ",
                45.0,
                -78.0,
                12.34,
                0.1,
                12.3,
                100.0,
                30.0,
                44.9,
                -77.9,
                0.0,
                3600,
            )
        ],
        dtype=dtype,
    )
    with h5py.File(h5path, "w") as h5f:
        group = h5f.create_group("Data")
        group.create_dataset("Table Layout", data=row)

    table = load_madrigal_table(h5path)
    assert table["station"].to_list() == ["ALGO"]


def test_station_case_normalisation_makes_an_otherwise_failing_join_succeed():
    """`require_station_match` compares upper-cased names; without that
    normalisation, our own uppercase 'ALGO' would never equal Madrigal's raw
    lowercase 'algo' and every station-gated match would fail."""
    obs = pl.DataFrame(
        [
            {
                "station": "ALGO",
                "lat_sta": 10.0,
                "lon_sta": -78.0,
                "sod": 100,
                "satele": 30.0,
                "satazi": 100.0,
            }
        ]
    )
    # Simulate an *unnormalised* Madrigal frame (as if load_madrigal_table's
    # upper-casing had not run) to prove normalisation is what makes this work.
    mad_unnormalised = pl.DataFrame(
        [
            {
                "station": "algo",
                "lat_sta": 10.0005,
                "lon_sta": -78.0,
                "sod": 100,
                "satele": 30.0,
                "satazi": 100.0,
                "los_tec": 12.34,
            }
        ]
    )
    result = match_nearest(
        obs, mad_unnormalised, lat_lon_tolerance_deg=0.01, require_station_match=True
    )
    assert result.n_matched == 0  # case mismatch defeats the station filter

    mad_normalised = mad_unnormalised.with_columns(pl.col("station").str.to_uppercase())
    result = match_nearest(
        obs, mad_normalised, lat_lon_tolerance_deg=0.01, require_station_match=True
    )
    assert result.n_matched == 1


def test_require_station_match_rejects_a_close_but_different_station():
    obs = pl.DataFrame(
        [
            {
                "station": "ALGO",
                "lat_sta": 10.0,
                "lon_sta": -78.0,
                "sod": 100,
                "satele": 30.0,
                "satazi": 100.0,
            }
        ]
    )
    mad = pl.DataFrame(
        [
            {
                "station": "OTHR",
                "lat_sta": 10.0005,
                "lon_sta": -78.0,
                "sod": 100,
                "satele": 30.0,
                "satazi": 100.0,
                "los_tec": 99.0,
            }
        ]
    )
    result = match_nearest(
        obs, mad, lat_lon_tolerance_deg=0.01, require_station_match=True
    )
    assert result.n_matched == 0


# --------------------------------------------------------------------------
# Tolerance of 0 reproduces exact-key behaviour exactly.
# --------------------------------------------------------------------------


def test_zero_tolerance_reproduces_exact_key_behavior_exactly():
    # A mix: an exact match, a same-bin near-match, and a boundary-straddling
    # pair that exact-key drops - zero tolerance must agree with exact-key on
    # every one of these, not just the easy cases.
    obs = _obs_frame(
        [
            {"sod": 100, "lat_sta": 1.00000},
            {"sod": 200, "lat_sta": 10.0005},
            {"sod": 300, "lat_sta": 3.00000},
        ]
    )
    mad = _madrigal_frame(
        [
            {"sod": 100, "lat_sta": 1.00000, "los_tec": 10.0},
            {"sod": 200, "lat_sta": 10.0015, "los_tec": 20.0},  # boundary-straddling
            {"sod": 999, "lat_sta": 3.00000, "los_tec": 30.0},  # sod mismatch
        ]
    )

    exact = match_exact_key(obs, mad)
    zero_tolerance = match_nearest(obs, mad, lat_lon_tolerance_deg=0.0)

    assert zero_tolerance.n_matched == exact.n_matched
    assert list(zero_tolerance.matched) == list(exact.matched)
    np.testing.assert_array_equal(zero_tolerance.stec, exact.stec)


def test_negative_tolerance_is_rejected():
    obs = _obs_frame([{}])
    mad = _madrigal_frame([{}])
    with pytest.raises(ValueError, match="lat_lon_tolerance_deg"):
        match_nearest(obs, mad, lat_lon_tolerance_deg=-0.001)


# --------------------------------------------------------------------------
# match_nearest picks the closest candidate when more than one is in range.
# --------------------------------------------------------------------------


def test_nearest_match_picks_the_closest_candidate_within_tolerance():
    obs = _obs_frame([{"lat_sta": 10.000}])
    mad = _madrigal_frame(
        [
            {"lat_sta": 10.004, "los_tec": 40.0},  # farther
            {"lat_sta": 10.002, "los_tec": 20.0},  # closer
        ]
    )
    result = match_nearest(obs, mad, lat_lon_tolerance_deg=0.01)
    assert result.n_matched == 1
    assert result.stec[0] == pytest.approx(20.0)


def test_no_satellite_identity_columns_are_produced_for_madrigal():
    """Madrigal has no satellite identity; `load_madrigal_table`'s output
    must not invent sat/slipc/gfphase placeholders for it."""
    mad = _madrigal_frame([{}])
    assert "sat" not in mad.columns
    assert "slipc" not in mad.columns
    assert "gfphase" not in mad.columns


# --------------------------------------------------------------------------
# find_madrigal_file
# --------------------------------------------------------------------------


def test_find_madrigal_file_returns_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "MADRIGAL_ROOT", tmp_path)
    assert find_madrigal_file(date(2099, 1, 1)) is None


def test_find_madrigal_file_finds_an_existing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "MADRIGAL_ROOT", tmp_path)
    year_dir = tmp_path / "2024"
    year_dir.mkdir()
    expected = year_dir / "los_20240101_IGS.h5"
    expected.touch()
    # find_madrigal_file routes through paths.madrigal_day, which builds the
    # path from MADRIGAL_ROOT directly rather than caching it at import time.
    found = find_madrigal_file(date(2024, 1, 1))
    assert found == paths.madrigal_day(2024, 1, 1)


# --------------------------------------------------------------------------
# Integration test against a real Madrigal file, skipped if unavailable.
# --------------------------------------------------------------------------

_SAMPLE_MADRIGAL_FILE = paths.MADRIGAL_ROOT / "2024" / "los_20240101_IGS.h5"


@pytest.mark.skipif(
    not _SAMPLE_MADRIGAL_FILE.exists(),
    reason="Real Madrigal data not present on this host",
)
def test_reading_a_small_slice_of_a_real_madrigal_file():
    """Reads only the first 2000 rows of one real file's `Table Layout` -
    the file itself is ~900 MB and the dataset ~13M rows, but h5py's slicing
    on a Dataset only materialises the rows actually requested."""
    import h5py

    with h5py.File(_SAMPLE_MADRIGAL_FILE, "r") as h5f:
        table = h5f["Data"]["Table Layout"][:2000]

    df_mad = pl.DataFrame(
        {
            "station": np.char.upper(table["gps_site"].astype(str)),
            "lat_sta": table["gdlatr"],
            "lon_sta": table["gdlonr"],
            "sod": table["sod"],
            "satele": table["elm"],
            "satazi": table["azm"],
            "los_tec": table["los_tec"],
        }
    )
    assert df_mad.height == 2000
    assert df_mad["station"].to_list()[0] == df_mad["station"].to_list()[0].upper()

    # Use the slice's own rows as observations: every row must find itself,
    # so this is a floor on the match rate, not a claim about real coverage.
    obs = df_mad.select(["station", "lat_sta", "lon_sta", "sod", "satele", "satazi"])
    result = match_exact_key(obs, df_mad)
    assert result.match_rate == pytest.approx(1.0)
