"""Tests for `stec.analysis.positioning_geography`, with hand-computed expected values.

Built for the owner's request (2026-08-28) to look at positioning performance
geographically - one map per station and a latitude stratification, geographic and
geomagnetic side by side, crossed with the recovered/original population split. Every
aggregation function here is exercised against tiny, hand-computable frames rather than
the real 43,101-row summary, the same convention `test_positioning_diagnostics.py` uses.
"""

from __future__ import annotations

from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import pytest

from stec.analysis import positioning_geography as pg

STEC = "Direct STEC"
GIM = "IGS GIM + Mapping"


def frame(rows: list[tuple]) -> pd.DataFrame:
    """rows of (station, Method, doy, error_3d_rms)."""
    return pd.DataFrame(rows, columns=["station", "Method", "doy", "error_3d_rms"])


# Three stations spanning three different latitude bins of `pg.LATITUDE_BINS`
# ([-90, -60, -40, -20, -10, 10, 20, 40, 60, 90]):
#   AAAA: lat=5    -> geographic bin (-10, 10];   sm_lat=15 -> geomagnetic bin (10, 20]
#   BBBB: lat=50   -> geographic bin (40, 60];    sm_lat=45 -> geomagnetic bin (40, 60]
#   CCCC: lat=-75  -> geographic bin (-90.001, -60]; sm_lat=-75 -> same bin
#   DDDD: lat=8    -> geographic bin (-10, 10] (same bin as AAAA, different population)
STATION_COORDS = pd.DataFrame(
    {
        "lat": [5.0, 50.0, -75.0, 8.0],
        "lon": [0.0, 10.0, -30.0, 2.0],
        "sm_lat": [15.0, 45.0, -75.0, 9.0],
    },
    index=pd.Index(["AAAA", "BBBB", "CCCC", "DDDD"], name="station"),
)


# ---------------------------------------------------------------------------
# Geomagnetic latitude
# ---------------------------------------------------------------------------


def test_add_geomagnetic_latitude_attaches_column_without_disturbing_others():
    coords = pd.DataFrame(
        {"lat": [45.0], "lon": [10.0]}, index=pd.Index(["ZZZZ"], name="station")
    )
    out = pg.add_geomagnetic_latitude(coords)
    assert list(out.columns) == ["lat", "lon", "sm_lat"]
    assert out.loc["ZZZZ", "lat"] == 45.0
    # A real spacepy transform, not a placeholder equal to the geographic latitude.
    assert out.loc["ZZZZ", "sm_lat"] != pytest.approx(45.0, abs=1e-6)


def test_geomagnetic_latitude_is_effectively_time_invariant():
    """The module docstring's own claim: SM latitude of a fixed station barely moves
    across time of day or season, because the GM->SM rotation that depends on time is
    purely about the shared dipole (Z) axis, so it changes SM longitude, not latitude.
    Checked directly rather than assumed - this justifies computing it once per station
    at a single reference epoch instead of per observation."""
    coords = pd.DataFrame(
        {"lat": [45.0], "lon": [10.0]}, index=pd.Index(["ZZZZ"], name="station")
    )
    reference = pg.add_geomagnetic_latitude(coords, pg.GEOMAGNETIC_REFERENCE_EPOCH).loc[
        "ZZZZ", "sm_lat"
    ]

    times = [
        datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 12),
        datetime(2024, 7, 1, 6), datetime(2024, 7, 1, 18),
        datetime(2024, 10, 15, 3),
    ]  # fmt: skip
    for t in times:
        sm_lat = pg.add_geomagnetic_latitude(coords, t).loc["ZZZZ", "sm_lat"]
        assert sm_lat == pytest.approx(reference, abs=0.02), (t, sm_lat, reference)


# ---------------------------------------------------------------------------
# build_geo_frame
# ---------------------------------------------------------------------------


def test_build_geo_frame_attaches_coordinates_and_warns_on_missing_station(caplog):
    positioning_frame = frame([("AAAA", STEC, 1, 1.0), ("EEEE", STEC, 1, 2.0)])
    with caplog.at_level("WARNING"):
        geo = pg.build_geo_frame(positioning_frame, STATION_COORDS)

    aaaa = geo[geo["station"] == "AAAA"].iloc[0]
    assert aaaa["lat"] == pytest.approx(5.0)
    assert aaaa["sm_lat"] == pytest.approx(15.0)

    eeee = geo[geo["station"] == "EEEE"].iloc[0]
    assert pd.isna(eeee["lat"])
    assert "EEEE" in caplog.text


def test_build_geo_frame_attaches_population_when_recovered_given():
    positioning_frame = frame([("AAAA", STEC, 1, 1.0), ("BBBB", STEC, 1, 2.0)])
    recovered = pd.DataFrame([{"year": 2024, "doy": 1, "station": "AAAA"}])

    geo = pg.build_geo_frame(positioning_frame, STATION_COORDS, recovered)
    assert geo.set_index("station").loc["AAAA", "population"] == "recovered"
    assert geo.set_index("station").loc["BBBB", "population"] == "original"


# ---------------------------------------------------------------------------
# 1. Per-station maps
# ---------------------------------------------------------------------------


def test_per_method_station_summary_hand_computed_with_outlier_rule():
    positioning_frame = frame(
        [
            ("AAAA", STEC, 1, 1.0),
            ("AAAA", STEC, 2, 3.0),
            ("AAAA", STEC, 3, 20.0),  # excluded by the default 10 m rule
            ("AAAA", GIM, 1, 2.0),
            ("AAAA", GIM, 2, 4.0),
            ("BBBB", STEC, 1, 5.0),
        ]
    )
    geo = pg.build_geo_frame(positioning_frame, STATION_COORDS)
    result = pg.per_method_station_summary(geo).set_index(["station", "Method"])

    aaaa_stec = result.loc[("AAAA", STEC)]
    assert aaaa_stec["mean_error_3d_m"] == pytest.approx(2.0)  # mean(1.0, 3.0)
    assert aaaa_stec["median_error_3d_m"] == pytest.approx(2.0)
    assert aaaa_stec["station_days"] == 2  # the 20.0 m day excluded
    assert aaaa_stec["lat"] == pytest.approx(5.0)

    aaaa_gim = result.loc[("AAAA", GIM)]
    assert aaaa_gim["mean_error_3d_m"] == pytest.approx(3.0)  # mean(2.0, 4.0)

    bbbb_stec = result.loc[("BBBB", STEC)]
    assert bbbb_stec["mean_error_3d_m"] == pytest.approx(5.0)
    assert bbbb_stec["lat"] == pytest.approx(50.0)


def test_station_diff_map_table_attaches_coordinates_and_keeps_gim_only_stations():
    positioning_frame = frame(
        [
            ("AAAA", STEC, 1, 1.0),
            ("AAAA", STEC, 2, 3.0),
            ("AAAA", GIM, 1, 2.0),
            ("AAAA", GIM, 2, 4.0),
            # EEEE has no coordinate row and no Direct STEC solution.
            ("EEEE", GIM, 1, 5.0),
        ]
    )
    result = pg.station_diff_map_table(positioning_frame, STATION_COORDS).set_index(
        "station"
    )

    aaaa = result.loc["AAAA"]
    assert aaaa["diff_mean_m"] == pytest.approx(-1.0)  # mean(1,3) - mean(2,4) = 2 - 3
    assert aaaa["lat"] == pytest.approx(5.0)

    eeee = result.loc["EEEE"]
    assert pd.isna(eeee["stec_mean_m"])
    assert pd.isna(eeee["lat"])  # not in STATION_COORDS
    assert eeee["gim_mean_m"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 2. Latitude stratification
# ---------------------------------------------------------------------------


def test_latitude_stratification_bins_by_the_requested_column():
    positioning_frame = frame(
        [
            ("AAAA", STEC, 1, 1.0),
            ("AAAA", STEC, 2, 3.0),
            ("AAAA", GIM, 1, 2.0),
            ("AAAA", GIM, 2, 4.0),
            ("BBBB", STEC, 1, 5.0),
            ("BBBB", GIM, 1, 1.0),
        ]
    )
    geo = pg.build_geo_frame(positioning_frame, STATION_COORDS)

    geographic = pg.latitude_stratification(geo, "lat").set_index(["lat_bin", "Method"])
    equatorial_bin = pd.cut([5.0], bins=pg.LATITUDE_BINS, include_lowest=True)[0]
    midlat_bin = pd.cut([50.0], bins=pg.LATITUDE_BINS, include_lowest=True)[0]

    assert geographic.loc[(equatorial_bin, STEC)]["mean"] == pytest.approx(2.0)
    assert geographic.loc[(equatorial_bin, STEC)]["n"] == 2
    assert geographic.loc[(equatorial_bin, GIM)]["mean"] == pytest.approx(3.0)
    assert geographic.loc[(midlat_bin, STEC)]["mean"] == pytest.approx(5.0)
    assert geographic.loc[(midlat_bin, GIM)]["mean"] == pytest.approx(1.0)

    # AAAA's sm_lat (15) falls in a different bin than its lat (5) - the geomagnetic
    # stratification must use sm_lat, not silently reuse the geographic bin.
    geomagnetic = pg.latitude_stratification(geo, "sm_lat").set_index(
        ["lat_bin", "Method"]
    )
    sm_equatorial_bin = pd.cut([15.0], bins=pg.LATITUDE_BINS, include_lowest=True)[0]
    assert sm_equatorial_bin != equatorial_bin
    assert geomagnetic.loc[(sm_equatorial_bin, STEC)]["mean"] == pytest.approx(2.0)


def test_latitude_stratification_applies_outlier_threshold():
    positioning_frame = frame(
        [("AAAA", STEC, 1, 6.0), ("AAAA", STEC, 2, 4.0), ("AAAA", GIM, 1, 1.0)]
    )
    geo = pg.build_geo_frame(positioning_frame, STATION_COORDS)
    result = pg.latitude_stratification(geo, "lat", threshold=5.0)
    stec_row = result[result["Method"] == STEC].iloc[0]
    # 6.0 m excluded at a 5 m threshold, so only the 4.0 m day survives.
    assert stec_row["n"] == 1
    assert stec_row["mean"] == pytest.approx(4.0)


def test_latitude_stratification_diff_hand_computed_mean_median_and_improvement():
    positioning_frame = frame(
        [
            ("AAAA", STEC, 1, 1.0),
            ("AAAA", STEC, 2, 3.0),
            ("AAAA", GIM, 1, 2.0),
            ("AAAA", GIM, 2, 4.0),
            ("BBBB", STEC, 1, 5.0),
            ("BBBB", GIM, 1, 1.0),
        ]
    )
    geo = pg.build_geo_frame(positioning_frame, STATION_COORDS)
    result = pg.latitude_stratification_diff(geo, "lat").set_index("lat_bin")

    equatorial_bin = pd.cut([5.0], bins=pg.LATITUDE_BINS, include_lowest=True)[0]
    equatorial = result.loc[equatorial_bin]
    # AAAA doy1: stec=1, gim=2, diff=-1; doy2: stec=3, gim=4, diff=-1.
    assert equatorial["n"] == 2
    assert equatorial["stec_mean_m"] == pytest.approx(2.0)
    assert equatorial["gim_mean_m"] == pytest.approx(3.0)
    assert equatorial["diff_mean_m"] == pytest.approx(-1.0)
    assert equatorial["mean_improvement_pct"] == pytest.approx(100 * (3.0 - 2.0) / 3.0)
    assert equatorial["stec_median_m"] == pytest.approx(2.0)
    assert equatorial["gim_median_m"] == pytest.approx(3.0)

    midlat_bin = pd.cut([50.0], bins=pg.LATITUDE_BINS, include_lowest=True)[0]
    midlat = result.loc[midlat_bin]
    # BBBB doy1: stec=5, gim=1, diff=+4 - Direct STEC much worse here.
    assert midlat["n"] == 1
    assert midlat["diff_mean_m"] == pytest.approx(4.0)
    assert midlat["mean_improvement_pct"] == pytest.approx(100 * (1.0 - 5.0) / 1.0)


def test_latitude_stratification_diff_drops_station_days_missing_either_method():
    # BBBB has only a Direct STEC row (no GIM that day) - must not appear at all,
    # the same pairwise handling `positioning_diagnostics.per_station_summary` uses.
    positioning_frame = frame(
        [("AAAA", STEC, 1, 1.0), ("AAAA", GIM, 1, 2.0), ("BBBB", STEC, 1, 9.0)]
    )
    geo = pg.build_geo_frame(positioning_frame, STATION_COORDS)
    result = pg.latitude_stratification_diff(geo, "lat")
    assert result["n"].sum() == 1


# ---------------------------------------------------------------------------
# 3. Crossing latitude with the recovered/original population split
# ---------------------------------------------------------------------------


def test_recovered_concentration_by_latitude_hand_computed():
    coverage = pd.DataFrame(
        [
            {"doy": 1, "station": "AAAA"},
            {"doy": 2, "station": "AAAA"},
            {"doy": 1, "station": "BBBB"},
            {"doy": 1, "station": "CCCC"},
        ]
    )
    recovered = pd.DataFrame([{"doy": 1, "station": "AAAA"}])

    result = pg.recovered_concentration_by_latitude(
        coverage, recovered, STATION_COORDS, "lat"
    ).set_index("lat_bin")

    equatorial_bin = pd.cut([5.0], bins=pg.LATITUDE_BINS, include_lowest=True)[0]
    equatorial = result.loc[equatorial_bin]
    assert equatorial["total"] == 2
    assert equatorial["recovered"] == 1
    assert equatorial["original"] == 1
    assert equatorial["pct_recovered"] == pytest.approx(50.0)

    midlat_bin = pd.cut([50.0], bins=pg.LATITUDE_BINS, include_lowest=True)[0]
    midlat = result.loc[midlat_bin]
    assert midlat["total"] == 1
    assert midlat["pct_recovered"] == pytest.approx(0.0)


def test_population_latitude_diff_separates_populations_within_the_same_bin():
    # AAAA and DDDD both fall in the equatorial (-10, 10] geographic bin (lat 5 and 8)
    # but only DDDD's station-day is recovered - the split must not average them
    # together just because they share a latitude bin.
    positioning_frame = frame(
        [
            ("AAAA", STEC, 1, 1.0),
            ("AAAA", GIM, 1, 2.0),
            ("DDDD", STEC, 1, 5.0),
            ("DDDD", GIM, 1, 1.0),
        ]
    )
    recovered = pd.DataFrame([{"year": 2024, "doy": 1, "station": "DDDD"}])
    geo = pg.build_geo_frame(positioning_frame, STATION_COORDS, recovered)

    result = pg.population_latitude_diff(geo, "lat").set_index(
        ["lat_bin", "population"]
    )
    equatorial_bin = pd.cut([5.0], bins=pg.LATITUDE_BINS, include_lowest=True)[0]

    original = result.loc[(equatorial_bin, "original")]
    assert original["n"] == 1
    assert original["diff_mean_m"] == pytest.approx(-1.0)  # AAAA: 1 - 2
    assert original["mean_improvement_pct"] == pytest.approx(50.0)  # (2-1)/2*100

    recovered_row = result.loc[(equatorial_bin, "recovered")]
    assert recovered_row["n"] == 1
    assert recovered_row["diff_mean_m"] == pytest.approx(4.0)  # DDDD: 5 - 1
    assert recovered_row["mean_improvement_pct"] == pytest.approx(-400.0)  # (1-5)/1*100


# ---------------------------------------------------------------------------
# load_recovered_station_days_cached
# ---------------------------------------------------------------------------


def _write_recovered_fixture(root, year: int, doy: int, stations: list[str]) -> None:
    dtype = np.dtype([("station", "S4"), ("satele", "f4")])
    data = np.zeros(len(stations), dtype=dtype)
    data["station"] = [s.encode("ascii") for s in stations]
    doy_dir = root / str(year) / f"{doy:03d}"
    doy_dir.mkdir(parents=True)
    with h5py.File(doy_dir / f"ccl_{year}{doy:03d}_30_5.h5", "w") as handle:
        group = handle.require_group(str(year)).require_group(f"{doy:03d}")
        group.create_dataset("all_data", data=data)


def test_load_recovered_station_days_cached_prefers_the_existing_csv(tmp_path):
    cache_csv = tmp_path / "recovered_station_days.csv"
    pd.DataFrame([{"year": 2024, "doy": 100, "station": "ZZZZ"}]).to_csv(
        cache_csv, index=False
    )
    # The scan root does not even need to exist - the cache must be preferred without
    # touching it.
    result = pg.load_recovered_station_days_cached(
        cache_csv, root=tmp_path / "does_not_exist"
    )
    assert result["station"].tolist() == ["ZZZZ"]


def test_load_recovered_station_days_cached_falls_back_to_scanning_the_tree(tmp_path):
    root = tmp_path / "recovered_stec_db"
    _write_recovered_fixture(root, 2024, 166, ["AAAA"])
    result = pg.load_recovered_station_days_cached(tmp_path / "no_cache.csv", root=root)
    assert list(zip(result["doy"], result["station"])) == [(166, "AAAA")]
