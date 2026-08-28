"""Tests for `stec.analysis.positioning_diagnostics`, with hand-computed expected values.

Built for the owner's request (2026-08-28) to evaluate the coverage-recovery positioning
result properly before drawing conclusions from it - see that module's docstring for the
four questions each group of tests below backs. Every aggregation function here is
exercised against tiny, hand-computable frames rather than the real 43,101-row summary,
so the expected numbers in this file are arithmetic, not a second implementation.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from stec.analysis import positioning_diagnostics as pd_diag


def frame(rows: list[tuple]) -> pd.DataFrame:
    """rows of (station, Method, doy, error_3d_rms)."""
    return pd.DataFrame(rows, columns=["station", "Method", "doy", "error_3d_rms"])


STEC = "Direct STEC"
GIM = "IGS GIM + Mapping"

# ---------------------------------------------------------------------------
# 1. daily_timeseries
# ---------------------------------------------------------------------------


def test_daily_timeseries_computes_mean_median_count_per_doy_and_method():
    df = frame(
        [
            ("AAAA", STEC, 100, 1.0),
            ("BBBB", STEC, 100, 20.0),  # excluded by the default 10 m rule
            ("AAAA", GIM, 100, 2.0),
            ("BBBB", GIM, 100, 5.0),
            ("AAAA", STEC, 101, 3.0),
            ("AAAA", GIM, 101, 4.0),
        ]
    )
    result = pd_diag.daily_timeseries(df).set_index(["doy", "Method"])

    assert result.loc[(100, STEC)]["mean"] == pytest.approx(1.0)
    assert result.loc[(100, STEC)]["median"] == pytest.approx(1.0)
    assert result.loc[(100, STEC)]["station_days"] == 1  # BBBB's 20 m excluded
    assert result.loc[(100, GIM)]["mean"] == pytest.approx(3.5)
    assert result.loc[(100, GIM)]["station_days"] == 2
    assert result.loc[(101, STEC)]["mean"] == pytest.approx(3.0)
    assert result.loc[(101, GIM)]["mean"] == pytest.approx(4.0)


def test_daily_timeseries_respects_a_narrower_threshold():
    df = frame([("AAAA", STEC, 100, 6.0), ("BBBB", STEC, 100, 4.0)])
    result = pd_diag.daily_timeseries(df, threshold=5.0).set_index(["doy", "Method"])
    # 6.0 m excluded at a 5 m threshold, so only BBBB's 4.0 m survives.
    assert result.loc[(100, STEC)]["station_days"] == 1
    assert result.loc[(100, STEC)]["mean"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# 2. per_station_summary
# ---------------------------------------------------------------------------


def test_per_station_summary_diff_and_missing_method_is_nan_not_dropped():
    df = frame(
        [
            ("AAAA", STEC, 100, 1.0),
            ("AAAA", STEC, 101, 3.0),
            ("AAAA", GIM, 100, 2.0),
            ("AAAA", GIM, 101, 4.0),
            # BBBB has no Direct STEC row at all for either day.
            ("BBBB", GIM, 100, 5.0),
            # CCCC's only Direct STEC row is an outlier, excluded by the 10 m rule.
            ("CCCC", STEC, 100, 15.0),
            ("CCCC", GIM, 100, 2.0),
        ]
    )
    result = pd_diag.per_station_summary(df).set_index("station")

    aaaa = result.loc["AAAA"]
    assert aaaa["stec_mean_m"] == pytest.approx(2.0)  # mean(1.0, 3.0)
    assert aaaa["gim_mean_m"] == pytest.approx(3.0)  # mean(2.0, 4.0)
    assert aaaa["diff_mean_m"] == pytest.approx(-1.0)
    assert aaaa["stec_station_days"] == 2
    assert aaaa["gim_station_days"] == 2

    bbbb = result.loc["BBBB"]
    assert pd.isna(bbbb["stec_mean_m"])
    assert pd.isna(bbbb["diff_mean_m"])
    assert bbbb["gim_mean_m"] == pytest.approx(5.0)

    cccc = result.loc["CCCC"]
    assert pd.isna(cccc["stec_mean_m"])  # 15.0 m excluded, no STEC row survives
    assert cccc["gim_mean_m"] == pytest.approx(2.0)


def test_per_station_summary_is_sorted_worst_first():
    df = frame(
        [
            ("LOSES", STEC, 100, 5.0),
            ("LOSES", GIM, 100, 1.0),  # diff = +4.0, STEC much worse
            ("WINS", STEC, 100, 1.0),
            ("WINS", GIM, 100, 5.0),  # diff = -4.0, STEC much better
        ]
    )
    result = pd_diag.per_station_summary(df)
    assert result["station"].tolist() == ["LOSES", "WINS"]


# ---------------------------------------------------------------------------
# 3. outlier_threshold_counts / outlier_headline_sensitivity / outlier_concentration
# ---------------------------------------------------------------------------


def test_outlier_threshold_counts_hand_computed():
    df = frame(
        [
            ("A", STEC, 1, 1.0),
            ("A", STEC, 2, 2.0),
            ("A", STEC, 3, 5.0),
            ("A", GIM, 1, 1.0),
            ("A", GIM, 2, 1.0),
        ]
    )
    result = pd_diag.outlier_threshold_counts(df, thresholds=(1.5, 3.5)).set_index(
        ["Method", "threshold_m"]
    )

    # Direct STEC total mass = 1.0 + 2.0 + 5.0 = 8.0.
    stec_15 = result.loc[(STEC, 1.5)]
    assert stec_15["n_exceeding"] == 2  # 2.0 and 5.0 exceed 1.5
    assert stec_15["total_station_days"] == 3
    assert stec_15["pct_station_days_exceeding"] == pytest.approx(200 / 3)
    assert stec_15["error_mass_fraction_pct"] == pytest.approx(100 * 7.0 / 8.0)

    stec_35 = result.loc[(STEC, 3.5)]
    assert stec_35["n_exceeding"] == 1  # only 5.0 exceeds 3.5
    assert stec_35["error_mass_fraction_pct"] == pytest.approx(100 * 5.0 / 8.0)

    # Neither GIM value (1.0, 1.0) exceeds either threshold.
    gim_15 = result.loc[(GIM, 1.5)]
    assert gim_15["n_exceeding"] == 0
    assert gim_15["error_mass_fraction_pct"] == pytest.approx(0.0)


def test_outlier_threshold_counts_can_group_by_extra_columns():
    df = pd.DataFrame(
        [
            {"Method": STEC, "population": "original", "error_3d_rms": 1.0},
            {"Method": STEC, "population": "recovered", "error_3d_rms": 20.0},
        ]
    )
    result = pd_diag.outlier_threshold_counts(
        df, thresholds=(10.0,), group_cols=("population", "Method")
    ).set_index(["population", "Method"])
    assert result.loc[("original", STEC)]["n_exceeding"] == 0
    assert result.loc[("recovered", STEC)]["n_exceeding"] == 1


def test_outlier_headline_sensitivity_none_vs_threshold():
    df = frame(
        [
            ("A", STEC, 1, 1.0),
            ("A", STEC, 2, 2.0),
            ("A", STEC, 3, 20.0),  # excluded once threshold reaches 10 m
            ("A", GIM, 1, 2.0),
            ("A", GIM, 2, 2.0),
            ("A", GIM, 3, 2.0),
        ]
    )
    result = pd_diag.outlier_headline_sensitivity(df, thresholds=(10.0,)).set_index(
        "threshold_label"
    )

    none_row = result.loc["none"]
    stec_mean_none = (1.0 + 2.0 + 20.0) / 3  # 7.6667
    assert none_row["n_stec"] == 3
    assert none_row["stec_mean_m"] == pytest.approx(stec_mean_none)
    assert none_row["mean_improvement_pct"] == pytest.approx(
        100 * (2.0 - stec_mean_none) / 2.0
    )
    assert none_row["median_improvement_pct"] == pytest.approx(
        0.0
    )  # both medians = 2.0

    ten_row = result.loc["10m"]
    assert ten_row["n_stec"] == 2  # 20.0 m excluded
    assert ten_row["stec_mean_m"] == pytest.approx(1.5)
    assert ten_row["mean_improvement_pct"] == pytest.approx(100 * (2.0 - 1.5) / 2.0)
    assert ten_row["median_improvement_pct"] == pytest.approx(100 * (2.0 - 1.5) / 2.0)


def test_outlier_concentration_counts_and_top_n_share():
    df = frame(
        [
            ("AAAA", STEC, 1, 15.0),
            ("AAAA", STEC, 2, 16.0),
            ("BBBB", STEC, 1, 17.0),
            ("CCCC", GIM, 1, 12.0),
        ]
    )
    by_station, by_day, summary = pd_diag.outlier_concentration(
        df, threshold=10.0, top_n=1
    )

    stec_by_station = by_station[by_station["Method"] == STEC].set_index("station")
    assert stec_by_station.loc["AAAA", "n_outlier_station_days"] == 2
    assert stec_by_station.loc["BBBB", "n_outlier_station_days"] == 1

    stec_by_day = by_day[by_day["Method"] == STEC].set_index("doy")
    assert stec_by_day.loc[1, "n_outlier_stations"] == 2  # AAAA and BBBB both doy 1
    assert stec_by_day.loc[2, "n_outlier_stations"] == 1

    stec_summary = summary.set_index("Method").loc[STEC]
    assert stec_summary["total_outlier_station_days"] == 3
    assert stec_summary["n_distinct_stations"] == 2
    # top_n=1: AAAA alone (2 of 3) is the largest single-station share.
    assert stec_summary["top_1_station_share_pct"] == pytest.approx(200 / 3)

    gim_summary = summary.set_index("Method").loc[GIM]
    assert gim_summary["total_outlier_station_days"] == 1


# ---------------------------------------------------------------------------
# 4. attach_population / population_split
# ---------------------------------------------------------------------------


def test_attach_population_labels_recovered_and_original_correctly():
    df = frame([("AAAA", STEC, 100, 1.0), ("BBBB", STEC, 100, 2.0)])
    recovered = pd.DataFrame([{"year": 2024, "doy": 100, "station": "AAAA"}])

    result = pd_diag.attach_population(df, recovered).set_index("station")
    assert result.loc["AAAA", "population"] == "recovered"
    assert result.loc["BBBB", "population"] == "original"


def test_attach_population_with_no_recovered_tree_marks_everything_original():
    df = frame([("AAAA", STEC, 100, 1.0)])
    result = pd_diag.attach_population(
        df, pd.DataFrame(columns=["year", "doy", "station"])
    )
    assert (result["population"] == "original").all()


def test_population_split_reports_separate_improvement_per_population():
    df = frame(
        [
            ("AAAA", STEC, 100, 1.0),
            ("AAAA", GIM, 100, 2.0),  # original: STEC beats GIM 1.0 vs 2.0
            ("BBBB", STEC, 101, 4.0),
            ("BBBB", GIM, 101, 2.0),  # recovered: STEC loses to GIM 4.0 vs 2.0
        ]
    )
    recovered = pd.DataFrame([{"year": 2024, "doy": 101, "station": "BBBB"}])
    with_population = pd_diag.attach_population(df, recovered)

    by_method, stec_vs_gim = pd_diag.population_split(with_population)
    stec_vs_gim = stec_vs_gim.set_index("population")

    original = stec_vs_gim.loc["original"]
    assert original["n_stec"] == 1
    assert original["stec_mean_m"] == pytest.approx(1.0)
    assert original["mean_improvement_pct"] == pytest.approx(50.0)  # (2-1)/2*100

    recovered_row = stec_vs_gim.loc["recovered"]
    assert recovered_row["stec_mean_m"] == pytest.approx(4.0)
    assert recovered_row["mean_improvement_pct"] == pytest.approx(-100.0)  # (2-4)/2*100

    by_method_indexed = by_method.set_index(["population", "Method"])
    assert by_method_indexed.loc[("original", STEC), "station_days"] == 1
    assert by_method_indexed.loc[("recovered", GIM), "mean_m"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# load_recovered_station_days: reads the real recovery tree layout
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


def test_load_recovered_station_days_reads_the_real_tree_layout(tmp_path):
    root = tmp_path / "recovered_stec_db"
    _write_recovered_fixture(root, 2024, 166, ["BIK0", "BRMG"])
    _write_recovered_fixture(root, 2024, 178, ["BIK0"])  # BIK0 recovered on two days

    result = pd_diag.load_recovered_station_days(root)

    assert len(result) == 3
    pairs = set(zip(result["doy"], result["station"]))
    assert pairs == {(166, "BIK0"), (166, "BRMG"), (178, "BIK0")}
    assert (result["year"] == 2024).all()


def test_load_recovered_station_days_missing_root_warns_and_returns_empty(tmp_path):
    result = pd_diag.load_recovered_station_days(tmp_path / "does_not_exist")
    assert result.empty
    assert list(result.columns) == ["year", "doy", "station"]


# ---------------------------------------------------------------------------
# overall_summary: a thin wrapper around stec.positioning.metrics.summarise
# ---------------------------------------------------------------------------


def test_overall_summary_reindexes_to_method_order_and_applies_10m_rule():
    # pm.summarise also reports 2D and Up error, so those columns must be present too.
    df = pd.DataFrame(
        [
            {
                "station": "A",
                "Method": STEC,
                "doy": 1,
                "error_3d_rms": 1.0,
                "error_2d_rms": 0.5,
                "u_rms": 0.5,
            },
            {
                "station": "A",
                "Method": STEC,
                "doy": 2,
                "error_3d_rms": 20.0,
                "error_2d_rms": 10.0,
                "u_rms": 10.0,
            },  # excluded
            {
                "station": "A",
                "Method": GIM,
                "doy": 1,
                "error_3d_rms": 2.0,
                "error_2d_rms": 1.0,
                "u_rms": 1.0,
            },
        ]
    )
    result = pd_diag.overall_summary(df)
    assert list(result.index) == pd_diag.METHOD_ORDER
    assert result.loc[STEC, "station_days"] == 1
    assert result.loc[STEC, "3D_mean_m"] == pytest.approx(1.0)
