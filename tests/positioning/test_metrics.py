"""Tests for `stec.positioning.metrics`.

All unit tests build their own tiny fixtures (a synthetic `.pos` file, a synthetic
metrics frame) rather than depending on the live checkout. One integration test reads a
real `.pos` file from the live PNN_STEC checkout and is skipped cleanly when that
checkout isn't present.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stec.positioning import metrics as pm

# A real PPPx .pos file, kinematic AIRA station, DOY 300/2024 - used only by the
# integration test below, which skips if this path doesn't exist.
_LIVE_POS_FILE = Path(
    "/scratch2/arrueegg/WP4/PNN_STEC/experiments/"
    "Finetune_STEC_2024_300_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr2e-4_bs512_"
    "GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI/"
    "positioning/results/2024300/model/AIRA/AIRA_model.pos"
)

_POS_FIXTURE = """\
 mjd     sod   nsat       x             y             z          stdx     stdy     stdz    rck(m)   zhd     zwd     dzwd
60609     0.00   4  -3530194.195   4118798.368   3344042.673    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
60609    30.00   4  -3530194.840   4118798.715   3344043.220    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
60609    60.00   4  -3530194.369   4118798.410   3344042.724    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
"""


# ---------------------------------------------------------------------------
# .pos parsing
# ---------------------------------------------------------------------------


def test_parse_pos_file_reads_the_documented_column_layout(tmp_path):
    """Columns line up with the header: sod/nsat/xyz/rck/zhd/zwd/dzwd, plus derived
    hour/ztd/e/n/u/error_2d/error_3d and a ref_source flag."""
    pos_file = tmp_path / "AIRA_model.pos"
    pos_file.write_text(_POS_FIXTURE)

    df = pm.parse_pos_file(pos_file)

    assert df is not None
    assert len(df) == 3
    assert list(df["sod"]) == [0.0, 30.0, 60.0]
    assert list(df["nsat"]) == [4, 4, 4]
    assert df["ref_source"].iloc[0] == "mean"
    # Day-mean reference => residuals must average to ~0 in each ENU component.
    assert df["e"].mean() == pytest.approx(0.0, abs=1e-6)
    assert df["n"].mean() == pytest.approx(0.0, abs=1e-6)
    assert df["u"].mean() == pytest.approx(0.0, abs=1e-6)
    assert (df["error_3d"] >= 0).all()
    assert (df["error_3d"] >= df["error_2d"]).all()  # 3D includes the vertical term


def test_parse_pos_file_uses_ground_truth_reference_when_given(tmp_path):
    pos_file = tmp_path / "AIRA_model.pos"
    pos_file.write_text(_POS_FIXTURE)

    # A reference far from every epoch's position - errors must be large and nonzero,
    # unlike the day-mean case above.
    ref_pos = np.array([-3530000.0, 4118800.0, 3344000.0])
    df = pm.parse_pos_file(pos_file, ref_pos=ref_pos)

    assert df["ref_source"].iloc[0] == "ground_truth"
    assert (df["error_3d"] > 1.0).all()


def test_parse_pos_file_returns_none_for_unreadable_file(tmp_path):
    missing = tmp_path / "does_not_exist.pos"
    assert pm.parse_pos_file(missing) is None


@pytest.mark.skipif(
    not _LIVE_POS_FILE.exists(), reason="live PNN_STEC checkout not present"
)
def test_parse_pos_file_reads_a_real_pppx_output():
    df = pm.parse_pos_file(_LIVE_POS_FILE)
    assert df is not None
    assert len(df) > 0
    assert {"sod", "nsat", "x", "y", "z", "e", "n", "u", "error_3d"} <= set(df.columns)


# ---------------------------------------------------------------------------
# Per-station-day RMSE (hand-computed)
# ---------------------------------------------------------------------------


def test_compute_metrics_rmse_matches_hand_computation():
    """Two epochs with known e/n/u errors; every RMS is checked against the formula
    computed independently here, not against `compute_metrics`'s own arithmetic."""
    df = pd.DataFrame(
        {
            "nsat": [5, 5],
            "e": [3.0, 0.0],
            "n": [4.0, 0.0],
            "u": [0.0, 12.0],
            "ref_source": ["ground_truth", "ground_truth"],
        }
    )
    df["error_2d"] = np.sqrt(df["e"] ** 2 + df["n"] ** 2)
    df["error_3d"] = np.sqrt(df["e"] ** 2 + df["n"] ** 2 + df["u"] ** 2)

    metrics = pm.compute_metrics(df)

    assert metrics is not None
    expected_2d_rms = math.sqrt((5.0**2 + 0.0**2) / 2)
    expected_3d_rms = math.sqrt((5.0**2 + 12.0**2) / 2)
    expected_u_rms = math.sqrt((0.0**2 + 12.0**2) / 2)
    assert metrics["error_2d_rms"] == pytest.approx(expected_2d_rms)
    assert metrics["error_3d_rms"] == pytest.approx(expected_3d_rms)
    assert metrics["u_rms"] == pytest.approx(expected_u_rms)
    assert metrics["n_epochs"] == 2


def test_compute_metrics_returns_none_for_empty_frame():
    assert pm.compute_metrics(None) is None
    assert pm.compute_metrics(pd.DataFrame()) is None


# ---------------------------------------------------------------------------
# Outlier rule: <=10 m kept, >10 m excluded, boundary pinned
# ---------------------------------------------------------------------------


def test_outlier_rule_boundary_is_inclusive_at_exactly_10m():
    frame = pd.DataFrame(
        {
            "station": ["A", "B", "C"],
            "error_3d_rms": [9.9999, 10.0, 10.0001],
        }
    )

    kept = pm.exclude_outlier_station_days(frame)

    assert list(kept["station"]) == ["A", "B"]  # 10.0 kept, 10.0001 excluded


def test_outlier_rule_uses_the_named_constant_by_default():
    assert pm.OUTLIER_3D_RMS_M == 10.0


# ---------------------------------------------------------------------------
# Aggregation: mean of station-days, not pooled over epochs
# ---------------------------------------------------------------------------


def test_summarise_is_the_mean_of_station_days_not_epoch_weighted():
    """Station A has far more epochs than station B, but `summarise` must weight both
    station-days equally - if it silently pooled by epoch count, the mean would sit much
    closer to A's value than to the unweighted midpoint asserted here."""
    frame = pd.DataFrame(
        {
            "Method": ["Direct STEC", "Direct STEC"],
            "n_epochs": [1000, 10],  # present to prove summarise ignores it
            "error_3d_rms": [2.0, 10.0],
            "error_2d_rms": [1.0, 5.0],
            "u_rms": [0.5, 2.5],
        }
    )

    summary = pm.summarise(frame, ["Method"])
    row = summary.loc["Direct STEC"]

    assert row["station_days"] == 2
    assert row["3D_mean_m"] == pytest.approx(6.0)  # unweighted mean of [2.0, 10.0]
    assert row["3D_median_m"] == pytest.approx(6.0)
    assert row["2D_mean_m"] == pytest.approx(3.0)
    assert row["Up_mean_m"] == pytest.approx(1.5)
    # pandas linear-interpolation quantile of [2.0, 10.0] at q=0.95
    assert row["3D_p95_m"] == pytest.approx(9.6)

    # The epoch-weighted (pooled) alternative would be far from the unweighted mean -
    # confirms the two conventions actually differ for this fixture, not just in theory.
    pooled_weighted_mean = np.average(frame["error_3d_rms"], weights=frame["n_epochs"])
    assert abs(row["3D_mean_m"] - pooled_weighted_mean) > 3.0


# ---------------------------------------------------------------------------
# SINEX parsing
# ---------------------------------------------------------------------------


def test_load_sinex_coords_parses_stax_stay_staz(tmp_path):
    snx_file = tmp_path / "IGS0OPSSNX_test.SNX"
    snx_file.write_text(
        "+SOLUTION/ESTIMATE\n"
        " 1 STAX  ZIMM  A    1  05:159:43200 m    01  4331297.0450 0.0011\n"
        " 2 STAY  ZIMM  A    1  05:159:43200 m    01   567555.6390 0.0011\n"
        " 3 STAZ  ZIMM  A    1  05:159:43200 m    01  4633133.9060 0.0011\n"
        " 4 STAX  AMC4  A    1  05:159:43200 m    01 ___ESTIMATED_VALUE___ 0.0011\n"
        "-SOLUTION/ESTIMATE\n"
    )

    coords = pm.load_sinex_coords(snx_file)

    assert coords["ZIMM"] == pytest.approx([4331297.0450, 567555.6390, 4633133.9060])
    # Placeholder value must not be recorded as a real coordinate, and AMC4 has no other
    # line to establish an entry at all.
    assert "AMC4" not in coords


def test_load_sinex_coords_missing_file_returns_empty_dict(tmp_path):
    assert pm.load_sinex_coords(tmp_path / "missing.SNX") == {}


# ---------------------------------------------------------------------------
# Per-station aggregation across .pos files
# ---------------------------------------------------------------------------


def test_aggregate_daily_metrics_across_station_subdirectories(tmp_path):
    results_dir = tmp_path / "results" / "2024300" / "model"
    for station in ("AIRA", "ZIMM"):
        station_dir = results_dir / station
        station_dir.mkdir(parents=True)
        (station_dir / f"{station}_model.pos").write_text(_POS_FIXTURE)

    metrics_df = pm.aggregate_daily_metrics(results_dir, 2024, 300, "model")

    assert metrics_df is not None
    assert sorted(metrics_df["station"]) == ["AIRA", "ZIMM"]
    assert (metrics_df["method"] == "model").all()
    assert (metrics_df["year"] == 2024).all()
    assert (metrics_df["doy"] == 300).all()
    assert list(metrics_df.columns[:4]) == ["station", "method", "year", "doy"]


def test_aggregate_daily_metrics_skips_stations_missing_from_sinex(tmp_path):
    results_dir = tmp_path / "results" / "2024300" / "model"
    for station in ("AIRA", "ZIMM"):
        station_dir = results_dir / station
        station_dir.mkdir(parents=True)
        (station_dir / f"{station}_model.pos").write_text(_POS_FIXTURE)

    snx_file = tmp_path / "ground_truth.SNX"
    snx_file.write_text(
        "+SOLUTION/ESTIMATE\n"
        " 1 STAX  AIRA  A    1  05:159:43200 m    01  -3530200.0000 0.0011\n"
        " 2 STAY  AIRA  A    1  05:159:43200 m    01   4118800.0000 0.0011\n"
        " 3 STAZ  AIRA  A    1  05:159:43200 m    01   3344040.0000 0.0011\n"
        "-SOLUTION/ESTIMATE\n"
    )

    metrics_df = pm.aggregate_daily_metrics(
        results_dir, 2024, 300, "model", snx_file=snx_file
    )

    # ZIMM has no SINEX coordinate and no true reference, so it must be excluded rather
    # than silently falling back to a day-mean reference.
    assert metrics_df is not None
    assert list(metrics_df["station"]) == ["AIRA"]
    assert metrics_df["ref_source"].iloc[0] == "ground_truth"


def test_aggregate_daily_metrics_returns_none_when_no_pos_files(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert pm.aggregate_daily_metrics(empty_dir, 2024, 300, "model") is None
