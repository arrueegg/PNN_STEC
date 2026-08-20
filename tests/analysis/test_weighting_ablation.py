"""Tests for `stec.analysis.weighting_ablation` (R2.5).

All fixtures are small synthetic frames in `tmp_path`. The point of this module is the
paired comparison, so the tests centre on pairing: dropping station-days that only one
arm solved, and reporting how many were dropped.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stec.analysis import weighting_ablation as wa


def _write_summary(path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(station: str, doy: int, method: str, error_3d_rms: float) -> dict:
    return {
        "station": station,
        "doy": doy,
        "method": method,
        "error_3d_rms": error_3d_rms,
    }


# ---------------------------------------------------------------------------
# Pairing: only station-days solved under every arm of a correction are kept.
# ---------------------------------------------------------------------------


def test_paired_ablation_drops_station_days_present_in_only_one_arm(tmp_path):
    summary_path = tmp_path / "summary.csv"
    _write_summary(
        summary_path,
        [
            # AMC4/132 and ZIMM/133 solved under both elev and iono -> paired.
            _row("AMC4", 132, "STEC_elev", 1.0),
            _row("AMC4", 132, "STEC_iono", 0.8),
            _row("ZIMM", 133, "STEC_elev", 1.2),
            _row("ZIMM", 133, "STEC_iono", 1.0),
            # WTZR/134 solved only under elev -> unpaired, must be dropped.
            _row("WTZR", 134, "STEC_elev", 2.0),
            # ONSA/135 solved only under iono -> unpaired, must be dropped.
            _row("ONSA", 135, "STEC_iono", 3.0),
        ],
    )

    table = wa.paired_ablation(summary_path)

    row = table.loc["Direct STEC"]
    # 4 raw STEC_elev/STEC_iono rows pivot to 3 station-days (AMC4/132, ZIMM/133,
    # WTZR/134, ONSA/135 = 4 unique station-days), of which 2 are unpaired (WTZR, ONSA)
    # and dropped, leaving 2 paired.
    assert row["paired_station_days"] == 2
    assert row["dropped_unpaired"] == 2
    assert row["elev_mean"] == pytest.approx((1.0 + 1.2) / 2)
    assert row["iono_mean"] == pytest.approx((0.8 + 1.0) / 2)


def test_paired_ablation_gain_percent_is_positive_when_iono_reduces_error(tmp_path):
    summary_path = tmp_path / "summary.csv"
    _write_summary(
        summary_path,
        [
            _row("AMC4", 132, "STEC_elev", 2.0),
            _row("AMC4", 132, "STEC_iono", 1.0),  # iono halves the error here
        ],
    )

    table = wa.paired_ablation(summary_path)

    row = table.loc["Direct STEC"]
    # gain_% = 100 * (elev - iono) / elev = 100 * (2 - 1) / 2 = 50
    assert row["gain_iono_%"] == pytest.approx(50.0)
    assert row["gain_%"] == pytest.approx(50.0)
    assert row["iono_better_frac_%"] == pytest.approx(100.0)


def test_paired_ablation_excludes_outlier_station_days_above_10m(tmp_path):
    summary_path = tmp_path / "summary.csv"
    _write_summary(
        summary_path,
        [
            _row("AMC4", 132, "STEC_elev", 1.0),
            _row("AMC4", 132, "STEC_iono", 1.0),
            _row("ZIMM", 133, "STEC_elev", 15.0),  # excluded: > 10 m
            _row("ZIMM", 133, "STEC_iono", 15.0),
        ],
    )

    table = wa.paired_ablation(summary_path)

    assert table.loc["Direct STEC", "paired_station_days"] == 1


def test_paired_ablation_ignores_unlabelled_methods(tmp_path):
    summary_path = tmp_path / "summary.csv"
    _write_summary(
        summary_path,
        [
            _row("AMC4", 132, "STEC_elev", 1.0),
            _row("AMC4", 132, "STEC_iono", 1.0),
            _row("AMC4", 132, "some_other_method", 99.0),
        ],
    )

    table = wa.paired_ablation(summary_path)

    assert table.loc["Direct STEC", "paired_station_days"] == 1


# ---------------------------------------------------------------------------
# Fixed-variance arm: read from per-day daily_summary_iono.csv files (weight_opt=iono
# provenance), not from a multiday_summary.csv.
# ---------------------------------------------------------------------------


def test_load_fixed_variance_concatenates_per_day_files_and_labels_the_arm(tmp_path):
    results_dir = tmp_path / "results"
    (results_dir / "2024132").mkdir(parents=True)
    (results_dir / "2024133").mkdir(parents=True)
    _write_summary(
        results_dir / "2024132" / "daily_summary_iono.csv",
        [_row("AMC4", 132, "STEC_iono", 1.5)],
    )
    _write_summary(
        results_dir / "2024133" / "daily_summary_iono.csv",
        [_row("ZIMM", 133, "STEC_iono", 2.5)],
    )

    frame = wa.load_fixed_variance(results_dir)

    assert len(frame) == 2
    assert set(frame["correction"]) == {"Direct STEC"}
    assert set(frame["weighting"]) == {"fixed"}
    assert set(frame["error_3d_rms"]) == {1.5, 2.5}


def test_load_fixed_variance_returns_empty_frame_when_no_files(tmp_path):
    frame = wa.load_fixed_variance(tmp_path / "does_not_exist")
    assert frame.empty


def test_fixed_variance_comparison_pairs_all_three_stochastic_models(tmp_path):
    summary_path = tmp_path / "summary.csv"
    _write_summary(
        summary_path,
        [
            _row("AMC4", 132, "STEC_elev", 3.0),
            _row("AMC4", 132, "STEC_iono", 1.0),
        ],
    )
    fixed_variance_dir = tmp_path / "fixed_variance_results"
    (fixed_variance_dir / "2024132").mkdir(parents=True)
    _write_summary(
        fixed_variance_dir / "2024132" / "daily_summary_iono.csv",
        [_row("AMC4", 132, "STEC_iono", 2.0)],
    )

    result = wa.fixed_variance_comparison(summary_path, fixed_variance_dir)

    assert result is not None
    assert result["paired_station_days"] == 1
    assert result["elev_mean_m"] == pytest.approx(3.0)
    assert result["fixed_variance_mean_m"] == pytest.approx(2.0)
    assert result["predicted_uncertainty_mean_m"] == pytest.approx(1.0)
    # iono (1.0) beats fixed (2.0): 100 * (2 - 1) / 2 = 50%.
    assert result["iono_vs_fixed_%"] == pytest.approx(50.0)


def test_fixed_variance_comparison_returns_none_when_arm_is_missing(tmp_path):
    summary_path = tmp_path / "summary.csv"
    _write_summary(summary_path, [_row("AMC4", 132, "STEC_elev", 3.0)])

    result = wa.fixed_variance_comparison(summary_path, tmp_path / "does_not_exist")

    assert result is None
