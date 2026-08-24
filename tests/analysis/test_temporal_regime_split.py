"""Pins the regime boundary and the streamed pooled statistics against a direct
whole-frame computation.

Follows the style of `test_daily_metrics.py` / `test_dstec_evaluation.py`: synthetic days
are written through `prediction_store.write_predictions` so the test exercises the real
on-disk format, and every statistic is also computed independently from the concatenated
frames rather than trusted from the module under test.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from stec.analysis import relative_error_metrics as rem
from stec.analysis import temporal_regime_split as trs
from stec.inference import prediction_store as ps

REGIME_LABEL = dict(rem.REGIME_LABELS)


def day_frame(rows: int, error: float, seed: int) -> pd.DataFrame:
    """A day where every prediction misses truth by exactly `error` (constant, so the
    day's RMSE/MAE are known analytically without a second implementation)."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, 60, rows)
    return pd.DataFrame(
        {
            "station": ["AMC4"] * rows,
            "sat": ["G01"] * rows,
            "satele": rng.uniform(5, 90, rows),
            "true_stec": truth,
            "stec_pred": truth + error,
        }
    )


# --------------------------------------------------------------------------
# Regime boundary: reused from relative_error_metrics, not a second literal
# --------------------------------------------------------------------------


def test_uses_the_same_boundary_constant_as_relative_error_metrics():
    assert trs.EXTRAPOLATION_START == datetime(2024, 5, 1) == rem.EXTRAPOLATION_START


def test_2024_doy_121_is_interpolation_and_doy_122_is_extrapolation():
    """May 1, 2024 (a leap year) is DOY 122 - the boundary `split_test_data_by_date`
    draws with `>=`, so 121 (April 30) is the last interpolation day."""
    assert trs.day_regime(2024, 121) == "interpolation"
    assert trs.day_regime(2024, 122) == "extrapolation"


def test_a_pre_2024_year_is_always_interpolation():
    assert trs.day_regime(2023, 366) == "interpolation"


def test_late_2024_is_extrapolation():
    assert trs.day_regime(2024, 366) == "extrapolation"


# --------------------------------------------------------------------------
# Pooled statistics: streamed accumulation vs. a direct whole-frame computation
# --------------------------------------------------------------------------


def test_collect_pools_within_a_regime_across_multiple_days(tmp_path):
    """Two interpolation days (different years) must pool into one RMSE/MAE, not be
    averaged day-by-day - matching `src/`'s `calculate_metrics` over the concatenated
    regime frame, not a mean-of-daily statistic."""
    frames = {
        (2023, 200): day_frame(100, error=2.0, seed=1),
        (2024, 50): day_frame(50, error=4.0, seed=2),  # before DOY 122: interpolation
        (2024, 200): day_frame(300, error=6.0, seed=3),  # extrapolation
    }
    for (year, doy), frame in frames.items():
        ps.write_predictions(frame, "pretrained_stec", "own", year, doy, root=tmp_path)

    table = trs.collect(store_root=tmp_path)

    interp_direct = pd.concat([frames[(2023, 200)], frames[(2024, 50)]])
    interp_error = (
        interp_direct["stec_pred"].to_numpy() - interp_direct["true_stec"].to_numpy()
    )
    interp_row = table[table["regime"] == REGIME_LABEL["interpolation"]].iloc[0]
    assert interp_row["count"] == len(interp_direct)
    assert interp_row["RMSE"] == pytest.approx(
        np.sqrt(np.mean(interp_error**2)), rel=1e-6
    )
    assert interp_row["MAE"] == pytest.approx(np.mean(np.abs(interp_error)), rel=1e-6)

    extrap_direct = frames[(2024, 200)]
    extrap_error = (
        extrap_direct["stec_pred"].to_numpy() - extrap_direct["true_stec"].to_numpy()
    )
    extrap_row = table[table["regime"] == REGIME_LABEL["extrapolation"]].iloc[0]
    assert extrap_row["count"] == len(extrap_direct)
    assert extrap_row["RMSE"] == pytest.approx(
        np.sqrt(np.mean(extrap_error**2)), rel=1e-6
    )


def test_r2_and_normalised_error_match_manual_formulas(tmp_path):
    frame = day_frame(500, error=3.0, seed=7)
    ps.write_predictions(frame, "pretrained_stec", "own", 2024, 200, root=tmp_path)

    table = trs.collect(store_root=tmp_path)
    row = table[table["regime"] == REGIME_LABEL["extrapolation"]].iloc[0]

    truth = frame["true_stec"].to_numpy()
    pred = frame["stec_pred"].to_numpy()
    error = pred - truth
    rmse = np.sqrt(np.mean(error**2))
    mean_stec = truth.mean()
    r2 = 1 - np.mean(error**2) / np.var(truth)

    assert row["R2"] == pytest.approx(r2, rel=1e-6)
    assert row["mean_STEC"] == pytest.approx(mean_stec, rel=1e-6)
    assert row["nRMSE_%"] == pytest.approx(100 * rmse / mean_stec, rel=1e-6)


def test_regime_with_no_stored_days_is_zero_count_not_a_crash(tmp_path):
    """The store in this test has only an extrapolation day - interpolation must come
    back as an explicit zero-count row with NaN statistics, not raise or be omitted."""
    ps.write_predictions(
        day_frame(20, error=1.0, seed=1),
        "pretrained_stec",
        "own",
        2024,
        200,
        root=tmp_path,
    )

    table = trs.collect(store_root=tmp_path)
    interp_row = table[table["regime"] == REGIME_LABEL["interpolation"]].iloc[0]
    assert interp_row["count"] == 0
    assert np.isnan(interp_row["RMSE"])


def test_collect_excludes_non_finite_pairs(tmp_path):
    frame = day_frame(10, error=2.0, seed=4)
    frame.loc[0, "stec_pred"] = np.nan
    ps.write_predictions(frame, "pretrained_stec", "own", 2024, 200, root=tmp_path)

    table = trs.collect(store_root=tmp_path)
    row = table[table["regime"] == REGIME_LABEL["extrapolation"]].iloc[0]
    assert row["count"] == 9


def test_collect_raises_file_not_found_when_the_store_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        trs.collect(store_root=tmp_path)


def test_rows_are_returned_in_interpolation_then_extrapolation_order(tmp_path):
    ps.write_predictions(
        day_frame(5, error=1.0, seed=1),
        "pretrained_stec",
        "own",
        2023,
        100,
        root=tmp_path,
    )
    ps.write_predictions(
        day_frame(5, error=1.0, seed=2),
        "pretrained_stec",
        "own",
        2024,
        200,
        root=tmp_path,
    )
    table = trs.collect(store_root=tmp_path)
    assert list(table["regime"]) == [
        REGIME_LABEL["interpolation"],
        REGIME_LABEL["extrapolation"],
    ]
