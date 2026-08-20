"""Pins streaming accumulation against a direct whole-frame computation.

Both `collect`/`summarise` and the direct check below build on the same synthetic store,
written through `prediction_store.write_predictions` so the test exercises the real
on-disk format rather than in-memory frames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stec.analysis import daily_metrics as dm
from stec.inference import prediction_store as ps


def day_frame(
    rows: int, error: float, seed: int, missing: list[str] | None = None
) -> pd.DataFrame:
    """A day where every prediction misses truth by exactly `error` (constant, so the
    day's RMSE and MAE are known analytically without a second implementation)."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, 60, rows)
    frame = pd.DataFrame(
        {
            "station": ["AMC4"] * rows,
            "sat": ["G01"] * rows,
            "satele": rng.uniform(5, 90, rows),
            "true_stec": truth,
            "stec_pred": truth + error,
            "pretrained_stec_pred": truth - error,
            "vtec_model_stec": truth + 2 * error,
            "gim_stec": truth - 2 * error,
        }
    )
    return frame.drop(columns=missing or [])


def test_pooled_metrics_match_direct_whole_frame_computation(tmp_path):
    """pooled_RMSE/pooled_MAE from streamed per-day accumulation must equal the same
    statistic computed on the days concatenated into one frame - that equivalence is the
    entire justification for streaming instead of reading the whole store."""
    days = [(2024, 132, 40, 3.0), (2024, 133, 160, 1.0), (2024, 134, 10, 5.0)]
    frames = {}
    for year, doy, rows, error in days:
        frame = day_frame(rows, error, seed=doy)
        frames[doy] = frame
        ps.write_predictions(frame, "finetuned_stec", "own", year, doy, root=tmp_path)

    per_day = dm.collect("finetuned_stec", tmp_path)
    summary = dm.summarise(per_day)

    whole = pd.concat(frames.values(), ignore_index=True)
    for column, name in dm.MODELS.items():
        row = summary[
            (summary["dataset"] == "own_vtec_gim") & (summary["Model"] == name)
        ].iloc[0]
        error = whole[column].to_numpy(float) - whole["true_stec"].to_numpy(float)
        direct_pooled_rmse = np.sqrt(np.mean(error**2))
        direct_pooled_mae = np.mean(np.abs(error))
        # rel tolerance loosened to float32 precision: the store casts numeric columns
        # to float32 on write, so the round trip alone costs ~1e-7 relative.
        assert row["pooled_RMSE"] == pytest.approx(direct_pooled_rmse, rel=1e-6)
        assert row["pooled_MAE"] == pytest.approx(direct_pooled_mae, rel=1e-6)
        assert row["observations"] == len(whole)


def test_rmse_mean_and_pooled_rmse_differ_for_uneven_day_sizes(tmp_path):
    """A sparse, error-prone day must not dominate RMSE_mean the way it dominates
    pooled_RMSE - that is precisely why the manuscript's headline number (mean of daily
    RMSE) and the observation-pooled number are reported as two separate columns."""
    ps.write_predictions(
        day_frame(2, error=3.0, seed=1),
        "finetuned_stec",
        "own",
        2024,
        132,
        root=tmp_path,
    )
    ps.write_predictions(
        day_frame(8, error=1.0, seed=2),
        "finetuned_stec",
        "own",
        2024,
        133,
        root=tmp_path,
    )

    per_day = dm.collect("finetuned_stec", tmp_path)
    summary = dm.summarise(per_day)
    row = summary[
        (summary["dataset"] == "own_vtec_gim")
        & (summary["Model"] == "Direct STEC Model")
    ].iloc[0]

    assert row["RMSE_mean"] == pytest.approx((3.0 + 1.0) / 2)
    assert row["pooled_RMSE"] == pytest.approx(np.sqrt((2 * 9 + 8 * 1) / 10))
    assert row["RMSE_mean"] != pytest.approx(row["pooled_RMSE"])


def test_missing_model_column_is_skipped_not_errored(tmp_path):
    """gim_stec is absent from the run that had no GIM comparison - it must be dropped
    for that day rather than crash the read or appear as a spurious all-NaN row."""
    ps.write_predictions(
        day_frame(20, error=2.0, seed=1, missing=["gim_stec"]),
        "finetuned_stec",
        "own",
        2024,
        132,
        root=tmp_path,
    )

    per_day = dm.collect("finetuned_stec", tmp_path)
    assert "IGS GIM" not in set(per_day["Model"])
    assert "Direct STEC Model" in set(per_day["Model"])


def test_nan_predictions_are_excluded_pairwise():
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    prediction = np.array([1.0, np.nan, 3.5, 4.5])
    metrics = dm.day_metrics(truth, prediction)
    assert metrics["Count"] == 3
    assert metrics["RMSE"] == pytest.approx(np.sqrt((0**2 + 0.5**2 + 0.5**2) / 3))


def test_day_metrics_returns_none_when_nothing_is_finite():
    truth = np.array([np.nan, np.nan])
    prediction = np.array([1.0, 2.0])
    assert dm.day_metrics(truth, prediction) is None
