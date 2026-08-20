"""Pins the streaming accumulation, the fixed uncertainty-bin edges and the pairwise
NaN handling for `uncertainty_error_relation`.

Follows the style of `test_daily_metrics.py`: the streaming-equivalence test writes
synthetic days through `prediction_store.write_predictions` so it exercises the real
on-disk format. The monotonic-MAE and epistemic-share tests build the frame directly,
since they need exact, noise-free placement into every fixed bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stec.analysis import uncertainty_error_relation as uer
from stec.inference import prediction_store as ps


def day_frame(rows: int, seed: int, missing: list[str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, 60, rows)
    frame = pd.DataFrame(
        {
            "satele": rng.uniform(5, 90, rows),
            "true_stec": truth,
            "stec_pred": truth + rng.normal(0, 3, rows),
            "pred_total_unc": rng.uniform(0.2, 32.0, rows),
            "pred_epistemic_unc": rng.uniform(0.05, 6.0, rows),
        }
    )
    return frame.drop(columns=missing or [])


def test_streaming_matches_whole_frame_direct_computation(tmp_path):
    """MAE/RMSE from streamed per-day accumulation must equal the same statistic
    computed on the days concatenated into one frame - the entire justification for
    streaming instead of reading the whole store."""
    days = [(2024, 140, 400, 11), (2024, 141, 600, 12), (2024, 142, 150, 13)]
    frames = {}
    for year, doy, rows, seed in days:
        frame = day_frame(rows, seed)
        frames[doy] = frame
        ps.write_predictions(frame, "finetuned_stec", "own", year, doy, root=tmp_path)

    rows_accumulated = uer.collect("finetuned_stec", "own", tmp_path)
    table = uer.finalise(rows_accumulated).set_index("bin")

    whole = pd.concat(frames.values(), ignore_index=True)
    binned = pd.cut(
        whole["pred_total_unc"], bins=uer.UNCERTAINTY_BINS_TECU, include_lowest=True
    ).astype(str)
    error = whole["stec_pred"].to_numpy(float) - whole["true_stec"].to_numpy(float)
    direct = (
        pd.DataFrame({"bin": binned, "_abs": np.abs(error), "_sq": error**2})
        .groupby("bin", observed=True)
        .agg(n=("_abs", "size"), sum_abs=("_abs", "sum"), sum_sq=("_sq", "sum"))
    )
    direct["MAE"] = direct.sum_abs / direct.n
    direct["RMSE"] = np.sqrt(direct.sum_sq / direct.n)

    for bin_label, row in direct.iterrows():
        # float32 round trip through the parquet store costs ~1e-7 relative.
        assert table.loc[bin_label, "MAE"] == pytest.approx(row["MAE"], rel=1e-5)
        assert table.loc[bin_label, "RMSE"] == pytest.approx(row["RMSE"], rel=1e-5)
        assert table.loc[bin_label, "observations"] == row["n"]


def test_missing_epistemic_column_is_skipped_not_errored(tmp_path):
    """A model variant that never predicted an epistemic term must still produce the
    error-vs-uncertainty numbers, with the epistemic columns reporting no data rather
    than crashing the read."""
    frame = day_frame(80, seed=9, missing=["pred_epistemic_unc"])
    ps.write_predictions(frame, "finetuned_stec", "own", 2024, 160, root=tmp_path)

    rows = uer.collect("finetuned_stec", "own", tmp_path)
    table = uer.finalise(rows)

    assert (table["observations"] > 0).any()
    assert (table["observations_epistemic"] == 0).all()
    assert table["epistemic_share"].isna().all()


def test_nans_are_excluded_pairwise_between_error_and_epistemic():
    """A NaN in the predicted mean must not remove that observation's uncertainty from
    the epistemic-share tally, and vice versa - the two are accumulated independently."""
    rows = 60
    rng = np.random.default_rng(4)
    truth = rng.uniform(0, 60, rows)
    frame = pd.DataFrame(
        {
            "true_stec": truth,
            "stec_pred": truth + rng.normal(0, 1, rows),
            "pred_total_unc": rng.uniform(3.1, 3.9, rows),  # all in the (3.0, 4.0] bin
            "pred_epistemic_unc": rng.uniform(0.5, 1.0, rows),
        }
    )
    frame.loc[0:9, "stec_pred"] = np.nan  # 10 rows invalid for the error tally only
    frame.loc[10:19, "pred_epistemic_unc"] = np.nan  # a different 10, epistemic only

    rows_out = uer.accumulate_day(frame, doy=100)
    assert len(rows_out) == 1  # every valid row lands in the single (3.0, 4.0] bin
    row = rows_out[0]
    assert row["n"] == rows - 10
    assert row["n_epistemic"] == rows - 10


def test_mae_increases_monotonically_and_epistemic_share_is_bounded():
    """Deterministic placement of one sigma value per fixed bin, with |error|
    proportional to sigma, pins two properties at once: the fixed bins partition the
    full range in ascending order, and MAE increases bin-to-bin as designed - not an
    incidental property of random data."""
    sigmas = [0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 8.5, 12.0, 17.0, 25.0, 40.0]
    assert len(sigmas) == len(uer.BIN_LABELS)
    rows_per_bin = 25
    epistemic_fraction = 0.3

    rng = np.random.default_rng(21)
    truth_parts, pred_parts, sigma_parts, epistemic_parts = [], [], [], []
    for sigma in sigmas:
        truth = rng.uniform(0, 60, rows_per_bin)
        truth_parts.append(truth)
        pred_parts.append(truth + 0.5 * sigma)  # |error| = 0.5 * sigma, no noise
        sigma_parts.append(np.full(rows_per_bin, sigma))
        epistemic_parts.append(np.full(rows_per_bin, sigma * epistemic_fraction))

    frame = pd.DataFrame(
        {
            "true_stec": np.concatenate(truth_parts),
            "stec_pred": np.concatenate(pred_parts),
            "pred_total_unc": np.concatenate(sigma_parts),
            "pred_epistemic_unc": np.concatenate(epistemic_parts),
        }
    )

    table = uer.finalise(uer.accumulate_day(frame, doy=100))

    assert list(table["bin"]) == uer.BIN_LABELS
    assert (table["MAE"].diff().dropna() > 0).all()
    assert table["epistemic_share"].between(0, 1).all()
    np.testing.assert_allclose(
        table["epistemic_share"], epistemic_fraction**2, rtol=1e-9
    )
