"""Pins the streaming accumulation, the fixed uncertainty-bin edges and the pairwise
NaN handling for `uncertainty_error_relation`.

Follows the style of `test_daily_metrics.py`: the streaming-equivalence test writes
synthetic days through `prediction_store.write_predictions` so it exercises the real
on-disk format. The monotonic-MAE and epistemic-share tests build the frame directly,
since they need exact, noise-free placement into every fixed bin.
"""

from __future__ import annotations

import sys

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
            "pred_aleatoric_unc": rng.uniform(0.1, 30.0, rows),
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


def test_elevation_view_has_expected_bins_and_columns():
    """Restores the source script's by-elevation breakdown, dropped by the port. Bins
    must be `ELEVATION_BINS` in ascending order and columns must match the source
    script's own names exactly, not the differently-named uncertainty-bin view above."""
    frame = day_frame(rows=900, seed=30)  # spans the full 5-90 degree elevation range

    table = uer.finalise_elevation(uer.accumulate_day_by_elevation(frame, doy=100))

    assert list(table["bin"]) == uer.ELEVATION_BIN_LABELS
    assert list(table.columns) == [
        "bin",
        "n",
        "mean_sigma",
        "RMSE",
        "MAE",
        "rmse_over_sigma",
        "rms_sigma",
        "rmse_over_rms_sigma",
        "mean_aleatoric",
        "mean_epistemic",
        "epistemic_share_%",
    ]


def test_elevation_streaming_matches_whole_frame_direct_computation(tmp_path):
    """RMSE/MAE/mean_sigma/epistemic_share_% from streamed per-day accumulation must
    equal the same statistics computed on the days concatenated into one frame -
    reproducing the source script's own by-elevation formulas, which differ from the
    uncertainty-bin view's (bin-mean-based, not sum-of-squares-based)."""
    days = [(2024, 140, 400, 11), (2024, 141, 600, 12), (2024, 142, 150, 13)]
    frames = {}
    for year, doy, rows, seed in days:
        frame = day_frame(rows, seed)
        frames[doy] = frame
        ps.write_predictions(frame, "finetuned_stec", "own", year, doy, root=tmp_path)

    rows_accumulated = uer.collect_by_elevation("finetuned_stec", "own", tmp_path)
    table = uer.finalise_elevation(rows_accumulated).set_index("bin")

    whole = pd.concat(frames.values(), ignore_index=True)
    binned = pd.cut(whole["satele"], bins=uer.ELEVATION_BINS).astype(str)
    error = whole["stec_pred"].to_numpy(float) - whole["true_stec"].to_numpy(float)
    direct = pd.DataFrame(
        {
            "bin": binned,
            "_abs": np.abs(error),
            "_sq": error**2,
            "_sigma": whole["pred_total_unc"],
            "_aleatoric": whole["pred_aleatoric_unc"],
            "_epistemic": whole["pred_epistemic_unc"],
        }
    )
    for bin_label, group in direct.groupby("bin", observed=True):
        rmse = np.sqrt(group["_sq"].mean())
        mae = group["_abs"].mean()
        mean_sigma = group["_sigma"].mean()
        mean_aleatoric = group["_aleatoric"].mean()
        mean_epistemic = group["_epistemic"].mean()
        epistemic_share_pct = 100 * (
            mean_epistemic**2 / (mean_epistemic**2 + mean_aleatoric**2)
        )
        # float32 round trip through the parquet store costs ~1e-7 relative.
        assert table.loc[bin_label, "RMSE"] == pytest.approx(rmse, rel=1e-5)
        assert table.loc[bin_label, "MAE"] == pytest.approx(mae, rel=1e-5)
        assert table.loc[bin_label, "mean_sigma"] == pytest.approx(mean_sigma, rel=1e-5)
        assert table.loc[bin_label, "epistemic_share_%"] == pytest.approx(
            epistemic_share_pct, rel=1e-4
        )
        assert table.loc[bin_label, "n"] == len(group)


def test_elevation_nans_are_excluded_pairwise_between_error_and_decomposition():
    """A NaN in the predicted mean must not remove that observation's aleatoric/
    epistemic uncertainty from the elevation view's decomposition tally, and vice
    versa - the same pairwise-exclusion discipline the uncertainty-bin view already
    applies above, extended to the restored elevation view."""
    rows = 60
    rng = np.random.default_rng(4)
    truth = rng.uniform(0, 60, rows)
    frame = pd.DataFrame(
        {
            "satele": rng.uniform(41, 49, rows),  # all in the (40, 50] elevation bin
            "true_stec": truth,
            "stec_pred": truth + rng.normal(0, 1, rows),
            "pred_total_unc": rng.uniform(3.0, 4.0, rows),
            "pred_aleatoric_unc": rng.uniform(0.5, 1.0, rows),
            "pred_epistemic_unc": rng.uniform(0.5, 1.0, rows),
        }
    )
    frame.loc[0:9, "stec_pred"] = np.nan  # 10 rows invalid for the error tally only
    frame.loc[10:19, "pred_epistemic_unc"] = (
        np.nan
    )  # a different 10, decomposition only

    rows_out = uer.accumulate_day_by_elevation(frame, doy=100)
    assert len(rows_out) == 1  # every valid row lands in the single (40, 50] bin
    row = rows_out[0]
    assert row["n"] == rows - 10
    assert row["n_decomposition"] == rows - 10


def test_missing_aleatoric_column_is_skipped_not_errored():
    """A model variant that never predicted an aleatoric term must still produce the
    elevation view's error statistics, with the decomposition columns reporting no
    data rather than crashing the read."""
    frame = day_frame(80, seed=9, missing=["pred_aleatoric_unc"])

    table = uer.finalise_elevation(uer.accumulate_day_by_elevation(frame, doy=160))

    assert (table["n"] > 0).any()
    assert table["mean_aleatoric"].isna().all()
    assert table["epistemic_share_%"].isna().all()


def test_main_pretrained_variant_writes_to_its_own_output_dir(tmp_path, monkeypatch):
    """`uncertainty_error_relation_pretrained` (stec/pipeline/stages.py) invokes `main()`
    with `--model-variant pretrained_stec --dataset own` and its own `--output-dir`. The
    module's output filename suffix keys only on `--dataset` (empty for "own"), so
    `by_uncertainty.csv`/`by_elevation.csv`/`calibrating_factor.csv` would collide with
    the default finetuned_stec/own invocation's files if both stages wrote to the same
    directory - this pins that the pretrained variant's own `--output-dir` is what keeps
    them apart, not a filename difference."""
    store_root = tmp_path / "store"
    frame = day_frame(rows=300, seed=5)
    ps.write_predictions(frame, "pretrained_stec", "own", 2024, 132, root=store_root)

    output_dir = tmp_path / "uncertainty_error_relation_pretrained"
    argv = [
        "uncertainty_error_relation.py",
        "--store-root",
        str(store_root),
        "--model-variant",
        "pretrained_stec",
        "--dataset",
        "own",
        "--output-dir",
        str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    uer.main()

    assert (output_dir / "by_uncertainty.csv").exists()
    assert (output_dir / "by_elevation.csv").exists()
    assert (output_dir / "calibrating_factor.csv").exists()
    by_uncertainty = pd.read_csv(output_dir / "by_uncertainty.csv")
    assert (by_uncertainty["observations"] > 0).any()


def test_calibrating_factor_reports_spread_across_elevation_bands():
    by_elevation = pd.DataFrame(
        {
            "bin": ["(0, 10]", "(10, 20]", "(20, 30]"],
            "n": [100, 200, 300],
            "rmse_over_sigma": [1.60, 1.65, 1.70],
            "rmse_over_rms_sigma": [1.50, 1.55, 1.58],
        }
    )
    factor = uer.calibrating_factor(by_elevation)
    assert factor["factor_median"] == pytest.approx(1.65)
    assert factor["factor_min"] == pytest.approx(1.60)
    assert factor["factor_max"] == pytest.approx(1.70)
    assert factor["factor_spread"] == pytest.approx(0.10)
    # Same spread statistics, computed for the owner-decided RMSE/RMS(sigma) ratio
    # alongside the existing RMSE/mean(sigma) ones above.
    assert factor["factor_median_rms_sigma"] == pytest.approx(1.55)
    assert factor["factor_min_rms_sigma"] == pytest.approx(1.50)
    assert factor["factor_max_rms_sigma"] == pytest.approx(1.58)
    assert factor["factor_spread_rms_sigma"] == pytest.approx(0.08)


# --- rmse_over_rms_sigma = sqrt(sum(e**2) / sum(sigma**2)), exact under streaming -------


def test_by_uncertainty_rms_sigma_equals_sqrt_ratio_of_sums():
    """Tiny synthetic single-bin frame: pin rms_sigma and rmse_over_rms_sigma to the
    exact closed form, not just to a value read back from the same code."""
    truth = np.array([0.0, 0.0, 0.0, 0.0])
    pred = np.array([1.0, -2.0, 3.0, -1.0])  # errors: 1, -2, 3, -1
    sigma = np.array([2.0, 2.0, 2.0, 2.0])  # all in the (1.0, 2.0] bin
    frame = pd.DataFrame(
        {"true_stec": truth, "stec_pred": pred, "pred_total_unc": sigma}
    )

    table = uer.finalise(uer.accumulate_day(frame, doy=100)).set_index("bin")
    row = table.loc["(1.0, 2.0]"]

    sum_sq_error = float(np.sum((pred - truth) ** 2))
    sum_sq_sigma = float(np.sum(sigma**2))
    expected_rms_sigma = np.sqrt(sum_sq_sigma / 4)
    expected_ratio = np.sqrt(sum_sq_error / sum_sq_sigma)

    assert row["rms_sigma"] == pytest.approx(expected_rms_sigma)
    assert row["rmse_over_rms_sigma"] == pytest.approx(expected_ratio)
    # rms_sigma != mean_pred_unc here only because sigma is constant in this fixture -
    # the two coincide exactly for a constant sigma, so this also pins that rms_sigma
    # was actually computed from sigma**2, not aliased to the existing mean.
    assert row["rms_sigma"] == pytest.approx(row["mean_pred_unc"])


def test_elevation_view_rms_sigma_equals_sqrt_ratio_of_sums_with_heterogeneous_sigma():
    """Same closed-form pin as the by-uncertainty test above, but for the by-elevation
    view's own accumulator, and with sigma genuinely varying within the one bin so
    RMS(sigma) is pinned to differ from mean(sigma)."""
    truth = np.zeros(4)
    pred = np.array([1.0, -2.0, 3.0, -1.0])
    sigma = np.array([1.0, 2.0, 3.0, 4.0])  # heterogeneous within one elevation bin
    frame = pd.DataFrame(
        {
            "satele": np.full(4, 15.0),  # all in the (10, 20] elevation bin
            "true_stec": truth,
            "stec_pred": pred,
            "pred_total_unc": sigma,
        }
    )

    table = uer.finalise_elevation(uer.accumulate_day_by_elevation(frame, doy=100))
    row = table.set_index("bin").loc["(10, 20]"]

    expected_rms_sigma = np.sqrt(np.mean(sigma**2))
    expected_ratio = np.sqrt(np.sum((pred - truth) ** 2) / np.sum(sigma**2))

    assert row["rms_sigma"] == pytest.approx(expected_rms_sigma)
    assert row["rmse_over_rms_sigma"] == pytest.approx(expected_ratio)
    # Pins that RMS(sigma) is not just an alias for the existing mean(sigma) column.
    assert row["rms_sigma"] != pytest.approx(row["mean_sigma"])
