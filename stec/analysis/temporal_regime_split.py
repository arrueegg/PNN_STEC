"""Interpolation vs extrapolation regime comparison (R2.1), computed from the store.

`docs/revision/response_to_reviewers.md`'s answer to R2.1 quotes RMSE 14.05 vs 7.65 TECU
and normalised error 26.9% vs 31.0% for the pretrained model's held-out test set, split
into two temporal regimes:

- **interpolation**: 2014-2023 test months, each surrounded by training data from the
  same year.
- **extrapolation**: all of 2024, predicted purely from *past* observations (the daily
  fine-tunes never see future data either).

That split is `src/training/training_utils.py`'s `split_test_data_by_date`: a single
`datetime(2024, 5, 1)` boundary, `< boundary` is interpolation and `>= boundary` is
extrapolation, applied to a `year`/`doy` pair converted with
`datetime(year, 1, 1) + timedelta(days=doy - 1)`. In practice this collapses to a
year split - the pretrained model's 2024 test days only ever start at DOY 122 (the
train/val/test date lists reserve 2024 test as DOY 122-366, see
`stec/data/splits/test_dates.list`), so no 2024 row falls before the May 1 boundary and
the "regime" and "calendar year" splits agree exactly. Checked directly against the real
store: every `predictions/pretrained_stec/own` file with `year=2024` has `doy` in
122-366, confirming this.

**Reading from the store instead of re-deriving `year`/`doy` from the model's inputs
sidesteps a real bug, not a hypothetical one.** `split_test_data_by_date` builds its
`year`/`doy` from `int(row["year"])`/`int(row["doy"])` - a *truncating* cast on values
that round-tripped through the model's `(doy-1)/365` normalisation and its float32
inverse, which lands 26 days of the year just under the integer (2024 DOY 189 comes back
as 188.99998). That is the exact defect
`stec.inference.prediction_store` documents and fixes at the write site, and the exact
one `compare_stec_vtec_gim.py` had before it was repaired (Table 4: 8.56 -> 8.28). Unlike
that site, `split_test_data_by_date` was never fixed - it is still `int()`, not
`round()`, in the checked-in `src/` today. It happens not to change the *headline*
numbers here - this module's `--compare-to` diff against the `src/`-produced CSV comes
out at 0.0000 TECU, and the row counts this module derives from the store's own
partition keys match that CSV exactly, 4,400,934 / 5,599,066 - because the
26 affected days of the year do not include the boundary itself (May 1 = DOY 122, not one
of the 12 truncation-affected days named in `compare_stec_vtec_gim.py`'s history: DOY
184-189 and 225-230) - but that is a property of *this specific boundary*, not something
the original code guaranteed. Reading `year`/`doy` from `prediction_store`'s directory
partition (`year=<YYYY>/doy=<DDD>.parquet`) avoids the question entirely: those keys are
the caller's authoritative values, overwritten at write time
(`prediction_store.write_predictions`), never reconstructed from a float.

Reads `predictions/pretrained_stec/own` - the pretrained model's own held-out test set,
spanning 2014-2024 - one day at a time via `prediction_store.iter_days`, and accumulates
per-regime sums (`sum |error|`, `sum error**2`, `sum y`, `sum y**2`, count) rather than
concatenating the store into memory. That store alone is ~10M rows across 544 days; the
full multi-variant store is ~580M rows and has OOM-killed an analysis that tried to read
it whole (see `prediction_store`'s module docstring). Every reported quantity here is a
sum or a count, so day-at-a-time accumulation is exact, matching the pooled statistic
`src/`'s `calculate_metrics(df)` (`src/viz/base.py`) computed over the concatenated
regime frame, not an approximation of it.

Relationship to `stec.analysis.relative_error_metrics`: that module already writes a
`temporal_regime_comparison.csv` and already answers R2.1's numbers correctly - but by
parsing `total_metrics_summary.txt` text files that `src/inference_testset.py`'s live run
wrote under `experiments/Pretrain_STEC_.../{interpolation,extrapolation}/`. It is a
reader of `src/`'s output, not an independent computation, so it cannot run once `src/`
is retired or the experiment directory is gone. This module computes the same comparison
directly from the prediction store, with no `src/` dependency at all. Both are kept for
now (their `canonical_for` values do not collide - `relative_error_metrics`'s Stage
leaves it unset): `relative_error_metrics` remains the source for the *yearly* breakdown
this module does not attempt (R2.2's per-year table), and this module is the
`src/`-independent source for the regime comparison specifically.

Usage::

    python -m stec.analysis.temporal_regime_split
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import paths
from ..inference import prediction_store as ps
from .relative_error_metrics import EXTRAPOLATION_START, REGIME_LABELS, regime_of

logger = logging.getLogger(__name__)

DEFAULT_MODEL_VARIANT = "pretrained_stec"
DEFAULT_DATASET = "own"
DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("temporal_regime_split", rebuilt=True)

# The `src/`-produced CSV this is verified against - see the module docstring.
DEFAULT_COMPARE_TO = paths.LEGACY_MULTIDAY / "temporal_regime_comparison.csv"

REQUIRED_COLUMNS = ["true_stec", "stec_pred"]


def day_regime(year: int, doy: int) -> str:
    """Which regime one store partition's (year, doy) falls in.

    Reproduces `split_test_data_by_date`'s date construction and boundary comparison
    (`regime_of`, imported from `relative_error_metrics` so the two modules can never
    disagree about where the line is), but starting from the store's own directory-key
    integers instead of a value reconstructed from the model's normalised input - see
    the module docstring for why that distinction matters here.
    """
    date = datetime(int(year), 1, 1) + timedelta(days=int(doy) - 1)
    return regime_of(date)


def _new_accumulator() -> dict[str, float]:
    return {
        "n": 0,
        "sum_abs_error": 0.0,
        "sum_sq_error": 0.0,
        "sum_truth": 0.0,
        "sum_truth_sq": 0.0,
    }


def _accumulate(
    accumulator: dict[str, float], truth: np.ndarray, pred: np.ndarray
) -> None:
    """Fold one day's (truth, pred) pair into a regime's running sums, in float64 -
    the store holds float32, and summing ~5.6M rows in float32 loses real precision."""
    error = pred - truth
    accumulator["n"] += truth.size
    accumulator["sum_abs_error"] += float(np.abs(error).sum())
    accumulator["sum_sq_error"] += float(np.square(error).sum())
    accumulator["sum_truth"] += float(truth.sum())
    accumulator["sum_truth_sq"] += float(np.square(truth).sum())


def _summarise_accumulator(accumulator: dict[str, float]) -> dict[str, float]:
    """Pooled RMSE/MAE/R2/mean_STEC/nRMSE%/nMAE% from one regime's running sums.

    R2 uses the population variance of the truth (`sum_truth_sq/n - mean**2`), matching
    `sklearn.metrics.r2_score`'s `1 - SS_res/SS_tot` - the formula `src/viz/base.py`'s
    `create_temporal_metrics_summaries` used to produce the numbers this reproduces.
    """
    n = accumulator["n"]
    if n == 0:
        return {
            "count": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "mean_STEC": np.nan,
            "nRMSE_%": np.nan,
            "nMAE_%": np.nan,
        }

    rmse = float(np.sqrt(accumulator["sum_sq_error"] / n))
    mae = accumulator["sum_abs_error"] / n
    mean_truth = accumulator["sum_truth"] / n
    variance_truth = accumulator["sum_truth_sq"] / n - mean_truth**2
    r2 = (
        1 - (accumulator["sum_sq_error"] / n) / variance_truth
        if variance_truth > 0
        else np.nan
    )
    nrmse_pct = 100 * rmse / mean_truth if mean_truth != 0 else np.nan
    nmae_pct = 100 * mae / mean_truth if mean_truth != 0 else np.nan
    return {
        "count": int(n),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "mean_STEC": mean_truth,
        "nRMSE_%": nrmse_pct,
        "nMAE_%": nmae_pct,
    }


def collect(
    model_variant: str = DEFAULT_MODEL_VARIANT,
    dataset: str = DEFAULT_DATASET,
    store_root: Path = DEFAULT_STORE_ROOT,
) -> pd.DataFrame:
    """Pooled interpolation/extrapolation statistics, streamed one store day at a time.

    Rows are in `REGIME_LABELS` order (interpolation, then extrapolation), matching the
    `src/`-produced `temporal_regime_comparison.csv` this is verified against.
    """
    accumulators = {
        "interpolation": _new_accumulator(),
        "extrapolation": _new_accumulator(),
    }

    days = ps.available_days(model_variant, dataset, root=store_root)
    if not days:
        raise FileNotFoundError(
            f"no prediction store at {store_root}/{model_variant}/{dataset}"
        )
    logger.info(f"{model_variant}/{dataset}: {len(days)} day(s)")

    for year, doy in days:
        regime = day_regime(year, doy)
        _, _, frame = next(
            ps.iter_days(
                model_variant,
                dataset,
                years=[year],
                doys=[doy],
                columns=REQUIRED_COLUMNS,
                root=store_root,
            )
        )
        truth = frame["true_stec"].to_numpy(dtype=np.float64)
        pred = frame["stec_pred"].to_numpy(dtype=np.float64)
        keep = np.isfinite(truth) & np.isfinite(pred)
        if keep.sum() == 0:
            logger.warning(
                f"{year}-{doy:03d}: no finite true_stec/stec_pred pairs, skipping"
            )
            continue
        _accumulate(accumulators[regime], truth[keep], pred[keep])

    rows = [
        {"regime": label, **_summarise_accumulator(accumulators[key])}
        for key, label in REGIME_LABELS
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-variant", type=str, default=DEFAULT_MODEL_VARIANT)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=DEFAULT_COMPARE_TO,
        help="src/-produced temporal_regime_comparison.csv to diff against, if present.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    table = collect(args.model_variant, args.dataset, args.store_root)
    if table["count"].sum() == 0:
        raise RuntimeError("the prediction store produced no regime observations")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "temporal_regime_comparison.csv"
    table.to_csv(output_path, index=False)

    print(
        f"=== Interpolation vs extrapolation regime (R2.1), split "
        f"{EXTRAPOLATION_START:%Y-%m-%d}, from {args.model_variant}/{args.dataset} ==="
    )
    print(table.round(4).to_string(index=False))

    interp, extrap = table.iloc[0], table.iloc[1]
    print(
        f"\nAbsolute RMSE is {extrap['RMSE'] / interp['RMSE']:.2f}x higher under extrapolation,"
        f"\nbut mean STEC is {extrap['mean_STEC'] / interp['mean_STEC']:.2f}x higher, so the"
        f"\nnormalised error is {extrap['nRMSE_%']:.1f}% vs {interp['nRMSE_%']:.1f}%"
        " - lower, not higher."
    )

    if args.compare_to.exists():
        legacy = pd.read_csv(args.compare_to)
        print(f"\n=== against {args.compare_to} (src/-produced) ===")
        print(legacy.round(4).to_string(index=False))
        delta_rmse_interp = interp["RMSE"] - legacy.iloc[0]["RMSE"]
        delta_rmse_extrap = extrap["RMSE"] - legacy.iloc[1]["RMSE"]
        print(
            f"\ndelta RMSE: interpolation {delta_rmse_interp:+.4f} TECU, "
            f"extrapolation {delta_rmse_extrap:+.4f} TECU"
        )
    else:
        logger.info(
            f"no legacy comparison file at {args.compare_to} - skipping the diff"
        )

    logger.info(f"wrote {output_path}")


if __name__ == "__main__":
    main()
