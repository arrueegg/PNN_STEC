"""Per-day and pooled STEC metrics for Tables 3 and 4, computed from the store.

The published `summary_statistics.csv` was aggregated from per-day metrics that each
evaluation run wrote at inference time. Two things make that no longer the right source:
its IGS GIM column was computed against the wrong day's map on 12 days of 2024 (see
`repair_gim_baseline.py`), and it cannot be recomputed without re-running inference. The
prediction store holds every quantity these tables need, so this derives them directly -
no GPU, and it picks up the repaired GIM automatically.

Reproduces the published aggregation exactly: Tables 3 and 4 report the **mean of the
daily RMSE**, not the observation-pooled RMSE. Both are written, because they answer
different questions and differ by several tenths of a TECU - the mean of daily values
weights a sparse day like a dense one, which is what the paper did and what a reader
comparing against it needs. `pooled_*` columns are the count-weighted alternative.

The store is read one day at a time via `prediction_store.iter_days` - concatenating all
242 days first is what OOM-killed the earlier driver at ~580 M rows. Every quantity here
is a per-day metric or a count, so day-at-a-time accumulation is exact, not approximate.

Usage::

    python -m stec.analysis.daily_metrics
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..inference import prediction_store as ps
from ..config import paths

logger = logging.getLogger(__name__)

# Store column -> the label Tables 3 and 4 use, so the output is directly comparable with
# the published summary_statistics.csv. All four are read from the same model_variant
# partition: pretrained_stec_pred lives alongside stec_pred in the finetuned_stec store,
# not in a separate pretrained_stec partition.
MODELS = {
    "stec_pred": "Direct STEC Model",
    "pretrained_stec_pred": "Pretrained STEC",
    "vtec_model_stec": "VTEC + Mapping",
    "gim_stec": "IGS GIM",
}
TRUTH_COLUMN = "true_stec"
DATASET_LABELS = {"own": "own_vtec_gim", "madrigal": "madrigal_vtec_gim"}

# The pre-rebuild store, resolved in one place so this file does not become a fifth
# copy of an absolute path. paths.py honours STEC_DATA_ROOT / STEC_ARTIFACT_ROOT, so a
# reader of the published code can point it elsewhere without editing source.
DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("daily_metrics", rebuilt=True)

# The aggregation this analysis supersedes (see the module docstring for why it is no
# longer the right source - it predates the GIM day-lookup repair and cannot be
# recomputed without re-running inference). Diffing against it is what turns "the GIM
# column changed" into a checkable number instead of an implicit claim.
PUBLISHED_SUMMARY = (
    paths.LEGACY_MULTIDAY
    / "with_pretrained_baseline"
    / "summary"
    / "summary_statistics.csv"
)


def day_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict | None:
    """RMSE/MAE/R2/Bias/Std for one model on one day, pairwise-excluding NaNs."""
    keep = np.isfinite(truth) & np.isfinite(prediction)
    if keep.sum() == 0:
        return None
    truth, prediction = truth[keep], prediction[keep]
    error = prediction - truth
    variance = float(np.var(truth))
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "R2": float(1 - np.mean(error**2) / variance) if variance > 0 else np.nan,
        "Bias": float(np.mean(error)),
        "Std": float(np.std(error)),
        "Count": int(truth.size),
    }


def _wanted_columns(path: Path) -> list[str]:
    """Restrict the read to columns this day's file actually has.

    Not every day carries every baseline - the pretrained column is absent where the run
    had no baseline, and Madrigal days differ again - so ask only for what is present
    rather than requesting a fixed column list that would fail on some days.
    """
    present = set(pq.ParquetFile(path).schema.names)
    return [column for column in (TRUTH_COLUMN, *MODELS) if column in present]


def collect(model_variant: str, store_root: Path) -> pd.DataFrame:
    """Per-day, per-model metrics across both datasets, streamed day by day."""
    rows: list[dict] = []
    for dataset, label in DATASET_LABELS.items():
        try:
            days = ps.available_days(model_variant, dataset, root=store_root)
        except FileNotFoundError:
            logger.warning(f"no {dataset} store under {store_root}")
            continue
        logger.info(f"{dataset}: {len(days)} day(s)")

        for year, doy in days:
            path = ps.store_path(model_variant, dataset, year, doy, store_root)
            wanted = _wanted_columns(path)
            if TRUTH_COLUMN not in wanted:
                logger.warning(f"{year}-{doy:03d} has no {TRUTH_COLUMN}, skipping")
                continue

            _, _, frame = next(
                ps.iter_days(
                    model_variant,
                    dataset,
                    years=[year],
                    doys=[doy],
                    columns=wanted,
                    root=store_root,
                )
            )
            truth = frame[TRUTH_COLUMN].to_numpy(float)
            for column, name in MODELS.items():
                if column not in frame.columns:
                    continue
                metrics = day_metrics(truth, frame[column].to_numpy(float))
                if metrics is None:
                    continue
                rows.append(
                    {
                        "date": f"{year}-{doy:03d}",
                        "year": year,
                        "doy": doy,
                        "dataset": label,
                        "Model": name,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def summarise(per_day: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-day rows to one row per (dataset, model).

    RMSE_mean/MAE_mean/R2_mean are the mean of the daily values, matching the published
    tables. pooled_RMSE/pooled_MAE recombine the per-day sums of squared/absolute error
    (recoverable from `RMSE**2 * Count` and `MAE * Count`) into the observation-weighted
    statistic - equivalent to computing it over the concatenated store, without holding
    the store in memory to do so.
    """

    def aggregate(group: pd.DataFrame) -> pd.Series:
        counts = group["Count"]
        return pd.Series(
            {
                "RMSE_mean": group["RMSE"].mean(),
                "RMSE_std": group["RMSE"].std(),
                "MAE_mean": group["MAE"].mean(),
                "MAE_std": group["MAE"].std(),
                "R2_mean": group["R2"].mean(),
                "R2_std": group["R2"].std(),
                "Num_days": group["doy"].nunique(),
                "pooled_RMSE": float(
                    np.sqrt((counts * group["RMSE"] ** 2).sum() / counts.sum())
                ),
                "pooled_MAE": float((counts * group["MAE"]).sum() / counts.sum()),
                "observations": int(counts.sum()),
            }
        )

    return (
        per_day.groupby(["dataset", "Model"], observed=True)
        .apply(aggregate, include_groups=False)
        .reset_index()
    )


def compare_to_published(
    summary: pd.DataFrame, published: pd.DataFrame
) -> pd.DataFrame:
    """Diff the recomputed summary against the pre-rebuild `summary_statistics.csv`.

    `delta` is the recomputed RMSE_mean minus the published one, so a change caused by
    the GIM day-lookup repair (or any other correction this module picks up
    automatically) is a checkable number rather than an implicit claim. Joined on
    (dataset, Model); `published`'s own column names (`Dataset`, capitalised) are
    renamed to match `summary`'s.
    """
    reference = published[["Dataset", "Model", "RMSE_mean", "Num_days"]].rename(
        columns={
            "Dataset": "dataset",
            "RMSE_mean": "RMSE_published",
            "Num_days": "days_published",
        }
    )
    comparison = summary.merge(reference, on=["dataset", "Model"], how="left")
    comparison["delta"] = comparison["RMSE_mean"] - comparison["RMSE_published"]
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--model-variant", type=str, default="finetuned_stec")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--published",
        type=Path,
        default=PUBLISHED_SUMMARY,
        help="pre-rebuild summary_statistics.csv to diff the recomputed RMSE against.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    per_day = collect(args.model_variant, args.store_root)
    if per_day.empty:
        raise RuntimeError("the prediction store produced no metrics")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_day.to_csv(args.output_dir / "per_day.csv", index=False)
    summary = summarise(per_day)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    print("=== Tables 3 and 4, recomputed from the prediction store ===")
    print(
        summary[["dataset", "Model", "RMSE_mean", "pooled_RMSE", "Num_days"]]
        .round(4)
        .to_string(index=False)
    )

    if args.published.exists():
        published = pd.read_csv(args.published)
        comparison = compare_to_published(summary, published)
        comparison.to_csv(args.output_dir / "vs_published.csv", index=False)
        print("\n=== against the published table ===")
        print(
            comparison[
                ["dataset", "Model", "RMSE_published", "RMSE_mean", "delta", "Num_days"]
            ]
            .round(4)
            .to_string(index=False)
        )
        incomplete = comparison[comparison["Num_days"] < comparison["days_published"]]
        if not incomplete.empty:
            logger.warning(
                "the store covers fewer days than the published table - these "
                "numbers are not yet the final correction"
            )
    else:
        logger.info(
            f"no published summary at {args.published} - skipping vs_published.csv"
        )

    logger.info(f"wrote per_day.csv and summary.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
