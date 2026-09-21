"""Per-day and pooled STEC metrics for Tables 3 and 4, computed from the store.

The published `summary_statistics.csv` was aggregated from per-day metrics that
each evaluation run wrote at inference time. Two things make that no longer the
right source: its IGS GIM column was computed against the wrong day's map on 12
days of 2024 (see `repair_gim_baseline.py`), and it cannot be recomputed without
re-running inference. The prediction store holds every quantity those tables
need, so this derives them directly - no GPU, and it picks up the repaired GIM
automatically.

Reproduces the published aggregation exactly: Tables 3 and 4 report the **mean
of the daily RMSE**, not the observation-pooled RMSE. Both are written, because
they answer different questions and differ by ~0.3 TECU here - the mean of daily
values weights a sparse day like a dense one, which is what the paper did and
what a reader comparing against it needs. `pooled_*` columns are the
count-weighted alternative.

Usage::

    python src/analysis/daily_metrics.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import prediction_store  # noqa: E402

logger = logging.getLogger(__name__)

# Store column -> the label Tables 3 and 4 use, so the output is directly
# comparable with the published summary_statistics.csv.
MODELS = {
    "stec_pred": "Direct STEC Model",
    "pretrained_stec_pred": "Pretrained STEC",
    "vtec_model_stec": "VTEC + Mapping",
    "gim_stec": "IGS GIM",
}
TRUTH_COLUMN = "true_stec"
DATASET_LABELS = {"own": "own_vtec_gim", "madrigal": "madrigal_vtec_gim"}


def day_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict | None:
    keep = np.isfinite(truth) & np.isfinite(prediction)
    if keep.sum() == 0:
        return None
    truth, prediction = truth[keep], prediction[keep]
    error = prediction - truth
    variance = float(np.var(truth))
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "R²": float(1 - np.mean(error**2) / variance) if variance > 0 else np.nan,
        "Bias": float(np.mean(error)),
        "Std": float(np.std(error)),
        "Count": int(truth.size),
    }


def collect(model_variant: str, root: Path) -> pd.DataFrame:
    rows = []
    for dataset, label in DATASET_LABELS.items():
        try:
            days = prediction_store.available_days(model_variant, dataset, root=root)
        except FileNotFoundError:
            logger.warning(f"⚠️  no {dataset} store under {root}")
            continue
        logger.info(f"{dataset}: {len(days)} day(s)")

        for year, doy in days:
            # Not every day carries every baseline - the pretrained column is
            # absent where the run had no baseline, and Madrigal days differ
            # again - so ask only for what this file actually has.
            path = prediction_store.store_path(model_variant, dataset, year, doy, root)
            present = set(pq.ParquetFile(path).schema.names)
            wanted = [c for c in (TRUTH_COLUMN, *MODELS) if c in present]
            if TRUTH_COLUMN not in wanted:
                logger.warning(f"⚠️  {year}-{doy:03d} has no {TRUTH_COLUMN}, skipping")
                continue
            frame = prediction_store.read_predictions(
                model_variant,
                dataset,
                years=[year],
                doys=[doy],
                root=root,
                columns=wanted,
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
    def aggregate(group: pd.DataFrame) -> pd.Series:
        counts = group["Count"]
        return pd.Series(
            {
                "RMSE_mean": group["RMSE"].mean(),
                "RMSE_std": group["RMSE"].std(),
                "MAE_mean": group["MAE"].mean(),
                "MAE_std": group["MAE"].std(),
                "R2_mean": group["R²"].mean(),
                "R2_std": group["R²"].std(),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store_root", type=Path, default=Path("predictions"))
    parser.add_argument("--model_variant", type=str, default="finetuned_stec")
    parser.add_argument(
        "--output_dir", type=Path, default=Path("multiday_results/daily_metrics")
    )
    parser.add_argument(
        "--published",
        type=Path,
        default=Path(
            "multiday_results/with_pretrained_baseline/summary/summary_statistics.csv"
        ),
        help="published table to diff against, so the correction is explicit",
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
        summary[["dataset", "Model", "RMSE_mean", "MAE_mean", "R2_mean", "Num_days"]]
        .round(4)
        .to_string(index=False)
    )

    if args.published.exists():
        published = pd.read_csv(args.published)
        comparison = summary.merge(
            published[["Dataset", "Model", "RMSE_mean", "Num_days"]].rename(
                columns={
                    "Dataset": "dataset",
                    "RMSE_mean": "RMSE_published",
                    "Num_days": "days_published",
                }
            ),
            on=["dataset", "Model"],
            how="left",
        )
        comparison["delta"] = comparison["RMSE_mean"] - comparison["RMSE_published"]
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
                "⚠️  the store covers fewer days than the published table - these "
                "numbers are not yet the final correction"
            )

    logger.info(f"💾 {args.output_dir}")


if __name__ == "__main__":
    main()
