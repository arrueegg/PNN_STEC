"""Predicted uncertainty against realised error, over the whole test period (R2.6).

The manuscript shows this only for the pretrained model on a scatter plot
(Figure 4) and per-day PNGs exist for the fine-tuned models, but there is no
aggregate over the 2024 test period - which is what the reviewer asked for. The
prediction store makes it a groupby.

Two views, because they answer different questions:

* **by predicted-sigma decile** - within each bin, the mean predicted sigma
  against the realised RMSE. A well-calibrated model puts these on the identity
  line; the ratio is reported so the direction and size of the miscalibration
  are readable per bin rather than only in aggregate.
* **by elevation** - the same comparison where the physics changes, since the
  low-elevation observations are both the hardest and the ones the positioning
  weighting leans on most.

Aleatoric and epistemic parts are reported separately. For the published
architecture the epistemic term is small by construction - only the output layer
is Bayesian - and saying so with a number is the evidence for R1.2.

Deciles are computed on a subsample of the first day's sigma distribution and
then applied to every day, so the bin edges are identical across days and the
per-day results can be pooled without re-reading everything into memory.

Usage::

    python src/analysis/uncertainty_error_relation.py
    python src/analysis/uncertainty_error_relation.py --model_variant pretrained_stec
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import prediction_store  # noqa: E402

logger = logging.getLogger(__name__)

ELEVATION_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
N_QUANTILES = 10
COLUMNS = [
    "true_stec",
    "stec_pred",
    "pred_total_unc",
    "pred_aleatoric_unc",
    "pred_epistemic_unc",
    "satele",
]


def sigma_bin_edges(frame: pd.DataFrame) -> np.ndarray:
    """Decile edges of the predicted sigma, shared by every day."""
    edges = np.quantile(
        frame["pred_total_unc"].to_numpy(float), np.linspace(0, 1, N_QUANTILES + 1)
    )
    # Duplicate edges collapse bins; nudging keeps pd.cut monotonic.
    return np.maximum.accumulate(edges + np.arange(edges.size) * 1e-9)


def accumulate(frame: pd.DataFrame, edges: np.ndarray) -> list[dict]:
    """Sum of squared error and of sigma per bin, so days can be pooled later."""
    error = frame["stec_pred"].to_numpy(float) - frame["true_stec"].to_numpy(float)
    frame = frame.assign(
        _sq_error=error**2,
        _abs_error=np.abs(error),
        sigma_bin=pd.cut(frame["pred_total_unc"], bins=edges, include_lowest=True),
        elevation_bin=pd.cut(frame["satele"], bins=ELEVATION_BINS),
    )

    rows = []
    for key, column in (("sigma", "sigma_bin"), ("elevation", "elevation_bin")):
        grouped = frame.groupby(column, observed=True).agg(
            n=("_sq_error", "size"),
            sum_sq_error=("_sq_error", "sum"),
            sum_abs_error=("_abs_error", "sum"),
            sum_sigma=("pred_total_unc", "sum"),
            sum_aleatoric=("pred_aleatoric_unc", "sum"),
            sum_epistemic=("pred_epistemic_unc", "sum"),
        )
        for label, row in grouped.iterrows():
            rows.append({"view": key, "bin": str(label), **row.to_dict()})
    return rows


def finalise(rows: list[dict]) -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(rows)
    pooled = frame.groupby(["view", "bin"], observed=True).sum().reset_index()
    pooled["RMSE"] = np.sqrt(pooled.sum_sq_error / pooled.n)
    pooled["MAE"] = pooled.sum_abs_error / pooled.n
    pooled["mean_sigma"] = pooled.sum_sigma / pooled.n
    pooled["mean_aleatoric"] = pooled.sum_aleatoric / pooled.n
    pooled["mean_epistemic"] = pooled.sum_epistemic / pooled.n
    # >1 means the model is over-confident in that bin: the error exceeds what
    # the predicted sigma claims.
    pooled["rmse_over_sigma"] = pooled.RMSE / pooled.mean_sigma
    pooled["epistemic_share_%"] = 100 * (
        pooled.mean_epistemic**2 / (pooled.mean_epistemic**2 + pooled.mean_aleatoric**2)
    )
    keep = [
        "bin",
        "n",
        "mean_sigma",
        "RMSE",
        "MAE",
        "rmse_over_sigma",
        "mean_aleatoric",
        "mean_epistemic",
        "epistemic_share_%",
    ]
    return {
        view: group[keep].reset_index(drop=True)
        for view, group in pooled.groupby("view", observed=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store_root", type=Path, default=Path("predictions"))
    parser.add_argument("--model_variant", type=str, default="finetuned_stec")
    parser.add_argument("--dataset", type=str, default="own")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("multiday_results/uncertainty_error_relation"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    days = prediction_store.available_days(
        args.model_variant, args.dataset, root=args.store_root
    )
    logger.info(f"{len(days)} day(s) of {args.model_variant}/{args.dataset}")

    edges, rows = None, []
    for year, doy in days:
        frame = prediction_store.read_predictions(
            args.model_variant,
            args.dataset,
            years=[year],
            doys=[doy],
            root=args.store_root,
            columns=COLUMNS,
        )
        if edges is None:
            edges = sigma_bin_edges(frame)
        rows.extend(accumulate(frame, edges))

    tables = finalise(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.dataset}" if args.dataset != "own" else ""
    for view, table in tables.items():
        table.to_csv(args.output_dir / f"by_{view}{suffix}.csv", index=False)
        print(f"\n=== predicted uncertainty vs realised error, by {view} ===")
        print(
            table[
                [
                    "bin",
                    "n",
                    "mean_sigma",
                    "RMSE",
                    "rmse_over_sigma",
                    "epistemic_share_%",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

    logger.info(f"💾 {args.output_dir}")


if __name__ == "__main__":
    main()
