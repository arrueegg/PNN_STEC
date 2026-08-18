"""All four methods compared across every stratifier the reviewer asks for (R1.4).

R1.4 says the aggregate density plots are not stratified enough. The manuscript
does stratify - Figures 5-8 cover elevation, geomagnetic latitude, local time and
season - but those figures are built from a single frame with one prediction
column (`src/viz/spatial.py`), so they show the model's own error and never the
baselines. They answer "where is the model worse" and not "where does the model
still beat the alternatives", which is the question the comment actually poses.

`activity_stratification.py` already does this for Dst and F10.7 from the daily
metrics. Those two are between-day quantities, so daily granularity is right for
them. Elevation, geomagnetic latitude, local time and season vary *within* a day
and need per-observation binning, which is what the prediction store provides.

Streams one day at a time and accumulates per (stratifier, bin, method) sums, so
memory is flat regardless of how many days the store holds. Pooling is by
observation count, matching activity_stratification: RMSE over a bin is
sqrt(sum(n_i * RMSE_i^2) / sum(n_i)), never a mean of per-day RMSEs.

Usage::

    python src/analysis/stratified_comparison.py
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

METHODS = {
    "stec_pred": "Direct STEC",
    "pretrained_stec_pred": "Pretrained Direct STEC",
    "vtec_model_stec": "VTEC + Mapping",
    "gim_stec": "IGS GIM + Mapping",
}
BASELINE = "IGS GIM + Mapping"

ELEVATION_BINS = [5, 20, 30, 40, 50, 60, 70, 90]
# Geomagnetic rather than geographic: the ionosphere organises by magnetic
# latitude, and the equatorial anomaly is the feature worth resolving.
GEOMAGNETIC_BINS = [-90, -60, -40, -20, -10, 10, 20, 40, 60, 90]
LOCAL_TIME_BINS = [0, 4, 8, 12, 16, 20, 24]
SEASON_BINS = [0, 80, 172, 264, 356, 367]
SEASON_LABELS = ["winter", "spring", "summer", "autumn", "winter "]

STRATIFIERS = {
    "elevation": ("satele", ELEVATION_BINS, None),
    "geomagnetic_latitude": ("sm_lat_ipp", GEOMAGNETIC_BINS, None),
    "local_time": ("local_time_hours", LOCAL_TIME_BINS, None),
    "season": ("doy", SEASON_BINS, SEASON_LABELS),
}


def accumulate_day(frame: pd.DataFrame) -> list[dict]:
    truth = frame["true_stec"].to_numpy(float)
    rows = []
    for name, (column, bins, labels) in STRATIFIERS.items():
        if column not in frame.columns:
            continue
        binned = pd.cut(frame[column], bins=bins, labels=labels, include_lowest=True)
        for method_column, method in METHODS.items():
            if method_column not in frame.columns:
                continue
            error = frame[method_column].to_numpy(float) - truth
            part = pd.DataFrame({"bin": binned, "_sq": error**2, "_abs": np.abs(error)})
            part = part.dropna(subset=["bin"])
            grouped = part.groupby("bin", observed=True).agg(
                n=("_sq", "size"), sum_sq=("_sq", "sum"), sum_abs=("_abs", "sum")
            )
            for value, row in grouped.iterrows():
                rows.append(
                    {
                        "stratifier": name,
                        "bin": str(value),
                        "Method": method,
                        **row.to_dict(),
                    }
                )
    return rows


def finalise(rows: list[dict]) -> dict[str, pd.DataFrame]:
    pooled = (
        pd.DataFrame(rows)
        .groupby(["stratifier", "bin", "Method"], observed=True)
        .sum()
        .reset_index()
    )
    pooled["RMSE"] = np.sqrt(pooled.sum_sq / pooled.n)
    pooled["MAE"] = pooled.sum_abs / pooled.n
    pooled = pooled.rename(columns={"n": "observations"})

    tables = {}
    for name, group in pooled.groupby("stratifier", observed=True):
        baseline = group[group.Method == BASELINE].set_index("bin")["RMSE"]
        group = group.assign(
            improvement_over_gim_pct=lambda g: 100
            * (g["bin"].map(baseline) - g["RMSE"])
            / g["bin"].map(baseline)
        )
        tables[name] = group[
            ["bin", "Method", "observations", "RMSE", "MAE", "improvement_over_gim_pct"]
        ].reset_index(drop=True)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store_root", type=Path, default=Path("predictions"))
    parser.add_argument("--model_variant", type=str, default="finetuned_stec")
    parser.add_argument("--dataset", type=str, default="own")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("multiday_results/stratified_comparison"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    days = prediction_store.available_days(
        args.model_variant, args.dataset, root=args.store_root
    )
    logger.info(f"streaming {len(days)} day(s)")

    wanted = ["true_stec", *METHODS, *(c for c, _, _ in STRATIFIERS.values())]
    rows = []
    for year, doy in days:
        path = prediction_store.store_path(
            args.model_variant, args.dataset, year, doy, args.store_root
        )
        available = set(pq.ParquetFile(path).schema.names)
        frame = prediction_store.read_predictions(
            args.model_variant,
            args.dataset,
            years=[year],
            doys=[doy],
            root=args.store_root,
            columns=[c for c in dict.fromkeys(wanted) if c in available],
        )
        rows.extend(accumulate_day(frame))

    tables = finalise(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"by_{name}.csv", index=False)
        print(f"\n=== RMSE [TECU] by {name.replace('_', ' ')} ===")
        print(
            table.pivot(index="bin", columns="Method", values="RMSE")
            .reindex(columns=list(METHODS.values()))
            .round(2)
            .to_string()
        )
        print(f"--- Direct STEC advantage over {BASELINE} [%] ---")
        margin = table[table.Method == "Direct STEC"].set_index("bin")
        print(margin["improvement_over_gim_pct"].round(1).to_string())

    logger.info(f"💾 {args.output_dir}")


if __name__ == "__main__":
    main()
