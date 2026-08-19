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
# Winter straddles the year boundary, so pd.cut needs two edges for it. The
# trailing-space label keeps them distinct for pd.cut and is stripped straight
# afterwards, which merges the two fragments into one bin. On the 2024 test
# period only the December fragment had data and the duplicate never showed;
# the pretrained model's 2014-2024 set has both, and reported "winter" twice.
SEASON_LABELS = ["winter", "spring", "summer", "autumn", "winter "]

STRATIFIERS = {
    "elevation": ("satele", ELEVATION_BINS, None),
    "geomagnetic_latitude": ("sm_lat_ipp", GEOMAGNETIC_BINS, None),
    "local_time": ("local_time_hours", LOCAL_TIME_BINS, None),
    "season": ("doy", SEASON_BINS, SEASON_LABELS),
}


def accumulate_day(frame: pd.DataFrame, doy: int) -> list[dict]:
    truth = frame["true_stec"].to_numpy(float)
    rows = []
    for name, (column, bins, labels) in STRATIFIERS.items():
        if column not in frame.columns:
            continue
        binned = pd.cut(frame[column], bins=bins, labels=labels, include_lowest=True)
        if labels is not None:
            # Strip the disambiguating whitespace so split bins merge.
            binned = binned.astype(str).str.strip().replace("nan", np.nan)
        for method_column, method in METHODS.items():
            if method_column not in frame.columns:
                continue
            error = frame[method_column].to_numpy(float) - truth
            # sum_truth / sum_truth_sq carry the total sum of squares, so R2 is
            # poolable across days without holding any observation in memory.
            part = pd.DataFrame(
                {
                    "bin": binned,
                    "_sq": error**2,
                    "_abs": np.abs(error),
                    "_truth": truth,
                    "_truth_sq": truth**2,
                }
            ).dropna(subset=["bin"])
            grouped = part.groupby("bin", observed=True).agg(
                n=("_sq", "size"),
                sum_sq=("_sq", "sum"),
                sum_abs=("_abs", "sum"),
                sum_truth=("_truth", "sum"),
                sum_truth_sq=("_truth_sq", "sum"),
            )
            for value, row in grouped.iterrows():
                rows.append(
                    {
                        "stratifier": name,
                        "bin": str(value),
                        "Method": method,
                        "doy": doy,
                        **row.to_dict(),
                    }
                )
    return rows


def finalise(rows: list[dict]) -> dict[str, pd.DataFrame]:
    if not rows:
        raise RuntimeError(
            "no observations were read - the prediction store for this model "
            "variant is empty. Run the inference pass that populates it first; "
            "a bare KeyError here hides which step actually failed."
        )
    frame = pd.DataFrame(rows)
    # Day count per bin, so an unevenly sampled stratifier cannot be read as if
    # it were balanced. The 2024 test period runs DOY 122-366, so "winter" is 11
    # December days and nothing else - that has to be visible.
    days = (
        frame.groupby(["stratifier", "bin"], observed=True)["doy"]
        .nunique()
        .rename("days")
        .reset_index()
    )
    pooled = (
        frame.drop(columns=["doy"])
        .groupby(["stratifier", "bin", "Method"], observed=True)
        .sum()
        .reset_index()
        .merge(days, on=["stratifier", "bin"], how="left")
    )
    pooled["RMSE"] = np.sqrt(pooled.sum_sq / pooled.n)
    pooled["MAE"] = pooled.sum_abs / pooled.n
    # SST from the running sums: sum((y - ybar)^2) = sum(y^2) - n*ybar^2
    total_sum_squares = pooled.sum_truth_sq - pooled.sum_truth**2 / pooled.n
    pooled["R2"] = np.where(total_sum_squares > 0, 1 - pooled.sum_sq / total_sum_squares, np.nan)
    pooled = pooled.rename(columns={"n": "observations"})

    tables = {}
    for name, group in pooled.groupby("stratifier", observed=True):
        # The pretrained model's own multi-year test set carries no baselines:
        # the VTEC baseline is fine-tuned per day and exists for 2024 only, so
        # there is nothing to compare against before then. Report the RMSE alone
        # rather than a margin against a baseline that is not there.
        if BASELINE in set(group["Method"]):
            baseline = group[group.Method == BASELINE].set_index("bin")["RMSE"]
            group = group.assign(
                improvement_over_gim_pct=lambda g: 100
                * (g["bin"].map(baseline) - g["RMSE"])
                / g["bin"].map(baseline)
            )
        else:
            group = group.assign(improvement_over_gim_pct=np.nan)
        tables[name] = group[
            ["bin", "Method", "days", "observations", "RMSE", "MAE", "R2",
             "improvement_over_gim_pct"]
        ].reset_index(drop=True)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store_root", type=Path, default=Path("predictions"))
    parser.add_argument("--model_variant", type=str, default="finetuned_stec")
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="rename stec_pred in the output, e.g. 'Pretrained Direct STEC'",
    )
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
        rows.extend(accumulate_day(frame, doy))

    tables = finalise(rows)
    if args.label:
        for table in tables.values():
            table["Method"] = table["Method"].replace({"Direct STEC": args.label})
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
        if table["improvement_over_gim_pct"].notna().any():
            print(f"--- Direct STEC advantage over {BASELINE} [%] ---")
            margin = table[table.Method == "Direct STEC"].set_index("bin")
            print(margin["improvement_over_gim_pct"].round(1).to_string())

    logger.info(f"💾 {args.output_dir}")


if __name__ == "__main__":
    main()
