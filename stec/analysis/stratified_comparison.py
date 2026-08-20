"""All four methods compared across every stratifier the reviewer asks for (R1.4).

Ported from ``src/analysis/stratified_comparison.py`` in the live checkout. R1.4 says
the aggregate density plots are not stratified enough - the manuscript's Figures 5-8
cover elevation, geomagnetic latitude, local time and season, but each is built from a
single frame with one prediction column, so they show the model's own error and never
the baselines. This answers "where does the model still beat the alternatives", pooling
by observation count so a bin's RMSE is ``sqrt(sum(n_i * RMSE_i^2) / sum(n_i))``, never a
mean of per-day RMSEs.

Elevation, geomagnetic latitude, local time and season vary *within* a day, so this
bins per observation rather than per day (contrast ``activity_stratification.py``,
which stratifies by Dst/F10.7 - between-day quantities - from the daily metrics).

Streamed one day at a time via ``prediction_store.iter_days``, so memory is flat
regardless of how many days the store holds; every quantity reported is a sum or a
count, so the accumulation is exact. Bin edges are fixed module constants, not derived
from any day's data, so every day is binned into the same partition.

NaNs are excluded **pairwise per method**: a method with a missing or invalid
prediction for one observation must not remove that observation from another method's
tally. The original driver computed ``error = frame[method] - truth`` without an
explicit finite check, so a NaN prediction silently poisoned that bin's sum (NaN
propagates through ``sum``); this port masks each method's error independently before
accumulating.

Usage::

    python -m stec.analysis.stratified_comparison
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..inference import prediction_store as ps

logger = logging.getLogger(__name__)

METHODS = {
    "stec_pred": "Direct STEC",
    "vtec_model_stec": "VTEC + Mapping",
    "gim_stec": "IGS GIM",
    "pretrained_stec_pred": "Pretrained",
}
BASELINE = "IGS GIM"
TRUTH_COLUMN = "true_stec"

ELEVATION_BINS = [5, 20, 30, 40, 50, 60, 70, 90]
# Geomagnetic rather than geographic: the ionosphere organises by magnetic latitude,
# and the equatorial anomaly is the feature worth resolving.
GEOMAGNETIC_BINS = [-90, -60, -40, -20, -10, 10, 20, 40, 60, 90]
LOCAL_TIME_BINS = [0, 4, 8, 12, 16, 20, 24]
SEASON_BINS = [0, 80, 172, 264, 356, 367]
# Winter straddles the year boundary, so pd.cut needs two edges for it. The
# trailing-space label keeps them distinct for pd.cut and is stripped straight
# afterwards, which merges the two fragments into one bin.
SEASON_LABELS = ["winter", "spring", "summer", "autumn", "winter "]

# column, bin edges, labels (season only - the rest use pd.cut's own interval labels).
STRATIFIERS = {
    "elevation": ("satele", ELEVATION_BINS, None),
    "geomagnetic_latitude": ("sm_lat_ipp", GEOMAGNETIC_BINS, None),
    "local_time": ("local_time_hours", LOCAL_TIME_BINS, None),
    "season": ("doy", SEASON_BINS, SEASON_LABELS),
}

DEFAULT_STORE_ROOT = Path("/scratch2/arrueegg/WP4/PNN_STEC/predictions")
DEFAULT_OUTPUT_DIR = Path("multiday_results/stratified_comparison_rebuilt")


def add_local_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive local time where the store lacks it.

    ``local_time_hours`` is a model *input*, so it is only stored for models
    configured with it - the pretrained STEC model was not, which would silently
    drop the local-time panel for its test set. It is a function of the observation,
    not of the model: solar local time at the pierce point is UTC plus 15 degrees of
    longitude per hour.
    """
    if "local_time_hours" in frame.columns or not {"sod", "lon_ipp"} <= set(
        frame.columns
    ):
        return frame
    local = (
        frame["sod"].to_numpy(float) / 3600.0 + frame["lon_ipp"].to_numpy(float) / 15.0
    ) % 24.0
    return frame.assign(local_time_hours=local)


def _wanted_columns(path: Path) -> list[str]:
    """Restrict the read to columns this day's file actually has (same reasoning as
    ``daily_metrics._wanted_columns``: not every day carries every baseline)."""
    present = set(pq.ParquetFile(path).schema.names)
    wanted = [
        TRUTH_COLUMN,
        *METHODS,
        *(column for column, _, _ in STRATIFIERS.values()),
        "sod",
        "lon_ipp",
    ]
    return [column for column in dict.fromkeys(wanted) if column in present]


def accumulate_day(frame: pd.DataFrame, doy: int) -> list[dict]:
    """Per (stratifier, bin, method) sums for one day, NaNs excluded pairwise."""
    frame = add_local_time(frame)
    truth = frame[TRUTH_COLUMN].to_numpy(float)
    rows = []
    for stratifier_name, (column, bins, labels) in STRATIFIERS.items():
        if column not in frame.columns:
            continue
        binned = pd.cut(frame[column], bins=bins, labels=labels, include_lowest=True)
        if labels is not None:
            # Strip the disambiguating whitespace so the two winter fragments merge.
            binned = binned.astype(str).str.strip().replace("nan", np.nan)

        for method_column, method in METHODS.items():
            if method_column not in frame.columns:
                continue
            error = frame[method_column].to_numpy(float) - truth
            # Pairwise exclusion: this method's own NaNs must not drop observations
            # that other methods scored validly.
            valid = np.isfinite(error)
            if not valid.any():
                continue
            part = pd.DataFrame(
                {
                    "bin": binned.to_numpy()[valid],
                    "_sq": error[valid] ** 2,
                    "_abs": np.abs(error[valid]),
                }
            ).dropna(subset=["bin"])
            if part.empty:
                continue
            grouped = part.groupby("bin", observed=True).agg(
                n=("_sq", "size"), sum_sq=("_sq", "sum"), sum_abs=("_abs", "sum")
            )
            for bin_value, row in grouped.iterrows():
                rows.append(
                    {
                        "stratifier": stratifier_name,
                        "bin": str(bin_value),
                        "Method": method,
                        "doy": doy,
                        **row.to_dict(),
                    }
                )
    return rows


def finalise(rows: list[dict]) -> dict[str, pd.DataFrame]:
    if not rows:
        raise RuntimeError(
            "no observations were read - the prediction store for this model variant "
            "is empty. Run the inference pass that populates it first; a bare KeyError "
            "here would hide which step actually failed."
        )
    frame = pd.DataFrame(rows)
    # Day count per bin, so an unevenly sampled stratifier cannot be read as if it
    # were balanced (e.g. "winter" on the 2024 test period is 11 December days only).
    days_covered = (
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
        .merge(days_covered, on=["stratifier", "bin"], how="left")
    )
    pooled["RMSE"] = np.sqrt(pooled.sum_sq / pooled.n)
    pooled["MAE"] = pooled.sum_abs / pooled.n
    pooled = pooled.rename(columns={"n": "observations"})

    tables = {}
    for stratifier_name, group in pooled.groupby("stratifier", observed=True):
        if BASELINE in set(group["Method"]):
            baseline_rmse = group[group.Method == BASELINE].set_index("bin")["RMSE"]
            group = group.assign(
                improvement_over_gim_pct=lambda g: 100
                * (g["bin"].map(baseline_rmse) - g["RMSE"])
                / g["bin"].map(baseline_rmse)
            )
        else:
            group = group.assign(improvement_over_gim_pct=np.nan)
        tables[stratifier_name] = group[
            [
                "bin",
                "Method",
                "days",
                "observations",
                "RMSE",
                "MAE",
                "improvement_over_gim_pct",
            ]
        ].reset_index(drop=True)
    return tables


def collect(
    model_variant: str,
    dataset: str,
    store_root: Path,
    doys: list[int] | None = None,
) -> list[dict]:
    """Stream the requested stored days of `model_variant`/`dataset` (or every day if
    `doys` is None), accumulating per-bin sums."""
    rows: list[dict] = []
    for path in ps.day_paths(model_variant, dataset, doys=doys, root=store_root):
        year = int(path.parent.name.split("=")[1])
        doy = int(path.stem.split("=")[1])
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
        rows.extend(accumulate_day(frame, doy))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--model-variant", type=str, default="finetuned_stec")
    parser.add_argument("--dataset", type=str, default="own")
    parser.add_argument(
        "--doys",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to these day-of-year values; default is every day in the store.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    rows = collect(args.model_variant, args.dataset, args.store_root, doys=args.doys)
    tables = finalise(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stratifier_name, table in tables.items():
        table.to_csv(args.output_dir / f"by_{stratifier_name}.csv", index=False)
        print(f"\n=== RMSE [TECU] by {stratifier_name.replace('_', ' ')} ===")
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

    logger.info(f"wrote {len(tables)} table(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
