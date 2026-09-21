"""STEC accuracy stratified by geomagnetic activity (Dst) and solar flux (F10.7).

Answers the part of reviewer comment R1.4 that Figures 5-8 do not already cover.
The published figures stratify by elevation, geomagnetic latitude, local time and
season, but never by activity level, so the manuscript cannot say what happens to
the STEC error itself during the disturbed periods of 2024. The positioning-domain
answer is in `storm_stratification.py`; this is the STEC-domain counterpart.

Uses the per-day metrics that already exist for all 242 test days, joined to the
hourly OMNI indices. No inference and no GPU.

Two design points worth stating in the paper:

* **F10.7 is constant within a day**, so stratifying by it is inherently a
  between-day comparison; the daily metrics are the natural granularity. Dst also
  enters at daily resolution here (the daily minimum, the conventional storm
  measure). A within-day Dst breakdown needs per-observation indices and is a
  separate analysis on the prediction store.
* **Absolute RMSE rises with activity partly because STEC itself does.** The
  improvement over the operational baseline is reported alongside, because it is
  scale-free and is what actually answers "does the advantage survive a storm".

Daily metrics are pooled by observation count, not averaged: RMSE over a bin is
sqrt(sum(n_i * RMSE_i^2) / sum(n_i)). Averaging daily RMSEs would weight a sparse
day the same as a dense one.

Usage::

    python src/analysis/activity_stratification.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SWI_PATH = Path("data/omni_hourly_2010-2025.h5")
STORE_DAILY_METRICS = Path("multiday_results/daily_metrics/per_day.csv")

# Conventional geomagnetic storm classification on the daily minimum Dst.
DST_BINS = [-1000, -100, -50, -30, 1000]
DST_LABELS = [
    "intense\n(≤ −100 nT)",
    "moderate\n(−100 to −50)",
    "weak\n(−50 to −30)",
    "quiet\n(> −30)",
]

MODEL_ORDER = ["Direct STEC Model", "Pretrained STEC", "VTEC + Mapping", "IGS GIM"]
GIM_MODEL = "IGS GIM"


def load_daily_indices(year: int, swi_path: Path = DEFAULT_SWI_PATH) -> pd.DataFrame:
    """Daily minimum Dst, maximum Kp and mean F10.7 for `year`."""
    with h5py.File(swi_path, "r") as handle:
        group = handle[str(year)]
        doys = sorted(group.keys(), key=int)
        columns = [
            c.decode() if isinstance(c, bytes) else c
            for c in group[doys[0]].attrs["columns"]
        ]
        dst_col = columns.index("Dst-index,_nT")
        kp_col = columns.index("Kp_index")
        f107_col = columns.index("f107_index")

        records = []
        for doy in doys:
            hourly = np.asarray(group[doy])
            records.append(
                {
                    "doy": int(doy),
                    "dst_min": float(np.nanmin(hourly[:, dst_col])),
                    "kp_max": float(np.nanmax(hourly[:, kp_col])),
                    "f107": float(np.nanmean(hourly[:, f107_col])),
                }
            )
    return pd.DataFrame(records)


def _pool(group: pd.DataFrame) -> pd.Series:
    """Count-weighted pooling of daily metrics within a bin."""
    n = group["Count"]
    return pd.Series(
        {
            "RMSE": float(np.sqrt((n * group["RMSE"] ** 2).sum() / n.sum())),
            "MAE": float((n * group["MAE"]).sum() / n.sum()),
            "R2": float((n * group["R²"]).sum() / n.sum()),
            "days": int(group["doy"].nunique()),
            "observations": int(n.sum()),
        }
    )


def resolve_results_csv(preferred: Path, fallback: Path) -> Path:
    """Prefer the per-day metrics derived from the prediction store.

    `all_results.csv` was aggregated at inference time and its IGS GIM column is
    wrong on 12 days of 2024 (see `repair_gim_baseline.py`). This used to be
    handled by patching those days from the repair report, which was a trap: the
    report only marks a day `repaired` when that run actually changed it, so once
    the store was fixed the report went empty, the patch substituted nothing and
    this analysis silently reverted to the contaminated numbers. A correction
    that erases itself once it succeeds is worse than none.

    `daily_metrics.py` recomputes the same per-day table directly from the
    repaired store with an identical schema, so it is simply the better input.
    It covers only the days the store holds, which is the honest constraint -
    fewer days, right numbers.
    """
    if preferred.exists():
        logger.info(f"using store-derived daily metrics: {preferred}")
        return preferred
    logger.warning(
        f"⚠️  {preferred} not found - falling back to {fallback}, whose IGS GIM "
        "column is wrong on DOY 184-189 and 225-230. Run daily_metrics.py first."
    )
    return fallback


def stratify(
    results_csv: Path,
    year: int,
    dataset: str = "own_vtec_gim",
    swi_path: Path = DEFAULT_SWI_PATH,
) -> dict[str, pd.DataFrame]:
    """Pool the daily metrics into Dst and F10.7 bins, per model."""
    results = pd.read_csv(results_csv)
    results = results[results["dataset"] == dataset]
    merged = results.merge(load_daily_indices(year, swi_path), on="doy", how="inner")

    merged["dst_bin"] = pd.cut(merged["dst_min"], bins=DST_BINS, labels=DST_LABELS)
    # Terciles rather than fixed edges: F10.7 range is specific to the test period.
    f107_edges = merged["f107"].quantile([0, 1 / 3, 2 / 3, 1.0]).values
    f107_labels = [
        f"low\n({f107_edges[0]:.0f}–{f107_edges[1]:.0f})",
        f"medium\n({f107_edges[1]:.0f}–{f107_edges[2]:.0f})",
        f"high\n({f107_edges[2]:.0f}–{f107_edges[3]:.0f})",
    ]
    merged["f107_bin"] = pd.cut(
        merged["f107"], bins=f107_edges, labels=f107_labels, include_lowest=True
    )

    tables = {}
    for key, column, labels in (
        ("dst", "dst_bin", DST_LABELS),
        ("f107", "f107_bin", f107_labels),
    ):
        pooled = (
            merged.groupby(["Model", column], observed=True)
            .apply(_pool, include_groups=False)
            .reset_index()
        )
        pooled = pooled[pooled["Model"].isin(MODEL_ORDER)]

        # Scale-free companion: how much better than the operational baseline,
        # inside each activity bin.
        baseline = pooled[pooled["Model"] == GIM_MODEL].set_index(column)["RMSE"]
        pooled["improvement_over_gim_%"] = pooled.apply(
            lambda r: 100 * (baseline[r[column]] - r["RMSE"]) / baseline[r[column]],
            axis=1,
        )
        # Present both stratifiers with activity increasing left to right, so the
        # Dst and F10.7 panels of the figure read in the same direction. The Dst
        # bins come out of pd.cut in ascending Dst, i.e. most disturbed first.
        display_order = list(reversed(labels)) if key == "dst" else labels
        pooled[column] = pd.Categorical(
            pooled[column], categories=display_order, ordered=True
        )
        tables[key] = pooled.sort_values(["Model", column])

    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="per-day metrics; defaults to the store-derived table when present",
    )
    parser.add_argument(
        "--fallback_results",
        type=Path,
        default=Path(
            "multiday_results/with_pretrained_baseline/summary/all_results.csv"
        ),
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--dataset", type=str, default="own_vtec_gim")
    parser.add_argument("--swi_path", type=Path, default=DEFAULT_SWI_PATH)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("multiday_results/activity_stratification"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    results_csv = args.results or resolve_results_csv(
        STORE_DAILY_METRICS, args.fallback_results
    )
    tables = stratify(results_csv, args.year, args.dataset, args.swi_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for key, table in tables.items():
        path = args.output_dir / f"by_{key}.csv"
        table.to_csv(path, index=False)
        logger.info(f"💾 {path}")
        bin_col = f"{key}_bin"
        print(f"\n=== STEC error stratified by {key.upper()} ===")
        print(
            table.pivot(index=bin_col, columns="Model", values="RMSE")
            .reindex(columns=MODEL_ORDER)
            .round(2)
            .to_string()
        )
        print(f"\n--- improvement over {GIM_MODEL} [%] ---")
        print(
            table.pivot(index=bin_col, columns="Model", values="improvement_over_gim_%")
            .reindex(columns=[m for m in MODEL_ORDER if m != GIM_MODEL])
            .round(1)
            .to_string()
        )
        counts = table[table.Model == "Direct STEC Model"][
            [bin_col, "days", "observations"]
        ]
        print(f"\n--- bin population ---\n{counts.to_string(index=False)}")


if __name__ == "__main__":
    main()
