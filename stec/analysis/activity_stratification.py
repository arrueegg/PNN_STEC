"""STEC accuracy stratified by geomagnetic activity (Dst) and solar flux (F10.7).

Ported from ``src/analysis/activity_stratification.py`` in the live checkout. Answers
the part of reviewer comment R1.4 that Figures 5-8 do not already cover. The published
figures stratify by elevation, geomagnetic latitude, local time and season, but never
by activity level, so the manuscript cannot say what happens to the STEC error itself
during the disturbed periods of 2024. The positioning-domain answer is
``storm_stratification.py``; this is the STEC-domain counterpart.

Uses the per-day metrics ``daily_metrics.py`` computes from the prediction store,
joined to the hourly OMNI indices. No inference and no GPU.

**Must run after `repair_gim_baseline`.** The un-repaired IGS GIM baseline - where a
truncating cast on a float32-denormalised ``doy`` loaded the *previous* day's IONEX map
on 12 days of 2024 (DOY 184-189 and 225-230, see ``stec/baselines/gim.py``) - inflated
the published IGS GIM RMSE from 8.28 to 8.56 TECU and **reversed this analysis's R1.4
conclusion**: whether the model's advantage over the operational GIM baseline survives
a storm depends on the size of the GIM error it is being compared against. Because
``daily_metrics.py`` reads the prediction store directly, this analysis has no way to
tell from ``per_day.csv`` alone whether the store it was computed from was ever
checked. ``require_repaired_daily_metrics`` below therefore looks for
``repair_gim_baseline``'s own report file and refuses to run without it, rather than
silently computing from a store that may still carry the bug - the previous version of
this module fell back to the un-recomputable, contaminated ``all_results.csv`` on a
mere warning, which is exactly how the reversed conclusion happened.

Two design points worth stating in the paper:

* **F10.7 is constant within a day**, so stratifying by it is inherently a
  between-day comparison; the daily metrics are the natural granularity. Dst also
  enters at daily resolution here (the daily minimum, the conventional storm
  measure). A within-day Dst breakdown needs per-observation indices and is a
  separate analysis on the prediction store.
* **Absolute RMSE rises with activity partly because STEC itself does.** The
  improvement over the operational baseline is reported alongside, because it is
  scale-free and is what actually answers "does the advantage survive a storm".

Both Dst and F10.7 bin edges are fixed module constants, never derived from the test
period's own distribution (contrast the original source, which computed F10.7
terciles from the data being summarised - two test periods with different solar
activity would then not be comparable, because "high" would mean something different
in each). Daily metrics are pooled by observation count, not averaged: RMSE over a bin
is ``sqrt(sum(n_i * RMSE_i^2) / sum(n_i))``. Averaging daily RMSEs would weight a
sparse day the same as a dense one.

Usage::

    python -m stec.analysis.activity_stratification
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..config import paths
from .daily_metrics import DEFAULT_OUTPUT_DIR as DAILY_METRICS_DIR

logger = logging.getLogger(__name__)

DEFAULT_SWI_PATH = paths.OMNI_INDICES
DEFAULT_DAILY_METRICS_CSV = DAILY_METRICS_DIR / "per_day.csv"

# Written by repair_gim_baseline.py (not yet ported to stec/analysis - see the module
# docstring), one row per stored day, regardless of whether that day needed a fix. Its
# mere presence is the evidence the check ran; an empty "repaired" column after a full
# repair is the *expected*, healthy outcome; it is not itself a failure signal.
DEFAULT_REPAIR_REPORT = (
    paths.analysis_result_dir("repair_gim_baseline", rebuilt=False)
    / "gim_repair_report.csv"
)

# Conventional geomagnetic storm classification on the daily minimum Dst.
DST_BINS = [-1000, -100, -50, -30, 1000]
DST_LABELS = [
    "intense\n(≤ −100 nT)",
    "moderate\n(−100 to −50)",
    "weak\n(−50 to −30)",
    "quiet\n(> −30)",
]

# Conventional F10.7 solar-flux activity classification (solar flux units, sfu),
# following the low/moderate/elevated/high bands NOAA/SWPC uses for solar-cycle
# context. Fixed for the same reason DST_BINS is fixed: a quiet stretch of the test
# period and a solar-maximum stretch must land in the same absolute band, not each be
# rescaled to its own tercile.
F107_BINS = [0, 100, 150, 200, 1000]
F107_LABELS = [
    "low\n(< 100 sfu)",
    "moderate\n(100–150)",
    "elevated\n(150–200)",
    "high\n(≥ 200 sfu)",
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
            "R2": float((n * group["R2"]).sum() / n.sum()),
            "days": int(group["doy"].nunique()),
            "observations": int(n.sum()),
        }
    )


def require_repaired_daily_metrics(
    daily_metrics_csv: Path, repair_report: Path
) -> None:
    """Fail loudly rather than compute this analysis from an unchecked GIM baseline.

    Two separate things must be true before a number here can be trusted: the per-day
    metrics must exist (``daily_metrics.py`` has run), and ``repair_gim_baseline`` must
    have run against the store those metrics were derived from. Neither is inferred
    from the other - a `per_day.csv` on disk says nothing about whether the store
    behind it was ever checked - so both are required explicitly. See the module
    docstring for why silently falling back on a missing repair report is the failure
    mode this replaces.
    """
    if not daily_metrics_csv.exists():
        raise FileNotFoundError(
            f"{daily_metrics_csv} does not exist. Run `python -m "
            "stec.analysis.daily_metrics` first - this analysis has no fallback to "
            "the un-recomputable summary_statistics.csv."
        )
    if not repair_report.exists():
        raise FileNotFoundError(
            f"{repair_report} does not exist: repair_gim_baseline has not been run "
            "against the prediction store that daily_metrics.py reads. The "
            "un-repaired IGS GIM baseline reverses this analysis's R1.4 conclusion "
            "(see module docstring), so this refuses to compute from an unchecked "
            "store. Run repair_gim_baseline.py --apply, then re-run daily_metrics, "
            "before this analysis."
        )


def stratify(
    results_csv: Path,
    year: int,
    dataset: str = "own_vtec_gim",
    swi_path: Path = DEFAULT_SWI_PATH,
    repair_report: Path = DEFAULT_REPAIR_REPORT,
) -> dict[str, pd.DataFrame]:
    """Pool the daily metrics into Dst and F10.7 bins, per model."""
    require_repaired_daily_metrics(results_csv, repair_report)

    results = pd.read_csv(results_csv)
    results = results[results["dataset"] == dataset]
    merged = results.merge(load_daily_indices(year, swi_path), on="doy", how="inner")

    merged["dst_bin"] = pd.cut(merged["dst_min"], bins=DST_BINS, labels=DST_LABELS)
    merged["f107_bin"] = pd.cut(merged["f107"], bins=F107_BINS, labels=F107_LABELS)

    tables = {}
    for key, column, labels in (
        ("dst", "dst_bin", DST_LABELS),
        ("f107", "f107_bin", F107_LABELS),
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
        "--daily-metrics-csv", type=Path, default=DEFAULT_DAILY_METRICS_CSV
    )
    parser.add_argument("--repair-report", type=Path, default=DEFAULT_REPAIR_REPORT)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--dataset", type=str, default="own_vtec_gim")
    parser.add_argument("--swi-path", type=Path, default=DEFAULT_SWI_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.analysis_result_dir("activity_stratification", rebuilt=True),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    tables = stratify(
        args.daily_metrics_csv,
        args.year,
        args.dataset,
        args.swi_path,
        args.repair_report,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for key, table in tables.items():
        path = args.output_dir / f"by_{key}.csv"
        bin_col = f"{key}_bin"
        # DST_LABELS/F107_LABELS embed a `\n` so the bin reads as two lines on a plot
        # axis - useful there, but a raw newline inside a CSV cell breaks any tool that
        # counts rows by counting lines, which is exactly how this pipeline's own
        # provenance row count (stec/pipeline/provenance.py) was getting them wrong until
        # that was fixed to parse CSV properly. Flatten to one line for the file; the
        # in-memory `table` used for the console printout below keeps the original
        # multi-line labels. A consumer that wants the line break back for plotting
        # (stec/viz/revision_figures.py's activity figures currently read this column
        # straight off disk into a tick label) needs to restore it, e.g. from
        # DST_LABELS/F107_LABELS keyed by the flattened form - not yet done as of this
        # change.
        to_write = table.copy()
        to_write[bin_col] = (
            to_write[bin_col].astype(str).str.replace("\n", " ", regex=False)
        )
        to_write.to_csv(path, index=False)
        logger.info(f"wrote {path}")
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
