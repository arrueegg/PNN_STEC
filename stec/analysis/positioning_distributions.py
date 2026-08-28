"""Distribution-based positioning summary, replacing Table 5's mean-with-10 m-exclusion
(owner decision, 2026-08-28).

**The decision this implements.** Table 5 currently reports the mean of per-station-day
3D positioning error after dropping any station-day above 10 m
(`stec.positioning.metrics.OUTLIER_3D_RMS_M`). `positioning_diagnostics.py` and
`positioning_quality_gate.py` (both landed the same day) established why that is the
wrong rule: it filters on the *outcome* being compared, and it is not neutral between
methods - Direct STEC carries 68 station-days above 10 m against IGS GIM's 16, so the
rule disproportionately trims Direct STEC's own worst outcomes. The headline mean swings
from -0.8% (no filter) to +16.4% (5 m) to +4.7% (10 m) depending on a threshold that has
no principled value, while the median sits in an 18.6-20.9% band under every filter
including none at all, and a method-blind quality gate (session completeness only, no
`error_3d_rms`-derived column) lands the mean near zero too. The owner's decision: report
**distributions and medians, with no outcome-based outlier filtering**.

**What must not be lost.** Dropping the filter and reporting only a median would hide
that Direct STEC fails catastrophically more often than GIM - that is a real property of
the method, and operationally the most important one for a positioning user relying on
it. So every table and figure here reports the tail as an explicit exceedance rate
(`exceedance_table`, reusing `positioning_diagnostics.outlier_threshold_counts` rather
than re-deriving the same count) alongside the median - never excluded, never glossed.

**No outcome filter is applied anywhere in this module.** Every function here reads the
full, unfiltered per-station-day table. Where a figure needs a bounded axis to stay
readable, the *view* is clipped (matplotlib axis limits), not the data - see
`stec.viz.positioning_distributions`, which reports how many points fall outside each
clipped view rather than silently cropping them.

Source: `multiday_results/analyses/positioning_coverage/rebuilt/multiday_summary.csv`,
the iono-weighted per-station-day table (the elev arm is mid-re-solve as of 2026-08-28 -
see CLAUDE.md - so this module, like `positioning_diagnostics.py`, states "iono" in every
output rather than leaving the weighting implicit).

Five sections, matching the five figures in `stec.viz.positioning_distributions`:

1. `boxplot_stats` - Tukey box statistics (median/Q1/Q3/whiskers) plus the individual
   outlier ("flier") values, computed by hand so the CSV a figure writes is exactly what
   `ax.bxp()` draws, not a second, possibly-diverging computation matplotlib does
   internally. Whisker convention: matplotlib's own default, 1.5xIQR.
2. `cdf_points` - the full empirical CDF, one row per station-day, so the plotted curve
   and the written CSV are the same thing.
3. Storm/quiet and 5. original/recovered are the same two functions above and
   `percentile_summary`/`exceedance_table` below, called with an extra grouping column
   (`regime`, `population`) rather than separate functions - the distributional
   contrasts are a grouping choice, not a different computation.
4. `percentile_summary` + `exceedance_table` - median, IQR, p95, p99 and exceedance
   counts/rates at 5/10/20/50 m, per group. This is what replaces Table 5's numeric
   content; `main()` also writes it out as `TABLE5_NUMBERS.md`, formatted to be read
   straight into the manuscript.

Usage::

    python -m stec.analysis.positioning_distributions
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import paths
from .positioning_diagnostics import (
    RECOVERED_STEC_DB_ROOT,
    attach_population,
    attach_storm_flag,
    load_positioning_table,
    load_recovered_station_days,
    outlier_threshold_counts,
)
from .positioning_summary import METHOD_ORDER, canonical_positioning_summary
from .storm_stratification import STORM_DST_THRESHOLD_NT

logger = logging.getLogger(__name__)

# The four thresholds Table 5's replacement reports exceedance at. Identical to
# `positioning_diagnostics.OUTLIER_THRESHOLDS_M` (10 m is the project's old rule, kept as
# one point of reference among the four, not a special case) - shared by re-using
# `outlier_threshold_counts` directly rather than redefining the tuple.
EXCEEDANCE_THRESHOLDS_M: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0)

# Tukey's rule, matplotlib's own `boxplot`/`bxp` default: whiskers reach the most extreme
# data point within this many IQRs of the box edges; everything further out is drawn as
# an individual outlier ("flier") rather than folded into an extended whisker. Chosen
# deliberately, not left at whatever matplotlib happens to default to - see
# `boxplot_stats`'s docstring for why "no exclusion" only makes sense paired with a
# whisker rule that keeps every point visible somewhere on the figure.
WHISKER_IQR_MULTIPLIER = 1.5

DEFAULT_OUTPUT_DIR = paths.analysis_result_dir(
    "positioning_distributions", rebuilt=True
)

# positioning_diagnostics.py already scanned the 3.2 GB geometry-only recovered-day tree
# once and wrote the (year, doy, station) membership it found; reading that cache keeps
# this module CPU-light (a live positioning chain is running elsewhere on this host) at
# the cost of depending on that module having been run at least once. Falls back to a
# fresh scan, with a warning, if the cache is absent.
RECOVERED_STATION_DAYS_CACHE = (
    paths.analysis_result_dir("positioning_diagnostics", rebuilt=True)
    / "recovered_station_days.csv"
)

STEC_LABEL = "Direct STEC"
GIM_LABEL = "IGS GIM + Mapping"


def load_recovered_station_days_cached(
    cache_path: Path = RECOVERED_STATION_DAYS_CACHE,
    root: Path = RECOVERED_STEC_DB_ROOT,
) -> pd.DataFrame:
    """(year, doy, station) triples reconstructed from geometry alone - see
    `positioning_diagnostics.load_recovered_station_days` for what this means. Prefers
    that module's cached CSV over re-walking the raw tree."""
    if cache_path.exists():
        logger.info(f"recovered station-days: {cache_path} (cached)")
        return pd.read_csv(cache_path)
    logger.warning(f"{cache_path} not found - scanning {root} directly")
    return load_recovered_station_days(root)


# --------------------------------------------------------------------------
# 1 & 3 & 5. Box statistics (reused across the overall, storm/quiet and
# original/recovered figures by varying group_cols)
# --------------------------------------------------------------------------


def boxplot_stats(
    frame: pd.DataFrame,
    group_cols: list[str],
    whisker_iqr_multiplier: float = WHISKER_IQR_MULTIPLIER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tukey box statistics per group, and the individual flier values.

    Returns `(stats, fliers)`. `stats` has one row per group: `n`, `median_m`, `q1_m`,
    `q3_m`, `whislo_m`, `whishi_m`, `min_m`, `max_m`, `n_fliers`, `pct_fliers`. `fliers`
    is long format, one row per outlier point (`*group_cols`, `error_3d_rms`) - the
    points a box-plot figure draws individually rather than hiding inside the whisker.
    Both are computed directly from quantiles, not from `matplotlib.cbook.boxplot_stats`,
    so the same numbers can be written to CSV and handed to `ax.bxp()` without importing
    a plotting library into an analysis module.
    """
    stats_rows: list[dict] = []
    flier_rows: list[dict] = []
    for keys, sub in frame.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        key_dict = dict(zip(group_cols, keys))
        values = sub["error_3d_rms"].dropna()
        if values.empty:
            continue

        q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        lo_fence = q1 - whisker_iqr_multiplier * iqr
        hi_fence = q3 + whisker_iqr_multiplier * iqr
        within_fences = values[(values >= lo_fence) & (values <= hi_fence)]
        # A whisker reaches the most extreme *data point* within the fence, not the
        # fence itself - matplotlib's convention, followed here so the CSV and the
        # figure agree. If nothing falls inside the fence (a degenerate, near-constant
        # group), the whisker collapses to the box edge.
        whislo = float(within_fences.min()) if not within_fences.empty else float(q1)
        whishi = float(within_fences.max()) if not within_fences.empty else float(q3)
        fliers = values[(values < whislo) | (values > whishi)]

        stats_rows.append(
            {
                **key_dict,
                "n": len(values),
                "median_m": round(float(median), 4),
                "q1_m": round(float(q1), 4),
                "q3_m": round(float(q3), 4),
                "whislo_m": round(whislo, 4),
                "whishi_m": round(whishi, 4),
                "min_m": round(float(values.min()), 4),
                "max_m": round(float(values.max()), 4),
                "n_fliers": len(fliers),
                "pct_fliers": round(100 * len(fliers) / len(values), 4),
            }
        )
        for value in fliers:
            flier_rows.append({**key_dict, "error_3d_rms": round(float(value), 4)})

    stats = pd.DataFrame(stats_rows)
    fliers_out = pd.DataFrame(flier_rows, columns=[*group_cols, "error_3d_rms"])
    return stats, fliers_out


# --------------------------------------------------------------------------
# 2. Empirical CDF
# --------------------------------------------------------------------------


def cdf_points(frame: pd.DataFrame, group_cols: list[str] = ["Method"]) -> pd.DataFrame:
    """Every station-day's `error_3d_rms` with its empirical cumulative percentile,
    sorted within each group - literally the (x, y) pairs a CDF figure plots, so the
    written CSV and the curve are the same object."""
    chunks = []
    for keys, sub in frame.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        key_dict = dict(zip(group_cols, keys))
        values = sub["error_3d_rms"].dropna().sort_values().to_numpy()
        if values.size == 0:
            continue
        cumulative_pct = 100 * np.arange(1, values.size + 1) / values.size
        chunk = pd.DataFrame({**key_dict, "error_3d_rms": values})
        chunk["cumulative_pct"] = cumulative_pct
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


# --------------------------------------------------------------------------
# 4. Percentile / exceedance summary - the Table 5 replacement
# --------------------------------------------------------------------------


def percentile_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """N, mean, median, IQR bounds and p95/p99 of `error_3d_rms`, one row per group.

    No outcome filter applied - every station-day in `frame` counts. Quantiles use
    pandas' default linear interpolation.
    """
    grouped = frame.groupby(group_cols, observed=True)["error_3d_rms"]
    quantiles = grouped.quantile([0.25, 0.5, 0.75, 0.95, 0.99]).unstack()
    quantiles.columns = ["q1_m", "median_m", "q3_m", "p95_m", "p99_m"]
    out = pd.DataFrame({"n": grouped.size(), "mean_m": grouped.mean()}).join(quantiles)
    out["iqr_m"] = out["q3_m"] - out["q1_m"]
    out = out.reset_index()
    return out[
        [
            *group_cols,
            "n",
            "mean_m",
            "median_m",
            "q1_m",
            "q3_m",
            "iqr_m",
            "p95_m",
            "p99_m",
        ]
    ].round(4)


def exceedance_table(
    frame: pd.DataFrame,
    group_cols: list[str],
    thresholds: tuple[float, ...] = EXCEEDANCE_THRESHOLDS_M,
) -> pd.DataFrame:
    """Per group and threshold: how many/what fraction of station-days exceed it, and
    what fraction of the group's total (unfiltered) error mass those exceedances carry.
    This is the number that replaces the 10 m exclusion rule - it reports the tail
    instead of removing it. Thin wrapper around `positioning_diagnostics.
    outlier_threshold_counts`, which already computes exactly this; not re-derived here.
    """
    return outlier_threshold_counts(
        frame, thresholds=thresholds, group_cols=tuple(group_cols)
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _format_table5_markdown(
    percentiles: pd.DataFrame, exceedances: pd.DataFrame, n_source_rows: int
) -> str:
    """The Table 5 replacement as markdown: one row per method, ready to read into the
    manuscript. Iono weighting, no outcome filter - stated explicitly rather than left
    implicit, per the owner's instruction."""
    order = [m for m in METHOD_ORDER if m in percentiles["Method"].unique()]
    exc_pivot = exceedances.pivot(
        index="Method", columns="threshold_m", values="pct_station_days_exceeding"
    )
    exc_counts = exceedances.pivot(
        index="Method", columns="threshold_m", values="n_exceeding"
    )

    lines = [
        "# Table 5 replacement: positioning error, unfiltered distributions",
        "",
        "Iono weighting (predicted-uncertainty-weighted PPP). Every station-day in the "
        "coverage-recovered population is included - **no outcome-based outlier "
        "filter is applied** (replacing the old 10 m station-day exclusion). Values are "
        "3D positioning RMSE per station-day, in metres.",
        "",
        f"Source: `multiday_results/analyses/positioning_coverage/rebuilt/"
        f"multiday_summary.csv` ({n_source_rows:,} rows across all four methods).",
        "",
        "| Method | N | Median | IQR (Q1-Q3) | P95 | P99 | Exceed 5 m | Exceed 10 m | "
        "Exceed 20 m | Exceed 50 m |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method in order:
        row = percentiles.set_index("Method").loc[method]
        exc_row = exc_pivot.loc[method]
        exc_n = exc_counts.loc[method]
        exceed_cells = " | ".join(
            f"{exc_row[t]:.2f}% (n={int(exc_n[t])})" for t in EXCEEDANCE_THRESHOLDS_M
        )
        lines.append(
            f"| {method} | {int(row['n']):,} | {row['median_m']:.3f} m | "
            f"{row['q1_m']:.3f}-{row['q3_m']:.3f} m | {row['p95_m']:.3f} m | "
            f"{row['p99_m']:.3f} m | {exceed_cells} |"
        )

    stec_median = percentiles.set_index("Method").loc[STEC_LABEL, "median_m"]
    gim_median = percentiles.set_index("Method").loc[GIM_LABEL, "median_m"]
    median_improvement = 100 * (gim_median - stec_median) / gim_median
    stec_exceed_10 = int(exc_counts.loc[STEC_LABEL, 10.0])
    gim_exceed_10 = int(exc_counts.loc[GIM_LABEL, 10.0])

    lines += [
        "",
        f"Direct STEC vs IGS GIM + Mapping, median improvement: **{median_improvement:.1f}%**.",
        "",
        f"Direct STEC exceeds 10 m on {stec_exceed_10} station-days against IGS GIM's "
        f"{gim_exceed_10} ({stec_exceed_10 / max(gim_exceed_10, 1):.1f}x) - the tail this "
        "table reports explicitly rather than filtering out. See "
        "`overall_boxplot_stats.csv`/`overall_boxplot_fliers.csv` and "
        "`overall_cdf_points.csv` for the full distributions these numbers summarise.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-path", type=Path, default=canonical_positioning_summary()
    )
    parser.add_argument(
        "--recovered-cache", type=Path, default=RECOVERED_STATION_DAYS_CACHE
    )
    parser.add_argument("--recovered-root", type=Path, default=RECOVERED_STEC_DB_ROOT)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--swi-path", type=Path, default=paths.OMNI_INDICES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_positioning_table(args.summary_path)
    logger.info(
        f"{args.summary_path}: {len(frame):,} rows, {frame['Method'].nunique()} methods, "
        "no outcome filter applied"
    )

    # --- overall (Figures 1, 2, 4) ---
    overall_percentiles = percentile_summary(frame, ["Method"])
    overall_percentiles.to_csv(
        args.output_dir / "overall_percentile_summary.csv", index=False
    )

    overall_exceedance = exceedance_table(frame, ["Method"])
    overall_exceedance.to_csv(args.output_dir / "overall_exceedance.csv", index=False)

    overall_box_stats, overall_fliers = boxplot_stats(frame, ["Method"])
    overall_box_stats.to_csv(args.output_dir / "overall_boxplot_stats.csv", index=False)
    overall_fliers.to_csv(args.output_dir / "overall_boxplot_fliers.csv", index=False)

    overall_cdf = cdf_points(frame, ["Method"])
    overall_cdf.to_csv(args.output_dir / "overall_cdf_points.csv", index=False)

    # --- storm/quiet (Figure 3) ---
    with_storm = attach_storm_flag(frame, args.year, args.swi_path)
    with_storm["regime"] = np.where(with_storm["storm"], "storm", "quiet")
    n_storm_doys = with_storm.loc[with_storm["storm"], "doy"].nunique()
    logger.info(
        f"storm regime: {n_storm_doys} DOYs with daily min Dst <= "
        f"{STORM_DST_THRESHOLD_NT:g} nT"
    )

    regime_percentiles = percentile_summary(with_storm, ["Method", "regime"])
    regime_percentiles.to_csv(
        args.output_dir / "regime_percentile_summary.csv", index=False
    )

    regime_exceedance = exceedance_table(with_storm, ["Method", "regime"])
    regime_exceedance.to_csv(args.output_dir / "regime_exceedance.csv", index=False)

    regime_box_stats, regime_fliers = boxplot_stats(with_storm, ["Method", "regime"])
    regime_box_stats.to_csv(args.output_dir / "regime_boxplot_stats.csv", index=False)
    regime_fliers.to_csv(args.output_dir / "regime_boxplot_fliers.csv", index=False)

    # --- original/recovered (Figure 5) ---
    recovered = load_recovered_station_days_cached(
        args.recovered_cache, args.recovered_root
    )
    with_population = attach_population(frame, recovered)
    logger.info(
        f"population split: {(with_population['population'] == 'recovered').sum():,} "
        "recovered station-day rows across all methods"
    )

    population_percentiles = percentile_summary(
        with_population, ["Method", "population"]
    )
    population_percentiles.to_csv(
        args.output_dir / "population_percentile_summary.csv", index=False
    )

    population_exceedance = exceedance_table(with_population, ["Method", "population"])
    population_exceedance.to_csv(
        args.output_dir / "population_exceedance.csv", index=False
    )

    population_box_stats, population_fliers = boxplot_stats(
        with_population, ["Method", "population"]
    )
    population_box_stats.to_csv(
        args.output_dir / "population_boxplot_stats.csv", index=False
    )
    population_fliers.to_csv(
        args.output_dir / "population_boxplot_fliers.csv", index=False
    )

    # --- Table 5 replacement, formatted for the manuscript ---
    table5_markdown = _format_table5_markdown(
        overall_percentiles, overall_exceedance, len(frame)
    )
    (args.output_dir / "TABLE5_NUMBERS.md").write_text(table5_markdown)

    print(table5_markdown)
    logger.info(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
