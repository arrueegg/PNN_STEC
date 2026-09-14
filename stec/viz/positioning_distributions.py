"""Figures for `stec.analysis.positioning_distributions` (owner decision, 2026-08-28).

Not a manuscript or revision-response figure set yet - lives in its own
`plots/positioning_distributions/` tree, the same convention `positioning_diagnostics.py`
and `positioning_geography.py` use and for the same reason (see their module
docstrings). Reads the CSVs `stec.analysis.positioning_distributions` writes to
`multiday_results/analyses/positioning_distributions/rebuilt/`; does not recompute
anything from the raw per-station-day table itself.

**The decision this implements**: positioning results are reported as distributions and
medians, with no outcome-based outlier filtering, replacing the mean-with-10 m-exclusion
Table 5 used. See `stec.analysis.positioning_distributions`'s module docstring for the
full reasoning. **No figure here drops a row.** Where a figure needs a bound to stay
readable it constrains only what is *drawn* - an axis limit, or omitting the individual
outlier markers - never which rows enter the statistics, and every such figure reports
how many points it leaves undrawn.

Style: `stec.viz.style.APPROACH_COLORS` (blue Direct STEC, orange VTEC + Mapping, green
IGS GIM + Mapping, purple Pretrained Direct STEC) means only that approach - never
reused for a condition. Storm/quiet and original/recovered are conditions, not
approaches, so they take `CONDITION_COLORS` (grey baseline / red contrast) instead, the
same pair `positioning_diagnostics.py` and `positioning_geography.py` already use for
their own non-approach contrasts.

Every figure is written twice via `_save` (titled working copy + `_notitle` copy), and
`_save` always writes the plotted values as a CSV alongside the PNGs - for the box plots,
that CSV is the box statistics (median/Q1/Q3/whiskers) plus every individual outlier
point, i.e. exactly what `ax.bxp()` draws, not a re-aggregation of it.

Usage::

    python -m stec.viz.positioning_distributions --output_dir plots/positioning_distributions
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import paths
from .revision_figures import analysis_dir
from .style import APPROACH_COLORS, CONDITION_COLORS, FIGSIZE_WIDE, configure_plotting

import matplotlib.pyplot as plt  # noqa: E402  (style.py sets the Agg backend on import)

logger = logging.getLogger(__name__)

SOURCE_DIRS = {"positioning": "positioning_2024"}

METHOD_ORDER = [
    "Direct STEC",
    "Pretrained Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]
FIGSIZE_TWO_BY_TWO = (16, 10)
FIGSIZE_TABLE = (18, 6)

STEC_LABEL = "Direct STEC"
GIM_LABEL = "IGS GIM + Mapping"

# The threshold Figure 13 reports its undrawn tail against. Matches one of
# `stec.analysis.positioning_distributions.EXCEEDANCE_THRESHOLDS_M`'s four thresholds, so
# `overall_exceedance.csv` always carries a `threshold_m == 10.0` row, and the count is
# the same one `TABLE5_NUMBERS.md` reports - the figure and the table quote one number,
# not two. Box/whisker statistics remain over the full unfiltered population.
BOXPLOT_EXCEEDANCE_THRESHOLD_M = 10.0

# The regime/population contrasts drawn inside a per-method subplot (Figures 3 and 5)
# both compare exactly two groups, so the same two `CONDITION_COLORS` cover both without
# needing a third palette - "baseline" is always the less-disturbed/more-complete case
# (quiet, original), "contrast" the other (storm, recovered).
_REGIME_COLORS = {
    "quiet": CONDITION_COLORS["baseline"],
    "storm": CONDITION_COLORS["contrast"],
}
_POPULATION_COLORS = {
    "original": CONDITION_COLORS["baseline"],
    "recovered": CONDITION_COLORS["contrast"],
}


def _save(
    fig: plt.Figure,
    name: str,
    source: str,
    output_dir: Path,
    provenance: str,
    data: pd.DataFrame | None = None,
) -> None:
    """Mirrors `positioning_diagnostics._save` / `positioning_geography._save` exactly;
    duplicated rather than imported for the same reason those two duplicate each other -
    `SOURCE_DIRS` is local to each figure family."""
    target = output_dir / SOURCE_DIRS[source]
    target.mkdir(parents=True, exist_ok=True)
    if data is not None:
        data.to_csv(target / f"{name}.csv", index=False)

    footnote = fig.text(
        0.0, -0.04, f"Data: {provenance}", fontsize=11, color="#555555", va="top"
    )
    fig.savefig(target / f"{name}.png", bbox_inches="tight")

    footnote.set_text("")
    for ax in fig.axes:
        for loc in ("center", "left", "right"):
            ax.set_title("", loc=loc)
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    fig.savefig(target / f"{name}_notitle.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {target / name}.png (+ _notitle)")


def _bxp_stats_list(
    stats: pd.DataFrame, fliers: pd.DataFrame, group_col: str, order: list[str]
) -> list[dict]:
    """Build the list of dicts `ax.bxp()` expects, straight from the analysis module's
    own box statistics - matplotlib never recomputes a quantile here, it only draws the
    numbers `positioning_distributions.boxplot_stats` already wrote to CSV."""
    fliers_by_group = (
        fliers.groupby(group_col)["error_3d_rms"].apply(list).to_dict()
        if not fliers.empty
        else {}
    )
    indexed = stats.set_index(group_col)
    out = []
    for label in order:
        if label not in indexed.index:
            out.append(
                {"label": label, "med": np.nan, "q1": np.nan, "q3": np.nan,
                 "whislo": np.nan, "whishi": np.nan, "fliers": np.array([])}
            )  # fmt: skip
            continue
        row = indexed.loc[label]
        out.append(
            {
                "label": label,
                "med": row["median_m"],
                "q1": row["q1_m"],
                "q3": row["q3_m"],
                "whislo": row["whislo_m"],
                "whishi": row["whishi_m"],
                "fliers": np.array(fliers_by_group.get(label, [])),
            }
        )
    return out


def _draw_boxplot(
    ax: plt.Axes,
    stats: pd.DataFrame,
    fliers: pd.DataFrame,
    group_col: str,
    order: list[str],
    colors: dict[str, str],
    widths: float = 0.6,
    showfliers: bool = True,
) -> None:
    """Draw one Tukey box per group in `order`, coloured by `colors[label]` - shared by
    the overall, storm/quiet and original/recovered box-plot figures, which differ only
    in what `order` and `colors` mean.

    `showfliers=False` omits the individual outlier points so the axis scales to the box
    and whisker geometry. It changes only what is drawn: the boxes, whiskers and medians
    still come from `stats`, computed over the full unfiltered population, and every
    flier value is still written to the figure's CSV sidecar by `_tidy_box_data`."""
    positions = np.arange(1, len(order) + 1)
    result = ax.bxp(
        _bxp_stats_list(stats, fliers, group_col, order),
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=showfliers,
    )
    for i, label in enumerate(order):
        color = colors[label]
        result["boxes"][i].set_facecolor(color)
        result["boxes"][i].set_edgecolor(color)
        result["boxes"][i].set_alpha(0.55)
        result["medians"][i].set_color("black")
        result["medians"][i].set_linewidth(2.2)
        for whisker in result["whiskers"][2 * i : 2 * i + 2]:
            whisker.set_color(color)
        for cap in result["caps"][2 * i : 2 * i + 2]:
            cap.set_color(color)
        if showfliers:
            result["fliers"][i].set_markeredgecolor(color)
            result["fliers"][i].set_markerfacecolor("none")
            result["fliers"][i].set_marker("o")
            result["fliers"][i].set_markersize(4)
            result["fliers"][i].set_alpha(0.45)
    ax.set_xticks(positions)
    ax.set_xticklabels(order)


def _tidy_box_data(
    stats: pd.DataFrame,
    fliers: pd.DataFrame,
    id_cols: list[str],
    exceedance: pd.DataFrame | None = None,
    exceedance_threshold_m: float | None = None,
) -> pd.DataFrame:
    """The literal plotted values for a box-plot figure's `_save(data=...)`: box
    geometry (one row per statistic) and every flier point, both long-format so they
    concatenate into one CSV.

    `exceedance`/`exceedance_threshold_m`, when given, append one more row per group:
    `stat="n_exceeding_threshold"`, the count of station-days whose `error_3d_rms` exceeds
    `exceedance_threshold_m`. This is how a figure that does not draw its outlier points
    (Figure 13) still reports the size of the tail it leaves undrawn - CLAUDE.md's "no
    in-plot explanatory text" rule keeps the count out of the PNG body, so the caption
    text reads it from here (and from the figure's own title, which carries the same
    counts but is stripped from the `_notitle` manuscript copy).
    """
    box_long = stats.melt(
        id_vars=id_cols,
        value_vars=["median_m", "q1_m", "q3_m", "whislo_m", "whishi_m", "min_m", "max_m", "n", "n_fliers"],
        var_name="stat",
        value_name="value",
    )  # fmt: skip
    flier_long = fliers.rename(columns={"error_3d_rms": "value"}).copy()
    flier_long["stat"] = "flier"
    flier_long = flier_long[[*id_cols, "stat", "value"]]
    combined = pd.concat([box_long, flier_long], ignore_index=True)

    if exceedance is not None and exceedance_threshold_m is not None:
        at_limit = exceedance[
            np.isclose(exceedance["threshold_m"], exceedance_threshold_m)
        ].copy()
        at_limit["stat"] = "n_exceeding_threshold"
        at_limit = at_limit.rename(columns={"n_exceeding": "value"})[
            [*id_cols, "stat", "value"]
        ]
        combined = pd.concat([combined, at_limit], ignore_index=True)

    return combined


# --------------------------------------------------------------------------
# 1. Overall box plot - the primary Table 5 replacement
# --------------------------------------------------------------------------


def fig_boxplot_3d_error(
    stats: pd.DataFrame,
    fliers: pd.DataFrame,
    output_dir: Path,
    provenance: str,
    exceedance: pd.DataFrame,
) -> None:
    """Box plot of 3D positioning error per method, unfiltered, linear y-axis scaled to
    the box and whisker geometry.

    Whisker convention: Tukey's rule (matplotlib's own default), 1.5xIQR - whiskers
    reach the most extreme data point within that fence. The data spans ~0.18 m to
    ~5,990 m (a ~33,000x range), all computed from the full unfiltered population
    regardless of what the axis shows.

    2026-09-14 (owner review, two rounds): first from a log y-axis to linear capped at
    `BOXPLOT_EXCEEDANCE_THRESHOLD_M` - the log axis compressed the box geometry, which is the
    comparison this figure exists to show, into a sliver beside a handful of
    multi-thousand-metre PPPx solve failures. Then to omitting the outlier points
    entirely (`showfliers=False`) and dropping the cap, so the axis scales naturally to
    the quantile bars instead of to a dense column of overlapping flier markers.

    **Display, not data.** The boxes, medians and whiskers are still computed over the
    full unfiltered population - no station-day is excluded, which would contradict the
    Table 5 methodology (`docs/revision/positioning_reporting.md`). Every omitted flier
    value is still written to this figure's CSV sidecar by `_tidy_box_data`, and the
    per-method counts beyond the whiskers (`n_fliers`) and beyond
    `BOXPLOT_EXCEEDANCE_THRESHOLD_M` (`n_exceeding_threshold`, from `exceedance`) are reported in the
    title - stripped from the `_notitle` manuscript copy, where the caption carries them
    instead, per CLAUDE.md's "no in-plot explanatory text" rule.
    """
    order = [m for m in METHOD_ORDER if m in stats["Method"].unique()]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    _draw_boxplot(
        ax,
        stats,
        fliers,
        "Method",
        order,
        APPROACH_COLORS,
        widths=0.55,
        showfliers=False,
    )
    ax.set_ylim(bottom=0)
    ax.set_ylabel("3D positioning error [m]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    n_fliers = {m: int(stats.set_index("Method").loc[m, "n_fliers"]) for m in order}
    exceeding_by_method = exceedance.loc[
        np.isclose(exceedance["threshold_m"], BOXPLOT_EXCEEDANCE_THRESHOLD_M)
    ].set_index("Method")["n_exceeding"]
    n_exceeding = {m: int(exceeding_by_method.get(m, 0)) for m in order}
    ax.set_title(
        "Positioning 3D error by method, unfiltered (iono weighting)\n"
        "Tukey whiskers (1.5×IQR); outlier points not drawn, none excluded from the "
        "statistics - "
        + ", ".join(f"{m}: {n_fliers[m]} beyond whiskers" for m in order)
        + f"\nn station-days > {BOXPLOT_EXCEEDANCE_THRESHOLD_M:g} m: "
        + ", ".join(f"{m}: {n_exceeding[m]}" for m in order)
    )
    _save(
        fig,
        "boxplot_3d_error",
        "positioning",
        output_dir,
        provenance,
        _tidy_box_data(
            stats,
            fliers,
            ["Method"],
            exceedance=exceedance,
            exceedance_threshold_m=BOXPLOT_EXCEEDANCE_THRESHOLD_M,
        ),
    )


# --------------------------------------------------------------------------
# 2. CDF - all four methods, unfiltered, median and p95 marked
# --------------------------------------------------------------------------


def fig_cdf_unfiltered(
    cdf: pd.DataFrame, percentiles: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """CDF of 3D positioning error, all four methods, unfiltered - the "typically
    better, worse in the tail" picture in one figure. Median (circle) and p95 (triangle)
    are marked on each curve. The x-axis view is clipped at 1.2x the pooled 99th
    percentile for readability; every value is still in the written CSV, and the count
    of points beyond the clipped view is annotated on the figure rather than silently
    cropped.
    """
    order = [m for m in METHOD_ORDER if m in cdf["Method"].unique()]
    view_limit = float(1.2 * cdf["error_3d_rms"].quantile(0.99))

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    beyond_view: dict[str, int] = {}
    for method in order:
        sub = cdf[cdf["Method"] == method]
        color = APPROACH_COLORS[method]
        ax.plot(
            sub["error_3d_rms"],
            sub["cumulative_pct"],
            color=color,
            linewidth=2.2,
            label=method,
        )

        row = percentiles.set_index("Method").loc[method]
        ax.scatter([row["median_m"]], [50], color=color, marker="o", s=90,
                   edgecolor="black", linewidth=0.8, zorder=5)  # fmt: skip
        ax.scatter([row["p95_m"]], [95], color=color, marker="^", s=100,
                   edgecolor="black", linewidth=0.8, zorder=5)  # fmt: skip
        beyond_view[method] = int((sub["error_3d_rms"] > view_limit).sum())

    ax.axhline(50, color="#999999", linestyle=":", linewidth=1, zorder=1)
    ax.axhline(95, color="#999999", linestyle=":", linewidth=1, zorder=1)
    ax.set_xlim(0, view_limit)
    ax.set_ylim(0, 101)
    ax.set_xlabel(f"3D positioning error [m] (view clipped at {view_limit:.1f} m)")
    ax.set_ylabel("Cumulative probability [%]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    beyond_note = "; ".join(f"{m}: {beyond_view[m]} beyond view" for m in order)
    ax.set_title(
        "Positioning error CDF, unfiltered (iono weighting)\n"
        f"circle = median, triangle = p95 - {beyond_note}"
    )
    _save(
        fig,
        "cdf_unfiltered",
        "positioning",
        output_dir,
        provenance,
        cdf[["Method", "error_3d_rms", "cumulative_pct"]],
    )


# --------------------------------------------------------------------------
# 3. Storm vs quiet distributions, per method
# --------------------------------------------------------------------------


def fig_storm_quiet_boxplot(
    stats: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """One box-plot panel per method: quiet vs storm 3D error, log y-axis shared across
    panels so magnitudes compare directly. Storm/quiet is a *condition*, not an
    approach, so it takes `CONDITION_COLORS` rather than the method's own approach hue -
    the point of this figure is what happens to one method's distribution across
    regimes, not a method-vs-method comparison.

    Answers R1.7 in distributional form: the published mean-based storm result
    collapsed to -0.3% (storm and quiet means nearly equal, because a few very large
    quiet-day outliers inflate the quiet mean); the median tells a materially different,
    physically sensible story - every method's storm-day median is higher than its
    quiet-day median.
    """
    order = [m for m in METHOD_ORDER if m in stats["Method"].unique()]
    fig, axes = plt.subplots(
        2, 2, figsize=FIGSIZE_TWO_BY_TWO, sharey=True, layout="constrained"
    )
    regimes = ["quiet", "storm"]
    for ax, method in zip(axes.flat, order):
        sub = stats[stats["Method"] == method]
        _draw_boxplot(
            ax,
            sub,
            pd.DataFrame(columns=["regime", "error_3d_rms"]),
            "regime",
            regimes,
            _REGIME_COLORS,
        )
        for i, regime in enumerate(regimes):
            row = sub.set_index("regime").loc[regime]
            ax.text(i + 1, row["median_m"], f"{row['median_m']:.2f} m", ha="center",
                     va="bottom", fontsize=12, fontweight="bold")  # fmt: skip
        quiet_median = sub.set_index("regime").loc["quiet", "median_m"]
        storm_median = sub.set_index("regime").loc["storm", "median_m"]
        degradation_pct = 100 * (storm_median - quiet_median) / quiet_median
        ax.set_title(
            f"{method}\nstorm vs quiet median: {degradation_pct:+.1f}%",
            color=APPROACH_COLORS[method],
        )
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    # A single figure-level y-label (not one per left-column axes) - `sharey=True`
    # already means the two rows are on one scale, and two independent `ax.set_ylabel`
    # calls in a tall 2x2 grid visually collide with each other once `constrained_layout`
    # tightens the row spacing.
    fig.supylabel("3D positioning error [m] (log scale)")
    fig.suptitle(
        "Storm vs quiet 3D positioning error by method, unfiltered\n"
        f"storm = daily min Dst ≤ -50 nT (Tukey box, {int(stats['n'].sum()):,} station-days total)"
    )
    plotted = stats[
        ["Method", "regime", "n", "median_m", "q1_m", "q3_m", "whislo_m", "whishi_m"]
    ]
    _save(fig, "storm_quiet_boxplot", "positioning", output_dir, provenance, plotted)


# --------------------------------------------------------------------------
# 4. Percentile / exceedance summary table
# --------------------------------------------------------------------------


def fig_percentile_exceedance_table(
    percentiles: pd.DataFrame,
    exceedance: pd.DataFrame,
    output_dir: Path,
    provenance: str,
) -> None:
    """The Table 5 replacement rendered as a figure: one row per method, median/IQR/p95/
    p99 plus the exceedance rate at each of 5/10/20/50 m. The same numbers are in
    `TABLE5_NUMBERS.md`; this is the one-glance version."""
    order = [m for m in METHOD_ORDER if m in percentiles["Method"].unique()]
    thresholds = sorted(exceedance["threshold_m"].unique())
    pct_pivot = exceedance.pivot(
        index="Method", columns="threshold_m", values="pct_station_days_exceeding"
    )
    n_pivot = exceedance.pivot(
        index="Method", columns="threshold_m", values="n_exceeding"
    )
    indexed = percentiles.set_index("Method")

    columns = ["N", "Median [m]", "IQR [m]", "P95 [m]", "P99 [m]"] + [
        f">{t:g} m" for t in thresholds
    ]
    cell_text, row_colors = [], []
    for method in order:
        row = indexed.loc[method]
        cells = [
            f"{int(row['n']):,}",
            f"{row['median_m']:.3f}",
            f"{row['q1_m']:.3f}–{row['q3_m']:.3f}",
            f"{row['p95_m']:.3f}",
            f"{row['p99_m']:.3f}",
        ]
        cells += [
            f"{pct_pivot.loc[method, t]:.2f}% (n={int(n_pivot.loc[method, t])})"
            for t in thresholds
        ]
        cell_text.append(cells)
        row_colors.append(APPROACH_COLORS[method])

    fig, ax = plt.subplots(figsize=FIGSIZE_TABLE)
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=order,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.4)
    for row_index, color in enumerate(row_colors, start=1):
        row_label_cell = table[row_index, -1].get_text()
        row_label_cell.set_color(color)
        row_label_cell.set_fontweight("bold")
    ax.set_title(
        "Positioning error summary, unfiltered (iono weighting) - "
        "the Table 5 replacement"
    )
    long = pd.concat(
        [
            percentiles.assign(kind="percentile"),
            exceedance.rename(
                columns={"pct_station_days_exceeding": "pct_exceeding"}
            ).assign(kind="exceedance"),
        ],
        ignore_index=True,
    )
    _save(
        fig, "percentile_exceedance_table", "positioning", output_dir, provenance, long
    )


# --------------------------------------------------------------------------
# 5. Original vs recovered population split, per method
# --------------------------------------------------------------------------


def fig_population_split_boxplot(
    stats: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """One box-plot panel per method: original (real STEC-database coverage) vs
    recovered (geometry-only, no DCB/target fields) 3D error, log y-axis shared across
    panels. Population is a *condition*, not an approach, so it takes
    `CONDITION_COLORS` rather than the method's own hue.

    This is the distributional form of the +19.4%/-31.9% mean-based split: the model's
    behaviour on geometry-only recovered station-days shown directly rather than
    summarised into two numbers.
    """
    order = [m for m in METHOD_ORDER if m in stats["Method"].unique()]
    fig, axes = plt.subplots(
        2, 2, figsize=FIGSIZE_TWO_BY_TWO, sharey=True, layout="constrained"
    )
    populations = ["original", "recovered"]
    for ax, method in zip(axes.flat, order):
        sub = stats[stats["Method"] == method]
        _draw_boxplot(
            ax, sub, pd.DataFrame(columns=["population", "error_3d_rms"]),
            "population", populations, _POPULATION_COLORS,
        )  # fmt: skip
        for i, population in enumerate(populations):
            row = sub.set_index("population").loc[population]
            ax.text(i + 1, row["median_m"], f"{row['median_m']:.2f} m", ha="center",
                     va="bottom", fontsize=12, fontweight="bold")  # fmt: skip
        original_median = sub.set_index("population").loc["original", "median_m"]
        recovered_median = sub.set_index("population").loc["recovered", "median_m"]
        change_pct = 100 * (recovered_median - original_median) / original_median
        ax.set_title(
            f"{method}\nrecovered vs original median: {change_pct:+.1f}%",
            color=APPROACH_COLORS[method],
        )
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    # See fig_storm_quiet_boxplot for why this is one figure-level label rather than one
    # per left-column axes.
    fig.supylabel("3D positioning error [m] (log scale)")
    fig.suptitle(
        "Original vs geometry-only recovered station-days, by method, unfiltered\n"
        f"recovered = no real STEC-database entry that day ({int(stats['n'].sum()):,} station-days total)"
    )
    plotted = stats[
        [
            "Method",
            "population",
            "n",
            "median_m",
            "q1_m",
            "q3_m",
            "whislo_m",
            "whishi_m",
        ]
    ]
    _save(
        fig, "population_split_boxplot", "positioning", output_dir, provenance, plotted
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _build_figures(args: argparse.Namespace, output_dir: Path) -> None:
    source = analysis_dir(args.results_dir, "positioning_distributions")
    if not source.is_dir():
        logger.warning(
            f"{source} not found - run stec/analysis/positioning_distributions.py first"
        )
        return
    prov = (
        f"{source} (stec.analysis.positioning_distributions), SF-PPP 2024 test period, "
        "iono weighting, no outcome filter"
    )

    overall_stats_path = source / "overall_boxplot_stats.csv"
    overall_fliers_path = source / "overall_boxplot_fliers.csv"
    overall_exceedance_path = source / "overall_exceedance.csv"
    if (
        overall_stats_path.exists()
        and overall_fliers_path.exists()
        and overall_exceedance_path.exists()
    ):
        fig_boxplot_3d_error(
            pd.read_csv(overall_stats_path),
            pd.read_csv(overall_fliers_path),
            output_dir,
            prov,
            pd.read_csv(overall_exceedance_path),
        )

    cdf_path = source / "overall_cdf_points.csv"
    percentile_path = source / "overall_percentile_summary.csv"
    if cdf_path.exists() and percentile_path.exists():
        fig_cdf_unfiltered(
            pd.read_csv(cdf_path), pd.read_csv(percentile_path), output_dir, prov
        )

    regime_stats_path = source / "regime_boxplot_stats.csv"
    if regime_stats_path.exists():
        fig_storm_quiet_boxplot(pd.read_csv(regime_stats_path), output_dir, prov)

    exceedance_path = source / "overall_exceedance.csv"
    if percentile_path.exists() and exceedance_path.exists():
        fig_percentile_exceedance_table(
            pd.read_csv(percentile_path), pd.read_csv(exceedance_path), output_dir, prov
        )

    population_stats_path = source / "population_boxplot_stats.csv"
    if population_stats_path.exists():
        fig_population_split_boxplot(
            pd.read_csv(population_stats_path), output_dir, prov
        )


FIGURE_BUILDERS = (_build_figures,)


def build_all(args: argparse.Namespace) -> None:
    configure_plotting()
    for build in FIGURE_BUILDERS:
        build(args, args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("plots/positioning_distributions")
    )
    parser.add_argument("--results_dir", type=Path, default=paths.RESULTS_ROOT)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    build_all(args)


if __name__ == "__main__":
    main()
