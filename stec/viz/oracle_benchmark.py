"""Deployment-ready figures and table for the R1.8 oracle-benchmark result (owner request,
2026-09-16: "the oracle results including nice plots ready if i use it later for the reviewer
comment addressing").

Not a manuscript or revision-response figure set. `stec.analysis.oracle_benchmark` already
has a companion bar chart (`stec.viz.revision_figures.fig_oracle_benchmark`,
`plots/revision/positioning_2024/oracle_benchmark.png`) that answers R1.8 at manuscript scale.
This module is a deeper, standalone look at the same result, held in its own
`plots/oracle_benchmark/` tree outside the manuscript until the owner decides whether and how
it enters the reviewer response. It must never write into `plots/manuscript/` or
`plots/revision/`, and it does not touch Figure 13 (`fig_boxplot_3d_error`) or any other
existing manuscript figure.

Reads `stec.analysis.oracle_benchmark`'s output
(`multiday_results/analyses/oracle_benchmark/rebuilt/`) directly - `paired_station_days.csv`
(the full unfiltered population, one column per method, one row per station-day solved by
every method) and `station_coverage.csv` - and never recomputes the analysis itself, the same
separation of concerns `positioning_diagnostics.py` and `revision_figures.py` use between
analysis (writes numbers) and viz (reads and draws them).

Four things, in build order:

1. `boxplot_oracle_benchmark` - all four methods plus the oracle floor, one Tukey box plot.
   Log y-scale, not linear - deliberately the opposite of Figure 13's
   (`stec.viz.positioning_distributions.fig_boxplot_3d_error`, untouched by this module)
   choice, and the contrast is intentional, not an inconsistency to "fix": here the floor's
   median (0.069 m) and VTEC's P99 (6.8 m) are about two orders of magnitude apart, so a
   linear axis would compress all four boxes into an unreadable strip at the bottom. Figure
   13's four methods all sit within roughly one decade of each other, which is exactly why
   linear is the right choice there. Outlier points are drawn (unlike Figure 13, which omits
   them on its linear axis to keep the box geometry visible) - on a log axis the tail
   doesn't dominate the picture the way it does on a linear one, and showing it is what makes
   IGS GIM's better tail reliability visible in the figure itself rather than only in the
   table. The y-axis is further cropped to `[BOXPLOT_Y_MIN_M, BOXPLOT_Y_MAX_M]` (owner
   review, 2026-09-16) so the boxes themselves - where the actual comparison lives - are
   not squeezed by the small number of ~6,000 m PPPx solve failures shared across all four
   methods; points above the cap are still counted, per method, in the CSV sidecar and the
   title, never silently dropped.
2. `paired_difference` - a paired log-log scatter of Direct STEC against each of the other
   two baselines, with the 1:1 diagonal, one point per station-day. Chosen over a difference
   histogram because R1.8 is about per-observation achievability, and a scatter keeps every
   station-day's own outcome visible (which points are wins, which are losses, and by how
   much) rather than collapsing that into one aggregate distribution.
3. `station_coverage` - every one of the 54 stations in the paired population, including the
   eight below 20 station-days, so a reviewer can see the population is no longer dominated
   by a handful of stations without taking that on faith.
4. `oracle_table.md` - the same numbers (median, Q1, Q3, P95, P99, ratio to the oracle floor,
   percentage exceeding 5 m) as a generated table, computed from the CSVs rather than
   transcribed by hand - this repository has a documented history of hand-typed numbers
   drifting from their source (CLAUDE.md's canonical-results table opens with exactly that
   story), and this module exists in part to not repeat it.

Style: `stec.viz.style` throughout - `APPROACH_COLORS` (blue Direct STEC, orange VTEC +
Mapping, green IGS GIM + Mapping) for the three positioning methods, and `ORACLE_COLOR` for
the reference-STEC floor, which is not a fifth approach and must never take an approach
colour. Each figure is written twice via this module's own `_save` (mirrors
`revision_figures._save` / `positioning_diagnostics._save`, duplicated rather than imported
for the same reason those two duplicate each other - this module's output layout is its own):
a titled working copy with a provenance footnote, and a `_notitle` copy with neither - the
`_notitle` copy is the one to use if this ever becomes a manuscript or reviewer-response
figure. No in-plot explanatory text beyond axis labels and the (stripped) title.

Honesty notes, so a reader of this docstring sees the same things a reader of the figures
should see, and so nobody re-derives them incorrectly later: Direct STEC is not uniformly
better than IGS GIM + Mapping - it exceeds 5 m on 1.3% of station-days against GIM's 0.5%,
and its P95 (3.400 m) is marginally worse than GIM's (3.389 m), even though Direct STEC wins
the pairwise comparison on 77.6% of station-days. GIM is the more reliable baseline in the
tail; Direct STEC is the more accurate baseline overall. Only 1.3% of station-days put Direct
STEC within 2x of the oracle floor. The floor itself is not uniformly tiny either - its own
P99 (1.064 m) is about fifteen times its median (0.069 m). No figure or table here clips a
view or a column in a way that would hide any of this.

Usage::

    python -m stec.viz.oracle_benchmark --output_dir plots/oracle_benchmark
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import paths
from .revision_figures import analysis_dir
from .style import (
    APPROACH_COLORS,
    CONDITION_COLORS,
    FIGSIZE_DOUBLE_WIDE,
    FIGSIZE_WIDE,
    ORACLE_COLOR,
    configure_plotting,
)

import matplotlib.pyplot as plt  # noqa: E402  (style.py sets the Agg backend on import)

logger = logging.getLogger(__name__)

ORACLE_LABEL = "Reference STEC (oracle)"
# Matches stec.analysis.oracle_benchmark.DISPLAY_ORDER and
# revision_figures.fig_oracle_benchmark's own ordering - the oracle floor first, then the
# three baselines in the paper's usual Direct STEC / VTEC / GIM order.
DISPLAY_ORDER = [ORACLE_LABEL, "Direct STEC", "VTEC + Mapping", "IGS GIM + Mapping"]
BASELINE_METHODS = ["IGS GIM + Mapping", "VTEC + Mapping"]

# Station-days below this count are called out in the coverage figure (owner's own framing
# of the 2026-09-16 recovery: "eight stations remain under 20 station-days - show them
# rather than hiding them").
LOW_COVERAGE_THRESHOLD_DAYS = 20

# The threshold the honesty requirements are stated against (Direct STEC 1.3% vs GIM 0.5%
# exceeding this), reused for both the box-plot title and the table's own column.
EXCEEDANCE_THRESHOLD_M = 5.0

# The box plot's y-axis window (owner review, 2026-09-16): the full data spans ~0.02 m to
# ~6,000 m (a handful of PPPx solve failures shared across all four methods), but every
# box, whisker and the oracle floor line lives between ~0.02 and ~4 m - two of six decades
# carrying the whole comparison, the rest a scatter of isolated points. Cropping to
# [1e-2, 1e2] m keeps four decades on screen (the boxes plus two decades of tail above the
# whiskers) and roughly doubles the on-screen height of every box, without dropping the
# data itself: points above the cap are still real rows in `paired`, still counted in the
# CSV sidecar and the title (this module's own version of Figure 13's "count what you
# don't draw" convention for its own excluded fliers) - only their marker is clipped by
# the axis.
BOXPLOT_Y_MIN_M = 1e-2
BOXPLOT_Y_MAX_M = 1e2


def _save(
    fig: plt.Figure,
    name: str,
    output_dir: Path,
    provenance: str,
    data: pd.DataFrame | None = None,
) -> None:
    """Write the titled working copy (+ provenance footnote), the `_notitle` copy, and the
    plotted values as a CSV - the number a reader checks is the number the figure draws."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if data is not None:
        data.to_csv(output_dir / f"{name}.csv", index=False)

    footnote = fig.text(
        0.0, -0.04, f"Data: {provenance}", fontsize=11, color="#555555", va="top"
    )
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")

    footnote.set_text("")
    for ax in fig.axes:
        for loc in ("center", "left", "right"):
            ax.set_title("", loc=loc)
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    fig.savefig(output_dir / f"{name}_notitle.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {output_dir / name}.png (+ _notitle)")


# --------------------------------------------------------------------------
# 1. Box plot, log scale, oracle floor marked
# --------------------------------------------------------------------------


def _box_stats(series: pd.Series) -> dict:
    """Tukey box geometry (1.5xIQR) - hand-computed so the CSV sidecar and the table can
    carry exactly the numbers `ax.boxplot` draws, without reaching into matplotlib's return
    structure to read them back out."""
    q1, median, q3 = series.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    within_fence = series[(series >= lower_fence) & (series <= upper_fence)]
    whislo, whishi = within_fence.min(), within_fence.max()
    fliers = series[(series < whislo) | (series > whishi)]
    return {
        "median": median,
        "q1": q1,
        "q3": q3,
        "whislo": whislo,
        "whishi": whishi,
        "n": len(series),
        "n_fliers": len(fliers),
        "fliers": fliers,
    }


def _boxplot_plotted_data(paired: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    rows = []
    for method in order:
        stats = _box_stats(paired[method].dropna())
        n_exceeding = int((paired[method] > EXCEEDANCE_THRESHOLD_M).sum())
        n_above_cap = int((paired[method] > BOXPLOT_Y_MAX_M).sum())
        for stat in ("median", "q1", "q3", "whislo", "whishi", "n", "n_fliers"):
            rows.append({"Method": method, "stat": stat, "value": stats[stat]})
        rows.append({"Method": method, "stat": "n_exceeding_5m", "value": n_exceeding})
        rows.append(
            {"Method": method, "stat": "n_above_axis_cap", "value": n_above_cap}
        )
        for value in stats["fliers"]:
            rows.append({"Method": method, "stat": "flier", "value": value})
    return pd.DataFrame(rows)


def fig_boxplot(paired: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """Box plot of 3D RMS positioning error, four methods plus the oracle floor, log
    y-axis. See the module docstring for why log (not linear, unlike Figure 13) is right
    here. Boxes and whiskers are Tukey (1.5xIQR, matplotlib's default), computed from the
    full 8,223-row unfiltered population - no outcome-based exclusion, matching
    `oracle_benchmark`'s own methodology.

    The y-axis is cropped to [`BOXPLOT_Y_MIN_M`, `BOXPLOT_Y_MAX_M`] (owner review,
    2026-09-16) so the boxes and whiskers - where the actual method comparison lives -
    are not squeezed into a sliver below a handful of ~6,000 m PPPx solve failures. The
    crop is a *display* choice only: every point above the cap is still a real row in
    `paired` and is still counted, per method, in both this title and the CSV sidecar
    (`n_above_axis_cap`) - never silently dropped."""
    order = [m for m in DISPLAY_ORDER if m in paired.columns]
    colors = [ORACLE_COLOR if m == ORACLE_LABEL else APPROACH_COLORS[m] for m in order]
    data = [paired[m].dropna().to_numpy() for m in order]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    result = ax.boxplot(
        data,
        positions=np.arange(1, len(order) + 1),
        widths=0.55,
        patch_artist=True,
        showfliers=True,
        whis=1.5,
    )
    for i, color in enumerate(colors):
        result["boxes"][i].set_facecolor(color)
        result["boxes"][i].set_edgecolor(color)
        result["boxes"][i].set_alpha(0.55)
        result["medians"][i].set_color("black")
        result["medians"][i].set_linewidth(2.2)
        for whisker in result["whiskers"][2 * i : 2 * i + 2]:
            whisker.set_color(color)
        for cap in result["caps"][2 * i : 2 * i + 2]:
            cap.set_color(color)
        result["fliers"][i].set_markeredgecolor(color)
        result["fliers"][i].set_markerfacecolor("none")
        result["fliers"][i].set_alpha(0.25)
        result["fliers"][i].set_markersize(3)

    ax.set_yscale("log")
    ax.set_ylim(BOXPLOT_Y_MIN_M, BOXPLOT_Y_MAX_M)
    floor_median = paired[ORACLE_LABEL].median()
    ax.axhline(
        floor_median, color=ORACLE_COLOR, linewidth=1.5, linestyle="--", zorder=4
    )
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(
        [m.replace(" (oracle)", "\n(oracle)").replace(" + ", "\n+ ") for m in order]
    )
    ax.set_ylabel("3D RMS positioning error [m], log scale")
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    n_exceeding = {m: int((paired[m] > EXCEEDANCE_THRESHOLD_M).sum()) for m in order}
    n_above_cap = {m: int((paired[m] > BOXPLOT_Y_MAX_M).sum()) for m in order}
    ax.set_title(
        "Positioning error against the observation-derived floor "
        f"(log scale, y-axis cropped to [{BOXPLOT_Y_MIN_M:g}, {BOXPLOT_Y_MAX_M:g}] m)\n"
        "Tukey whiskers (1.5xIQR), full unfiltered population, outlier points drawn - "
        f"n > {EXCEEDANCE_THRESHOLD_M:g} m: "
        + ", ".join(f"{m}: {n_exceeding[m]}" for m in order)
        + f"\nn above the axis cap ({BOXPLOT_Y_MAX_M:g} m, off-scale, not visible): "
        + ", ".join(f"{m}: {n_above_cap[m]}" for m in order)
    )
    _save(
        fig,
        "boxplot_oracle_benchmark",
        output_dir,
        provenance,
        _boxplot_plotted_data(paired, order),
    )


# --------------------------------------------------------------------------
# 2. Paired-difference scatter, Direct STEC vs each baseline
# --------------------------------------------------------------------------


def fig_paired_difference(
    paired: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Log-log scatter, one panel per baseline: Direct STEC's error against that
    baseline's, same station-day, with the 1:1 diagonal. Points below the diagonal are
    station-days where Direct STEC wins - the same per-station-day comparison behind the
    77.6%/85.1% win-rate figures, rendered directly rather than collapsed to one number, so
    the shape of the exceptions (how far above the line the GIM/VTEC-wins points sit) is
    visible too, not just their count."""
    fig, axes = plt.subplots(1, len(BASELINE_METHODS), figsize=FIGSIZE_DOUBLE_WIDE)
    direct = paired["Direct STEC"]
    tables = []
    for ax, baseline in zip(axes, BASELINE_METHODS):
        other = paired[baseline]
        color = APPROACH_COLORS[baseline]
        ax.scatter(direct, other, s=6, alpha=0.25, color=color, linewidths=0, zorder=2)
        lo = min(direct.min(), other.min()) * 0.8
        hi = max(direct.max(), other.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.2, zorder=3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_aspect("equal")
        ax.set_xlabel("Direct STEC 3D RMS error [m]")
        ax.set_ylabel(f"{baseline} 3D RMS error [m]")
        win_pct = (direct < other).mean() * 100
        ax.set_title(
            f"vs {baseline}\nDirect STEC lower on {win_pct:.1f}% of station-days"
        )
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

        table = paired[["station", "doy"]].copy()
        table["baseline_method"] = baseline
        table["direct_stec_m"] = direct
        table["baseline_m"] = other
        table["direct_stec_wins"] = direct < other
        tables.append(table)

    fig.suptitle(
        "Direct STEC vs. each baseline, per station-day "
        "(points below the diagonal: Direct STEC wins)"
    )
    _save(
        fig,
        "paired_difference",
        output_dir,
        provenance,
        pd.concat(tables, ignore_index=True),
    )


# --------------------------------------------------------------------------
# 3. Station coverage
# --------------------------------------------------------------------------


def fig_station_coverage(
    coverage: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Horizontal bar, one per station, sorted by station-day count. Stations below
    `LOW_COVERAGE_THRESHOLD_DAYS` are marked rather than dropped - the point of this figure
    is to show the population is no longer dominated by a handful of stations, which
    requires showing the thin end too, not just the well-covered majority."""
    ordered = coverage.sort_values("station_days", ascending=True).reset_index(
        drop=True
    )
    colors = [
        CONDITION_COLORS["contrast"]
        if v < LOW_COVERAGE_THRESHOLD_DAYS
        else CONDITION_COLORS["baseline"]
        for v in ordered["station_days"]
    ]
    fig, ax = plt.subplots(figsize=(10, max(8, 0.22 * len(ordered))))
    ax.barh(ordered["station"], ordered["station_days"], color=colors, zorder=3)
    ax.axvline(
        LOW_COVERAGE_THRESHOLD_DAYS,
        color="black",
        linewidth=1.0,
        linestyle=":",
        zorder=4,
    )
    ax.set_xlabel("Station-days in the paired population")
    ax.set_ylabel("Station")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    n_low = int((ordered["station_days"] < LOW_COVERAGE_THRESHOLD_DAYS).sum())
    ax.set_title(
        f"Station coverage in the oracle-benchmark paired population "
        f"(N={len(ordered)} stations, {n_low} below "
        f"{LOW_COVERAGE_THRESHOLD_DAYS} station-days)"
    )
    _save(
        fig,
        "station_coverage",
        output_dir,
        provenance,
        ordered[["station", "station_days"]],
    )


# --------------------------------------------------------------------------
# 4. Generated table
# --------------------------------------------------------------------------


def build_oracle_table(paired: pd.DataFrame, source_csv: Path) -> str:
    """Median/Q1/Q3/P95/P99/ratio-to-floor/exceedance-of-5m per method, generated straight
    from `paired_station_days.csv` - never hand-transcribed, so it cannot silently drift
    from the CSV the way CLAUDE.md's canonical-results table records this repository's
    numbers drifting before."""
    order = [m for m in DISPLAY_ORDER if m in paired.columns]
    floor_median = paired[ORACLE_LABEL].median()

    rows = []
    for method in order:
        series = paired[method]
        q1, median, q3, p95, p99 = series.quantile([0.25, 0.5, 0.75, 0.95, 0.99])
        pct_exceeding = (series > EXCEEDANCE_THRESHOLD_M).mean() * 100
        rows.append(
            {
                "Method": method,
                "Median [m]": median,
                "Q1 [m]": q1,
                "Q3 [m]": q3,
                "P95 [m]": p95,
                "P99 [m]": p99,
                "Ratio to floor": median / floor_median,
                f"% > {EXCEEDANCE_THRESHOLD_M:g} m": pct_exceeding,
            }
        )
    table = pd.DataFrame(rows)

    lines = [
        "# Oracle benchmark: positioning error against the observation-derived floor",
        "",
        "| Method | Median [m] | Q1 [m] | Q3 [m] | P95 [m] | P99 [m] | Ratio to floor "
        f"| % > {EXCEEDANCE_THRESHOLD_M:g} m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['Method']} | {row['Median [m]']:.3f} | {row['Q1 [m]']:.3f} | "
            f"{row['Q3 [m]']:.3f} | {row['P95 [m]']:.3f} | {row['P99 [m]']:.3f} | "
            f"{row['Ratio to floor']:.1f}x | "
            f"{row[f'% > {EXCEEDANCE_THRESHOLD_M:g} m']:.2f}% |"
        )
    lines += [
        "",
        f"*Source: `{source_csv}`. Elevation (`elev`) weighting throughout - see "
        "`stec.analysis.oracle_benchmark`'s module docstring for why the reference STEC's "
        "placeholder sigma rules out `iono` weighting here. "
        f"N={len(paired):,} station-days, {paired['station'].nunique()} stations, "
        f"{paired['doy'].nunique()} days of 2024. Generated by "
        "`stec.viz.oracle_benchmark.build_oracle_table` - not hand-transcribed.*",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _build_figures(args: argparse.Namespace, output_dir: Path) -> None:
    source = analysis_dir(args.results_dir, "oracle_benchmark")
    paired_path = source / "paired_station_days.csv"
    if not paired_path.exists():
        logger.warning(
            f"{paired_path} not found - run `python -m stec.analysis.oracle_benchmark` first"
        )
        return
    paired = pd.read_csv(paired_path)
    prov = (
        f"{paired_path} (stec.analysis.oracle_benchmark), SF-PPP 2024 test period, "
        f"elevation weighting, {len(paired):,} station-days, "
        f"{paired['station'].nunique()} stations"
    )

    fig_boxplot(paired, output_dir, prov)
    fig_paired_difference(paired, output_dir, prov)

    coverage_path = source / "station_coverage.csv"
    if coverage_path.exists():
        fig_station_coverage(pd.read_csv(coverage_path), output_dir, prov)
    else:
        logger.warning(f"{coverage_path} not found - skipping station_coverage figure")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "oracle_table.md"
    table_path.write_text(build_oracle_table(paired, paired_path))
    logger.info(f"wrote {table_path}")


FIGURE_BUILDERS = (_build_figures,)


def build_all(args: argparse.Namespace) -> None:
    configure_plotting()
    for build in FIGURE_BUILDERS:
        build(args, args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("plots/oracle_benchmark")
    )
    parser.add_argument("--results_dir", type=Path, default=paths.RESULTS_ROOT)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    build_all(args)


if __name__ == "__main__":
    main()
