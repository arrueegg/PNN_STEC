"""Figures for `stec.analysis.positioning_diagnostics` (owner request, 2026-08-28).

Not manuscript or revision-response figures - this is diagnostic output built to look at
the coverage-recovery result before drawing any conclusion from it, so it lives in its
own `plots/positioning_diagnostics/` tree rather than `plots/manuscript/` or
`plots/revision/`. Nothing here is wired into `stec/pipeline/stages.py` yet.

Reads the CSVs `stec.analysis.positioning_diagnostics` writes to
`multiday_results/analyses/positioning_diagnostics/rebuilt/` - it does not recompute
anything from the raw per-station-day table itself, the same separation of concerns as
`revision_figures.py` and `manuscript_figures.py` (analysis writes numbers, viz reads and
draws them).

Style: `stec.viz.style.PLOT_CONFIG` and `APPROACH_COLORS` (blue Direct STEC, orange
VTEC + Mapping, green IGS GIM + Mapping, purple Pretrained Direct STEC) - an approach
colour means only that approach. The mean-vs-median and quiet-vs-storm contrasts use
`CONDITION_COLORS` (grey/red) instead, never an approach colour. Each figure is written
twice via `_save`: a titled working copy with a provenance footnote, and a `_notitle`
copy with neither - and `_save` always writes the plotted values as a CSV alongside the
PNGs, so the number a reader checks is the number on the axes.

Usage::

    python -m stec.viz.positioning_diagnostics --output_dir plots/positioning_diagnostics
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import paths
from .revision_figures import _grouped_bars, analysis_dir
from .style import (
    APPROACH_COLORS,
    CONDITION_COLORS,
    FIGSIZE_WIDE,
    configure_plotting,
)

import matplotlib.pyplot as plt  # noqa: E402  (style.py sets the Agg backend on import)

logger = logging.getLogger(__name__)

SOURCE_DIRS = {"positioning": "positioning_2024"}

METHOD_ORDER = [
    "Direct STEC",
    "Pretrained Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]
_METHOD_MARKERS = {
    "Direct STEC": "o",
    "VTEC + Mapping": "s",
    "IGS GIM + Mapping": "^",
    "Pretrained Direct STEC": "d",
}
FIGSIZE_TWO_PANEL = (16, 12)


def _save(
    fig: plt.Figure,
    name: str,
    source: str,
    output_dir: Path,
    provenance: str,
    data: pd.DataFrame | None = None,
) -> None:
    """Write the working copy (title + provenance), the notitle copy, and the plotted
    numbers. Mirrors `revision_figures._save` / `manuscript_figures._save` exactly;
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


# --------------------------------------------------------------------------
# 1. Per-DOY timeseries - mean and median panels, storm days shaded
# --------------------------------------------------------------------------


def fig_daily_timeseries(df: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """Two panels sharing a DOY x-axis: mean 3D error on top, median on the bottom, one
    line per method. Storm days (from `storm_stratification`'s daily-Dst rule) are
    shaded so the mean/median divergence can be read against geomagnetic activity."""
    fig, (ax_mean, ax_median) = plt.subplots(
        2, 1, figsize=FIGSIZE_TWO_PANEL, sharex=True
    )
    order = [m for m in METHOD_ORDER if m in df["Method"].unique()]

    storm_doys = sorted(df.loc[df["storm"], "doy"].unique())
    for ax in (ax_mean, ax_median):
        for doy in storm_doys:
            ax.axvspan(
                doy - 0.5,
                doy + 0.5,
                color=CONDITION_COLORS["contrast"],
                alpha=0.12,
                zorder=0,
            )

    for method in order:
        subset = df[df["Method"] == method].sort_values("doy")
        color = APPROACH_COLORS[method]
        marker = _METHOD_MARKERS[method]
        ax_mean.plot(
            subset["doy"], subset["mean"], color=color, marker=marker,
            markersize=3, linewidth=1, label=method,
        )  # fmt: skip
        ax_median.plot(
            subset["doy"], subset["median"], color=color, marker=marker,
            markersize=3, linewidth=1, label=method,
        )  # fmt: skip

    ax_mean.set_ylabel("Mean 3D error [m]")
    ax_median.set_ylabel("Median 3D error [m]")
    ax_median.set_xlabel("Day of year (2024)")
    for ax in (ax_mean, ax_median):
        ax.grid(True, linestyle="--", alpha=0.3)
    ax_mean.legend(loc="upper right", ncol=2)
    fig.suptitle("Daily positioning 3D error by method (storm days shaded)")
    _save(
        fig,
        "daily_timeseries",
        "positioning",
        output_dir,
        provenance,
        df[["doy", "Method", "mean", "median", "station_days", "storm"]],
    )


# --------------------------------------------------------------------------
# 2. Per-station Direct-STEC-minus-GIM difference
# --------------------------------------------------------------------------


def fig_per_station_diff(df: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """One dot per station: Direct STEC mean 3D error minus IGS GIM mean 3D error,
    sorted so stations where the model loses (positive) are grouped together. Marker
    size scales with the station's Direct-STEC station-day count, so a 3-day station
    reads differently from a 200-day one; stations with no Direct STEC solution at all
    are listed separately rather than silently dropped."""
    plotted = df.dropna(subset=["diff_mean_m"]).copy()
    plotted = plotted.sort_values("diff_mean_m", ascending=True).reset_index(drop=True)
    missing_stec = df[df["diff_mean_m"].isna()]["station"].tolist()

    fig, ax = plt.subplots(figsize=(10, max(8, 0.28 * len(plotted))))
    colors = [
        CONDITION_COLORS["contrast"] if v > 0 else CONDITION_COLORS["baseline"]
        for v in plotted["diff_mean_m"]
    ]
    sizes = 15 + 45 * np.sqrt(
        plotted["stec_station_days"] / plotted["stec_station_days"].max()
    )
    ax.scatter(plotted["diff_mean_m"], plotted["station"], c=colors, s=sizes, zorder=3)
    ax.axvline(0, color="black", linewidth=1.0, zorder=2)
    ax.set_xlabel(
        "Direct STEC minus IGS GIM, mean 3D error [m]\n(positive = Direct STEC worse)"
    )
    ax.set_ylabel("Station")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_title(
        "Per-station Direct STEC vs IGS GIM"
        + (f"  ({len(missing_stec)} station(s) with no Direct STEC solution not shown)" if missing_stec else "")
    )  # fmt: skip
    _save(
        fig,
        "per_station_diff",
        "positioning",
        output_dir,
        provenance,
        df[
            [
                "station", "stec_station_days", "gim_station_days",
                "stec_mean_m", "gim_mean_m", "diff_mean_m", "diff_median_m",
            ]
        ],
    )  # fmt: skip


# --------------------------------------------------------------------------
# 3. Outlier characterisation
# --------------------------------------------------------------------------


_THRESHOLD_ORDER = ["none", "5m", "10m", "20m", "50m"]


def fig_outlier_threshold_sensitivity(
    df: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Direct STEC vs IGS GIM headline improvement, mean and median, as the exclusion
    threshold widens from no exclusion to 50 m. The project's own 10 m rule is marked
    with a vertical line rather than assumed to be where the curve settles."""
    ordered = df.set_index("threshold_label").reindex(_THRESHOLD_ORDER).reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(ordered))
    ax.plot(
        x, ordered["mean_improvement_pct"], color=CONDITION_COLORS["baseline"],
        marker="o", markersize=8, linewidth=2, label="Mean improvement",
    )  # fmt: skip
    ax.plot(
        x, ordered["median_improvement_pct"], color=CONDITION_COLORS["contrast"],
        marker="s", markersize=8, linewidth=2, label="Median improvement",
    )  # fmt: skip
    ax.axhline(0, color="black", linewidth=1.0)
    ten_m_index = _THRESHOLD_ORDER.index("10m")
    ax.axvline(
        ten_m_index, color="#555555", linestyle=":", linewidth=1.5,
        label="Project's 10 m rule",
    )  # fmt: skip
    ax.set_xticks(x)
    ax.set_xticklabels(_THRESHOLD_ORDER)
    ax.set_xlabel("Station-day exclusion threshold")
    ax.set_ylabel("Direct STEC improvement over IGS GIM [%]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("Headline improvement vs. outlier exclusion threshold")
    _save(
        fig,
        "outlier_threshold_sensitivity",
        "positioning",
        output_dir,
        provenance,
        ordered[
            [
                "threshold_label", "threshold_m", "n_stec", "n_gim",
                "mean_improvement_pct", "median_improvement_pct",
            ]
        ],
    )  # fmt: skip


def fig_outlier_pct_by_threshold(
    df: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Grouped bars: fraction of each method's station-days exceeding each threshold.
    Shows directly whether Direct STEC produces more extreme station-days than the
    other methods, rather than only how many exceed the one 10 m rule."""
    order = [m for m in METHOD_ORDER if m in df["Method"].unique()]
    thresholds = sorted(df["threshold_m"].unique())
    pivot = df.pivot(
        index="threshold_m", columns="Method", values="pct_station_days_exceeding"
    )
    pivot = pivot.reindex(thresholds)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        [f"{t:g} m" for t in thresholds],
        order,
        {m: pivot[m].values for m in order},
        [APPROACH_COLORS[m] for m in order],
        "Station-days exceeding threshold [%]",
        xlabel="Threshold",
    )
    ax.legend(loc="upper right")
    ax.set_title("Share of station-days exceeding each outlier threshold")
    _save(
        fig, "outlier_pct_by_threshold", "positioning", output_dir, provenance, plotted
    )


# --------------------------------------------------------------------------
# 4. Population split - recovered (geometry-only) vs original station-days
# --------------------------------------------------------------------------


def fig_population_split(df: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """Two panels (mean top, median bottom): 3D error by method, grouped by whether the
    station-day's STEC-database row is original or was reconstructed by
    `build_recovered_day.py` from geometry alone."""
    order = [m for m in METHOD_ORDER if m in df["Method"].unique()]
    populations = ["original", "recovered"]
    mean_pivot = df.pivot(
        index="population", columns="Method", values="mean_m"
    ).reindex(populations)
    median_pivot = df.pivot(
        index="population", columns="Method", values="median_m"
    ).reindex(populations)

    fig, (ax_mean, ax_median) = plt.subplots(1, 2, figsize=(20, 8))
    group_labels = ["Original\n(in STEC database)", "Recovered\n(geometry only)"]
    plotted_mean = _grouped_bars(
        ax_mean, group_labels, order,
        {m: mean_pivot[m].values for m in order},
        [APPROACH_COLORS[m] for m in order],
        "Mean 3D error [m]",
    )  # fmt: skip
    plotted_median = _grouped_bars(
        ax_median, group_labels, order,
        {m: median_pivot[m].values for m in order},
        [APPROACH_COLORS[m] for m in order],
        "Median 3D error [m]",
    )  # fmt: skip
    ax_mean.legend(loc="upper left")
    ax_mean.set_title("Mean")
    ax_median.set_title("Median")
    fig.suptitle("Positioning error: original vs. recovered station-days")
    plotted_mean["statistic"] = "mean"
    plotted_median["statistic"] = "median"
    _save(
        fig,
        "population_split",
        "positioning",
        output_dir,
        provenance,
        pd.concat([plotted_mean, plotted_median], ignore_index=True),
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _build_figures(args: argparse.Namespace, output_dir: Path) -> None:
    source = analysis_dir(args.results_dir, "positioning_diagnostics")
    if not source.is_dir():
        logger.warning(
            f"{source} not found - run stec/analysis/positioning_diagnostics.py first"
        )
        return
    prov = f"{source} (stec.analysis.positioning_diagnostics), SF-PPP 2024 test period"

    daily_path = source / "daily_timeseries.csv"
    if daily_path.exists():
        fig_daily_timeseries(pd.read_csv(daily_path), output_dir, prov)

    per_station_path = source / "per_station_summary.csv"
    if per_station_path.exists():
        fig_per_station_diff(pd.read_csv(per_station_path), output_dir, prov)

    sensitivity_path = source / "outlier_headline_sensitivity.csv"
    if sensitivity_path.exists():
        fig_outlier_threshold_sensitivity(
            pd.read_csv(sensitivity_path), output_dir, prov
        )

    counts_path = source / "outlier_threshold_counts.csv"
    if counts_path.exists():
        fig_outlier_pct_by_threshold(pd.read_csv(counts_path), output_dir, prov)

    population_path = source / "population_split_by_method.csv"
    if population_path.exists():
        fig_population_split(pd.read_csv(population_path), output_dir, prov)


FIGURE_BUILDERS = (_build_figures,)


def build_all(args: argparse.Namespace) -> None:
    configure_plotting()
    for build in FIGURE_BUILDERS:
        build(args, args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("plots/positioning_diagnostics")
    )
    parser.add_argument("--results_dir", type=Path, default=paths.RESULTS_ROOT)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    build_all(args)


if __name__ == "__main__":
    main()
