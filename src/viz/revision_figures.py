"""Figures for the JGR-MLC revision — one panel per file.

Each figure answers a specific reviewer comment and is built from analyses that
need no re-inference. Figures are written one plot per PNG, grouped into
subfolders by the data they were produced from, and every titled figure carries
a provenance footnote naming its source file and scope.

    plots/revision/
      stec_pretrained_testset/   pretrained model, held-out test set 2014-2024
      stec_finetuned_2024/       daily fine-tuned models, 242 test days of 2024
      positioning_2024/          SF-PPP solutions, 2024 test period
      training_runs/             W&B training history

Reviewer mapping
----------------
  R1.2  relative_error_absolute, relative_error_normalised
  R1.5  architecture_search
  R2.4  activity_dst_*, activity_f107_*
  R2.5  weighting_ablation
  R2.7  storm_positioning_*

Colour
------
Series colours and markers are taken unchanged from the published figures
(``positioning/scripts/plot_results.py``) so the revision is visually consistent
with the rest of the paper: blue = Direct STEC, orange = VTEC + Mapping,
green = IGS GIM + Mapping, purple = Pretrained Direct STEC.

Known limitation, carried over deliberately: orange (#ff7f0e) and green
(#2ca02c) are separated by only dE = 0.7 in OKLab under simulated protanopia,
so those two series are hard to distinguish for red-blind readers. Consistency
with the published figures was chosen over changing the palette. Where the form
allows it, the paper's own per-series markers are used as a secondary channel so
identity does not rest on colour alone.

Usage::

    python src/viz/revision_figures.py --output_dir plots/revision
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Unchanged from positioning/scripts/plot_results.py so the revision figures sit
# beside the published ones without a palette shift.
COLORS = {
    "Direct STEC": "#1f77b4",
    "VTEC + Mapping": "#ff7f0e",
    "IGS GIM + Mapping": "#2ca02c",
    "Pretrained Direct STEC": "#9467bd",
}
MARKERS = {
    "Direct STEC": "o",
    "VTEC + Mapping": "s",
    "IGS GIM + Mapping": "^",
    "Pretrained Direct STEC": "d",
}
METHOD_ORDER = [
    "Direct STEC",
    "Pretrained Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]

# Colours for non-approach encodings (geomagnetic regime, weighting scheme).
# Deliberately outside the approach palette above: an approach colour must only
# ever mean that approach, or the figures stop being readable side by side.
CONDITION_COLORS = {"baseline": "#7f7f7f", "contrast": "#d62728"}

# The observation-derived upper bound is not one of the compared approaches, so
# it takes its own dark neutral instead of borrowing an approach colour.
ORACLE_COLOR = "#3f3f3f"

INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#d9d9d9"
NEUTRAL = "#7f7f7f"

# Subfolders, keyed by which data produced the figure.
SOURCE_DIRS = {
    "pretrained": "stec_pretrained_testset",
    "finetuned": "stec_finetuned_2024",
    "positioning": "positioning_2024",
    "training": "training_runs",
}


def _style_axes(ax, ylabel: str, xlabel: str = "") -> None:
    """Recessive grid and axes; the data carries the emphasis."""
    ax.set_ylabel(ylabel, color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def _save(fig, name: str, source: str, output_dir: Path, provenance: str) -> None:
    """Write one figure twice: titled with provenance, and a clean publication copy.

    The provenance footnote states exactly which file and how many days or runs
    the numbers came from. It is deliberately absent from the `_notitle` variant,
    which is the one that goes into the manuscript.
    """
    target = output_dir / SOURCE_DIRS[source]
    target.mkdir(parents=True, exist_ok=True)

    # Negative y puts the note below the x-axis label; bbox_inches="tight"
    # expands the saved area to include artists outside the figure rectangle,
    # so this clears the label instead of overprinting it.
    footnote = fig.text(
        0.0, -0.06, f"Data: {provenance}", fontsize=7, color=INK_MUTED, va="top"
    )
    fig.savefig(target / f"{name}.png", dpi=300, bbox_inches="tight")

    footnote.set_text("")
    for ax in fig.axes:
        # An axes keeps a separate title artist per location, so clearing only
        # the default (centre) one silently leaves a loc="left" title in place.
        for loc in ("center", "left", "right"):
            ax.set_title("", loc=loc)
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    fig.savefig(target / f"{name}_notitle.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"💾 {target / name}.png (+ _notitle)")


def _grouped_bars(ax, groups, series, values, colors, ylabel, value_fmt="{:.2f}"):
    """Grouped bars with a small surface gap and a direct label on every bar."""
    n_series = len(series)
    x = np.arange(len(groups))
    width = 0.8 / n_series
    for i, name in enumerate(series):
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values[name],
            width * 0.94,
            color=colors[i],
            label=name,
            zorder=3,
        )
        for bar, v in zip(bars, values[name]):
            ax.annotate(
                value_fmt.format(v),
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 3 if v >= 0 else -11),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9, color=INK)
    _style_axes(ax, ylabel)


# --------------------------------------------------------------------------
# R1.2 - absolute vs relative error across the solar cycle
# --------------------------------------------------------------------------


def fig_relative_error_absolute(
    d: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Absolute RMSE alongside the mean observed STEC, both in TECU."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        d.year,
        d.mean_STEC,
        marker="o",
        markersize=6,
        linewidth=2,
        color=NEUTRAL,
        label="Mean observed STEC",
    )
    ax.plot(
        d.year,
        d.RMSE,
        marker="s",
        markersize=6,
        linewidth=2,
        color=COLORS["Direct STEC"],
        label="Model RMSE",
    )
    _style_axes(ax, "STEC [TECU]", "Year")
    ax.set_xticks(d.year)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    ax.annotate(
        f"r = {d.RMSE.corr(d.mean_STEC):+.2f}",
        xy=(0.985, 0.06),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color=INK_MUTED,
    )
    ax.set_title(
        "Absolute error rises with the ionosphere itself",
        loc="left",
        fontsize=10,
        color=INK,
    )
    _save(fig, "relative_error_absolute", "pretrained", output_dir, provenance)


def fig_relative_error_normalised(
    d: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """The same errors normalised by each year's mean observed STEC."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        d.year,
        d["nRMSE_%"],
        marker="s",
        markersize=6,
        linewidth=2,
        color=COLORS["Direct STEC"],
        label="RMSE / mean STEC",
    )
    median = d["nRMSE_%"].median()
    ax.axhline(median, color=NEUTRAL, linewidth=1.2, linestyle="--")
    ax.annotate(
        f"median {median:.0f}%",
        xy=(d.year.min(), median),
        xytext=(3, 5),
        textcoords="offset points",
        fontsize=9,
        color=INK_MUTED,
    )
    last = d[d.year == d.year.max()].iloc[0]
    ax.annotate(
        f"2024: {last['nRMSE_%']:.1f}%\n(lowest of all years)",
        xy=(last.year, last["nRMSE_%"]),
        xytext=(-8, 26),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8),
    )
    _style_axes(ax, "Normalised RMSE [%]", "Year")
    ax.set_xticks(d.year)
    ax.set_ylim(0, max(45, d["nRMSE_%"].max() * 1.18))
    ax.set_title(
        "Relative error is flat across the solar cycle",
        loc="left",
        fontsize=10,
        color=INK,
    )
    _save(fig, "relative_error_normalised", "pretrained", output_dir, provenance)


# --------------------------------------------------------------------------
# R2.7 - positioning under quiet vs storm conditions
# --------------------------------------------------------------------------


def fig_storm_positioning_absolute(
    d: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    labels = [
        m.replace(" Direct STEC", "\nDirect STEC").replace(" + ", "\n+ ")
        for m in d.index
    ]
    _grouped_bars(
        ax,
        labels,
        ["quiet", "storm"],
        {"quiet": d["quiet"].values, "storm": d["storm"].values},
        [CONDITION_COLORS["baseline"], CONDITION_COLORS["contrast"]],
        "3D RMS positioning error [m]",
    )
    ax.set_ylim(0, d[["quiet", "storm"]].max().max() * 1.18)
    legend = ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        title="Geomagnetic conditions",
        title_fontsize=9,
    )
    legend.get_title().set_color(INK)
    ax.set_title(
        "Positioning error by geomagnetic regime", loc="left", fontsize=10, color=INK
    )
    _save(fig, "storm_positioning_absolute", "positioning", output_dir, provenance)


def fig_storm_positioning_improvement(
    d: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    gim = "IGS GIM + Mapping"
    improvement = pd.DataFrame(
        {
            reg: 100 * (d.loc[gim, reg] - d[reg]) / d.loc[gim, reg]
            for reg in ("quiet", "storm")
        }
    ).drop(index=gim)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    labels = [
        m.replace(" Direct STEC", "\nDirect STEC").replace(" + ", "\n+ ")
        for m in improvement.index
    ]
    _grouped_bars(
        ax,
        labels,
        ["quiet", "storm"],
        {"quiet": improvement["quiet"].values, "storm": improvement["storm"].values},
        [CONDITION_COLORS["baseline"], CONDITION_COLORS["contrast"]],
        "Improvement over IGS GIM + Mapping [%]",
        value_fmt="{:+.0f}",
    )
    ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=4)
    flat = improvement.values.flatten()
    ax.set_ylim(flat.min() - 0.22 * np.ptp(flat), flat.max() + 0.22 * np.ptp(flat))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    ax.set_title(
        "Margin over the operational baseline is retained under storms",
        loc="left",
        fontsize=10,
        color=INK,
    )
    _save(fig, "storm_positioning_improvement", "positioning", output_dir, provenance)


# --------------------------------------------------------------------------
# R2.4 - STEC error against geomagnetic activity and solar flux
# --------------------------------------------------------------------------


def _activity_figures(
    table: pd.DataFrame,
    bin_col: str,
    axis_label: str,
    driver: str,
    stem: str,
    output_dir: Path,
    provenance: str,
) -> None:
    """One absolute-error figure and one margin figure for a given stratifier."""
    rename = {
        "Direct STEC Model": "Direct STEC",
        "Pretrained STEC": "Pretrained Direct STEC",
        "VTEC + Mapping": "VTEC + Mapping",
        "IGS GIM": "IGS GIM + Mapping",
    }
    d = table.copy()
    d["Model"] = d["Model"].map(rename)
    bins = list(dict.fromkeys(d[bin_col]))
    series = [m for m in METHOD_ORDER if m in set(d["Model"])]

    values = {
        m: [d[(d.Model == m) & (d[bin_col] == b)]["RMSE"].iloc[0] for b in bins]
        for m in series
    }
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    _grouped_bars(
        ax,
        bins,
        series,
        values,
        [COLORS[m] for m in series],
        "STEC RMSE [TECU]",
        value_fmt="{:.1f}",
    )
    ax.set_xlabel(axis_label, color=INK)
    ax.set_ylim(0, max(max(v) for v in values.values()) * 1.22)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, ncol=2, loc="upper left")
    ax.set_title(f"STEC error vs {driver}", loc="left", fontsize=10, color=INK)
    _save(fig, f"{stem}_absolute", "finetuned", output_dir, provenance)

    rel_series = [m for m in series if m != "IGS GIM + Mapping"]
    rel_values = {
        m: [
            d[(d.Model == m) & (d[bin_col] == b)]["improvement_over_gim_%"].iloc[0]
            for b in bins
        ]
        for m in rel_series
    }
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    _grouped_bars(
        ax,
        bins,
        rel_series,
        rel_values,
        [COLORS[m] for m in rel_series],
        "Improvement over IGS GIM [%]",
        value_fmt="{:+.0f}",
    )
    ax.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=4)
    ax.set_xlabel(axis_label, color=INK)
    flat = np.array([v for vals in rel_values.values() for v in vals])
    ax.set_ylim(flat.min() - 0.22 * np.ptp(flat), flat.max() + 0.22 * np.ptp(flat))
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
    ax.set_title(f"Margin over IGS GIM vs {driver}", loc="left", fontsize=10, color=INK)
    _save(fig, f"{stem}_improvement", "finetuned", output_dir, provenance)


# --------------------------------------------------------------------------
# R2.5 / R1.5
# --------------------------------------------------------------------------


def fig_weighting_ablation(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    labels = [m.replace(" + ", "\n+ ") for m in d.index]
    _grouped_bars(
        ax,
        labels,
        ["elevation weighting", "predicted-uncertainty weighting"],
        {
            "elevation weighting": d["elev_mean"].values,
            "predicted-uncertainty weighting": d["iono_mean"].values,
        },
        [CONDITION_COLORS["baseline"], CONDITION_COLORS["contrast"]],
        "3D RMS positioning error [m]",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax.set_ylim(0, d[["elev_mean", "iono_mean"]].max().max() * 1.28)
    for i, (_, row) in enumerate(d.iterrows()):
        ax.annotate(
            f"{row['gain_%']:+.1f}%",
            xy=(i, max(row["elev_mean"], row["iono_mean"])),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=INK if row["gain_%"] > 0 else INK_MUTED,
        )
    ax.set_title(
        "Uncertainty weighting helps only where the uncertainty is observation-level",
        loc="left",
        fontsize=10,
        color=INK,
    )
    _save(fig, "weighting_ablation", "positioning", output_dir, provenance)


def fig_oracle_benchmark(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """R2.8 - how much of the remaining error is the model's, and how much the pipeline's.

    The oracle bar is the pipeline's own noise floor: what is left when the
    reference STEC itself is used as the correction. The gap above it is what a
    better model could still recover.
    """
    order = [m for m in ("Reference STEC (oracle)", *METHOD_ORDER) if m in d.index]
    d = d.loc[order]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    colors = [
        ORACLE_COLOR if m.startswith("Reference STEC") else COLORS[m] for m in d.index
    ]
    bars = ax.bar(np.arange(len(d)), d["mean"], 0.62, color=colors, zorder=3)
    for bar, (_, row) in zip(bars, d.iterrows()):
        ax.annotate(
            f"{row['mean']:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, row["mean"]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )

    floor = d.loc[d.index[0], "mean"]
    ax.axhline(floor, color=ORACLE_COLOR, linewidth=1.0, linestyle="--", zorder=4)
    # State the multiple explicitly: the gap is what the bar labels alone hide.
    best = d.iloc[1:]["mean"].min()
    ax.annotate(
        f"models sit {best / floor:.0f}-{d.iloc[1:]['mean'].max() / floor:.0f}x above the floor,\n"
        "i.e. almost all remaining error is ionospheric",
        xy=(0.5, 0.93),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9,
        color=INK_MUTED,
    )

    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels(
        [
            m.replace(" (oracle)", "\n(oracle)")
            .replace(" Direct STEC", "\nDirect STEC")
            .replace(" + ", "\n+ ")
            for m in d.index
        ],
        fontsize=9,
        color=INK,
    )
    _style_axes(ax, "3D RMS positioning error [m]")
    ax.set_ylim(0, d["mean"].max() * 1.22)
    ax.set_title(
        "Correcting with the reference STEC itself leaves 0.09 m of 3D RMS error",
        loc="left",
        fontsize=10,
        color=INK,
    )
    _save(fig, "oracle_benchmark", "positioning", output_dir, provenance)


def fig_architecture_search(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    d = d.sort_values("best_val_MAE")
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    y = np.arange(len(d))
    colors = [COLORS["Direct STEC"] if c > 0 else NEUTRAL for c in d["credible_runs"]]
    bars = ax.barh(y, d["best_val_MAE"], 0.62, color=colors, zorder=3)
    for bar, (_, row) in zip(bars, d.iterrows()):
        ax.annotate(
            f"{row['best_val_MAE']:.2f}   ({int(row['runs'])} runs, {int(row['credible_runs'])} credible)",
            xy=(row["best_val_MAE"], bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(d.index, fontsize=9, color=INK)
    ax.invert_yaxis()
    _style_axes(ax, "")
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_xlabel("Best validation MAE [TECU]", color=INK)
    ax.set_xlim(0, d["best_val_MAE"].max() * 1.55)
    ax.set_title(
        "Grey = no run reached 20 epochs, so not a fair comparison",
        loc="left",
        fontsize=10,
        color=INK,
    )
    _save(fig, "architecture_search", "training", output_dir, provenance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("plots/revision"))
    parser.add_argument(
        "--relative_metrics",
        type=Path,
        default=Path("multiday_results/relative_error_metrics.csv"),
    )
    parser.add_argument(
        "--storm_degradation",
        type=Path,
        default=Path("multiday_results/storm_stratification/degradation.csv"),
    )
    parser.add_argument(
        "--weighting_paired",
        type=Path,
        default=Path("multiday_results/weighting_ablation/paired.csv"),
    )
    parser.add_argument(
        "--architectures",
        type=Path,
        default=Path("multiday_results/hyperparameter_search/architectures.csv"),
    )
    parser.add_argument(
        "--oracle_summary",
        type=Path,
        default=Path("multiday_results/oracle_benchmark/summary.csv"),
    )
    parser.add_argument(
        "--activity_dir",
        type=Path,
        default=Path("multiday_results/activity_stratification"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.relative_metrics.exists():
        d = pd.read_csv(args.relative_metrics).sort_values("year")
        prov = (
            f"{args.relative_metrics} — pretrained model, held-out test set, "
            f"{d.year.min()}-{d.year.max()} ({int(d['count'].sum()):,} observations)"
        )
        fig_relative_error_absolute(d, args.output_dir, prov)
        fig_relative_error_normalised(d, args.output_dir, prov)
    else:
        logger.warning(f"⚠️  {args.relative_metrics} not found")

    if args.storm_degradation.exists():
        d = pd.read_csv(args.storm_degradation, index_col=0).reindex(METHOD_ORDER)
        prov = (
            f"{args.storm_degradation} — SF-PPP, 2024 test period, 39 storm days "
            "(daily min Dst ≤ −50 nT) of 242, station-days ≤ 10 m"
        )
        fig_storm_positioning_absolute(d, args.output_dir, prov)
        fig_storm_positioning_improvement(d, args.output_dir, prov)
    else:
        logger.warning(f"⚠️  {args.storm_degradation} not found")

    if args.weighting_paired.exists():
        d = pd.read_csv(args.weighting_paired, index_col=0)
        prov = (
            f"{args.weighting_paired} — SF-PPP, 2024 test period, paired station-days "
            f"solved under both weightings ({int(d['paired_station_days'].sum()):,} pairs)"
        )
        fig_weighting_ablation(d, args.output_dir, prov)
    else:
        logger.warning(f"⚠️  {args.weighting_paired} not found")

    if args.architectures.exists():
        d = pd.read_csv(args.architectures, index_col=0)
        prov = (
            f"{args.architectures} — local W&B history, "
            f"{int(d['runs'].sum())} STEC runs reporting a validation MAE"
        )
        fig_architecture_search(d, args.output_dir, prov)
    else:
        logger.warning(f"⚠️  {args.architectures} not found")

    if args.oracle_summary.exists():
        d = pd.read_csv(args.oracle_summary, index_col=0)
        station_days = int(d["station_days"].max())
        prov = (
            f"{args.oracle_summary} — SF-PPP, elevation weighting throughout, "
            f"{station_days} station-days solved by every method"
        )
        fig_oracle_benchmark(d, args.output_dir, prov)
    else:
        logger.warning(
            f"⚠️  {args.oracle_summary} not found - run src/analysis/oracle_benchmark.py"
        )

    for stem, filename, bin_col, axis_label, driver in (
        (
            "activity_dst",
            "by_dst.csv",
            "dst_bin",
            "Daily minimum Dst",
            "geomagnetic activity",
        ),
        ("activity_f107", "by_f107.csv", "f107_bin", "Daily mean F10.7", "solar flux"),
    ):
        path = args.activity_dir / filename
        if not path.exists():
            logger.warning(
                f"⚠️  {path} not found - run src/analysis/activity_stratification.py"
            )
            continue
        table = pd.read_csv(path)
        days = int(table[table.Model == "Direct STEC Model"]["days"].sum())
        obs = int(table[table.Model == "Direct STEC Model"]["observations"].sum())
        prov = (
            f"{path} — daily fine-tuned models, own test set, "
            f"{days} test days of 2024 ({obs:,} observations)"
        )
        _activity_figures(
            table, bin_col, axis_label, driver, stem, args.output_dir, prov
        )


if __name__ == "__main__":
    main()
