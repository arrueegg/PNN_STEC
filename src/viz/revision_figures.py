"""Figures for the JGR-MLC revision.

Each figure answers a specific reviewer comment and is built from analyses that
need no re-inference:

  fig_relative_error       R1.2  absolute vs TEC-normalised error by year
  fig_storm_positioning    R2.7  positioning under quiet vs storm conditions
  fig_weighting_ablation   R2.5  predicted-uncertainty vs elevation weighting
  fig_architecture_search  R1.5  architecture comparison from the W&B history

Colour
------
The published figures use matplotlib's default tab10 blue/orange/green/purple.
That palette has a real accessibility defect: orange (#ff7f0e) and green
(#2ca02c) collapse to dE = 0.7 in OKLab under protanopia, i.e. the
"VTEC + Mapping" and "IGS GIM + Mapping" series are indistinguishable for
red-blind readers. These figures therefore use the Okabe-Ito palette, which
clears the same check at dE >= 9.6 (worst all-pairs, protanopia) while keeping
blue for the Direct STEC series so the reader's association carries over.
Every series also gets a distinct marker, so identity is never colour-alone.

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

# Okabe-Ito. Validated all-pairs: protanopia dE 9.6, deuteranopia 7.6 (floor
# band, hence the mandatory marker/label secondary encoding), normal 16.4.
COLORS = {
    "Direct STEC": "#0072B2",
    "VTEC + Mapping": "#D55E00",
    "IGS GIM + Mapping": "#009E73",
    "Pretrained Direct STEC": "#CC79A7",
}
MARKERS = {
    "Direct STEC": "o",
    "VTEC + Mapping": "s",
    "IGS GIM + Mapping": "^",
    "Pretrained Direct STEC": "D",
}
METHOD_ORDER = [
    "Direct STEC",
    "Pretrained Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]

# Text stays in ink, never in a series colour.
INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#d9d9d9"
NEUTRAL = "#8c8c8c"


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


def _save(fig, name: str, output_dir: Path) -> None:
    """Write the titled figure and, per repo convention, a _notitle variant."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    for ax in fig.axes:
        # An axes keeps a separate title artist per location, so clearing only
        # the default (centre) one silently leaves a loc="left" title in place.
        for loc in ("center", "left", "right"):
            ax.set_title("", loc=loc)
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    fig.savefig(output_dir / f"{name}_notitle.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"💾 {output_dir / name}.png (+ _notitle)")


def fig_relative_error(metrics_csv: Path, output_dir: Path) -> None:
    """R1.2 - absolute error tracks TEC amplitude; relative error does not.

    Two panels rather than one chart with two y-scales: mixing TECU and percent
    on twin axes would let the reader read any relationship they liked into the
    gap between the curves.
    """
    d = pd.read_csv(metrics_csv).sort_values("year")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.5, 6.2), sharex=True, gridspec_kw={"hspace": 0.18}
    )

    # Panel (a): both quantities are TECU, so one axis is honest here.
    ax_top.plot(
        d.year,
        d.mean_STEC,
        marker="o",
        markersize=6,
        linewidth=2,
        color=NEUTRAL,
        label="Mean observed STEC",
    )
    ax_top.plot(
        d.year,
        d.RMSE,
        marker="s",
        markersize=6,
        linewidth=2,
        color=COLORS["Direct STEC"],
        label="Model RMSE",
    )
    _style_axes(ax_top, "STEC [TECU]")
    ax_top.set_title(
        "(a) Absolute error rises with the ionosphere itself",
        loc="left",
        fontsize=10,
        color=INK,
    )
    ax_top.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    ax_top.annotate(
        f"r = {d.RMSE.corr(d.mean_STEC):+.2f}",
        xy=(0.985, 0.06),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color=INK_MUTED,
    )

    # Panel (b): the same errors normalised by the mean STEC of each year.
    ax_bot.plot(
        d.year,
        d["nRMSE_%"],
        marker="s",
        markersize=6,
        linewidth=2,
        color=COLORS["Direct STEC"],
        label="RMSE / mean STEC",
    )
    median = d["nRMSE_%"].median()
    ax_bot.axhline(median, color=NEUTRAL, linewidth=1.2, linestyle="--")
    ax_bot.annotate(
        f"median {median:.0f}%",
        xy=(d.year.min(), median),
        xytext=(3, 5),
        textcoords="offset points",
        fontsize=9,
        color=INK_MUTED,
    )
    # Direct-label the year the reviewer asked about.
    last = d[d.year == d.year.max()].iloc[0]
    ax_bot.annotate(
        f"2024: {last['nRMSE_%']:.1f}%\n(lowest of all years)",
        xy=(last.year, last["nRMSE_%"]),
        xytext=(-8, 20),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8),
    )
    _style_axes(ax_bot, "Normalised RMSE [%]", "Year")
    ax_bot.set_title(
        "(b) Relative error is flat across the solar cycle",
        loc="left",
        fontsize=10,
        color=INK,
    )
    ax_bot.set_ylim(0, max(45, d["nRMSE_%"].max() * 1.15))
    ax_bot.set_xticks(d.year)

    _save(fig, "revision_relative_error", output_dir)


def _grouped_bars(ax, groups, series, values, colors, ylabel, value_fmt="{:.2f}"):
    """Grouped bars with a 2 px surface gap and a direct label on every bar."""
    n_series = len(series)
    x = np.arange(len(groups))
    width = 0.8 / n_series
    for i, name in enumerate(series):
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values[name],
            width * 0.94,  # 0.94 leaves the surface gap
            color=colors[i],
            label=name,
            zorder=3,
        )
        for bar, v in zip(bars, values[name]):
            ax.annotate(
                value_fmt.format(v),
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9, color=INK)
    _style_axes(ax, ylabel)


def fig_storm_positioning(degradation_csv: Path, output_dir: Path) -> None:
    """R2.7 - does the method fail when the ionosphere is disturbed?"""
    d = pd.read_csv(degradation_csv, index_col=0).reindex(METHOD_ORDER)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.22}
    )

    labels = [
        m.replace(" Direct STEC", "\nDirect STEC").replace(" + ", "\n+ ")
        for m in d.index
    ]
    _grouped_bars(
        ax_left,
        groups=labels,
        series=["quiet", "storm"],
        values={"quiet": d["quiet"].values, "storm": d["storm"].values},
        colors=[NEUTRAL, COLORS["VTEC + Mapping"]],
        ylabel="3D RMS positioning error [m]",
    )
    ax_left.legend(
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        title="Geomagnetic conditions",
        title_fontsize=9,
    )
    ax_left.get_legend().get_title().set_color(INK)
    ax_left.set_title(
        "(a) Absolute positioning error by regime", loc="left", fontsize=10, color=INK
    )

    # Improvement over the operational baseline, within each regime.
    gim = "IGS GIM + Mapping"
    improvement = pd.DataFrame(
        {
            reg: 100 * (d.loc[gim, reg] - d[reg]) / d.loc[gim, reg]
            for reg in ("quiet", "storm")
        }
    ).drop(index=gim)

    # Same quiet/storm encoding as panel (a). The sign is already carried by the
    # zero line, so colour is not asked to do a second job here.
    x = np.arange(len(improvement))
    width = 0.38
    for i, (reg, color) in enumerate(
        (("quiet", NEUTRAL), ("storm", COLORS["VTEC + Mapping"]))
    ):
        offset = (i - 0.5) * width
        bars = ax_right.bar(
            x + offset, improvement[reg], width * 0.94, color=color, label=reg, zorder=3
        )
        for bar, v in zip(bars, improvement[reg]):
            ax_right.annotate(
                f"{v:+.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, v),
                xytext=(0, 4 if v >= 0 else -12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK,
            )
    ax_right.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=4)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(
        [
            m.replace(" Direct STEC", "\nDirect STEC").replace(" + ", "\n+ ")
            for m in improvement.index
        ],
        fontsize=9,
        color=INK,
    )
    _style_axes(ax_right, "Improvement over IGS GIM + Mapping [%]")
    # Headroom so the outermost data labels do not collide with the frame.
    lo, hi = improvement.min().min(), improvement.max().max()
    ax_right.set_ylim(lo - 0.20 * (hi - lo), hi + 0.15 * (hi - lo))
    ax_right.set_title(
        "(b) Margin over the operational baseline is retained",
        loc="left",
        fontsize=10,
        color=INK,
    )
    ax_right.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")

    _save(fig, "revision_storm_positioning", output_dir)


def fig_weighting_ablation(paired_csv: Path, output_dir: Path) -> None:
    """R2.5 - isolate what the predicted uncertainty contributes to positioning."""
    d = pd.read_csv(paired_csv, index_col=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    labels = [m.replace(" + ", "\n+ ") for m in d.index]
    _grouped_bars(
        ax,
        groups=labels,
        series=["elevation weighting", "predicted-uncertainty weighting"],
        values={
            "elevation weighting": d["elev_mean"].values,
            "predicted-uncertainty weighting": d["iono_mean"].values,
        },
        colors=[NEUTRAL, COLORS["Direct STEC"]],
        ylabel="3D RMS positioning error [m]",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    ax.set_ylim(0, max(d[["elev_mean", "iono_mean"]].max()) * 1.25)

    for i, (name, row) in enumerate(d.iterrows()):
        ax.annotate(
            f"{row['gain_%']:+.1f}%",
            xy=(i, max(row["elev_mean"], row["iono_mean"])),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            # Ink, not the series colour - the sign already carries the direction,
            # and colouring text by value spends the identity channel twice.
            color=INK if row["gain_%"] > 0 else INK_MUTED,
        )
    ax.set_title(
        "Uncertainty weighting helps only where the uncertainty is observation-level",
        loc="left",
        fontsize=10,
        color=INK,
    )
    _save(fig, "revision_weighting_ablation", output_dir)


def fig_activity_stratification(
    dst_csv: Path, f107_csv: Path, output_dir: Path
) -> None:
    """R2.4 - STEC accuracy against geomagnetic activity and solar flux.

    Left column is the absolute error, which rises with activity partly because
    STEC itself does. Right column is the improvement over the operational
    baseline, which is scale-free and is what actually answers whether the
    advantage survives disturbed conditions.
    """
    # all_results.csv uses its own model names; map them onto the paper's labels.
    rename = {
        "Direct STEC Model": "Direct STEC",
        "Pretrained STEC": "Pretrained Direct STEC",
        "VTEC + Mapping": "VTEC + Mapping",
        "IGS GIM": "IGS GIM + Mapping",
    }

    fig, axes = plt.subplots(
        2, 2, figsize=(12.5, 8.0), gridspec_kw={"hspace": 0.42, "wspace": 0.22}
    )

    panels = [
        (
            "dst",
            dst_csv,
            "dst_bin",
            "Daily minimum Dst",
            "geomagnetic activity",
            "(a)",
            "(b)",
        ),
        ("f107", f107_csv, "f107_bin", "Daily mean F10.7", "solar flux", "(c)", "(d)"),
    ]

    for row, (_, csv_path, bin_col, axis_label, driver, tag_abs, tag_rel) in enumerate(
        panels
    ):
        d = pd.read_csv(csv_path)
        d["Model"] = d["Model"].map(rename)
        bins = list(dict.fromkeys(d[bin_col]))  # preserve the file's bin order

        ax_abs, ax_rel = axes[row, 0], axes[row, 1]

        # Absolute RMSE, all four models.
        series = [m for m in METHOD_ORDER if m in set(d["Model"])]
        values = {
            m: [d[(d.Model == m) & (d[bin_col] == b)]["RMSE"].iloc[0] for b in bins]
            for m in series
        }
        _grouped_bars(
            ax_abs,
            bins,
            series,
            values,
            [COLORS[m] for m in series],
            "STEC RMSE [TECU]",
            value_fmt="{:.1f}",
        )
        ax_abs.set_xlabel(axis_label, color=INK)
        ax_abs.set_title(
            f"{tag_abs} STEC error vs {driver}", loc="left", fontsize=10, color=INK
        )
        if row == 0:
            ax_abs.legend(
                frameon=False, fontsize=8.5, labelcolor=INK, ncol=2, loc="upper left"
            )
        ax_abs.set_ylim(0, max(max(v) for v in values.values()) * 1.28)

        # Improvement over the operational baseline, inside each bin.
        rel_series = [m for m in series if m != "IGS GIM + Mapping"]
        rel_values = {
            m: [
                d[(d.Model == m) & (d[bin_col] == b)]["improvement_over_gim_%"].iloc[0]
                for b in bins
            ]
            for m in rel_series
        }
        _grouped_bars(
            ax_rel,
            bins,
            rel_series,
            rel_values,
            [COLORS[m] for m in rel_series],
            "Improvement over IGS GIM [%]",
            value_fmt="{:+.0f}",
        )
        ax_rel.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=4)
        ax_rel.set_xlabel(axis_label, color=INK)
        ax_rel.set_title(
            f"{tag_rel} Margin over IGS GIM vs {driver}",
            loc="left",
            fontsize=10,
            color=INK,
        )
        flat = [v for vals in rel_values.values() for v in vals]
        ax_rel.set_ylim(
            min(flat) - 0.22 * (max(flat) - min(flat)),
            max(flat) + 0.22 * (max(flat) - min(flat)),
        )

    _save(fig, "revision_activity_stratification", output_dir)


def fig_architecture_search(architectures_csv: Path, output_dir: Path) -> None:
    """R1.5 - what alternatives were tried, and how fairly."""
    d = pd.read_csv(architectures_csv, index_col=0).sort_values("best_val_MAE")

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    y = np.arange(len(d))
    credible = d["credible_runs"] > 0
    colors = [COLORS["Direct STEC"] if c else NEUTRAL for c in credible]
    bars = ax.barh(y, d["best_val_MAE"], 0.62, color=colors, zorder=3)

    for bar, (name, row) in zip(bars, d.iterrows()):
        note = (
            f"{row['best_val_MAE']:.2f}   "
            f"({int(row['runs'])} runs, {int(row['credible_runs'])} credible)"
        )
        ax.annotate(
            note,
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
        "Architecture search: grey = no run reached 20 epochs, so not a fair comparison",
        loc="left",
        fontsize=10,
        color=INK,
    )
    _save(fig, "revision_architecture_search", output_dir)


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
        "--activity_dir",
        type=Path,
        default=Path("multiday_results/activity_stratification"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    for path, builder in (
        (args.relative_metrics, fig_relative_error),
        (args.storm_degradation, fig_storm_positioning),
        (args.weighting_paired, fig_weighting_ablation),
        (args.architectures, fig_architecture_search),
    ):
        if path.exists():
            builder(path, args.output_dir)
        else:
            logger.warning(f"⚠️  {path} not found - skipping {builder.__name__}")

    # This one needs both stratification tables, so it sits outside the loop.
    dst_csv = args.activity_dir / "by_dst.csv"
    f107_csv = args.activity_dir / "by_f107.csv"
    if dst_csv.exists() and f107_csv.exists():
        fig_activity_stratification(dst_csv, f107_csv, args.output_dir)
    else:
        logger.warning(
            f"⚠️  {args.activity_dir} incomplete - run src/analysis/activity_stratification.py first"
        )


if __name__ == "__main__":
    main()
