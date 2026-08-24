"""Publication figures for the JGR-MLC revision - one panel per file.

Ported from `src/viz/revision_figures.py` in the live checkout. Each figure answers a
specific reviewer comment and is built from analyses that need no re-inference. Figures
are written one plot per PNG, grouped into subfolders by the data that produced them:

    plots/revision/
      stec_pretrained_testset/   pretrained model, held-out test set 2014-2024
      stec_finetuned_2024/       daily fine-tuned models, 242 test days of 2024
      positioning_2024/          SF-PPP solutions, 2024 test period
      training_runs/             W&B training history

Reviewer mapping
----------------
  R2.2  relative_error_absolute, relative_error_normalised
  R2.5  architecture_search
  R1.4  activity_dst_*, activity_f107_*, stratified_*
  R1.5  weighting_ablation
  R1.7  storm_positioning_*, positioning_tail
  R1.8  oracle_benchmark
  R1.3  madrigal_reference_offset, reference_precision, dstec_absolute_comparison,
        dstec_win_rate
  R1.6  calibration_coverage, calibration_pit, ionex_rms_*
  R2.3  station_independence
  R2.6  uncertainty_vs_error

Style
-----
Uses `stec.viz.style.PLOT_CONFIG`, the same configuration behind the published figures, so
these sit beside them without a visual break. Series colours are `stec.viz.style`'s approach
palette, taken unchanged from `positioning/scripts/plot_results.py`: blue = Direct STEC,
orange = VTEC + Mapping, green = IGS GIM + Mapping, purple = Pretrained Direct STEC. An
approach colour must only ever mean that approach - conditions, datasets and the oracle
bound are drawn from `stec.viz.style.NON_APPROACH_COLORS` instead (see style.py).

The plot area carries no explanatory text - no value labels, correlation figures,
reference-line captions or interpretive notes. All of that belongs in the manuscript
caption and body. Each figure is written twice: a working copy with a title and a
provenance footnote naming the source file and scope, and a `_notitle` copy with neither,
which is the one for the manuscript.

Known limitation, carried over deliberately: orange (#ff7f0e) and green (#2ca02c) are
separated by only dE = 0.7 in OKLab under simulated protanopia, so those two series are
hard to distinguish for red-blind readers. Consistency with the published figures was
chosen over changing the palette - see `stec/viz/style.py`.

Figure table
------------
`FIGURE_BUILDERS` is the entry point's data table: adding a new figure *family* is adding
one callable to that tuple. Within a family, the source already iterates a table rather
than hand-unrolling one call per case - the two activity stratifiers (`_ACTIVITY_STRATA`),
the four axes in `STRATIFIER_AXES`, and the two `stratified_comparison` sources
(`_STRATIFIED_SOURCES`) - and that shape is preserved here, so extending any of those three
is adding a tuple entry, not a function. What is *not* uniform across figures is which and
how many CSVs each one reads (0-3, some optional) and what its provenance sentence says, so
each family keeps its own small loader; a fully uniform "one row = one figure" table across
all ~19 kinds would need to paper over that with per-row lambdas that just re-implement the
per-family logic in another place.

Usage::

    python -m stec.viz.revision_figures --output_dir plots/revision --results_dir multiday_results
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import paths
from .style import (
    APPROACH_COLORS,
    CODE_GIM_COLOR,
    CONDITION_COLORS,
    DATASET_COLORS,
    FIGSIZE_WIDE,
    METHOD_ORDER,
    ORACLE_COLOR,
    configure_plotting,
)

import matplotlib.pyplot as plt  # noqa: E402  (style.py sets the Agg backend on import)

logger = logging.getLogger(__name__)

SOURCE_DIRS = {
    "stec_finetuned": "stec_finetuned_2024",
    "pretrained": "stec_pretrained_testset",
    "finetuned": "stec_finetuned_2024",
    "positioning": "positioning_2024",
    "training": "training_runs",
}


def analysis_dir(results_dir: Path, analysis: str) -> Path:
    """Where `analysis` left its output, preferring the rebuilt run over its predecessor.

    Mirrors `stec.config.paths.analysis_result_dir`, but resolved against a caller-given
    `results_dir` rather than the fixed `paths.RESULTS_ROOT` - the same reason
    `stec.runs.restructure_results.classify` takes its root as a parameter instead of
    importing the destination directly, so this also works against a caller pointing
    `--results_dir` at an alternate tree.

    The ported analyses write to `analyses/<name>/rebuilt/` so a rebuild cannot clobber
    the results the paper was submitted from. Without this the figures would be built
    from the pre-rebuild CSVs while every table came from the rebuilt ones - two
    implementations behind one manuscript, which is the ambiguity this package exists to
    remove.

    The stages that deliberately stay on pre-rebuild scripts (`repair_gim_baseline`,
    `hyperparameter_search`) have no `rebuilt/` sibling, so they resolve to
    `pre_rebuild/`. That fallback is not silent: every figure carries a provenance
    footnote naming the CSV it was drawn from, so which of the two was used is visible in
    the artifact itself.
    """
    base = results_dir / paths.ANALYSES_RESULTS.name / analysis
    rebuilt = base / "rebuilt"
    return rebuilt if rebuilt.is_dir() else base / "pre_rebuild"


def _save(
    fig: plt.Figure,
    name: str,
    source: str,
    output_dir: Path,
    provenance: str,
    data: pd.DataFrame | None = None,
) -> None:
    """Write the working copy (title + provenance), the manuscript copy, and the numbers.

    The CSV holds what the figure actually draws, not the analysis table it came from -
    those often carry extra models, bins or columns that never reach the axes. Writing the
    plotted values means the number a reader checks is the number they see, and the two
    cannot drift apart.
    """
    target = output_dir / SOURCE_DIRS[source]
    target.mkdir(parents=True, exist_ok=True)
    if data is not None:
        data.to_csv(target / f"{name}.csv", index=False)

    # Negative y places the note below the x-axis label; bbox_inches="tight" expands the
    # saved area to include artists outside the figure rectangle.
    footnote = fig.text(
        0.0, -0.04, f"Data: {provenance}", fontsize=11, color="#555555", va="top"
    )
    fig.savefig(target / f"{name}.png", bbox_inches="tight")

    footnote.set_text("")
    for ax in fig.axes:
        # An axes keeps a separate title artist per location, so clearing only the default
        # (centre) one leaves a loc="left" title in place.
        for loc in ("center", "left", "right"):
            ax.set_title("", loc=loc)
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    fig.savefig(target / f"{name}_notitle.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {target / name}.png (+ _notitle)")


def _grouped_bars(
    ax, groups, series, values, colors, ylabel, xlabel=""
) -> pd.DataFrame:
    """Grouped bars in the paper's style: no value labels, y-grid only.

    Returns the plotted values tidied to one row per bar, so the caller can hand them
    straight to `_save`.
    """
    n_series = len(series)
    x = np.arange(len(groups))
    width = 0.8 / n_series
    for i, name in enumerate(series):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(
            x + offset,
            values[name],
            width * 0.94,
            color=colors[i],
            label=name,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    return pd.DataFrame(
        [
            {
                "group": str(g).replace("\n", " "),
                "series": name,
                "value": values[name][i],
            }
            for i, g in enumerate(groups)
            for name in series
        ]
    )


def _method_labels(names) -> list[str]:
    return [
        n.replace(" Direct STEC", "\nDirect STEC").replace(" + ", "\n+ ") for n in names
    ]


# --------------------------------------------------------------------------
# R2.2 - absolute vs TEC-normalised error across the solar cycle
# --------------------------------------------------------------------------


def fig_relative_error_absolute(
    d: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(
        d.year,
        d.mean_STEC,
        marker="o",
        markersize=9,
        color=CONDITION_COLORS["baseline"],
        label="Mean observed STEC",
    )
    ax.plot(
        d.year,
        d.RMSE,
        marker="s",
        markersize=9,
        color=APPROACH_COLORS["Direct STEC"],
        label="Model RMSE",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("STEC [TECU]")
    ax.set_xticks(d.year)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title("Absolute error and mean observed STEC by year")
    _save(
        fig,
        "relative_error_absolute",
        "pretrained",
        output_dir,
        provenance,
        d[["year", "mean_STEC", "RMSE"]],
    )


def fig_relative_error_normalised(
    d: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(
        d.year,
        d["nRMSE_%"],
        marker="s",
        markersize=9,
        color=APPROACH_COLORS["Direct STEC"],
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalised RMSE [%]")
    ax.set_xticks(d.year)
    ax.set_ylim(0, max(45, d["nRMSE_%"].max() * 1.15))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("TEC-normalised error by year")
    _save(
        fig,
        "relative_error_normalised",
        "pretrained",
        output_dir,
        provenance,
        d[["year", "nRMSE_%"]],
    )


def _build_relative_error_figures(args: argparse.Namespace, output_dir: Path) -> None:
    # The one analysis whose port changed both the location and the filename: the
    # predecessor wrote a flat relative_error_metrics.csv, the port writes
    # relative_error_metrics_rebuilt/yearly_metrics.csv. analysis_dir cannot express that,
    # so the two candidates are named here rather than papered over.
    rebuilt = args.results_dir / "relative_error_metrics_rebuilt" / "yearly_metrics.csv"
    path = (
        rebuilt if rebuilt.exists() else args.results_dir / "relative_error_metrics.csv"
    )
    if not path.exists():
        logger.warning(f"{path} not found")
        return
    d = pd.read_csv(path).sort_values("year")
    prov = (
        f"{path} - pretrained model, held-out test set, "
        f"{d.year.min()}-{d.year.max()} ({int(d['count'].sum()):,} observations)"
    )
    fig_relative_error_absolute(d, output_dir, prov)
    fig_relative_error_normalised(d, output_dir, prov)


# --------------------------------------------------------------------------
# R1.7 - positioning under quiet vs storm conditions
# --------------------------------------------------------------------------


def fig_storm_positioning_absolute(
    d: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        _method_labels(d.index),
        ["quiet", "storm"],
        {"quiet": d["quiet"].values, "storm": d["storm"].values},
        [CONDITION_COLORS["baseline"], CONDITION_COLORS["contrast"]],
        "3D RMS positioning error [m]",
    )
    ax.legend(title="Geomagnetic conditions")
    ax.set_title("Positioning error by geomagnetic regime")
    _save(
        fig,
        "storm_positioning_absolute",
        "positioning",
        output_dir,
        provenance,
        plotted,
    )


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

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        _method_labels(improvement.index),
        ["quiet", "storm"],
        {"quiet": improvement["quiet"].values, "storm": improvement["storm"].values},
        [CONDITION_COLORS["baseline"], CONDITION_COLORS["contrast"]],
        "Improvement over IGS GIM + Mapping [%]",
    )
    ax.axhline(0, color="black", linewidth=1.2, zorder=4)
    ax.legend(title="Geomagnetic conditions", loc="lower left")
    ax.set_title("Margin over the operational baseline by geomagnetic regime")
    _save(
        fig,
        "storm_positioning_improvement",
        "positioning",
        output_dir,
        provenance,
        plotted,
    )


def _build_storm_positioning_figures(
    args: argparse.Namespace, output_dir: Path
) -> None:
    path = analysis_dir(args.results_dir, "storm_stratification") / "degradation.csv"
    if not path.exists():
        logger.warning(f"{path} not found")
        return
    d = pd.read_csv(path, index_col=0).reindex(METHOD_ORDER)
    prov = (
        f"{path} - SF-PPP, 2024 test period, 39 storm days (daily min Dst <= -50 nT) "
        "of 242, station-days <= 10 m"
    )
    fig_storm_positioning_absolute(d, output_dir, prov)
    fig_storm_positioning_improvement(d, output_dir, prov)


# --------------------------------------------------------------------------
# R1.5, R1.8, R2.5 - single-CSV figures
# --------------------------------------------------------------------------


def fig_weighting_ablation(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        [m.replace(" + ", "\n+ ") for m in d.index],
        # Two bars only: elevation weighting is the operational default, so it is the
        # comparison a reader needs. The fixed-variance arm answers a different,
        # mechanistic question and is reported as a number in
        # multiday_results/analyses/weighting_ablation/rebuilt/fixed_variance.csv rather
        # than as a bar for a scheme nobody actually uses.
        ["Elevation weighting", "Predicted-uncertainty weighting"],
        {
            "Elevation weighting": d["elev_mean"].values,
            "Predicted-uncertainty weighting": d["iono_mean"].values,
        },
        [CONDITION_COLORS["baseline"], CONDITION_COLORS["contrast"]],
        "3D RMS positioning error [m]",
    )
    ax.legend(loc="upper left")
    ax.set_title("Observation weighting scheme")
    _save(fig, "weighting_ablation", "positioning", output_dir, provenance, plotted)


def _build_weighting_ablation_figure(
    args: argparse.Namespace, output_dir: Path
) -> None:
    path = analysis_dir(args.results_dir, "weighting_ablation") / "paired.csv"
    if not path.exists():
        logger.warning(f"{path} not found")
        return
    d = pd.read_csv(path, index_col=0)
    prov = (
        f"{path} - SF-PPP, 2024 test period, paired station-days solved under both "
        f"weightings ({int(d['paired_station_days'].sum()):,} pairs)"
    )
    fig_weighting_ablation(d, output_dir, prov)


def fig_oracle_benchmark(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    order = [m for m in ("Reference STEC (oracle)", *METHOD_ORDER) if m in d.index]
    d = d.loc[order]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colors = [
        ORACLE_COLOR if m.startswith("Reference STEC") else APPROACH_COLORS[m]
        for m in d.index
    ]
    ax.bar(np.arange(len(d)), d["mean"], 0.62, color=colors, zorder=3)
    ax.axhline(
        d["mean"].iloc[0], color=ORACLE_COLOR, linewidth=1.5, linestyle="--", zorder=4
    )
    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels(
        [m.replace(" (oracle)", "\n(oracle)").replace(" + ", "\n+ ") for m in d.index]
    )
    ax.set_ylabel("3D RMS positioning error [m]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("Positioning against the observation-derived bound")
    _save(
        fig,
        "oracle_benchmark",
        "positioning",
        output_dir,
        provenance,
        d.reset_index()[
            [
                c
                for c in (
                    d.index.name or "index",
                    "mean",
                    "median",
                    "p95",
                    "station_days",
                    "above_oracle_m",
                    "ratio_to_oracle",
                )
                if c in d.reset_index().columns
            ]
        ],
    )


def _build_oracle_benchmark_figure(args: argparse.Namespace, output_dir: Path) -> None:
    path = analysis_dir(args.results_dir, "oracle_benchmark") / "summary.csv"
    if not path.exists():
        logger.warning(f"{path} not found - run src/analysis/oracle_benchmark.py")
        return
    d = pd.read_csv(path, index_col=0)
    prov = (
        f"{path} - SF-PPP, elevation weighting throughout, "
        f"{int(d['station_days'].max())} station-days solved by every method"
    )
    fig_oracle_benchmark(d, output_dir, prov)


def fig_architecture_search(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    d = d.sort_values("best_val_MAE")
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    y = np.arange(len(d))
    colors = [
        APPROACH_COLORS["Direct STEC"] if c > 0 else CONDITION_COLORS["baseline"]
        for c in d["credible_runs"]
    ]
    ax.barh(y, d["best_val_MAE"], 0.62, color=colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d.index)
    ax.invert_yaxis()
    ax.set_xlabel("Best validation MAE [TECU]")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("Architecture comparison")
    _save(
        fig,
        "architecture_search",
        "training",
        output_dir,
        provenance,
        d.reset_index()[
            [
                c
                for c in (
                    d.index.name or "index",
                    "best_val_MAE",
                    "runs",
                    "credible_runs",
                )
                if c in d.reset_index().columns
            ]
        ],
    )


def _build_architecture_search_figure(
    args: argparse.Namespace, output_dir: Path
) -> None:
    path = analysis_dir(args.results_dir, "hyperparameter_search") / "architectures.csv"
    if not path.exists():
        logger.warning(f"{path} not found")
        return
    d = pd.read_csv(path, index_col=0)
    prov = (
        f"{path} - local W&B history, {int(d['runs'].sum())} STEC runs "
        "reporting a validation MAE"
    )
    fig_architecture_search(d, output_dir, prov)


# --------------------------------------------------------------------------
# R1.4 - STEC error against geomagnetic activity and solar flux
# --------------------------------------------------------------------------

# (stem, filename, bin column, axis label) - the table the source already iterated.
_ACTIVITY_STRATA = (
    ("activity_dst", "by_dst.csv", "dst_bin", "Daily minimum Dst"),
    ("activity_f107", "by_f107.csv", "f107_bin", "Daily mean F10.7"),
)


def _activity_figures(
    table: pd.DataFrame,
    bin_col: str,
    axis_label: str,
    stem: str,
    output_dir: Path,
    provenance: str,
) -> None:
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
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        bins,
        series,
        values,
        [APPROACH_COLORS[m] for m in series],
        "STEC RMSE [TECU]",
        axis_label,
    )
    ax.legend(ncol=2)
    ax.set_title(f"STEC error by {axis_label.lower()}")
    _save(fig, f"{stem}_absolute", "finetuned", output_dir, provenance, plotted)

    rel_series = [m for m in series if m != "IGS GIM + Mapping"]
    rel_values = {
        m: [
            d[(d.Model == m) & (d[bin_col] == b)]["improvement_over_gim_%"].iloc[0]
            for b in bins
        ]
        for m in rel_series
    }
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        bins,
        rel_series,
        rel_values,
        [APPROACH_COLORS[m] for m in rel_series],
        "Improvement over IGS GIM [%]",
        axis_label,
    )
    ax.axhline(0, color="black", linewidth=1.2, zorder=4)
    ax.legend(loc="lower left")
    ax.set_title(f"Margin over IGS GIM by {axis_label.lower()}")
    _save(fig, f"{stem}_improvement", "finetuned", output_dir, provenance, plotted)


def _build_activity_figures(args: argparse.Namespace, output_dir: Path) -> None:
    activity_dir = analysis_dir(args.results_dir, "activity_stratification")
    for stem, filename, bin_col, axis_label in _ACTIVITY_STRATA:
        path = activity_dir / filename
        if not path.exists():
            logger.warning(
                f"{path} not found - run src/analysis/activity_stratification.py"
            )
            continue
        table = pd.read_csv(path)
        days = int(table[table.Model == "Direct STEC Model"]["days"].sum())
        obs = int(table[table.Model == "Direct STEC Model"]["observations"].sum())
        prov = (
            f"{path} - daily fine-tuned models, own test set, "
            f"{days} test days of 2024 ({obs:,} observations)"
        )
        _activity_figures(table, bin_col, axis_label, stem, output_dir, prov)


# --------------------------------------------------------------------------
# R1.4 - all four methods across every stratifier, both model families
# --------------------------------------------------------------------------

STRATIFIER_AXES = {
    "elevation": "Satellite elevation [°]",
    "geomagnetic_latitude": "Geomagnetic latitude of the pierce point [°]",
    "local_time": "Local time [h]",
    "season": "Season",
}

# (results subdir, filename suffix, provenance description) - the table the source already
# iterated over both the fine-tuned and the pretrained stratified_comparison output.
_STRATIFIED_SOURCES = (
    ("stratified_comparison", "", "daily fine-tuned models, own test set"),
    (
        "stratified_comparison_pretrained",
        "_pretrained",
        "pretrained model, multi-year held-out test set",
    ),
)


def _interval_label(text: str) -> tuple[float, str]:
    """Turn a pandas interval string into a sort key and a readable label.

    Ranges spanning negative values cannot use a bare dash - "-90--60" is unreadable - so a
    negative endpoint switches the separator to "to"; the unit is dropped since the axis
    label already carries it.
    """
    if not text.startswith("("):
        return (0.0, text)
    left, right = text.strip("()[]").split(", ")
    lo, hi = float(left), float(right)
    # pandas pads the lowest edge (4.999, -90.001); show the intended value.
    lo = round(lo) if abs(lo - round(lo)) < 0.01 else lo

    def minus(value: float) -> str:
        return f"{value:g}".replace("-", "−")

    separator = " to " if lo < 0 or hi < 0 else "–"
    return (lo, f"{minus(lo)}{separator}{minus(hi)}")


def _stratified_figures(
    table: pd.DataFrame, name: str, output_dir: Path, provenance: str
) -> None:
    keys = {b: _interval_label(b) for b in table["bin"].unique()}
    if name == "season":
        # Source quirk, preserved as-is: this only matches the fine-tuned-model source's
        # bare "season" name, not "season_pretrained" - so the pretrained stratified figure
        # falls through to the generic (unordered, unlabelled-day-count) bin handling below.
        # Flagged in the port report rather than silently widened to startswith("season").
        # Order by the calendar, and carry the day count: the test period is May-December,
        # so winter is ten December days and must not read as a quarter of the year.
        order = ["spring", "summer", "autumn", "winter"]
        days = table.drop_duplicates("bin").set_index("bin")["days"]
        keys = {
            b: (
                order.index(b.strip()) if b.strip() in order else 99,
                f"{b.strip()}\n({int(days[b])} d)",
            )
            for b in table["bin"].unique()
        }
    bins = sorted(keys, key=lambda b: keys[b][0])
    labels = [keys[b][1] for b in bins]
    series = [m for m in METHOD_ORDER if m in set(table["Method"])]
    axis_label = STRATIFIER_AXES[name.removesuffix("_pretrained")]

    def values(column: str) -> dict[str, list[float]]:
        return {
            m: [
                table[(table.Method == m) & (table.bin == b)][column].iloc[0]
                for b in bins
            ]
            for m in series
        }

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        labels,
        series,
        values("RMSE"),
        [APPROACH_COLORS[m] for m in series],
        "STEC RMSE [TECU]",
        axis_label,
    )
    ax.legend(ncol=2)
    ax.set_title(f"STEC error by {axis_label.split(' [')[0].lower()}")
    _save(
        fig, f"stratified_{name}_absolute", "finetuned", output_dir, provenance, plotted
    )

    relative = [m for m in series if m != "IGS GIM + Mapping"]
    # Without the GIM baseline - the pretrained model's own multi-year test set has none -
    # every margin is NaN and the panel would be blank. Skip it rather than ship an empty
    # figure that looks like a result.
    if not table["improvement_over_gim_pct"].notna().any() or not relative:
        logger.info(f"no baseline for {name}; margin panel skipped")
        return
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        labels,
        relative,
        values("improvement_over_gim_pct"),
        [APPROACH_COLORS[m] for m in relative],
        "Improvement over IGS GIM [%]",
        axis_label,
    )
    ax.axhline(0, color="black", linewidth=1.2, zorder=4)
    ax.legend(loc="lower left")
    ax.set_title(f"Margin over IGS GIM by {axis_label.split(' [')[0].lower()}")
    _save(
        fig,
        f"stratified_{name}_improvement",
        "finetuned",
        output_dir,
        provenance,
        plotted,
    )


def _build_stratified_figures(args: argparse.Namespace, output_dir: Path) -> None:
    for subdir, suffix, description in _STRATIFIED_SOURCES:
        source_dir = args.results_dir / subdir
        for name in STRATIFIER_AXES:
            path = source_dir / f"by_{name}.csv"
            if not path.exists():
                if not suffix:
                    logger.warning(f"{path} not found - run stratified_comparison.py")
                continue
            table = pd.read_csv(path)
            lead = table[table.Method == table.Method.iloc[0]]
            prov = (
                f"{path} - {description}, {int(lead['days'].max())} days "
                f"({int(lead['observations'].sum()):,} observations)"
            )
            _stratified_figures(table, f"{name}{suffix}", output_dir, prov)


# --------------------------------------------------------------------------
# R2.6 - predicted uncertainty against realised error
# --------------------------------------------------------------------------


def fig_uncertainty_vs_error(
    d: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Mean predicted sigma against realised RMSE, per predicted-sigma decile."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    limit = float(max(d.RMSE.max(), d.mean_sigma.max())) * 1.05
    ax.plot(
        [0, limit],
        [0, limit],
        linestyle="--",
        color=CONDITION_COLORS["baseline"],
        linewidth=1.5,
        label="Perfect calibration",
        zorder=2,
    )
    ax.plot(
        d.mean_sigma,
        d.RMSE,
        marker="o",
        markersize=9,
        color=APPROACH_COLORS["Direct STEC"],
        label="Direct STEC, by predicted-σ decile",
        zorder=3,
    )
    ax.set_xlabel("Mean predicted σ [TECU]")
    ax.set_ylabel("Realised RMSE [TECU]")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title("Predicted uncertainty against realised error")
    _save(
        fig,
        "uncertainty_vs_error",
        "finetuned",
        output_dir,
        provenance,
        d[["bin", "n", "mean_sigma", "RMSE", "rmse_over_sigma"]],
    )


def _build_uncertainty_vs_error_figure(
    args: argparse.Namespace, output_dir: Path
) -> None:
    path = analysis_dir(args.results_dir, "uncertainty_error_relation") / "by_sigma.csv"
    if not path.exists():
        logger.warning(
            f"{path} not found - run src/analysis/uncertainty_error_relation.py"
        )
        return
    d = pd.read_csv(path)
    prov = f"{path} - daily fine-tuned models, own test set ({int(d['n'].sum()):,} observations)"
    fig_uncertainty_vs_error(d, output_dir, prov)


# --------------------------------------------------------------------------
# IONEX RMS benchmark - our uncertainty against the GIM products' own
# --------------------------------------------------------------------------

_IONEX_COLORS = {
    "Direct STEC": APPROACH_COLORS["Direct STEC"],
    "VTEC + Mapping": APPROACH_COLORS["VTEC + Mapping"],
    "IGS GIM + Mapping": APPROACH_COLORS["IGS GIM + Mapping"],
    "CODE GIM + Mapping": CODE_GIM_COLOR,
}


def fig_ionex_coverage(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """Empirical against nominal coverage; the diagonal is perfect calibration."""
    levels = [50, 68, 90, 95]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    span = [levels[0] - 5, 100]
    ax.plot(
        span,
        span,
        linestyle="--",
        color=CONDITION_COLORS["baseline"],
        linewidth=1.5,
        label="Perfect calibration",
        zorder=2,
    )
    ax.set_xlim(*span)
    for product in [p for p in d.index if p in _IONEX_COLORS]:
        ax.plot(
            levels,
            [100 * d.loc[product, f"cov_{lv}"] for lv in levels],
            marker="o",
            markersize=9,
            color=_IONEX_COLORS[product],
            label=product,
            zorder=3,
        )
    ax.set_xlabel("Nominal coverage [%]")
    ax.set_ylabel("Empirical coverage [%]")
    ax.set_xticks(levels)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title("Interval coverage of each product's own uncertainty")
    _save(
        fig,
        "ionex_rms_coverage",
        "finetuned",
        output_dir,
        provenance,
        pd.DataFrame(
            [
                {
                    "product": p,
                    "nominal_%": lv,
                    "empirical_%": 100 * d.loc[p, f"cov_{lv}"],
                    "days": d.loc[p, "days"],
                }
                for p in d.index
                if p in _IONEX_COLORS
                for lv in levels
            ]
        ),
    )


def fig_ionex_crps_skill(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """CRPS skill against each product's own constant-sigma reference.

    Positive means the per-observation uncertainty beats a single constant for that same
    set of predictions; negative means it is worse than no uncertainty at all.
    """
    bins = ["5-20", "20-40", "40-60", "60-90"]
    products = [p for p in _IONEX_COLORS if p in d.index.get_level_values(0)]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        bins,
        products,
        {p: [100 * d.loc[(p, b), "CRPS_skill"] for b in bins] for p in products},
        [_IONEX_COLORS[p] for p in products],
        "CRPS skill over constant σ [%]",
        xlabel="Satellite elevation [°]",
    )
    ax.axhline(0, color="black", linewidth=1.0, zorder=4)
    ax.legend(loc="upper right")
    ax.set_title("Value of the per-observation uncertainty, by elevation")
    _save(fig, "ionex_rms_crps_skill", "finetuned", output_dir, provenance, plotted)


def _build_ionex_figures(args: argparse.Namespace, output_dir: Path) -> None:
    benchmark_dir = analysis_dir(args.results_dir, "ionex_rms_benchmark")
    overall = benchmark_dir / "overall_IGS.csv"
    if not overall.exists():
        logger.warning(f"{overall} not found - run src/analysis/ionex_rms_benchmark.py")
        return
    igs = pd.read_csv(overall, index_col=0)
    code_path = benchmark_dir / "overall_CODE.csv"
    if code_path.exists():
        code = pd.read_csv(code_path, index_col=0)
        igs = pd.concat([igs, code.loc[[i for i in code.index if i not in igs.index]]])
    prov = (
        f"{benchmark_dir}/overall_*.csv - daily fine-tuned models, own test set, "
        f"{int(igs['days'].max())} test days of 2024 ({int(igs['observations'].max()):,} "
        "observations)"
    )
    fig_ionex_coverage(igs, output_dir, prov)

    by_elev = [
        pd.read_csv(benchmark_dir / f"by_elevation_{g}.csv", index_col=[0, 1])
        for g in ("IGS", "CODE")
        if (benchmark_dir / f"by_elevation_{g}.csv").exists()
    ]
    if by_elev:
        merged = pd.concat(by_elev)
        merged = merged[~merged.index.duplicated()]
        fig_ionex_crps_skill(merged, output_dir, prov)


# --------------------------------------------------------------------------
# R1.3 - per-station offsets against Madrigal, and against its own precision
# --------------------------------------------------------------------------


def fig_madrigal_reference_offset(
    offsets: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Two unrelated estimates disagree with Madrigal the same way.

    Each point is a station. Agreement along the 1:1 line means the discrepancy is a
    property of the Madrigal reference, since the model and the GIM share nothing in how
    they are produced.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.scatter(
        offsets["offset_gim"],
        offsets["offset_model"],
        s=90,
        color=CONDITION_COLORS["contrast"],
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    lo = float(min(offsets["offset_gim"].min(), offsets["offset_model"].min())) - 2
    hi = float(max(offsets["offset_gim"].max(), offsets["offset_model"].max())) + 2
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.5, zorder=2)
    ax.axhline(0, color=CONDITION_COLORS["baseline"], linewidth=1.0, zorder=1)
    ax.axvline(0, color=CONDITION_COLORS["baseline"], linewidth=1.0, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("IGS GIM − Madrigal, per station [TECU]")
    ax.set_ylabel("Direct STEC − Madrigal, per station [TECU]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("Per-station disagreement with Madrigal")
    _save(
        fig,
        "madrigal_reference_offset",
        "stec_finetuned",
        output_dir,
        provenance,
        offsets.reset_index()[
            ["station", "observations", "offset_gim", "offset_model"]
        ],
    )


def fig_reference_precision(
    offsets: pd.DataFrame, precision: pd.Series, output_dir: Path, provenance: str
) -> None:
    """The offsets dwarf the reference's own stated precision.

    One point per station, absolute disagreement with Madrigal, on a log axis so the two
    scales fit together. The band is the reference product's own claimed slant precision
    (median to p90). Every station sits one to two orders of magnitude to the right of it,
    for two products that share nothing in their construction - which is what makes this an
    inter-product bias rather than reference noise or model error.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    rows = [
        (
            "Direct STEC − Madrigal",
            offsets["offset_model"].abs(),
            APPROACH_COLORS["Direct STEC"],
        ),
        (
            "IGS GIM − Madrigal",
            offsets["offset_gim"].abs(),
            APPROACH_COLORS["IGS GIM + Mapping"],
        ),
    ]
    ax.axvspan(
        precision["slant_stddev_median_TECU"],
        precision["slant_stddev_p90_TECU"],
        color=CONDITION_COLORS["baseline"],
        alpha=0.35,
        zorder=1,
    )
    ax.axvline(
        precision["slant_stddev_median_TECU"],
        color=CONDITION_COLORS["baseline"],
        linewidth=2.0,
        label="Reference product's own stated precision",
        zorder=2,
    )
    rng = np.linspace(-0.16, 0.16, len(offsets))
    for i, (label, values, colour) in enumerate(rows):
        ax.scatter(
            values,
            np.full(len(values), i) + rng,
            s=70,
            color=colour,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.6,
            zorder=3,
            label=label,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0].replace(" − ", "\n− ") for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Absolute per-station disagreement with Madrigal [TECU]")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    ax.set_title("Per-station offsets against the reference's own precision")
    _save(
        fig,
        "reference_precision",
        "stec_finetuned",
        output_dir,
        provenance,
        pd.concat(
            [
                offsets.reset_index()[["station"]].assign(
                    product="Direct STEC − Madrigal",
                    abs_offset_TECU=offsets["offset_model"].abs().values,
                ),
                offsets.reset_index()[["station"]].assign(
                    product="IGS GIM − Madrigal",
                    abs_offset_TECU=offsets["offset_gim"].abs().values,
                ),
            ],
            ignore_index=True,
        ).assign(
            reference_slant_precision_median_TECU=precision["slant_stddev_median_TECU"],
            reference_slant_precision_p90_TECU=precision["slant_stddev_p90_TECU"],
        ),
    )


def _build_madrigal_offset_figures(args: argparse.Namespace, output_dir: Path) -> None:
    offset_dir = analysis_dir(args.results_dir, "madrigal_reference_offset")
    offsets_path = offset_dir / "per_station_offsets.csv"
    if not offsets_path.exists():
        logger.warning(f"{offsets_path} not found - run madrigal_reference_offset.py")
        return
    offsets = pd.read_csv(offsets_path, index_col=0)
    prov = (
        f"{offsets_path} - daily fine-tuned models on Madrigal geometries, "
        f"{len(offsets)} stations, {int(offsets['observations'].sum()):,} observations"
    )
    fig_madrigal_reference_offset(offsets, output_dir, prov)

    precision_path = offset_dir / "reference_precision.csv"
    if not precision_path.exists():
        logger.warning(f"{precision_path} not found - skipping precision figure")
        return
    precision = pd.read_csv(precision_path, index_col=0)["value"]
    fig_reference_precision(offsets, precision, output_dir, prov)


# --------------------------------------------------------------------------
# R1.3 - dSTEC: does the model get a pass's shape right independent of any per-arc
# offset a different processing chain (Madrigal, a receiver/satellite DCB) would
# calibrate differently? See stec.analysis.dstec_evaluation's module docstring for why
# this is the direct rebuttal to the "Madrigal comparison confounds OOD and reference
# chain" criticism: dSTEC cancels a per-arc constant by construction.
#
# The predecessor script (positioning/scripts/evaluate_dstec.py, deleted as superseded)
# additionally produced per-arc scatter plots, error histograms, density hexbins, a
# Q-Q plot and a heteroscedasticity view. None of that is reproduced here: the
# reviewer's actual concern is narrow - is dSTEC an easier metric than absolute STEC,
# so that a good dSTEC number overstates the model? Two figures answer that directly
# and a bigger gallery would not: the pooled-RMSE margin and the per-arc win rate,
# each computed for dSTEC and absolute STEC side by side on the identical masked
# observations. If dSTEC's margin over IGS GIM were an artefact of the metric, it
# would not survive being placed next to the absolute-STEC margin on the same axes.
# --------------------------------------------------------------------------


def fig_dstec_absolute_comparison(
    summary: pd.Series, output_dir: Path, provenance: str
) -> None:
    """Pooled RMSE, dSTEC and absolute STEC, model against IGS GIM, side by side.

    Both bars share the identical masked observations (dstec_evaluation.compute_arc_dstec
    computes both from the same per-arc mask): if the model's dSTEC advantage were an
    artefact of dSTEC being an easier metric, the absolute-STEC bars would not show the
    same ordering.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        ["dSTEC", "Absolute STEC"],
        ["Direct STEC", "IGS GIM + Mapping"],
        {
            "Direct STEC": [
                summary["model_dstec_rmse_pooled"],
                summary["model_abs_rmse_pooled"],
            ],
            "IGS GIM + Mapping": [
                summary["gim_dstec_rmse_pooled"],
                summary["gim_abs_rmse_pooled"],
            ],
        },
        [APPROACH_COLORS["Direct STEC"], APPROACH_COLORS["IGS GIM + Mapping"]],
        "RMSE [TECU]",
    )
    ax.legend()
    ax.set_title("dSTEC removes the common-mode offset; absolute STEC keeps it")
    _save(
        fig, "dstec_absolute_comparison", "finetuned", output_dir, provenance, plotted
    )


def fig_dstec_win_rate(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """Share of arcs where Direct STEC beats IGS GIM, dSTEC and absolute STEC.

    A pooled RMSE can be dominated by a handful of extreme arcs; the per-arc win rate
    is a second, independent read on the same question. The two win rates tracking
    each other closely (73.4% dSTEC / 73.0% absolute STEC in the current 18-day run)
    is itself evidence against "dSTEC is just an easier metric."
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    # Return value unused: unlike every other _grouped_bars call, the CSV worth writing
    # here is `d` itself (it also carries n_arcs), not the tidied group/series/value
    # frame _grouped_bars would hand back.
    _grouped_bars(
        ax,
        d["metric"].tolist(),
        ["Direct STEC"],
        {"Direct STEC": d["win_rate_pct"].to_numpy()},
        [APPROACH_COLORS["Direct STEC"]],
        "Arcs where Direct STEC beats IGS GIM [%]",
    )
    ax.axhline(
        50,
        color=CONDITION_COLORS["baseline"],
        linestyle="--",
        linewidth=1.5,
        zorder=4,
        label="Chance (50%)",
    )
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    ax.set_title("Per-arc win rate against IGS GIM")
    _save(fig, "dstec_win_rate", "finetuned", output_dir, provenance, d)


def _build_dstec_evaluation_figures(args: argparse.Namespace, output_dir: Path) -> None:
    dstec_dir = analysis_dir(args.results_dir, "dstec_evaluation")
    summary_path = dstec_dir / "summary.csv"
    arcs_path = dstec_dir / "pass_statistics.csv"
    if not summary_path.exists() or not arcs_path.exists():
        logger.warning(f"{summary_path} not found - run stec.analysis.dstec_evaluation")
        return

    summary = pd.read_csv(summary_path, index_col=0)["value"]
    if "gim_dstec_rmse_pooled" not in summary.index:
        # dstec_evaluation.summarise only adds the gim_* keys when at least one arc had
        # a usable gim_stec value (see its docstring) - without them there is no
        # comparison to draw, only a one-sided model number.
        logger.info("dstec_evaluation summary has no GIM columns; skipping")
        return

    arcs = pd.read_csv(arcs_path)
    n_days = int(summary["n_days"])
    n_arcs = int(summary["n_arcs"])
    n_obs = int(summary["n_masked_obs"])
    prov = (
        f"{dstec_dir}/{{summary,pass_statistics}}.csv - daily fine-tuned models, own "
        f"test set, {n_days} days ({n_arcs:,} arcs, {n_obs:,} masked observations)"
    )
    fig_dstec_absolute_comparison(summary, output_dir, prov)

    valid = arcs[arcs["gim_dstec_rmse"].notna() & arcs["gim_abs_rmse"].notna()]
    if valid.empty:
        logger.info("no arcs with a valid GIM value; skipping the win-rate figure")
        return
    win_rates = pd.DataFrame(
        {
            "metric": ["dSTEC", "Absolute STEC"],
            "win_rate_pct": [
                100 * (valid["model_dstec_rmse"] < valid["gim_dstec_rmse"]).mean(),
                100 * (valid["model_abs_rmse"] < valid["gim_abs_rmse"]).mean(),
            ],
            "n_arcs": [len(valid), len(valid)],
        }
    )
    fig_dstec_win_rate(win_rates, output_dir, prov)


# --------------------------------------------------------------------------
# R1.6 - calibration: coverage and PIT
# --------------------------------------------------------------------------


def fig_calibration_coverage(
    own: pd.DataFrame,
    madrigal: pd.DataFrame | None,
    corrected: pd.DataFrame | None,
    output_dir: Path,
    provenance: str,
) -> None:
    """Reliability: nominal against empirical interval coverage."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.5, zorder=2)
    ax.plot(
        own["nominal"],
        own["empirical"],
        marker="o",
        markersize=10,
        color=DATASET_COLORS["own"],
        label="Own test set",
        zorder=3,
    )
    if madrigal is not None:
        ax.plot(
            madrigal["nominal"],
            madrigal["empirical"],
            marker="s",
            markersize=10,
            color=DATASET_COLORS["madrigal"],
            label="Madrigal",
            zorder=3,
        )
    if corrected is not None:
        ax.plot(
            corrected["nominal"],
            corrected["empirical_offset_removed"],
            marker="^",
            markersize=10,
            color=DATASET_COLORS["madrigal_corrected"],
            label="Madrigal, station offset removed",
            zorder=3,
        )
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    ax.set_title("Interval coverage reliability")
    series = [own[["nominal", "empirical"]].assign(series="own test set")]
    if madrigal is not None:
        series.append(madrigal[["nominal", "empirical"]].assign(series="Madrigal"))
    if corrected is not None:
        series.append(
            corrected[["nominal", "empirical_offset_removed"]]
            .rename(columns={"empirical_offset_removed": "empirical"})
            .assign(series="Madrigal, station offset removed")
        )
    _save(
        fig,
        "calibration_coverage",
        "stec_finetuned",
        output_dir,
        provenance,
        pd.concat(series, ignore_index=True),
    )


def fig_calibration_pit(
    own: pd.DataFrame, madrigal: pd.DataFrame | None, output_dir: Path, provenance: str
) -> None:
    """PIT histogram; uniform under calibration."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    centres = 0.5 * (own["bin_left"] + own["bin_right"])
    width = float(own["bin_right"].iloc[0] - own["bin_left"].iloc[0])
    ax.bar(
        centres,
        own["density"],
        width * 0.94,
        color=DATASET_COLORS["own"],
        label="Own test set",
        zorder=3,
    )
    if madrigal is not None:
        ax.step(
            centres,
            madrigal["density"],
            where="mid",
            linewidth=2.5,
            color=DATASET_COLORS["madrigal"],
            label="Madrigal",
            zorder=4,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, zorder=5)
    ax.set_xlabel("Probability integral transform")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title("PIT histogram")
    _save(
        fig,
        "calibration_pit",
        "stec_finetuned",
        output_dir,
        provenance,
        pd.DataFrame(
            {
                "pit_centre": centres,
                "density_own": own["density"].to_numpy(),
                **(
                    {"density_madrigal": madrigal["density"].to_numpy()}
                    if madrigal is not None
                    else {}
                ),
            }
        ),
    )


def _build_calibration_figures(args: argparse.Namespace, output_dir: Path) -> None:
    calibration_dir = analysis_dir(args.results_dir, "uncertainty_calibration")
    own_cov = calibration_dir / "finetuned_stec_own" / "coverage_all.csv"
    if not own_cov.exists():
        logger.warning(f"{own_cov} not found - run uncertainty_calibration.py")
        return
    mad_cov = calibration_dir / "finetuned_stec_madrigal" / "coverage_all.csv"
    own_pit = calibration_dir / "finetuned_stec_own" / "pit_all.csv"
    mad_pit = calibration_dir / "finetuned_stec_madrigal" / "pit_all.csv"
    coverage_path = (
        analysis_dir(args.results_dir, "madrigal_reference_offset")
        / "coverage_before_after.csv"
    )

    prov = f"{calibration_dir} - daily fine-tuned models, prediction store"
    fig_calibration_coverage(
        pd.read_csv(own_cov),
        pd.read_csv(mad_cov) if mad_cov.exists() else None,
        pd.read_csv(coverage_path) if coverage_path.exists() else None,
        output_dir,
        prov,
    )
    if own_pit.exists():
        fig_calibration_pit(
            pd.read_csv(own_pit),
            pd.read_csv(mad_pit) if mad_pit.exists() else None,
            output_dir,
            prov,
        )


# --------------------------------------------------------------------------
# R2.3 - station independence
# --------------------------------------------------------------------------


def fig_station_independence(
    per_station: pd.DataFrame, binned: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Does error grow with distance from the nearest training station?"""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.scatter(
        per_station["distance_km"],
        per_station["nRMSE_%"],
        s=70,
        color=CONDITION_COLORS["baseline"],
        alpha=0.65,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
    )
    ax.plot(
        binned["median_distance_km"],
        binned["nRMSE_pct"],
        marker="o",
        markersize=11,
        linewidth=2.5,
        color=CONDITION_COLORS["contrast"],
        zorder=4,
        label="Distance-bin mean",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Distance to nearest training station [km]")
    ax.set_ylabel("Normalised RMSE [%]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title("Test-station error against separation from the training network")
    _save(
        fig,
        "station_independence",
        "stec_finetuned",
        output_dir,
        provenance,
        pd.concat(
            [
                per_station.reset_index()[["station", "distance_km", "nRMSE_%"]].assign(
                    series="station"
                ),
                binned.rename(
                    columns={
                        "median_distance_km": "distance_km",
                        "nRMSE_pct": "nRMSE_%",
                    }
                )[["distance_km", "nRMSE_%"]].assign(series="distance-bin mean"),
            ],
            ignore_index=True,
        ),
    )


def _build_station_independence_figure(
    args: argparse.Namespace, output_dir: Path
) -> None:
    station_dir = analysis_dir(args.results_dir, "station_independence")
    per_station = station_dir / "per_station.csv"
    binned = station_dir / "by_distance_bin.csv"
    if not (per_station.exists() and binned.exists()):
        logger.warning(f"{per_station} not found - run station_independence.py")
        return
    d = pd.read_csv(per_station)
    prov = (
        f"{per_station} - daily fine-tuned models, own test set, {len(d)} test stations"
    )
    fig_station_independence(d, pd.read_csv(binned), output_dir, prov)


# --------------------------------------------------------------------------
# R1.7 - tail of the positioning error distribution
# --------------------------------------------------------------------------


def fig_positioning_tail(
    tails: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Tail behaviour, not just the mean."""
    quantiles = ["median", "p90", "p95", "p99"]
    order = [m for m in METHOD_ORDER if m in tails.index]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(quantiles))
    for i, method in enumerate(order):
        offset = (i - (len(order) - 1) / 2) * (0.8 / len(order))
        ax.bar(
            x + offset,
            tails.loc[method, quantiles].values,
            0.8 / len(order) * 0.94,
            color=APPROACH_COLORS[method],
            label=method,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Median", "90th", "95th", "99th"])
    ax.set_xlabel("Percentile of the daily 3D RMS across station-days")
    ax.set_ylabel("3D RMS positioning error [m]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(ncol=2)
    ax.set_title("Tail of the positioning error distribution")
    _save(
        fig,
        "positioning_tail",
        "positioning",
        output_dir,
        provenance,
        tails.loc[order, quantiles]
        .reset_index()
        .melt(
            id_vars=tails.index.name or "index",
            var_name="percentile",
            value_name="error_3d_rms_m",
        ),
    )


def _build_positioning_tail_figure(args: argparse.Namespace, output_dir: Path) -> None:
    path = (
        analysis_dir(args.results_dir, "positioning_robustness")
        / "tail_distribution.csv"
    )
    if not path.exists():
        logger.warning(f"{path} not found - run positioning_robustness.py")
        return
    d = pd.read_csv(path, index_col=0)
    prov = (
        f"{path} - SF-PPP, 2024 test period, "
        f"{int(d['station_days'].max()):,} station-days per method"
    )
    fig_positioning_tail(d, output_dir, prov)


# --------------------------------------------------------------------------
# Entry point - see the "Figure table" note in the module docstring.
# --------------------------------------------------------------------------

FIGURE_BUILDERS: tuple[Callable[[argparse.Namespace, Path], None], ...] = (
    _build_relative_error_figures,
    _build_storm_positioning_figures,
    _build_weighting_ablation_figure,
    _build_oracle_benchmark_figure,
    _build_architecture_search_figure,
    _build_activity_figures,
    _build_stratified_figures,
    _build_uncertainty_vs_error_figure,
    _build_ionex_figures,
    _build_madrigal_offset_figures,
    _build_dstec_evaluation_figures,
    _build_calibration_figures,
    _build_station_independence_figure,
    _build_positioning_tail_figure,
)


def build_all(args: argparse.Namespace) -> None:
    configure_plotting()
    for build in FIGURE_BUILDERS:
        build(args, args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("plots/revision"))
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=paths.RESULTS_ROOT,
        help="Root that all figure inputs below are resolved against.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    build_all(args)


if __name__ == "__main__":
    main()
