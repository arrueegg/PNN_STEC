"""The manuscript's own numbered figures - all 14 code-generated figures of the 15 defined
in `STEC_Modelling/PNN_main.tex` (everything except the hand-drawn Figure 3).

Distinct from `revision_figures.py`, which builds a separate set for the JGR-MLC response
letter - none of its ~19 figure kinds correspond to a numbered manuscript figure. This
module is the other half: reproducing the figures the paper itself embeds. See
`docs/revision/figure_coverage.md` for the audit that maps all 15 to their pre-rebuild
generator and confirmed none had a `stec/` counterpart before this module.

Coverage
--------

  Figure  1  fig_temporal_split                  train/val/test split, DOY 2014-2024
  Figure  2  fig_spatial_split                    train/val/test stations on a world map
  Figure  4  fig_pred_density                     hexbin predicted vs. true STEC density
  Figure  5  fig_residuals_elev                   residual boxplots by 5-degree elevation bin
  Figure  6  fig_residuals_lat                    residual boxplots by 10-degree sm_lat_ipp bin
  Figure  7  fig_residuals_localtime               residual boxplots by hourly local solar time
  Figure  8  fig_residuals_year_month              monthly residual boxplots
  Figure  9  fig_uncertainty                       abs. error vs. predicted-sigma bin, 4 curves
  Figure 10  fig_improvement_by_date              daily % RMSE/MAE improvement vs. date
  Figure 11  fig_mae_rmse_finetuned                RMSE/MAE vs. elevation, mean +/- across-day std
  Figure 12  fig_positioning_trend                daily 3D RMS, 4 methods, median line only
  Figure 13  fig_boxplot_3d_error                  overall 3D RMS distribution, unfiltered, view capped at 10 m
  Figure 14  fig_positioning_improvement_timeseries daily % improvement over GIM (median)
  Figure 15  fig_cdf_unfiltered                    3D RMS CDF, 4 methods, unfiltered

Figure 3 (`network`) is hand-drawn (`docs/ResNet.drawio`) and needs no code.

**Figures 12-15 report the median, not the mean, with no >10 m outcome-based station-day
exclusion (owner decision, 2026-08-28, `docs/revision/positioning_reporting.md`)** - the
mean is not a robust summary of this comparison, see that document's Sec 1. Figures 13 and
15 are not separately implemented here any more; they call `stec.viz.positioning_distributions.
fig_boxplot_3d_error`/`fig_cdf_unfiltered` directly against that module's own CSVs, so the
distribution figure the manuscript embeds and the one `positioning_distributions_figures`
produces standalone are the same rendering, not two implementations that could drift apart.
Figures 12 and 14 keep their own generators (no per-day equivalent exists in
`positioning_distributions.py`) but no longer apply the outlier filter and plot the median
rather than the mean. **2026-09-14 styling pass (owner review)**: Figure 12 dropped its
shaded Q1-Q3 band (four overlapping translucent fills were unreadable) and draws a thinner,
less marker-dense line - the underlying daily median is unchanged, still the exact value in
`pos_trend.csv`, and no temporal smoothing was applied (see `fig_positioning_trend`'s
docstring for why). Figure 13 switched from a log y-axis to a linear one capped at 10 m -
box/whisker statistics still come from the full unfiltered population, only the view is
clipped, and the count of station-days beyond 10 m per method is written to the plotted CSV
and the figure's own title (stripped from the `_notitle` manuscript copy, like every other
figure here).

**2026-09-15 (owner instruction, `docs/revision/results_register.md` consistency item A):
restricted to the common set.** Figures 12-15 used to read the full per-method population
(10,717-10,853 station-days depending on method) straight from `positioning_coverage`'s
`multiday_summary.csv`, while Tables 5-7 had already moved to the smaller common set - the
same positioning chapter reporting two different N's for the same station-days. All four
now read `stec.analysis.positioning_distributions`'s common-set outputs
(`common_set_daily_rows.csv` for Figures 12/14; `common_set_boxplot_stats.csv`/
`common_set_boxplot_fliers.csv`/`common_set_exceedance.csv` for Figure 13;
`common_set_cdf_points.csv`/`common_set_percentile_summary.csv` for Figure 15) - the
station-days solved by all four methods under both weighting schemes that Tables 5-7
already use (N=10,387; `common_set_positioning.coverage_common_station_days`). No
outcome-based outlier filter is applied on top of that restriction, unchanged from the
2026-08-28 decision. `oracle_benchmark` is not part of this change - it is permanently not
comparable with Table 5 by design (elev weighting, placeholder reference sigma) and stays
on its own population.

**Figures 4-9 are wired via one shared streaming cache**
(`stec/analysis/pretrained_test_diagnostics.py`,
`_build_pretrained_diagnostics_figures` below). Their pre-rebuild source
(`src/viz/{performance,distributions,spatial,uncertainty}.py`, driven by
`src/inference_testset.py`) consumes the full per-observation dataframe of the
*pretrained* model's held-out test set - every year `test_df` holds, never filtered to
2024 - which is `predictions/pretrained_stec/own/` in the prediction store: 544 sampled
days spanning 2014-2024, 10,000,000 observations (`data.test_size: 10000000` in the
paper's pretrain config; earlier revisions of this docstring guessed `_sub500K`, which is
actually `train_subset_size` - a training-time resample, not the test set's size). This is
not the 242-day, ~400M-row `finetuned_stec` store Figures 10-11 draw pooled/per-day
metrics from. No aggregate CSV anywhere carries per-observation residuals, so each
`fig_*` below takes that dataframe directly, as every figure in this module does -
`pretrained_test_diagnostics.py` is what builds it, one day at a time via
`prediction_store.iter_days`, narrowed to the ~11 columns Figures 4-9 actually read
*before* concatenating, which is what keeps the result (measured 10 M rows, ~450 MB) two
orders of magnitude below the 580 M-row whole-store read that OOM-killed the original
analysis driver. `fig_pred_density` (Figure 4) is the one hexbin/scatter figure in the
group; the other five are boxplots that need every row to compute exact quantiles, so
only Figure 4's builder additionally subsamples (2,000,000 points, seed 42 - see
`_PRED_DENSITY_SAMPLE_CAP`) purely to bound the render, not because the cache itself is
too large to hold. Each `fig_*` function is still tested against synthetic
per-observation frames only (`tests/viz/test_manuscript_figures.py`); the streaming and
column-narrowing behaviour is tested separately, against a synthetic on-disk store, in
`tests/analysis/test_pretrained_test_diagnostics.py`.

**Figure 11 needed a new small aggregate that did not exist.** It needs per-day,
per-elevation-bin RMSE/MAE **and their std across days** (the error bars in the published
figure). The only elevation-binned aggregate that already existed
(`stratified_comparison.py`, `multiday_results/stratified_comparison_rebuilt/
by_elevation.csv`) pools all 242 days into one RMSE/MAE value per bin with no variance
term, by design - it answers a different question (R1.4: where does the model still beat
the alternatives) and dropping the error bars to reuse it would be a different figure, not
a port of this one. `stec/analysis/elevation_metrics_finetuned.py` is a new streaming
stage, added here for this figure specifically: same day-at-a-time accumulation via
`prediction_store.iter_days` as `stratified_comparison.py`, same running-sum exactness,
but keeping `doy` as the finest unit instead of pooling it away, and using the
publication's original 5-degree elevation bins (`np.arange(0, 91, 5)`,
`src/multiday_evaluation.py::extract_elevation_metrics_from_experiment`) rather than
`stratified_comparison`'s coarser ones. It is exercised in `tests/analysis/
test_elevation_metrics_finetuned.py` against a synthetic on-disk store, never the real
one, per the same resource limit as everything else in this port; running it for real is,
again, a follow-up for whoever has the budget to stream the full store.

Colour rule
-----------
Approach colours (Direct STEC / VTEC + Mapping / IGS GIM + Mapping / Pretrained Direct
STEC) come from `stec.viz.style.APPROACH_COLORS` only - never a literal hex, never
seaborn's `colorblind` palette - wherever a figure compares those approaches against each
other (Figures 10, 11 and 12-15). This matters here specifically because the pre-rebuild
generator for Figures 10 and 11 (`src/multiday_evaluation.py`) colours its four series
from `seaborn.color_palette("colorblind")` indices 0/1/2/4, not the pinned hex values - a
divergence from the already-published figures that this port does not carry over.
Figures 12-15's pre-rebuild generator (`positioning/scripts/plot_results.py`) already used
the same hex constants `style.py` pins, so no divergence exists there.

Figures 4-9 are single-model residual/uncertainty diagnostics, not approach comparisons -
there is no "VTEC + Mapping" series to protect a colour for. Their green/orange/red/black/
blue markers are literal matplotlib colour names, ported unchanged from
`src/viz/{distributions,spatial,uncertainty}.py`, and are deliberately not routed through
`APPROACH_COLORS`: the named "green"/"orange" render as different hex values
(`#008000`/`#FFA500`) than `GIM_COLOR`/`VTEC_COLOR` (`#2ca02c`/`#ff7f0e`), so there is no
literal collision, and reusing the approach palette here would be applying an approach
colour to a quantity (MAE, RMSE, mean sigma) that is not an approach - the same category
error the colour rule exists to prevent, just from the other direction.

The train/val/test split colours in Figures 1-2 (`#215ACC`/`#5ACC21`/`#CC215A`) are a
separate, non-approach palette carried over unchanged from `src/data_processing/
visualize_temporal_splits.py` and `split_new.py`. Both source files' own inline comments
mislabel two of the three (calling the blue "red" or the green "orange", depending on the
file) - a pre-existing bug in code comments only, not in what is actually drawn. Not fixed
here, per the same reasoning `figure_coverage.md` used: the rendered colours are what the
manuscript published, and fixing a comment is not this port's job.

Usage::

    python -m stec.viz.manuscript_figures --output_dir plots/manuscript --results_dir multiday_results
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from ..config import paths
from .positioning_distributions import fig_boxplot_3d_error, fig_cdf_unfiltered
from .revision_figures import analysis_dir
from .style import (
    APPROACH_COLORS,
    FIGSIZE_DAILY_IMPROVEMENT,
    FIGSIZE_HISTOGRAM,
    FIGSIZE_POSITIONING_TREND,
    FIGSIZE_SQUARE,
    FIGSIZE_WIDE,
    configure_plotting,
)

import matplotlib.pyplot as plt  # noqa: E402  (style.py sets the Agg backend on import)

logger = logging.getLogger(__name__)

SOURCE_DIRS = {
    "dataset": "dataset_construction",
    "pretrained": "stec_pretrained_testset",
    "finetuned": "stec_finetuned_2024",
    "positioning": "positioning_2024",
}


def _save(
    fig: plt.Figure,
    name: str,
    source: str,
    output_dir: Path,
    provenance: str,
    data: pd.DataFrame | None = None,
) -> None:
    """Write the working copy (title + provenance), the manuscript copy, and the numbers.

    Mirrors `revision_figures._save` exactly (title/notitle pair, footnote, plotted-data
    CSV) - duplicated rather than imported because the two modules cover disjoint figure
    sets and `SOURCE_DIRS` differs between them; sharing the function would mean sharing
    that dict too, coupling two things that change independently.
    """
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
# Figure 1 - temporal train/val/test split
# --------------------------------------------------------------------------

# White/blue/green/red, ported unchanged from visualize_temporal_splits.py's `colors` list
# (0: no data, 1: train, 2: val, 3: test). Not APPROACH_COLORS - a split is not an approach.
_SPLIT_COLORS = ["white", "#215ACC", "#5ACC21", "#CC215A"]
_SPLIT_NAMES = ["Training", "Validation", "Test"]
_MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]  # fmt: skip


def fig_temporal_split(
    train_dates: Sequence[str],
    val_dates: Sequence[str],
    test_dates: Sequence[str],
    output_dir: Path,
    provenance: str,
    start_year: int = 2014,
    end_year: int = 2024,
) -> None:
    """Timeline heatmap of which split each (year, month) belongs to.

    Ported from `visualize_temporal_splits.py::create_timeline_heatmap`, whose defaults
    (`start_year=2014, end_year=2024`) match the caption's stated range exactly.
    """
    years = list(range(start_year, end_year + 1))
    matrix = np.zeros((len(years), 12))
    rows: list[dict[str, object]] = []
    for split_index, dates in enumerate((train_dates, val_dates, test_dates), start=1):
        for date_str in dates:
            year, month = (int(x) for x in date_str.split("-"))
            if year in years:
                matrix[year - start_year, month - 1] = split_index
                rows.append(
                    {
                        "year": year,
                        "month": month,
                        "split": _SPLIT_NAMES[split_index - 1],
                    }
                )

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = mcolors.ListedColormap(_SPLIT_COLORS)
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=3)
    ax.set_xticks(range(12))
    ax.set_xticklabels(_MONTH_LABELS)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.8, alpha=0.8)
    ax.tick_params(which="minor", length=0)

    total = int((matrix > 0).sum())
    counts = [int((matrix == i).sum()) for i in (1, 2, 3)]
    pcts = [100 * c / total if total else 0.0 for c in counts]
    handles = [
        mpatches.Patch(color=_SPLIT_COLORS[i + 1], label=f"{name} ({pct:.1f}%)")
        for i, (name, pct) in enumerate(zip(_SPLIT_NAMES, pcts))
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=True,
    )
    ax.set_title("Temporal dataset split")
    _save(
        fig,
        "temp_split",
        "dataset",
        output_dir,
        provenance,
        pd.DataFrame(rows).sort_values(["year", "month"]),
    )


def _build_temporal_split_figure(args: argparse.Namespace, output_dir: Path) -> None:
    train_path, val_path, test_path = (
        paths.date_list("train"),
        paths.date_list("val"),
        paths.date_list("test"),
    )
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        logger.warning(f"split date lists not found under {paths.SPLIT_LISTS}")
        return
    train_dates = train_path.read_text().split()
    val_dates = val_path.read_text().split()
    test_dates = test_path.read_text().split()
    prov = (
        f"{train_path.name}, {val_path.name}, {test_path.name} - temporal train/val/test "
        f"split, {len(train_dates)}/{len(val_dates)}/{len(test_dates)} months"
    )
    fig_temporal_split(train_dates, val_dates, test_dates, output_dir, prov)


# --------------------------------------------------------------------------
# Figure 2 - spatial train/val/test split
# --------------------------------------------------------------------------


def fig_spatial_split(
    train_stations: pd.DataFrame,
    val_stations: pd.DataFrame,
    test_stations: pd.DataFrame,
    output_dir: Path,
    provenance: str,
) -> None:
    """Train/val/test IGS stations on a world map.

    Ported from `split_new.py::plot_station_distribution`. Each argument is a dataframe
    with `name`, `lat`, `lon` columns. Cartopy is imported lazily - it is the only figure
    in this module that needs it, and importing it at module load would make every other
    figure pay for a dependency it does not use.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig, ax = plt.subplots(
        figsize=(14, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="#FFFFFF")
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff")
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.8, color="gray", alpha=0.6, linestyle="--"
    )
    gl.top_labels, gl.right_labels = False, False

    total = len(train_stations) + len(val_stations) + len(test_stations)
    groups = [
        (train_stations, "#215ACC", "Training"),
        (val_stations, "#5ACC21", "Validation"),
        (test_stations, "#CC215A", "Test"),
    ]
    plotted = []
    for stations, color, name in groups:
        pct = 100 * len(stations) / total if total else 0.0
        ax.scatter(
            stations["lon"],
            stations["lat"],
            s=35,
            c=color,
            label=f"{name}: {len(stations)} ({pct:.1f}%)",
            zorder=3,
        )
        plotted.append(stations.assign(split=name))

    ax.legend(
        title=f"Total stations: {total}",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=True,
    )
    ax.set_global()
    ax.set_title("IGS station distribution for the STEC database")
    _save(
        fig,
        "spatial_split",
        "dataset",
        output_dir,
        provenance,
        pd.concat(plotted, ignore_index=True)[["name", "lat", "lon", "split"]],
    )


def _load_split_stations(split: str, coordinates: pd.DataFrame) -> pd.DataFrame:
    names = paths.station_list(split).read_text().split()
    return coordinates[coordinates["name"].isin(names)]


def _build_spatial_split_figure(args: argparse.Namespace, output_dir: Path) -> None:
    if not paths.IGS_STATION_COORDINATES.exists():
        logger.warning(f"{paths.IGS_STATION_COORDINATES} not found")
        return
    coordinates = pd.read_csv(
        paths.IGS_STATION_COORDINATES,
        usecols=["#StationName", "Latitude", "Longitude"],
    ).rename(columns={"#StationName": "name", "Latitude": "lat", "Longitude": "lon"})
    coordinates["name"] = coordinates["name"].str[:4]
    coordinates = coordinates.drop_duplicates("name")

    train = _load_split_stations("train", coordinates)
    val = _load_split_stations("val", coordinates)
    test = _load_split_stations("test", coordinates)
    prov = (
        f"{paths.IGS_STATION_COORDINATES.name} + train/val/test_station.list - "
        f"{len(train)}/{len(val)}/{len(test)} IGS stations"
    )
    fig_spatial_split(train, val, test, output_dir, prov)


# --------------------------------------------------------------------------
# Figure 4 - hexbin density of predicted vs. true STEC
# --------------------------------------------------------------------------


def fig_pred_density(
    df: pd.DataFrame,
    output_dir: Path,
    provenance: str,
    max_limit: float | None = None,
) -> None:
    """Hexbin density of predicted vs. true STEC with the 1:1 line, Pearson r and R2.

    Ported from `src/viz/performance.py::plot_prediction_density`. `df` needs `true_stec`
    and `stec_pred`; r and R2 are computed here with plain numpy rather than
    `scipy.stats.pearsonr`/`sklearn.metrics.r2_score` - the source's choices - because
    neither scikit-learn nor an extra scipy import is otherwise needed by this module and
    both statistics are a few lines of arithmetic (matching how `daily_metrics.day_
    metrics` already computes R2 elsewhere in this package). `max_limit` reproduces the
    source's optional 300 TECU zoomed variant; `None` uses the data's own maximum, as it
    does.
    """
    true_stec = df["true_stec"].to_numpy(dtype=float)
    pred_stec = df["stec_pred"].to_numpy(dtype=float)

    correlation = float(np.corrcoef(true_stec, pred_stec)[0, 1])
    residual_sum_sq = float(np.sum((true_stec - pred_stec) ** 2))
    total_sum_sq = float(np.sum((true_stec - true_stec.mean()) ** 2))
    r_squared = 1 - residual_sum_sq / total_sum_sq if total_sum_sq > 0 else np.nan

    max_val = (
        float(max_limit)
        if max_limit is not None
        else float(max(true_stec.max(), pred_stec.max()))
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    hexbin = ax.hexbin(
        true_stec,
        pred_stec,
        gridsize=100,
        cmap="BuGn",
        mincnt=1,
        extent=[0, max_val, 0, max_val],
        norm=mcolors.LogNorm(),
    )
    ax.plot([0, max_val], [0, max_val], "r-", linewidth=3, alpha=0.9)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.set_xlabel("True STEC [TECU]")
    ax.set_ylabel("Predicted STEC [TECU]")
    fig.colorbar(hexbin, ax=ax, label="Count")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="red", linewidth=3, label="Perfect prediction"),
            plt.Line2D([0], [0], color="none", label=f"Pearson r = {correlation:.3f}"),
            plt.Line2D([0], [0], color="none", label=f"R² = {r_squared:.3f}"),
        ],
        loc="upper left",
        framealpha=0.9,
    )
    ax.grid(True, alpha=0.3)
    ax.set_title("Prediction density: predicted vs. observed STEC")
    name = "pred_density" if max_limit is None else "pred_density_limited"
    _save(
        fig,
        name,
        "pretrained",
        output_dir,
        provenance,
        pd.DataFrame({"true_stec": true_stec, "stec_pred": pred_stec}),
    )


# --------------------------------------------------------------------------
# Figures 5-8 - residual boxplots vs. elevation / sm latitude / local time / date
# --------------------------------------------------------------------------

# Boxplot styling shared by Figures 5-8, ported unchanged from the identical boilerplate
# block repeated in `src/viz/distributions.py` and `spatial.py`.


def _style_residual_boxplot(bp: dict) -> None:
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    for element in ("whiskers", "caps"):
        for item in bp[element]:
            item.set_linewidth(2)
            item.set_color("black")
    for item in bp["medians"]:
        item.set_linewidth(2)
        item.set_color("midnightblue")


def _plot_mae_rmse_overlay(
    ax: plt.Axes,
    positions: Sequence[float],
    mae_values: Sequence[float],
    rmse_values: Sequence[float],
) -> None:
    """MAE (green circles) / RMSE (orange squares) overlay, ported unchanged from the
    same boilerplate repeated in `src/viz/distributions.py` and `spatial.py`. Literal
    colour names, not `APPROACH_COLORS` - see the module docstring's colour-rule section
    for why: these are metrics of one model's own residuals, not a multi-approach
    comparison, so the approach palette does not apply.
    """
    ax.plot(
        positions,
        mae_values,
        color="green",
        marker="o",
        linewidth=3,
        markersize=8,
        label="MAE",
        alpha=0.9,
        zorder=10,
    )
    ax.plot(
        positions,
        rmse_values,
        color="orange",
        marker="s",
        linewidth=3,
        markersize=8,
        label="RMSE",
        alpha=0.9,
        zorder=10,
    )


# get_default_bin_ranges()["satele"] and src/viz/__init__.py's feature-specific bin count
# for "satele" (17 bins over a (5, 90) range is a 5-degree bin width).
_ELEVATION_BIN_RANGE = (5.0, 90.0)
_ELEVATION_NUM_BINS = 17


def fig_residuals_elev(
    df: pd.DataFrame,
    output_dir: Path,
    provenance: str,
    num_bins: int = _ELEVATION_NUM_BINS,
    bin_range: tuple[float, float] = _ELEVATION_BIN_RANGE,
) -> None:
    """Residual boxplots by elevation bin, with MAE/RMSE line overlay.

    Ported from `src/viz/distributions.py::plot_residuals_vs_feature(df, "satele", ...)`
    -> `plot_binned_boxplot`. `df` needs `true_stec`, `stec_pred`, `satele`.
    """
    residual = df["true_stec"] - df["stec_pred"]
    bin_edges = np.linspace(bin_range[0], bin_range[1], num_bins + 1)
    elevation_bin = pd.cut(df["satele"], bins=bin_edges, include_lowest=True)

    grouped = pd.DataFrame(
        {"elevation_bin": elevation_bin, "residual": residual}
    ).groupby("elevation_bin", observed=True)
    box_data = [grouped.get_group(b)["residual"].to_numpy() for b in grouped.groups]
    x_labels = [f"{b.left:.0f}–{b.right:.0f}" for b in grouped.groups]
    mae_values = [np.abs(v).mean() for v in box_data]
    rmse_values = [np.sqrt(np.mean(v**2)) for v in box_data]

    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)
    ax.axhline(y=0, color="red", linewidth=2, zorder=1, alpha=0.8)
    bp = ax.boxplot(
        box_data, tick_labels=x_labels, showfliers=False, zorder=2, patch_artist=True
    )
    _style_residual_boxplot(bp)
    _plot_mae_rmse_overlay(ax, range(1, len(box_data) + 1), mae_values, rmse_values)
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Elevation angle [degrees]")
    ax.set_ylabel("Residual [TECU]")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Residual analysis vs. elevation angle")
    _save(
        fig,
        "residuals_elev",
        "pretrained",
        output_dir,
        provenance,
        pd.DataFrame(
            {
                "elevation_bin": x_labels,
                "mae": mae_values,
                "rmse": rmse_values,
                "n": [len(v) for v in box_data],
            }
        ),
    )


# spatial.py::plot_box_by_lat's fixed 10-degree bins, -90 to 90.
_GEOMAGNETIC_LAT_BIN_EDGES = np.arange(-90, 91, 10)


def fig_residuals_lat(df: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """Residual boxplots by 10-degree solar-magnetic-latitude bin, MAE/RMSE overlay.

    Ported from `src/viz/spatial.py::plot_box_by_lat`. `df` needs `true_stec`,
    `stec_pred`, `sm_lat_ipp`. Every bin in the fixed range is kept even if empty (as a
    NaN box), matching the source's `reindex(all_bins)` - a gap in the plot is
    informative (no observations at that latitude), not something to silently compress
    away.
    """
    residual = df["true_stec"] - df["stec_pred"]
    all_bins = pd.IntervalIndex.from_breaks(_GEOMAGNETIC_LAT_BIN_EDGES, closed="left")
    lat_bin = pd.cut(
        df["sm_lat_ipp"],
        bins=_GEOMAGNETIC_LAT_BIN_EDGES,
        include_lowest=True,
        right=False,
    )

    grouped = (
        pd.DataFrame({"lat_bin": lat_bin, "residual": residual})
        .groupby("lat_bin", observed=False)["residual"]
        .apply(list)
        .reindex(all_bins)
    )
    bin_centers = [(b.left + b.right) / 2 for b in grouped.index]
    box_data: list[np.ndarray | list[float]] = []
    mae_values: list[float] = []
    rmse_values: list[float] = []
    for b in grouped.index:
        values = np.asarray(grouped[b], dtype=float)
        if values.size == 0:
            box_data.append([np.nan])
            mae_values.append(np.nan)
            rmse_values.append(np.nan)
        else:
            box_data.append(values)
            mae_values.append(float(np.abs(values).mean()))
            rmse_values.append(float(np.sqrt(np.mean(values**2))))

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.axhline(y=0, color="red", linewidth=2, zorder=1, alpha=0.8)
    bp = ax.boxplot(
        box_data,
        positions=bin_centers,
        widths=5,
        showfliers=False,
        zorder=2,
        patch_artist=True,
    )
    _style_residual_boxplot(bp)
    _plot_mae_rmse_overlay(ax, bin_centers, mae_values, rmse_values)
    ax.set_xticks(_GEOMAGNETIC_LAT_BIN_EDGES)
    ax.set_xticklabels([str(b) for b in _GEOMAGNETIC_LAT_BIN_EDGES], rotation=45)
    ax.set_ylim(-30, 30)
    ax.set_xlabel("Solar magnetic latitude [degrees]")
    ax.set_ylabel("Residual [TECU]")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Residual analysis vs. solar magnetic latitude")
    _save(
        fig,
        "residuals_lat",
        "pretrained",
        output_dir,
        provenance,
        pd.DataFrame(
            {"lat_bin_center": bin_centers, "mae": mae_values, "rmse": rmse_values}
        ),
    )


def fig_residuals_localtime(
    df: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Residual boxplots by hourly local solar time, MAE/RMSE overlay.

    Ported from `src/viz/distributions.py::plot_residuals_vs_local_time`. `df` needs
    `true_stec`, `stec_pred` and either `local_time_hours` or (`sod`, `lon_ipp`) to
    derive it - the pretrained model was not configured with `local_time_hours` as an
    input feature, so its store rows lack the column, exactly the case
    `stratified_comparison.add_local_time` already handles; reused here rather than
    reimplemented so the derivation is defined in one place. An hour needs at least 5
    observations to draw a box (matching the source), but the MAE/RMSE lines cover all 24
    hours regardless, with a gap where a hour has none.
    """
    from ..analysis.stratified_comparison import add_local_time

    df = add_local_time(df)
    if "local_time_hours" not in df.columns:
        logger.warning(
            "no local_time_hours (and no sod/lon_ipp to derive it) - "
            "skipping residuals_localtime"
        )
        return

    residual = (df["true_stec"] - df["stec_pred"]).to_numpy()
    hour_bin = pd.cut(df["local_time_hours"], bins=24, labels=range(24)).to_numpy()

    box_data, mae_values, rmse_values = [], [], []
    for hour in range(24):
        hour_residual = residual[hour_bin == hour]
        box_data.append(hour_residual if len(hour_residual) >= 5 else None)
        if len(hour_residual) > 0:
            mae_values.append(np.abs(hour_residual).mean())
            rmse_values.append(np.sqrt(np.mean(hour_residual**2)))
        else:
            mae_values.append(np.nan)
            rmse_values.append(np.nan)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    valid_hours = [h for h, data in enumerate(box_data) if data is not None]
    valid_box_data = [box_data[h] for h in valid_hours]
    if valid_box_data:
        bp = ax.boxplot(
            valid_box_data,
            positions=valid_hours,
            widths=0.6,
            showfliers=False,
            patch_artist=True,
        )
        _style_residual_boxplot(bp)
    _plot_mae_rmse_overlay(ax, range(24), mae_values, rmse_values)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, linewidth=2)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([str(h) for h in range(0, 24, 3)])
    ax.set_xlabel("Local solar time [hours]")
    ax.set_ylabel("Residual [TECU]")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Residual analysis vs. local solar time")
    _save(
        fig,
        "residuals_localtime",
        "pretrained",
        output_dir,
        provenance,
        pd.DataFrame({"hour": range(24), "mae": mae_values, "rmse": rmse_values}),
    )


def fig_residuals_year_month(
    df: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Monthly residual boxplots, MAE/RMSE line overlay.

    Ported from `src/viz/distributions.py::plot_box_by_date`. `df` needs `true_stec`,
    `stec_pred`, `year`, `doy` - the store's identity columns are already true integers
    (cast at write time), not the sin/cos-encoded model *input* the CLAUDE.md
    "round(), never int()" gotcha warns about, so no rounding is needed here.
    """
    date = pd.to_datetime(df["year"], format="%Y") + pd.to_timedelta(
        df["doy"] - 1, unit="D"
    )
    residual = (df["true_stec"] - df["stec_pred"]).to_numpy()
    year_month = date.dt.to_period("M").astype(str)
    order = sorted(year_month.unique())
    pos = np.arange(len(order))

    grouped = pd.Series(residual).groupby(year_month.to_numpy()).apply(list)
    box_data = [grouped[label] for label in order]
    mae_values = [np.mean(np.abs(v)) for v in box_data]
    rmse_values = [np.sqrt(np.mean(np.square(v))) for v in box_data]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.axhline(y=0, color="red", linewidth=2, zorder=1, alpha=0.8)
    bp = ax.boxplot(
        box_data,
        widths=0.5,
        positions=pos,
        showfliers=False,
        zorder=2,
        patch_artist=True,
    )
    _style_residual_boxplot(bp)
    _plot_mae_rmse_overlay(ax, pos, mae_values, rmse_values)
    ax.set_xticks(pos)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylim(-30, 30)
    ax.set_xlabel("Year-month")
    ax.set_ylabel("Residual [TECU]")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Residual analysis: monthly evolution")
    _save(
        fig,
        "residuals_year_month",
        "pretrained",
        output_dir,
        provenance,
        pd.DataFrame({"year_month": order, "mae": mae_values, "rmse": rmse_values}),
    )


# --------------------------------------------------------------------------
# Figure 9 - absolute error vs. predicted uncertainty, binned
# --------------------------------------------------------------------------


def fig_uncertainty(df: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """Absolute error vs. predicted-uncertainty bin: boxplots plus MAE, mean predicted
    sigma, mean epistemic and mean aleatoric curves.

    Ported from `src/viz/uncertainty.py::plot_binned_uncertainty_error_analysis`
    (`show_components=True` branch only - the manuscript figure has all four curves).
    `df` needs `true_stec`, `stec_pred`, `pred_total_unc`; `pred_epistemic_unc` and
    `pred_aleatoric_unc` are optional and each only drawn if present with a positive
    maximum. Fixed 1 TECU bin width from 0 up to the 95th percentile of predicted
    uncertainty, matching the source; a bin needs at least 5 observations to be plotted.
    """
    if "pred_total_unc" not in df.columns:
        logger.warning("no pred_total_unc column - skipping fig_uncertainty")
        return

    abs_error = (df["true_stec"] - df["stec_pred"]).abs().to_numpy()
    total_unc = df["pred_total_unc"].to_numpy(dtype=float)
    max_unc = float(np.quantile(total_unc, 0.95))
    if max_unc < 1e-6:
        logger.warning(
            f"max predicted uncertainty is {max_unc:.2e}, too small to bin - "
            "skipping fig_uncertainty (likely a deterministic model)"
        )
        return

    bin_width = 1.0
    bin_edges = np.arange(0, max(np.ceil(max_unc), 1.0) + bin_width, bin_width)
    unc_bin = pd.cut(total_unc, bins=bin_edges, include_lowest=True, labels=False)

    has_epistemic = "pred_epistemic_unc" in df.columns
    has_aleatoric = "pred_aleatoric_unc" in df.columns
    epistemic = (
        df["pred_epistemic_unc"].to_numpy(dtype=float) if has_epistemic else None
    )
    aleatoric = (
        df["pred_aleatoric_unc"].to_numpy(dtype=float) if has_aleatoric else None
    )

    positions, box_data, mean_abs_error, mean_total_unc = [], [], [], []
    mean_epistemic, mean_aleatoric = [], []
    for bin_index in range(len(bin_edges) - 1):
        in_bin = unc_bin == bin_index
        if in_bin.sum() < 5:
            continue
        center = (bin_edges[bin_index] + bin_edges[bin_index + 1]) / 2
        positions.append(center)
        box_data.append(abs_error[in_bin])
        mean_abs_error.append(abs_error[in_bin].mean())
        mean_total_unc.append(total_unc[in_bin].mean())
        mean_epistemic.append(epistemic[in_bin].mean() if has_epistemic else np.nan)
        mean_aleatoric.append(aleatoric[in_bin].mean() if has_aleatoric else np.nan)

    if not positions:
        logger.warning("no bin has >= 5 observations - skipping fig_uncertainty")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.boxplot(
        box_data,
        positions=positions,
        widths=bin_width * 0.8,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "lightgrey", "edgecolor": "black", "alpha": 0.7},
        whiskerprops={"linewidth": 0},
        capprops={"linewidth": 0},
        medianprops={"color": "red", "linewidth": 1.5},
    )
    ax.plot(
        positions,
        mean_abs_error,
        color="orange",
        marker="o",
        label="Mean absolute error",
        linewidth=2,
        markersize=6,
        zorder=20,
    )
    ax.plot(
        positions,
        mean_total_unc,
        color="red",
        marker="s",
        label="Mean predicted uncertainty",
        linewidth=2,
        markersize=6,
        zorder=20,
    )
    if has_epistemic and max(mean_epistemic) > 0:
        ax.plot(
            positions,
            mean_epistemic,
            color="black",
            marker="^",
            label="Mean epistemic uncertainty",
            linewidth=2,
            markersize=6,
            zorder=20,
        )
    if has_aleatoric and max(mean_aleatoric) > 0:
        ax.plot(
            positions,
            mean_aleatoric,
            color="blue",
            marker="v",
            label="Mean aleatoric uncertainty",
            linewidth=2,
            markersize=6,
            zorder=20,
        )

    ax.set_xlabel("Predicted uncertainty [TECU]")
    ax.set_ylabel("Absolute error [TECU]")
    ax.set_xlim(0, max(positions) + bin_width)
    ax.legend(framealpha=0.9, loc="upper left")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_title("Error analysis vs. predicted uncertainty (binned)")
    _save(
        fig,
        "uncertainty",
        "pretrained",
        output_dir,
        provenance,
        pd.DataFrame(
            {
                "unc_bin_center": positions,
                "mean_abs_error": mean_abs_error,
                "mean_total_unc": mean_total_unc,
                "mean_epistemic_unc": mean_epistemic,
                "mean_aleatoric_unc": mean_aleatoric,
            }
        ),
    )


# --------------------------------------------------------------------------
# Figures 4-9 wiring - one shared per-observation cache, six figures
# --------------------------------------------------------------------------

# Cap and seed for fig_pred_density's hexbin. The cache holds the model's whole 2014-2024
# test set (10,000,000 rows as of this writing - see
# stec.analysis.pretrained_test_diagnostics), and a hexbin rendered at gridsize=100 does
# not resolve any finer past a couple of million points, so beyond this cap only render
# cost grows, not the pattern the figure shows. This subsample is specific to
# fig_pred_density: the other five figures in this group are boxplots computed from every
# row in the cache (see that module's docstring for why a boxplot needs the full data
# where a hexbin does not).
_PRED_DENSITY_SAMPLE_CAP = 2_000_000
_PRED_DENSITY_SAMPLE_SEED = 42


def _build_pretrained_diagnostics_figures(
    args: argparse.Namespace, output_dir: Path
) -> None:
    """Figures 4-9: read the shared per-observation cache once, draw all six from it.

    `stec.analysis.pretrained_test_diagnostics` is the one stage that streams
    `predictions/pretrained_stec/own` (2014-2024, the pretrained model's entire held-out
    test set - the data Figures 4-9 were originally built from) into a bounded,
    narrow-column parquet cache; see that module's docstring for why a cache is needed at
    all instead of the sum-accumulator pattern the rest of `stec.analysis` uses, and for
    the measured size. Reading it once here and passing it to all six `fig_*` functions,
    rather than one `_build_*_figure` per figure each re-reading the cache, is what keeps
    a ~10 M-row read to a single pass.
    """
    diagnostics_dir = analysis_dir(args.results_dir, "pretrained_test_diagnostics")
    cache_path = diagnostics_dir / "observations.parquet"
    manifest_path = diagnostics_dir / "manifest.csv"
    if not cache_path.exists() or not manifest_path.exists():
        logger.warning(
            f"{cache_path} not found - run stec/analysis/pretrained_test_diagnostics.py"
        )
        return

    observations = pd.read_parquet(cache_path)
    manifest = pd.read_csv(manifest_path)
    prov = (
        f"{cache_path} - pretrained model, own held-out test set, "
        f"{manifest['year'].min()}-{manifest['year'].max()} "
        f"({int(manifest['n_days'].sum())} days, {len(observations):,} observations)"
    )

    fig_residuals_elev(observations, output_dir, prov)
    fig_residuals_lat(observations, output_dir, prov)
    fig_residuals_localtime(observations, output_dir, prov)
    fig_residuals_year_month(observations, output_dir, prov)
    fig_uncertainty(observations, output_dir, prov)

    # fig_pred_density alone needs a bounded input - see _PRED_DENSITY_SAMPLE_CAP above.
    if len(observations) > _PRED_DENSITY_SAMPLE_CAP:
        density_sample = observations.sample(
            n=_PRED_DENSITY_SAMPLE_CAP, random_state=_PRED_DENSITY_SAMPLE_SEED
        )
        density_prov = (
            f"{prov}, subsampled to {_PRED_DENSITY_SAMPLE_CAP:,} points "
            f"(seed {_PRED_DENSITY_SAMPLE_SEED}) for the hexbin render"
        )
    else:
        density_sample = observations
        density_prov = prov
    fig_pred_density(density_sample, output_dir, density_prov)
    fig_pred_density(density_sample, output_dir, density_prov, max_limit=300)


# --------------------------------------------------------------------------
# Figure 10 - daily % RMSE/MAE improvement of Direct STEC over the two baselines
# --------------------------------------------------------------------------

# daily_metrics reports Model names inherited from the analysis it replaces ("Direct STEC
# Model", "IGS GIM") rather than the palette's spellings - see
# tests/viz/test_style.py::test_analysis_method_labels_are_palette_keys, which documents
# this as an intentional, preserved inconsistency. Normalised here the same way
# revision_figures._activity_figures normalises the same column.
_MODEL_RENAME = {
    "Direct STEC Model": "Direct STEC",
    "Pretrained STEC": "Pretrained Direct STEC",
    "VTEC + Mapping": "VTEC + Mapping",
    "IGS GIM": "IGS GIM + Mapping",
}


def fig_improvement_by_date(
    daily: pd.DataFrame,
    metric: str,
    output_dir: Path,
    provenance: str,
    *,
    suffix: str = "",
) -> None:
    """Daily % improvement of Direct STEC over VTEC + Mapping and IGS GIM + Mapping.

    Ported from `src/multiday_evaluation.py::generate_aggregate_plots`, section 3
    ("Improvement statistics"): `improvement = (1 - stec / baseline) * 100`, one point per
    test day. `daily` is one dataset's slice of `daily_metrics_rebuilt/per_day.csv`, model
    names already normalised to the palette's spellings and pivoted so each row is one
    date. Sorted explicitly by date before plotting - the source relies on `.unique()`
    preserving already-chronological row order, which a groupby is not guaranteed to;
    sorting cannot change a correctly-ordered source's output and removes that latent
    dependency.

    `suffix` distinguishes the output filename across the multiple datasets
    `_build_improvement_by_date_figures` loops over (empty for the manuscript's own-dataset
    variant, `_madrigal` for the Madrigal one) - see `_IMPROVEMENT_DATASET_SUFFIXES`. Without
    it, two datasets plotted through this function collide on the same
    `improvements_{metric}` name and the second write silently discards the first.
    """
    pivot = daily.pivot(index="date", columns="Model", values=metric).sort_index()
    stec = pivot["Direct STEC"]

    fig, ax = plt.subplots(figsize=FIGSIZE_DAILY_IMPROVEMENT)
    rows = []
    for baseline, color in (
        ("VTEC + Mapping", APPROACH_COLORS["VTEC + Mapping"]),
        ("IGS GIM + Mapping", APPROACH_COLORS["IGS GIM + Mapping"]),
    ):
        if baseline not in pivot.columns:
            continue
        # A day where the baseline itself is 0 TECU RMSE/MAE cannot define a percentage
        # improvement; the source skips such points rather than plotting +/-inf.
        baseline_values = pivot[baseline].replace(0, np.nan)
        improvement = (1 - stec / baseline_values) * 100
        ax.plot(
            improvement.index,
            improvement.values,
            marker="o",
            markersize=5,
            color=color,
            label=f"Imp. by {baseline}",
        )
        rows.append(
            pd.DataFrame(
                {
                    "date": improvement.index,
                    "baseline": baseline,
                    "improvement_pct": improvement.values,
                }
            )
        )

    ax.axhline(0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_ylabel(f"{metric} improvement [%]")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="Baseline", loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax.set_title(f"Direct STEC {metric} improvement over baselines")
    _save(
        fig,
        f"improvements_{metric.lower()}{suffix}",
        "finetuned",
        output_dir,
        provenance,
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
    )


# `daily_metrics.per_day.csv`'s dataset labels (`daily_metrics.DATASET_LABELS`) to output
# filename suffix. Empty for "own_vtec_gim" so the manuscript's Figure 10 (own dataset,
# Tables 3-4's scope) keeps its pre-existing `improvements_{metric}` name - matching the
# convention `revision_figures.py` already uses for its own multi-source loops (e.g.
# `_DSTEC_SOURCES`'s `("", "", ...)` primary entry, `("_pretrained", ...)` for the rest):
# suffix is empty for the canonical variant, `_<qualifier>` for every other one, so no two
# variants can ever land on the same path. Any dataset label not listed here still gets a
# distinct name rather than colliding.
_IMPROVEMENT_DATASET_SUFFIXES = {"own_vtec_gim": "", "madrigal_vtec_gim": "_madrigal"}


def _build_improvement_by_date_figures(
    args: argparse.Namespace, output_dir: Path
) -> None:
    path = analysis_dir(args.results_dir, "daily_metrics") / "per_day.csv"
    if not path.exists():
        logger.warning(f"{path} not found - run stec/analysis/daily_metrics.py")
        return
    table = pd.read_csv(path, parse_dates=False)
    table["Model"] = table["Model"].map(_MODEL_RENAME)
    # per_day.csv's "date" is a "YYYY-DDD" label, not a calendar date; year+doy already
    # carry the real calendar day, which mdates needs to place ticks correctly.
    table["date"] = pd.to_datetime(table["year"], format="%Y") + pd.to_timedelta(
        table["doy"] - 1, unit="D"
    )
    for dataset, dataset_table in table.groupby("dataset"):
        suffix = _IMPROVEMENT_DATASET_SUFFIXES.get(dataset, f"_{dataset}")
        for metric in ("RMSE", "MAE"):
            days = dataset_table["date"].nunique()
            prov = (
                f"{path} - daily fine-tuned models, {dataset}, {days} test days of 2024"
            )
            fig_improvement_by_date(
                dataset_table, metric, output_dir, prov, suffix=suffix
            )


# --------------------------------------------------------------------------
# Figure 11 - RMSE/MAE vs. elevation, mean +/- across-day std
# --------------------------------------------------------------------------

_ELEVATION_METRIC_MARKERS = {
    "Direct STEC": "o",
    "Pretrained Direct STEC": "D",
    "VTEC + Mapping": "s",
    "IGS GIM + Mapping": "^",
}
# Fixed per-method x-offset, ported unchanged from the "Combined RMSE/MAE Plot" block of
# `src/multiday_evaluation.py::generate_aggregate_plots`: keyed to the method, not to its
# position among the methods actually present, so an absent Pretrained Direct STEC series
# does not shift the other three - matching the source exactly.
_ELEVATION_JITTER_OFFSETS = {
    "Direct STEC": -1.5,
    "Pretrained Direct STEC": -0.5,
    "VTEC + Mapping": 0.5,
    "IGS GIM + Mapping": 1.5,
}
_ELEVATION_JITTER_STEP = 0.8


def fig_mae_rmse_finetuned(
    daily_by_elevation: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """RMSE (top) / MAE (bottom) vs. 5-degree elevation bin, mean +/- across-day std.

    Ported from `generate_aggregate_plots`'s "Combined RMSE/MAE Plot" block.
    `daily_by_elevation` is one dataset's per-(doy, elevation_bin, Method) RMSE/MAE table
    - `stec.analysis.elevation_metrics_finetuned`'s output, or an equivalent synthetic
    frame in tests - with columns `doy`, `elevation_bin`, `Method`, `n`, `RMSE`, `MAE`.
    The across-day mean and std this figure needs are computed here, from that per-day
    table: the only pooled elevation aggregate that exists elsewhere
    (`stratified_comparison.py`) discards the day axis by design and cannot supply the
    error bars - see the module docstring.
    """
    agg = (
        daily_by_elevation.groupby(["elevation_bin", "Method"])
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            days=("doy", "nunique"),
            observations=("n", "sum"),
        )
        .reset_index()
    )
    # Bin centre: the source's `elevation_bin + 2.5` places the marker in the middle of
    # the 5-degree bin its left edge labels.
    agg["x"] = agg["elevation_bin"] + 2.5

    fig, (ax_rmse, ax_mae) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    order = [m for m in _ELEVATION_JITTER_OFFSETS if m in agg["Method"].unique()]
    for method in order:
        subset = agg[agg.Method == method].sort_values("x")
        x = subset["x"] + _ELEVATION_JITTER_OFFSETS[method] * _ELEVATION_JITTER_STEP
        color = APPROACH_COLORS[method]
        marker = _ELEVATION_METRIC_MARKERS[method]
        ax_rmse.errorbar(
            x,
            subset["RMSE_mean"],
            yerr=subset["RMSE_std"],
            label=method,
            marker=marker,
            capsize=4,
            color=color,
            markersize=6,
            alpha=0.9,
        )
        ax_mae.errorbar(
            x,
            subset["MAE_mean"],
            yerr=subset["MAE_std"],
            label=method,
            marker=marker,
            capsize=4,
            color=color,
            markersize=6,
            alpha=0.9,
        )

    ax_rmse.set_ylabel("RMSE [TECU]")
    ax_rmse.grid(True, linestyle="--", alpha=0.5)
    ax_mae.set_ylabel("MAE [TECU]")
    ax_mae.set_xlabel("Elevation angle [degrees]")
    ax_mae.grid(True, linestyle="--", alpha=0.5)
    ax_mae.set_xlim(0, 90)
    ax_mae.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(order), frameon=True
    )
    fig.suptitle("Elevation-dependent RMSE and MAE, daily fine-tuned models")
    _save(
        fig,
        "mae_rmse_finetuned",
        "finetuned",
        output_dir,
        provenance,
        agg[
            [
                "elevation_bin",
                "Method",
                "days",
                "observations",
                "RMSE_mean",
                "RMSE_std",
                "MAE_mean",
                "MAE_std",
            ]
        ],
    )


def _build_mae_rmse_finetuned_figure(
    args: argparse.Namespace, output_dir: Path
) -> None:
    path = (
        analysis_dir(args.results_dir, "elevation_metrics_finetuned")
        / "per_day_by_elevation.csv"
    )
    if not path.exists():
        logger.warning(
            f"{path} not found - run stec/analysis/elevation_metrics_finetuned.py"
        )
        return
    table = pd.read_csv(path)
    # The manuscript figure is the own test set specifically (Tables 3-4's scope);
    # Madrigal rows, if present, are left for a caller who wants that variant, avoiding a
    # same-filename collision if this were looped over every dataset in the CSV.
    own = table[table["dataset"] == "own"]
    if own.empty:
        logger.warning(f"{path} has no 'own' dataset rows")
        return
    days = own["doy"].nunique()
    prov = f"{path} - daily fine-tuned models, own test set, {days} test days of 2024"
    fig_mae_rmse_finetuned(own, output_dir, prov)


# --------------------------------------------------------------------------
# Figures 12-15 - SF-PPP positioning, 4 methods, 2024 test period
# --------------------------------------------------------------------------

# Plotting/z-order ported unchanged from plot_trends/plot_extended_analysis's
# `ordered_methods` sort key (pretrained, then stec, then vtec, then gim).
_POSITIONING_ORDER = [
    "Pretrained Direct STEC",
    "Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]
_POSITIONING_METHOD_MAP = {
    "STEC_iono": "Direct STEC",
    "Pretrained_STEC_iono": "Pretrained Direct STEC",
    "VTEC_iono": "VTEC + Mapping",
    "gim_iono": "IGS GIM + Mapping",
}


def _load_positioning_frame(path: Path) -> pd.DataFrame:
    """Read one station-day per row, normalise method names. No outcome-based filter.

    Used to apply `stec.positioning.metrics.exclude_outlier_station_days` (>10 m). Dropped
    per the owner's 2026-08-28 decision (`docs/revision/positioning_reporting.md`): the
    10 m rule filters on the *outcome* being compared and is not neutral between methods
    (Direct STEC carried 4.4x as many station-days above 10 m as IGS GIM), so every
    positioning figure reads a population with no outcome-based filter, matching Table 5.

    `path` is now `stec.analysis.positioning_distributions`'s `common_set_daily_rows.csv`
    (2026-09-15, results_register.md consistency item A) rather than `positioning_coverage`'s
    raw `multiday_summary.csv` - the common-set restriction (station-days solved by all
    four methods under both weighting schemes) already happened upstream, in that analysis
    module; this function still only reads `date`/`method`/`error_3d_rms` and maps method
    codes to display names exactly as before, since the extra `station`/`doy` columns that
    CSV also carries are for the common-set restriction itself, not for this figure.
    """
    frame = pd.read_csv(path, usecols=["date", "method", "error_3d_rms"])
    frame["method"] = frame["method"].map(_POSITIONING_METHOD_MAP)
    frame = frame.dropna(subset=["method"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


# Figure 12 styling pass, 2026-09-14 (owner review): the shaded Q1-Q3 band and the
# rcParams-default linewidth/markersize made four overlapping methods an unreadable
# smear. Both trimmed here; the x-axis formatter/rotation two lines below are unchanged
# from the figure's very first version - checked against `git show
# b0a610e^:stec/viz/manuscript_figures.py` (the last pre-median-switch commit) and
# against the original pre-rebuild `positioning/scripts/plot_results.py` (lines 220-221):
# both use the identical `mdates.DateFormatter("%m-%d")` + `rotation=45`, nothing to
# restore.
_POSITIONING_TREND_LINEWIDTH = 1.1
# Owner review 2026-09-15: a marker on every daily point, small enough that 242 points x 4
# series stay legible. An earlier pass drew one on every sixth point, which reads as an
# inconsistency - it invites the question of what is special about those days, and nothing is.
_POSITIONING_TREND_MARKERSIZE = 2.5
_POSITIONING_TREND_LEGEND_FONTSIZE = 12
# Only every Nth day gets a marker symbol; the line itself still connects every raw
# daily value, so no data point's shape is hidden, only the marker clutter is reduced.


def fig_positioning_trend(df: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """Daily 3D RMS positioning error, 4 methods, median line (no band).

    Ported from `plot_trends`, part 1, then updated for the owner's 2026-08-28 decision
    (`docs/revision/positioning_reporting.md`): the mean is not a robust summary of this
    comparison (Sec 1 of that document), so the daily statistic is the median, and `df`
    carries no outcome-based filter - `_load_positioning_frame` no longer applies the old
    >10 m exclusion. The y-limit (0, 3.5 m) is hardcoded, as in the source, and does real
    clipping work: a handful of station-days run into the thousands of metres (genuine
    PPPx solve failures), and this view bound keeps the typical-case trend readable
    without dropping any of them from the underlying median computation or the written
    CSV.

    2026-09-14 (owner review): the shaded Q1-Q3 band is gone - `q1`/`q3` are still
    computed and still written to `pos_trend.csv` (useful context, and what
    `verification/gate_f_figures.py`'s Figure 12 check still cross-checks), just no
    longer drawn - and the line is thinner with sparser markers (see the module-level
    `_POSITIONING_TREND_*` constants). **No temporal smoothing was applied**, after
    deliberately considering it: a rolling window over the *daily median* would damp
    exactly the single-day spikes this figure's hardcoded 3.5 m ceiling exists to keep
    visible (see this docstring's previous paragraph, and PPPx solve-failure days
    specifically) - smoothing them away would misrepresent the trend it is showing, not
    just declutter it.

    2026-09-14 (owner review, second round): markers are gone entirely. They had been
    drawn on every sixth point, which reads as an inconsistency rather than as a series
    cue - a marker on some days and not others invites the question of what is special
    about those days, and nothing is. A glyph on all 242 daily points x 4 series is
    unreadable, so the series are distinguished by colour alone, which is what
    `APPROACH_COLORS` is for.

    2026-09-15 (owner instruction, results_register.md consistency item A): `df` is now
    the common set (N=10,387 station-days, all four methods solved under both weighting
    schemes) rather than the full per-method population, matching Tables 5-7 - see
    `_load_positioning_frame` and `stec.analysis.positioning_distributions`'s
    `common_set_daily_rows.csv`. The grouping/median computation below is unchanged;
    only which rows reach it moved.
    """
    daily = (
        df.groupby(["date", "method"])["error_3d_rms"]
        .agg(
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
            count="count",
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_POSITIONING_TREND)
    order = [m for m in _POSITIONING_ORDER if m in daily["method"].unique()]
    for i, method in enumerate(order):
        subset = daily[daily.method == method].sort_values("date")
        color = APPROACH_COLORS[method]
        ax.plot(
            subset["date"],
            subset["median"],
            linewidth=_POSITIONING_TREND_LINEWIDTH,
            marker="o",
            markersize=_POSITIONING_TREND_MARKERSIZE,
            color=color,
            label=method,
            zorder=len(order) - i,
        )
    ax.set_ylabel("3D RMS error [m]")
    ax.set_xlabel("Date")
    # Ticks land on month boundaries, so the day adds nothing and "%Y-%m-%d" cost two
    # pages of the manuscript: the longer rotated labels make the figure taller, and at
    # width=\linewidth that cascades through float placement.
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_ylim(0, 3.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=_POSITIONING_TREND_LEGEND_FONTSIZE)
    ax.set_title("Daily positioning accuracy (median), common set, unfiltered")
    _save(
        fig,
        "pos_trend",
        "positioning",
        output_dir,
        provenance,
        daily[["date", "method", "median", "q1", "q3", "count"]],
    )


def fig_positioning_improvement_timeseries(
    df: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Daily % improvement over IGS GIM + Mapping, for every other method, using each
    day's median 3D error rather than its mean.

    Ported from `plot_trends`, part 2, then updated for the owner's 2026-08-28 decision
    (`docs/revision/positioning_reporting.md`): `df` carries no outcome-based filter -
    `_load_positioning_frame` no longer applies the old >10 m exclusion - and the daily
    statistic behind each point is the median, not the mean, for the same robustness
    reason `fig_positioning_trend` switched. Draw order is alphabetical-then-reversed,
    exactly reproducing `daily_pivot.columns` (pandas sorts pivoted string columns
    alphabetically) fed through `model_cols[::-1]` in the source - computed here rather
    than hardcoded so it stays correct if a method's display name changes. The y-axis is
    clipped to +/-1.2x the pooled 99th percentile of |improvement| - the same
    view-only-clip convention `stec.viz.positioning_distributions.fig_cdf_unfiltered`
    uses - so a single day's extreme swing cannot compress the rest of the year onto a
    flat line; every value still counts toward the plotted median and the written CSV.

    2026-09-15 (owner instruction, results_register.md consistency item A): `df` is now
    the common set, same population and same reasoning as `fig_positioning_trend` above.
    """
    daily_median = df.groupby(["date", "method"])["error_3d_rms"].median()
    pivot = daily_median.unstack("method").sort_index()
    if "IGS GIM + Mapping" not in pivot.columns:
        logger.info("no IGS GIM + Mapping series; improvement timeseries skipped")
        return
    gim = pivot["IGS GIM + Mapping"]
    model_cols = sorted(c for c in pivot.columns if c != "IGS GIM + Mapping")

    rows = []
    for method in model_cols[::-1]:
        improvement = (gim - pivot[method]) / gim * 100
        rows.append(
            pd.DataFrame(
                {
                    "date": improvement.index,
                    "method": method,
                    "improvement_pct": improvement.values,
                }
            )
        )
    plotted = pd.concat(rows, ignore_index=True)
    view_limit = float(1.2 * plotted["improvement_pct"].abs().quantile(0.99))
    beyond_view = int((plotted["improvement_pct"].abs() > view_limit).sum())

    fig, ax = plt.subplots(figsize=FIGSIZE_POSITIONING_TREND)
    for method in model_cols[::-1]:
        subset = plotted[plotted.method == method].sort_values("date")
        ax.plot(
            subset["date"],
            subset["improvement_pct"],
            color=APPROACH_COLORS[method],
            linewidth=_POSITIONING_TREND_LINEWIDTH,
            label=f"Imp. by {method}",
        )
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylim(-view_limit, view_limit)
    ax.set_ylabel("Improvement over IGS GIM + Mapping [%]")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    ax.set_title(
        "Daily relative improvement over IGS GIM + Mapping (median), common set, "
        f"unfiltered\n{beyond_view} day(s) beyond the clipped view"
    )
    _save(
        fig,
        "pos_improvement_timeseries",
        "positioning",
        output_dir,
        provenance,
        plotted,
    )


def _build_positioning_figures(args: argparse.Namespace, output_dir: Path) -> None:
    """Figures 12-15. 12 and 14 are this module's own (no per-day equivalent exists in
    `positioning_distributions.py`); 13 and 15 call
    `stec.viz.positioning_distributions.fig_boxplot_3d_error`/`fig_cdf_unfiltered`
    directly against that module's own CSVs (owner decision, 2026-08-28,
    `docs/revision/positioning_reporting.md`) - the manuscript's distribution/CDF figures
    and the ones `positioning_distributions_figures` produces standalone are therefore the
    same rendering, not two implementations of the same boxplot/CDF that could drift
    apart. All four now read the common set (2026-09-15, results_register.md consistency
    item A) - N=10,387 station-days solved by all four methods under both weighting
    schemes, matching Tables 5-7 - rather than the full per-method population.
    """
    path = (
        analysis_dir(args.results_dir, "positioning_distributions")
        / "common_set_daily_rows.csv"
    )
    if not path.exists():
        logger.warning(
            f"{path} not found - run stec/analysis/positioning_distributions.py"
        )
        return
    df = _load_positioning_frame(path)
    n_station_days = len(df) // df["method"].nunique()
    prov = (
        f"{path} - SF-PPP, iono weighting, SINEX ground truth, 2024 test period, "
        f"common set (N={n_station_days:,} station-days solved by all four methods "
        "under both weighting schemes), no outcome-based outlier filter "
        "(docs/revision/positioning_reporting.md, docs/revision/results_register.md "
        "item A)"
    )
    fig_positioning_trend(df, output_dir, prov)
    fig_positioning_improvement_timeseries(df, output_dir, prov)

    distributions_dir = analysis_dir(args.results_dir, "positioning_distributions")
    box_stats_path = distributions_dir / "common_set_boxplot_stats.csv"
    box_fliers_path = distributions_dir / "common_set_boxplot_fliers.csv"
    box_exceedance_path = distributions_dir / "common_set_exceedance.csv"
    if (
        box_stats_path.exists()
        and box_fliers_path.exists()
        and box_exceedance_path.exists()
    ):
        fig_boxplot_3d_error(
            pd.read_csv(box_stats_path),
            pd.read_csv(box_fliers_path),
            output_dir,
            f"{box_stats_path} (stec.analysis.positioning_distributions) - {prov}",
            pd.read_csv(box_exceedance_path),
        )
    else:
        logger.warning(
            f"positioning box stats/fliers/exceedance not found under "
            f"{distributions_dir} - run stec/analysis/positioning_distributions.py"
        )

    cdf_path = distributions_dir / "common_set_cdf_points.csv"
    percentile_path = distributions_dir / "common_set_percentile_summary.csv"
    if cdf_path.exists() and percentile_path.exists():
        fig_cdf_unfiltered(
            pd.read_csv(cdf_path),
            pd.read_csv(percentile_path),
            output_dir,
            f"{cdf_path} (stec.analysis.positioning_distributions) - {prov}",
        )
    else:
        logger.warning(
            f"{cdf_path} not found - run stec/analysis/positioning_distributions.py"
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

FIGURE_BUILDERS = (
    _build_temporal_split_figure,
    _build_spatial_split_figure,
    _build_pretrained_diagnostics_figures,
    _build_improvement_by_date_figures,
    _build_mae_rmse_finetuned_figure,
    _build_positioning_figures,
)


def build_all(args: argparse.Namespace) -> None:
    configure_plotting()
    for build in FIGURE_BUILDERS:
        build(args, args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("plots/manuscript"))
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
