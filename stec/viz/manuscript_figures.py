"""The manuscript's own numbered figures - Figures 1, 2, 10 and 12-15 of the 15 defined
in `STEC_Modelling/PNN_main.tex`.

Distinct from `revision_figures.py`, which builds a separate set for the JGR-MLC response
letter - none of its ~19 figure kinds correspond to a numbered manuscript figure. This
module is the other half: reproducing the figures the paper itself embeds. See
`docs/revision/figure_coverage.md` for the audit that maps all 15 to their pre-rebuild
generator and confirmed none had a `stec/` counterpart before this module.

Coverage
--------
Of the 15, this module ports 7:

  Figure  1  fig_temporal_split                  train/val/test split, DOY 2014-2024
  Figure  2  fig_spatial_split                    train/val/test stations on a world map
  Figure 10  fig_improvement_by_date              daily % RMSE/MAE improvement vs. date
  Figure 12  fig_positioning_trend                daily 3D RMS, 4 methods, SEM band
  Figure 13  fig_positioning_distribution_boxplot  overall 3D RMS distribution
  Figure 14  fig_positioning_improvement_timeseries daily % improvement over GIM
  Figure 15  fig_positioning_cdf_3d_rms            3D RMS CDF, 4 methods

Figure 3 (`network`) is hand-drawn (`docs/ResNet.drawio`) and needs no code.

**Not ported - Figures 4-9 and 11.** Figures 4-9 (`src/viz/{performance,distributions,
spatial,uncertainty}.py`) all consume the full per-observation test-set dataframe that
`src/inference_testset.py` assembles from the prediction store: residual boxplots by
elevation/latitude/local-time/date and the uncertainty-vs-error panel need every
observation's residual and bin membership, not an aggregate. No aggregate CSV in
`multiday_results/` carries that (`daily_metrics_rebuilt/per_day.csv` is whole-day RMSE/
MAE only, no per-observation binning). Building them requires reading the prediction
store, which this port is explicitly barred from doing. Figure 11
(`mae_rmse_finetuned`, `src/multiday_evaluation.py::extract_elevation_metrics_from_
experiment`) is the same shape of problem one level removed: it needs per-day,
per-elevation-bin RMSE/MAE **and their std across days** (the error bars in the
published figure), which only exists by reading each day's `detailed_predictions.csv` -
itself per-observation, store-scale data. The only elevation-binned aggregate that exists
anywhere (`multiday_results/bugfix_effects/stratified_comparison_rebuilt/by_elevation.csv`,
confirmed to share `stec/analysis/stratified_comparison.py`'s exact output schema) pools
all 242 days into one RMSE/MAE value per bin with no variance term - building the figure
from it would mean dropping the error bars, which is a different figure, not a port of
this one. Both are reported here rather than approximated.

Colour rule
-----------
Approach colours come from `stec.viz.style.APPROACH_COLORS` only - never a literal hex,
never seaborn's `colorblind` palette. This matters here specifically because the
pre-rebuild generator for Figures 10 and 11 (`src/multiday_evaluation.py`) colours its four
series from `seaborn.color_palette("colorblind")` indices 0/1/2/4, not the pinned hex
values - a divergence from the already-published figures that this port does not carry
over. Figures 12-15's pre-rebuild generator (`positioning/scripts/plot_results.py`)
already used the same hex constants `style.py` pins, so no divergence exists there.

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
from ..positioning.metrics import OUTLIER_3D_RMS_M, exclude_outlier_station_days
from .revision_figures import analysis_dir
from .style import APPROACH_COLORS, FIGSIZE_WIDE, configure_plotting

import matplotlib.pyplot as plt  # noqa: E402  (style.py sets the Agg backend on import)

logger = logging.getLogger(__name__)

SOURCE_DIRS = {
    "dataset": "dataset_construction",
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
    """
    pivot = daily.pivot(index="date", columns="Model", values=metric).sort_index()
    stec = pivot["Direct STEC"]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
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
        f"improvements_{metric.lower()}",
        "finetuned",
        output_dir,
        provenance,
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
    )


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
        for metric in ("RMSE", "MAE"):
            days = dataset_table["date"].nunique()
            prov = (
                f"{path} - daily fine-tuned models, {dataset}, {days} test days of 2024"
            )
            fig_improvement_by_date(dataset_table, metric, output_dir, prov)


# --------------------------------------------------------------------------
# Figures 12-15 - SF-PPP positioning, 4 methods, 2024 test period
# --------------------------------------------------------------------------

# Marker shapes ported unchanged from positioning/scripts/plot_results.py::get_style.
_POSITIONING_MARKERS = {
    "Direct STEC": "o",
    "VTEC + Mapping": "s",
    "IGS GIM + Mapping": "^",
    "Pretrained Direct STEC": "d",
}
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
    """Read one station-day per row, normalise method names, apply the paper's 10 m rule.

    `exclude_outlier_station_days`/`OUTLIER_3D_RMS_M` are `stec.positioning.metrics`'s -
    reused rather than reimplemented so the >10 m exclusion behind Figures 12-15 and
    Table 5 stays one rule in one place, not two copies that could drift apart.
    """
    frame = pd.read_csv(path, usecols=["date", "method", "error_3d_rms"])
    frame["method"] = frame["method"].map(_POSITIONING_METHOD_MAP)
    frame = frame.dropna(subset=["method"])
    frame["date"] = pd.to_datetime(frame["date"])
    return exclude_outlier_station_days(frame)


def fig_positioning_trend(df: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """Daily 3D RMS positioning error, 4 methods, mean +/- SEM across stations.

    Ported from `plot_trends`, part 1. The y-limit (0, 3.5 m) is hardcoded in the source
    regardless of the data range and is kept as-is.
    """
    daily = (
        df.groupby(["date", "method"])["error_3d_rms"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    daily["sem"] = daily["std"] / np.sqrt(daily["count"])

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    order = [m for m in _POSITIONING_ORDER if m in daily["method"].unique()]
    for i, method in enumerate(order):
        subset = daily[daily.method == method].sort_values("date")
        color = APPROACH_COLORS[method]
        ax.plot(
            subset["date"],
            subset["mean"],
            marker=_POSITIONING_MARKERS[method],
            markersize=6,
            color=color,
            label=method,
            zorder=len(order) - i,
        )
        ax.fill_between(
            subset["date"],
            subset["mean"] - subset["sem"],
            subset["mean"] + subset["sem"],
            color=color,
            alpha=0.2,
        )
    ax.set_ylabel("3D RMS error [m]")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_ylim(0, 3.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("Daily positioning accuracy")
    _save(
        fig,
        "pos_trend",
        "positioning",
        output_dir,
        provenance,
        daily[["date", "method", "mean", "std", "count", "sem"]],
    )


def fig_positioning_improvement_timeseries(
    df: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Daily % improvement over IGS GIM + Mapping, for every other method.

    Ported from `plot_trends`, part 2. Draw order is alphabetical-then-reversed, exactly
    reproducing `daily_pivot.columns` (pandas sorts pivoted string columns
    alphabetically) fed through `model_cols[::-1]` in the source - computed here rather
    than hardcoded so it stays correct if a method's display name changes.
    """
    daily_mean = df.groupby(["date", "method"])["error_3d_rms"].mean()
    pivot = daily_mean.unstack("method").sort_index()
    if "IGS GIM + Mapping" not in pivot.columns:
        logger.info("no IGS GIM + Mapping series; improvement timeseries skipped")
        return
    gim = pivot["IGS GIM + Mapping"]
    model_cols = sorted(c for c in pivot.columns if c != "IGS GIM + Mapping")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    rows = []
    for method in model_cols[::-1]:
        improvement = (gim - pivot[method]) / gim * 100
        ax.plot(
            improvement.index,
            improvement.values,
            marker=_POSITIONING_MARKERS[method],
            markersize=4,
            color=APPROACH_COLORS[method],
            label=f"Imp. by {method}",
        )
        rows.append(
            pd.DataFrame(
                {
                    "date": improvement.index,
                    "method": method,
                    "improvement_pct": improvement.values,
                }
            )
        )
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("Improvement over IGS GIM + Mapping [%]")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    ax.set_title("Daily relative improvement over IGS GIM + Mapping")
    _save(
        fig,
        "pos_improvement_timeseries",
        "positioning",
        output_dir,
        provenance,
        pd.concat(rows, ignore_index=True),
    )


def fig_positioning_distribution_boxplot(
    df: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Overall 3D RMS error distribution, 4 methods, one box per method.

    Ported from `plot_extended_analysis`, part 1. The source draws this with
    `seaborn.boxplot`; this uses `Axes.boxplot` directly instead of adding seaborn as a
    dependency for what is, for a single non-hued boxplot, a styling wrapper around it -
    same box/whisker statistics (median, IQR, 1.5x IQR whiskers, fliers hidden), same
    per-method colours and order.
    """
    order = [m for m in _POSITIONING_ORDER if m in df["method"].unique()]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    data = [df.loc[df.method == m, "error_3d_rms"].to_numpy() for m in order]
    boxes = ax.boxplot(
        data,
        tick_labels=order,
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black"},
    )
    for patch, method in zip(boxes["boxes"], order):
        patch.set_facecolor(APPROACH_COLORS[method])
    ax.set_ylabel("3D RMS error [m]")
    ax.set_xlabel("Correction method")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_title("Overall positioning accuracy distribution")
    _save(
        fig,
        "pos_distribution_boxplot",
        "positioning",
        output_dir,
        provenance,
        df[df.method.isin(order)][["method", "error_3d_rms"]],
    )


def fig_positioning_cdf_3d_rms(
    df: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """CDF of 3D RMS positioning error, 4 methods.

    Ported from `plot_extended_analysis`, part 2, with `threshold_cm=None` - the
    x-axis-limit branch that only fires when `plot_results.py --exclude_threshold` is
    passed. CLAUDE.md's own reproduction command
    (`plot_results.py --input multiday_results/positioning_comparison_3way/
    multiday_summary.csv`) does not pass it, so the published figure used the robust-limit
    branch (98th percentile x1.2) this reproduces, not a hardcoded x-axis cap.
    """
    order = [m for m in _POSITIONING_ORDER if m in df["method"].unique()]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    robust_max = 0.0
    rows = []
    for method in order:
        values = df.loc[df.method == method, "error_3d_rms"].dropna().sort_values()
        if values.empty:
            continue
        percentile = np.arange(1, len(values) + 1) / len(values) * 100
        ax.plot(
            values,
            percentile,
            label=method,
            linewidth=2.5,
            color=APPROACH_COLORS[method],
        )
        robust_max = max(robust_max, float(np.percentile(values, 98)))
        rows.append(
            pd.DataFrame(
                {
                    "method": method,
                    "error_3d_rms": values.values,
                    "cumulative_pct": percentile,
                }
            )
        )
    ax.set_xlabel("3D RMS error [m]")
    ax.set_ylabel("Cumulative probability [%]")
    ax.set_xlim(0, robust_max * 1.2)
    ax.set_ylim(0, 102)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Positioning error cumulative distribution")
    _save(
        fig,
        "pos_cdf_3d_rms",
        "positioning",
        output_dir,
        provenance,
        pd.concat(rows, ignore_index=True),
    )


def _build_positioning_figures(args: argparse.Namespace, output_dir: Path) -> None:
    path = (
        analysis_dir(args.results_dir, "positioning_coverage") / "multiday_summary.csv"
    )
    if not path.exists():
        logger.warning(f"{path} not found - run stec/analysis/positioning_coverage.py")
        return
    raw = pd.read_csv(path, usecols=["method"])
    kept_methods = raw["method"].isin(_POSITIONING_METHOD_MAP).sum()
    df = _load_positioning_frame(path)
    prov = (
        f"{path} - SF-PPP, iono weighting, SINEX ground truth, 2024 test period, "
        f"{len(df):,} of {kept_methods:,} station-days after the {OUTLIER_3D_RMS_M:g} m "
        "outlier rule"
    )
    fig_positioning_trend(df, output_dir, prov)
    fig_positioning_improvement_timeseries(df, output_dir, prov)
    fig_positioning_distribution_boxplot(df, output_dir, prov)
    fig_positioning_cdf_3d_rms(df, output_dir, prov)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

FIGURE_BUILDERS = (
    _build_temporal_split_figure,
    _build_spatial_split_figure,
    _build_improvement_by_date_figures,
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
        default=Path("multiday_results"),
        help="Root that all figure inputs below are resolved against.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    build_all(args)


if __name__ == "__main__":
    main()
