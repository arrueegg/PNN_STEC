"""Diagnostic-plot parity: everything `src/viz/{spatial,performance,distributions,
uncertainty}.py`'s `plot_test_metrics` chain drew that has no `stec/` counterpart yet.

The project owner's requirement is explicit: `stec/` must be able to reproduce every plot
`src/` produced, "if needed" - not only the 14 numbered manuscript figures
(`stec.viz.manuscript_figures`) or the JGR-MLC response-letter figures
(`stec.viz.revision_figures`). A parity audit found ~20 diagnostic plots reachable from
`cli.py inference -> src/inference_testset.py -> plot_test_metrics()` with no `stec/`
equivalent. This module is that port.

Coverage
--------

  src function                                       stec function(s)
  spatial.py::plot_spatial_error_map                 fig_spatial_error_map (3 PNGs)
  spatial.py::plot_spatial_error_map_by_local_time    fig_spatial_error_map_by_local_time
  spatial.py::plot_solar_magnetic_ipp_error_map       NOT PORTED - see below
  performance.py::plot_prediction_scatter             fig_prediction_scatter
  performance.py::plot_az_el_heatmap                  fig_az_el_heatmap (residual, mae)
  performance.py::plot_residuals_vs_date               fig_residuals_vs_date
  distributions.py::plot_residuals_vs_feature          fig_residuals_vs_feature x8 (doy,
                                                        time, satazi, kp, f107, dst,
                                                        target_stec, pred_stec - satele is
                                                        already manuscript Figure 5)
  distributions.py::plot_histogram_of_residuals        fig_histogram_of_residuals
  distributions.py::plot_residuals_vs_solar_indices    fig_residuals_vs_solar_indices
  uncertainty.py::plot_uncertainty_calibration_binned  fig_uncertainty_calibration_binned
  uncertainty.py::plot_uncertainty_calibration         fig_uncertainty_calibration
  uncertainty.py::plot_coverage_probability            fig_coverage_probability
  uncertainty.py::plot_sigma_coverage_comparison       fig_sigma_coverage_comparison
  uncertainty.py::plot_uncertainty_distributions       fig_uncertainty_distributions

Not ported: `plot_solar_magnetic_ipp_error_map` needs `sm_lon_ipp`, which the real
`predictions/pretrained_stec/own` store does not carry (checked directly against the
store's own parquet schema, 2026-08-25 - only `sm_lat_ipp` is present; the geometry was
apparently never written for this partition). Deriving it would mean a live
station/IPP solar-magnetic coordinate transform over 10,000,000 rows, which is out of
scope for a no-GPU session and is the same class of cost CLAUDE.md's own
`sm_lat_ipp`-offset gotcha describes for the analogous problem elsewhere in this
codebase. `plot_binned_uncertainty_error_analysis` (uncertainty.py) is also not ported:
it is the same "absolute error vs. predicted-uncertainty bin" analysis
`stec.viz.manuscript_figures.fig_uncertainty` (manuscript Figure 9) already provides for
this exact model/dataset, so porting it again would be a second copy of Figure 9's logic,
not new coverage. `plot_residuals_vs_feature_clipped`'s axis-clipped variant of
`target_stec` was not requested and is not ported either.

A src-side bug intentionally not carried over: `plot_uncertainty_distributions`
(uncertainty.py) and `plot_binned_uncertainty_analysis` (uncertainty.py, not ported here
at all - see above) both write to the literal filename `uncertainty_distributions.png`,
so whichever runs second in `plot_test_metrics` silently clobbers the first - two
different histograms sharing one name on disk. Only one of the two source functions is
ported here (`plot_uncertainty_distributions`, the three-component total/epistemic/
aleatoric one - `plot_binned_uncertainty_analysis` is a strict subset of Figure 9's
"Mean predicted uncertainty" curve, not new information), so this collision cannot
recur in this module, but the filename is otherwise unchanged from the source.

A second src-side bug, fixed rather than reproduced: `plot_uncertainty_calibration_binned`
references an undefined name `ax` (should be `ax1`) between creating its two-panel figure
and adding the legend, which raises `NameError` on the one branch of `src/viz/__init__.py`
that calls it (`plot_comprehensive_uncertainty_analysis`, itself unreachable from
`plot_test_metrics` - grepped, 2026-08-25 - so this bug has apparently never fired in
production). `fig_uncertainty_calibration_binned` below uses `ax1`, the panel the labels
were clearly meant for.

Data source
-----------
Same store, same read pattern as manuscript Figures 4-9: `predictions/pretrained_stec/own`
(544 days, 2014-2024, 10,000,000 observations), streamed one day at a time via
`prediction_store.iter_days` and narrowed to a fixed column list before concatenation -
see `stec.analysis.pretrained_test_diagnostics`'s docstring for the full accounting of why
these plots need real per-observation values (they are boxplots, heatmaps and a 2D
histogram, none of which can be built from a running sum) and why holding the narrowed
result in memory is a bounded, disclosed cost rather than the whole-store read that
OOM-killed the pre-rebuild driver.

This module reads a *second*, wider cache
(`stec.analysis.diagnostic_test_observations`) rather than extending
`pretrained_test_diagnostics.WANTED_COLUMNS` in place: `stec.viz.manuscript_figures`
(Figures 4-9) reads that module's cache directly, and both files were being worked on
elsewhere in this codebase at the time this module was written, so widening the shared
column list - and doubling its on-disk size for the Figures 4-9 caller that does not need
the extra columns - was avoided in favour of a second, self-contained pass over the same
544 files. See that module's docstring for the exact columns and the measured cost.

Colour rule
-----------
None of these are approach comparisons (there is no "Direct STEC vs. VTEC vs. GIM vs.
Pretrained" series anywhere in this module) - they are single-model residual/spatial/
uncertainty diagnostics of the pretrained model's own held-out test set, the same category
manuscript Figures 4-9 are in. `stec.viz.style.APPROACH_COLORS` is therefore not imported
here, for the same reason `manuscript_figures.py`'s own docstring gives for Figures 4-9:
applying an approach colour to a quantity that is not an approach would be the category
error the colour rule exists to prevent, from the other direction. Colours are literal
matplotlib names/hexes, ported unchanged from the source modules.

No in-plot explanatory text beyond axis labels, legends and a title - the `_notitle`
copy `_save` writes (via `stec.viz.style.save_plot`) is the clean version; the titled copy
is the working one.

Usage::

    python -m stec.viz.diagnostic_figures
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm, pearsonr

from ..analysis import diagnostic_test_observations as dto
from ..analysis.stratified_comparison import add_local_time
from .style import (
    FIGSIZE_HISTOGRAM,
    FIGSIZE_QUAD,
    FIGSIZE_SQUARE,
    FIGSIZE_WIDE,
    configure_plotting,
    save_plot,
)

import matplotlib.pyplot as plt  # noqa: E402  (style.py sets the Agg backend on import)

logger = logging.getLogger(__name__)

CATEGORY_DIRS = {
    "spatial": "spatial_analysis",
    "performance": "prediction_analysis",
    "temporal": "temporal_analysis",
    "feature": "feature_analysis",
    "uncertainty": "uncertainty_analysis",
}

DEFAULT_OUTPUT_DIR = Path("plots/diagnostics")


def _save(
    fig: plt.Figure, filename: str, category: str, output_dir: Path, data: pd.DataFrame
) -> dict:
    """Write `filename` (+ `_notitle`) via `stec.viz.style.save_plot`, plus the plotted
    data as CSV - the comparison surface a future equivalence gate needs. Returns a
    manifest record; `build_all` collects one of these per figure written."""
    target = output_dir / CATEGORY_DIRS[category]
    target.mkdir(parents=True, exist_ok=True)
    data.to_csv(target / f"{Path(filename).stem}.csv", index=False)
    save_plot(fig, filename, target)
    logger.info(
        f"wrote {target / filename} (+ _notitle, + .csv, {len(data):,} data rows)"
    )
    return {
        "figure": Path(filename).stem,
        "category": category,
        "filename": str(target / filename),
        "n_data_rows": len(data),
    }


# --------------------------------------------------------------------------
# Shared boxplot styling - duplicated from `stec.viz.manuscript_figures` deliberately
# (same reasoning that module gives for duplicating it from `revision_figures.py`): the
# two figure sets are worked on independently and importing across that boundary would
# couple them for a few lines of styling.
# --------------------------------------------------------------------------


def _style_boxplot(bp: dict) -> None:
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


def _mae_rmse_overlay(ax, positions, mae_values, rmse_values) -> None:
    ax.plot(
        positions, mae_values, color="green", marker="o", linewidth=3, markersize=8,
        label="MAE", alpha=0.9, zorder=10,
    )  # fmt: skip
    ax.plot(
        positions, rmse_values, color="orange", marker="s", linewidth=3, markersize=8,
        label="RMSE", alpha=0.9, zorder=10,
    )  # fmt: skip


def _binned_residual_stats(
    residual: np.ndarray, values: np.ndarray, bin_edges: np.ndarray
) -> list[dict]:
    """Per-bin residual boxes plus MAE/RMSE/count, dropping bins with no observations -
    the same binning `src/viz/distributions.py::plot_binned_boxplot` does, generalised
    to any feature axis."""
    bin_series = pd.cut(values, bins=bin_edges, include_lowest=True)
    frame = pd.DataFrame({"bin": bin_series, "residual": residual})
    stats = []
    for interval, group in frame.groupby("bin", observed=True):
        vals = group["residual"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        stats.append(
            {
                "bin_left": float(interval.left),
                "bin_right": float(interval.right),
                "bin_center": float((interval.left + interval.right) / 2),
                "values": vals,
                "mae": float(np.abs(vals).mean()),
                "rmse": float(np.sqrt(np.mean(vals**2))),
                "n": int(vals.size),
            }
        )
    stats.sort(key=lambda s: s["bin_center"])
    return stats


# --------------------------------------------------------------------------
# Spatial error maps - src/viz/spatial.py
# --------------------------------------------------------------------------

_SPATIAL_LAT_EDGES = np.linspace(-90, 90, 37)  # 5-degree bins
_SPATIAL_LON_EDGES = np.linspace(-180, 180, 73)  # 5-degree bins
_SPATIAL_MIN_COUNT = 10  # matches plot_spatial_error_map's own >=10-per-bin filter


def _spatial_bin_table(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    min_count: int,
) -> pd.DataFrame:
    residual = df["true_stec"] - df["stec_pred"]
    mae = residual.abs()
    lat_bin = pd.cut(df[lat_col], bins=lat_edges, include_lowest=True)
    lon_bin = pd.cut(df[lon_col], bins=lon_edges, include_lowest=True)
    grouped = (
        pd.DataFrame(
            {"lat_bin": lat_bin, "lon_bin": lon_bin, "mae": mae, "residual": residual}
        )
        .groupby(["lat_bin", "lon_bin"], observed=True)
        .agg(
            mae_mean=("mae", "mean"),
            residual_mean=("residual", "mean"),
            count=("mae", "size"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["count"] >= min_count].copy()
    grouped["lat_center"] = grouped["lat_bin"].apply(lambda b: b.mid).astype(float)
    grouped["lon_center"] = grouped["lon_bin"].apply(lambda b: b.mid).astype(float)
    return grouped.drop(columns=["lat_bin", "lon_bin"])


def fig_spatial_error_map(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    """Geographic MAE / mean-residual / sample-count maps, 5-degree lat/lon bins.

    Ported from `src/viz/spatial.py::plot_spatial_error_map`. `df` needs `lat_ipp`,
    `lon_ipp`, `true_stec`, `stec_pred`. Writes three PNGs, one per statistic - the
    source's own three-file output.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if "lat_ipp" not in df.columns or "lon_ipp" not in df.columns:
        logger.warning("no lat_ipp/lon_ipp - skipping spatial_error_map")
        return []

    stats = _spatial_bin_table(
        df,
        "lat_ipp",
        "lon_ipp",
        _SPATIAL_LAT_EDGES,
        _SPATIAL_LON_EDGES,
        _SPATIAL_MIN_COUNT,
    )
    if stats.empty:
        logger.warning(
            f"no spatial bin reached the >= {_SPATIAL_MIN_COUNT} minimum count - "
            "skipping spatial_error_map"
        )
        return []

    records = []
    for column, cmap, label, filename, symmetric in (
        ("mae_mean", "viridis", "MAE [TECU]", "spatial_error_map_mae.png", False),
        (
            "residual_mean",
            "RdBu_r",
            "Residual [TECU]",
            "spatial_error_map_residual.png",
            True,
        ),
        ("count", "plasma", "Count", "spatial_error_map_count.png", False),
    ):
        fig, ax = plt.subplots(
            figsize=FIGSIZE_WIDE, subplot_kw={"projection": ccrs.PlateCarree()}
        )
        limits = {}
        if symmetric:
            vmax = float(stats[column].abs().max())
            limits = {"vmin": -vmax, "vmax": vmax}
        scatter = ax.scatter(
            stats["lon_center"], stats["lat_center"], c=stats[column], s=50, cmap=cmap,
            transform=ccrs.PlateCarree(), alpha=0.8, **limits,
        )  # fmt: skip
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_global()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label(label)
        ax.set_title(f"Spatial analysis: {label.split(' [')[0]}")
        data = stats[["lat_center", "lon_center", column, "count"]].rename(
            columns={column: "value"}
        )
        records.append(_save(fig, filename, "spatial", output_dir, data))
    return records


_LOCAL_TIME_PERIODS = {
    "Dawn (04-08h)": [(4, 8)],
    "Day (08-16h)": [(8, 16)],
    "Dusk (16-20h)": [(16, 20)],
    "Night (20-04h)": [(20, 24), (0, 4)],
}
_LOCAL_TIME_LAT_EDGES = np.linspace(-90, 90, 19)  # 10-degree bins
_LOCAL_TIME_LON_EDGES = np.linspace(-180, 180, 37)  # 10-degree bins


def fig_spatial_error_map_by_local_time(
    df: pd.DataFrame, output_dir: Path
) -> list[dict]:
    """MAE map split into four local-time sectors (dawn/day/dusk/night), one 2x2 figure.

    Ported from `src/viz/spatial.py::plot_spatial_error_map_by_local_time`. `df` needs
    `lat_ipp`, `lon_ipp`, `true_stec`, `stec_pred` and `local_time_hours` (already added
    by `build_all` via `add_local_time` before this runs). No per-bin minimum count -
    the source does not apply one here, unlike `fig_spatial_error_map`.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if "lat_ipp" not in df.columns or "lon_ipp" not in df.columns:
        logger.warning("no lat_ipp/lon_ipp - skipping spatial_error_by_local_time")
        return []
    if "local_time_hours" not in df.columns:
        logger.warning(
            "no local_time_hours (and no sod/lon_ipp to derive it) - "
            "skipping spatial_error_by_local_time"
        )
        return []

    mae_all = (df["true_stec"] - df["stec_pred"]).abs()
    vmax = float(mae_all.quantile(0.95))
    local_time = df["local_time_hours"].to_numpy(dtype=float)

    fig = plt.figure(figsize=FIGSIZE_QUAD)
    rows: list[dict] = []
    for panel, (period_name, ranges) in enumerate(_LOCAL_TIME_PERIODS.items(), start=1):
        mask = np.zeros(len(df), dtype=bool)
        for start, end in ranges:
            mask |= (local_time >= start) & (local_time < end)
        subset = df.loc[mask]

        ax = fig.add_subplot(2, 2, panel, projection=ccrs.PlateCarree())
        if not subset.empty:
            stats = _spatial_bin_table(
                subset, "lat_ipp", "lon_ipp",
                _LOCAL_TIME_LAT_EDGES, _LOCAL_TIME_LON_EDGES, min_count=1,
            )  # fmt: skip
            if not stats.empty:
                scatter = ax.scatter(
                    stats["lon_center"], stats["lat_center"], c=stats["mae_mean"], s=80,
                    cmap="viridis", transform=ccrs.PlateCarree(), alpha=0.8,
                    vmin=0, vmax=vmax,
                )  # fmt: skip
                divider = make_axes_locatable(ax)
                cax = divider.append_axes(
                    "right", size="5%", pad=0.1, axes_class=plt.Axes
                )
                cbar = fig.colorbar(scatter, cax=cax)
                cbar.set_label("MAE [TECU]")
                rows.extend(
                    {
                        "period": period_name,
                        "lat_center": r.lat_center,
                        "lon_center": r.lon_center,
                        "mae": r.mae_mean,
                        "count": r.count,
                    }
                    for r in stats.itertuples()
                )
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_global()
        ax.set_title(f"{period_name}\n(N={len(subset):,})")

    fig.suptitle("Spatial analysis: MAE by local-time sector")
    data = pd.DataFrame(
        rows, columns=["period", "lat_center", "lon_center", "mae", "count"]
    )
    return [_save(fig, "spatial_error_by_local_time.png", "spatial", output_dir, data)]


# --------------------------------------------------------------------------
# Azimuth/elevation heatmaps - src/viz/performance.py::plot_az_el_heatmap
# --------------------------------------------------------------------------

_AZ_EDGES = np.linspace(0, 360, 37)  # 10-degree bins
_EL_EDGES = np.linspace(5, 90, 18)  # 5-degree bins


def fig_az_el_heatmap(df: pd.DataFrame, output_dir: Path, metric: str) -> list[dict]:
    """Mean residual or MAE by (azimuth, elevation) bin, `metric` in {"residual", "mae"}.

    Ported from `src/viz/performance.py::plot_az_el_heatmap`. `df` needs `satazi`,
    `satele`, `true_stec`, `stec_pred`.
    """
    if metric not in ("residual", "mae"):
        raise ValueError(f"metric must be 'residual' or 'mae', got {metric!r}")
    if "satazi" not in df.columns or "satele" not in df.columns:
        logger.warning(
            f"no satazi/satele - skipping {metric}_azimuth_elevation_heatmap"
        )
        return []

    residual = df["true_stec"] - df["stec_pred"]
    value = residual.abs() if metric == "mae" else residual

    az_bin = pd.cut(df["satazi"], bins=_AZ_EDGES, include_lowest=True)
    el_bin = pd.cut(df["satele"], bins=_EL_EDGES, include_lowest=True)
    frame = pd.DataFrame({"az_bin": az_bin, "el_bin": el_bin, "value": value})
    grid = frame.groupby(["el_bin", "az_bin"], observed=False)["value"].mean().unstack()
    counts = (
        frame.groupby(["el_bin", "az_bin"], observed=False)["value"].size().unstack()
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    cmap = "RdBu_r" if metric == "residual" else "viridis"
    array = grid.to_numpy(dtype=float)
    im = ax.imshow(
        array, cmap=cmap, aspect="auto", origin="lower", interpolation="bilinear"
    )
    if metric == "residual":
        finite = array[np.isfinite(array)]
        if finite.size:
            vmax = float(np.abs(finite).max())
            im.set_clim(-vmax, vmax)

    az_ticks = np.arange(0, len(grid.columns), 6)
    el_ticks = np.arange(0, len(grid.index), 3)
    ax.set_xticks(az_ticks)
    ax.set_yticks(el_ticks)
    ax.set_xticklabels([f"{int(_AZ_EDGES[i]):d}" for i in az_ticks])
    ax.set_yticklabels([f"{int(_EL_EDGES[i]):d}" for i in el_ticks])
    ax.set_xlabel("Azimuth [deg]")
    ax.set_ylabel("Elevation [deg]")
    label = "Residual [TECU]" if metric == "residual" else "MAE [TECU]"
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(label)
    title = "Residual analysis" if metric == "residual" else "Spatial analysis"
    ax.set_title(f"{title}: azimuth & elevation")

    rows = [
        {
            "az_center": float(az_interval.mid),
            "el_center": float(el_interval.mid),
            "value": float(grid.loc[el_interval, az_interval]),
            "count": int(counts.loc[el_interval, az_interval]),
        }
        for el_interval in grid.index
        for az_interval in grid.columns
        if pd.notna(grid.loc[el_interval, az_interval])
    ]
    filename = f"{metric}_azimuth_elevation_heatmap.png"
    return [_save(fig, filename, "spatial", output_dir, pd.DataFrame(rows))]


# --------------------------------------------------------------------------
# Prediction scatter - src/viz/performance.py::plot_prediction_scatter
# --------------------------------------------------------------------------


def fig_prediction_scatter(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    """2D log-count histogram of predicted vs. true STEC, 50x50 bins, with the 1:1 line.

    Ported from `src/viz/performance.py::plot_prediction_scatter` (distinct from
    manuscript Figure 4's hexbin density - a different binning/rendering choice already
    present in the source, kept as its own figure). `df` needs `true_stec`, `stec_pred`.
    The saved data is the rendered histogram's own bin grid (50x50 = <=2,500 rows), not
    the underlying ~10,000,000 points - it is exactly what the figure draws and is the
    right size for a comparison CSV.
    """
    true_stec = df["true_stec"].to_numpy(dtype=float)
    pred_stec = df["stec_pred"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    counts, xedges, yedges, image = ax.hist2d(
        true_stec, pred_stec, bins=50, cmap="Blues", norm=mcolors.LogNorm()
    )
    min_val = float(min(true_stec.min(), pred_stec.min()))
    max_val = float(max(true_stec.max(), pred_stec.max()))
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r-", linewidth=3, alpha=0.8,
        label="Perfect prediction",
    )  # fmt: skip
    ax.set_xlabel("True STEC [TECU]")
    ax.set_ylabel("Predicted STEC [TECU]")
    ax.legend(framealpha=0.9)
    fig.colorbar(image, ax=ax, label="Count (log scale)")
    ax.set_title("Prediction analysis: predicted vs. observed STEC")

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    rows = [
        {
            "true_stec_bin_center": float(x_centers[i]),
            "pred_stec_bin_center": float(y_centers[j]),
            "count": int(counts[i, j]),
        }
        for i in range(counts.shape[0])
        for j in range(counts.shape[1])
        if counts[i, j] > 0
    ]
    return [
        _save(
            fig, "prediction_scatter.png", "performance", output_dir, pd.DataFrame(rows)
        )
    ]


# --------------------------------------------------------------------------
# Residuals vs. date - src/viz/performance.py::plot_residuals_vs_date
# --------------------------------------------------------------------------


def fig_residuals_vs_date(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    """Monthly residual boxplots + MAE/RMSE lines + sample-count bars, 3 panels.

    Ported from `src/viz/performance.py::plot_residuals_vs_date`. `df` needs `year`,
    `doy`, `true_stec`, `stec_pred`. Distinct from manuscript Figure 8
    (`fig_residuals_year_month`, ported from `plot_box_by_date`): that is a single-panel
    boxplot-only figure; this one adds the MAE/RMSE and sample-count panels the source
    keeps as a second, separate function.
    """
    if "year" not in df.columns or "doy" not in df.columns:
        logger.warning("no year/doy - skipping residuals_vs_date")
        return []

    residual = (df["true_stec"] - df["stec_pred"]).to_numpy(dtype=float)
    date = pd.to_datetime(df["year"], format="%Y") + pd.to_timedelta(
        df["doy"] - 1, unit="D"
    )
    month = date.dt.to_period("M")

    monthly = []
    for period, group in pd.DataFrame({"month": month, "residual": residual}).groupby(
        "month", observed=True
    ):
        vals = group["residual"].to_numpy(dtype=float)
        monthly.append(
            {
                "month": str(period),
                "date": period.start_time,
                "count": int(vals.size),
                "mae": float(np.abs(vals).mean()),
                "rmse": float(np.sqrt(np.mean(vals**2))),
                "values": vals,
            }
        )
    monthly.sort(key=lambda m: m["date"])
    if not monthly:
        logger.warning("no month has data - skipping residuals_vs_date")
        return []

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=FIGSIZE_WIDE, sharex=True)
    positions = list(range(len(monthly)))
    box_positions = [i for i, m in enumerate(monthly) if m["count"] >= 5]
    box_data = [m["values"] for m in monthly if m["count"] >= 5]
    if box_data:
        bp = ax1.boxplot(
            box_data, positions=box_positions, widths=0.8, showfliers=False,
            patch_artist=True,
        )  # fmt: skip
        _style_boxplot(bp)
    ax1.axhline(0, color="red", linestyle="--", alpha=0.7, linewidth=2)
    ax1.set_ylabel("Residual [TECU]")
    ax1.grid(True, alpha=0.3)

    ax2.plot(
        positions,
        [m["mae"] for m in monthly],
        color="green",
        marker="o",
        linewidth=2,
        markersize=6,
        label="MAE",
    )
    ax2.plot(
        positions,
        [m["rmse"] for m in monthly],
        color="orange",
        marker="s",
        linewidth=2,
        markersize=6,
        label="RMSE",
    )
    ax2.set_ylabel("Error [TECU]")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    ax3.bar(
        positions,
        [m["count"] for m in monthly],
        color="purple",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    ax3.set_ylabel("Sample count")
    ax3.set_xlabel("Month")
    ax3.grid(True, alpha=0.3)

    n_ticks = min(12, len(monthly))
    tick_indices = np.linspace(0, len(monthly) - 1, n_ticks, dtype=int)
    for ax in (ax1, ax2, ax3):
        ax.set_xticks(tick_indices)
        ax.set_xlim(-0.5, len(monthly) - 0.5)
    ax3.set_xticklabels(
        [monthly[i]["date"].strftime("%Y-%m") for i in tick_indices], rotation=45
    )
    ax1.set_title("Residual analysis: temporal evolution")

    data = pd.DataFrame(
        [{k: v for k, v in m.items() if k != "values"} for m in monthly]
    )
    return [_save(fig, "residuals_vs_date.png", "temporal", output_dir, data)]


# --------------------------------------------------------------------------
# Residuals vs. feature axis - src/viz/distributions.py::plot_residuals_vs_feature
# --------------------------------------------------------------------------


class FeatureAxis(NamedTuple):
    name: str
    column: str
    label: str
    num_bins: int
    bin_range: tuple[float, float] | None = None
    transform: Callable[[np.ndarray], np.ndarray] | None = None


# Kp_index is stored as Kp*10 (OMNI2 convention, CLAUDE.md's own note) - divided back to
# the true 0-9 Kp scale here, matching the /10.0 correction
# `plot_residuals_vs_solar_indices` applies in the source before binning. The generic
# `features_to_analyze` loop in `plot_test_metrics` bins a pre-engineered "kp" feature
# whose exact upstream construction this port does not have access to; Kp_index/10 is
# the same physical quantity and is the documented, reasonable substitute.
_FEATURE_AXES: list[FeatureAxis] = [
    FeatureAxis("doy", "doy", "Day of year", 24, (1.0, 366.0)),
    FeatureAxis("time", "local_time_hours", "Local solar time [h]", 24, (0.0, 24.0)),
    FeatureAxis("satazi", "satazi", "Azimuth angle [deg]", 24, (0.0, 360.0)),
    FeatureAxis(
        "kp", "Kp_index", "Kp index", 9, (0.0, 9.0), transform=lambda v: v / 10.0
    ),
    FeatureAxis("f107", "f107_index", "F10.7 solar flux [sfu]", 10, (50.0, 300.0)),
    FeatureAxis("dst", "Dst-index,_nT", "Dst index [nT]", 20, (-200.0, 100.0)),
    FeatureAxis("target_stec", "true_stec", "True STEC [TECU]", 20, None),
    FeatureAxis("pred_stec", "stec_pred", "Predicted STEC [TECU]", 20, None),
]


def fig_residuals_vs_feature(
    df: pd.DataFrame, axis: FeatureAxis, output_dir: Path
) -> list[dict]:
    """Residual boxplot binned by one feature axis, with MAE/RMSE line overlay.

    Ported from `src/viz/distributions.py::plot_residuals_vs_feature` ->
    `plot_binned_boxplot`. `df` needs `true_stec`, `stec_pred` and `axis.column`.
    """
    if axis.column not in df.columns:
        logger.warning(f"no {axis.column} column - skipping residuals_vs_{axis.name}")
        return []

    residual = (df["true_stec"] - df["stec_pred"]).to_numpy(dtype=float)
    values = df[axis.column].to_numpy(dtype=float)
    if axis.transform is not None:
        values = axis.transform(values)
    if axis.bin_range is not None:
        bin_edges = np.linspace(axis.bin_range[0], axis.bin_range[1], axis.num_bins + 1)
    else:
        bin_edges = np.histogram_bin_edges(values, bins=axis.num_bins)

    stats = _binned_residual_stats(residual, values, bin_edges)
    if not stats:
        logger.warning(f"no bin has data - skipping residuals_vs_{axis.name}")
        return []

    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)
    ax.axhline(0, color="red", linewidth=2, zorder=1, alpha=0.8)
    positions = [s["bin_center"] for s in stats]
    bin_width = float(bin_edges[1] - bin_edges[0])
    bp = ax.boxplot(
        [s["values"] for s in stats], positions=positions, widths=bin_width * 0.8,
        showfliers=False, zorder=2, patch_artist=True,
    )  # fmt: skip
    _style_boxplot(bp)
    _mae_rmse_overlay(
        ax, positions, [s["mae"] for s in stats], [s["rmse"] for s in stats]
    )
    ax.set_xlabel(axis.label)
    ax.set_ylabel("Residual [TECU]")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title(f"Residual analysis vs. {axis.label}")

    data = pd.DataFrame(
        {
            "bin_left": [s["bin_left"] for s in stats],
            "bin_right": [s["bin_right"] for s in stats],
            "bin_center": positions,
            "mae": [s["mae"] for s in stats],
            "rmse": [s["rmse"] for s in stats],
            "n": [s["n"] for s in stats],
        }
    )
    filename = f"residual_vs_{axis.name}_boxplot.png"
    return [_save(fig, filename, "feature", output_dir, data)]


# --------------------------------------------------------------------------
# Histogram of residuals - src/viz/distributions.py::plot_histogram_of_residuals
# --------------------------------------------------------------------------


def fig_histogram_of_residuals(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    """Density histogram of `true_stec - stec_pred`, 50 bins, mean/zero reference lines.

    Ported from `src/viz/distributions.py::plot_histogram_of_residuals`. `df` needs
    `true_stec`, `stec_pred`.
    """
    residual = (df["true_stec"] - df["stec_pred"]).to_numpy(dtype=float)
    mean_residual = float(residual.mean())
    std_residual = float(residual.std())

    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)
    hist_range = None
    if float(residual.max() - residual.min()) < 1e-9:
        # An (effectively) constant residual - a deterministic prediction offset, where
        # true_stec - (true_stec + offset) is only equal to -offset up to floating-point
        # cancellation noise, or a tiny slice of real data - has no meaningful width for
        # np.histogram to bin over, which raises "Too many bins for data range" rather
        # than drawing a degenerate one-bin histogram. Pad it symmetrically instead.
        pad = max(abs(mean_residual) * 0.5, 0.5)
        hist_range = (mean_residual - pad, mean_residual + pad)
    density, edges, _ = ax.hist(
        residual, bins=50, range=hist_range, density=True, alpha=0.7, color="skyblue",
        edgecolor="black",
    )  # fmt: skip
    ax.axvline(
        mean_residual,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_residual:.3f}",
    )
    ax.axvline(0, color="black", linewidth=2, alpha=0.8)
    ax.set_xlabel("Residual [TECU]")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.9)
    ax.set_title("Residual distribution analysis")

    centers = (edges[:-1] + edges[1:]) / 2
    data = pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "bin_center": centers,
            "density": density,
            "mean_residual": mean_residual,
            "std_residual": std_residual,
            "n": len(residual),
        }
    )
    return [_save(fig, "residuals_histogram.png", "feature", output_dir, data)]


# --------------------------------------------------------------------------
# Residuals vs. solar/geomagnetic indices - src/viz/distributions.py
# --------------------------------------------------------------------------

# Fixed bins ported from `plot_residuals_vs_solar_indices`'s per-index branches. Unlike
# the source, an index with fewer than 5 observations in every bin - or a completely
# absent column - is simply omitted from the panel grid rather than drawn as an all-zero
# row; a zero MAE/RMSE for zero observations is not a meaningful value to plot.
_SOLAR_INDEX_AXES: list[FeatureAxis] = [
    FeatureAxis(
        "kp", "Kp_index", "Kp index", 10, (0.0, 10.0), transform=lambda v: v / 10.0
    ),
    FeatureAxis("dst", "Dst-index,_nT", "Dst index [nT]", 12, (-500.0, 100.0)),
    FeatureAxis("ae", "AE-index,_nT", "AE index [nT]", 13, (0.0, 2600.0)),
    FeatureAxis("f107", "f107_index", "F10.7 solar flux [sfu]", 12, (60.0, 420.0)),
    FeatureAxis("sunspot", "R_Sunspot_No", "Sunspot number", 12, (0.0, 300.0)),
]


def fig_residuals_vs_solar_indices(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    """One residual-boxplot panel per available solar/geomagnetic index.

    Ported from `src/viz/distributions.py::plot_residuals_vs_solar_indices`. `df` needs
    `true_stec`, `stec_pred` and at least one of `Kp_index`, `Dst-index,_nT`,
    `AE-index,_nT`, `f107_index`, `R_Sunspot_No`.
    """
    available = [axis for axis in _SOLAR_INDEX_AXES if axis.column in df.columns]
    if not available:
        logger.warning(
            "no solar index column present - skipping residuals_vs_solar_indices"
        )
        return []

    residual_all = (df["true_stec"] - df["stec_pred"]).to_numpy(dtype=float)
    fig, axes = plt.subplots(
        len(available), 1, figsize=(FIGSIZE_WIDE[0], 6 * len(available))
    )
    axes = [axes] if len(available) == 1 else list(axes)

    rows = []
    for ax, axis in zip(axes, available):
        values = df[axis.column].to_numpy(dtype=float)
        if axis.transform is not None:
            values = axis.transform(values)
        bin_edges = np.linspace(axis.bin_range[0], axis.bin_range[1], axis.num_bins + 1)
        stats = [
            s
            for s in _binned_residual_stats(residual_all, values, bin_edges)
            if s["n"] >= 5
        ]
        if not stats:
            ax.set_axis_off()
            continue

        positions = [s["bin_center"] for s in stats]
        bin_width = float(bin_edges[1] - bin_edges[0])
        ax.axhline(0, color="red", linestyle="--", alpha=0.7, linewidth=2)
        bp = ax.boxplot(
            [s["values"] for s in stats], positions=positions, widths=bin_width * 0.7,
            showfliers=False, patch_artist=True,
        )  # fmt: skip
        _style_boxplot(bp)
        _mae_rmse_overlay(
            ax, positions, [s["mae"] for s in stats], [s["rmse"] for s in stats]
        )
        ax.set_xlabel(axis.label)
        ax.set_ylabel("Residual [TECU]")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        rows.extend(
            {
                "index": axis.name,
                "bin_center": s["bin_center"],
                "mae": s["mae"],
                "rmse": s["rmse"],
                "n": s["n"],
            }
            for s in stats
        )

    if not rows:
        plt.close(fig)
        logger.warning(
            "no solar index bin reached the minimum count - skipping residuals_vs_solar_indices"
        )
        return []

    fig.suptitle("Residual analysis vs. solar/geomagnetic indices")
    return [
        _save(
            fig,
            "residuals_vs_solar_indices.png",
            "feature",
            output_dir,
            pd.DataFrame(rows),
        )
    ]


# --------------------------------------------------------------------------
# Uncertainty diagnostics - src/viz/uncertainty.py
# --------------------------------------------------------------------------


def fig_uncertainty_calibration_binned(
    df: pd.DataFrame, output_dir: Path
) -> list[dict]:
    """Mean predicted uncertainty vs. mean observed error, 20 quantile bins, + bin counts.

    Ported from `src/viz/uncertainty.py::plot_uncertainty_calibration_binned` - see the
    module docstring for the undefined-`ax` bug fixed here (uses `ax1`, not `ax`).
    """
    if "pred_total_unc" not in df.columns:
        logger.warning("no pred_total_unc - skipping uncertainty_calibration_binned")
        return []

    abs_residual = (df["true_stec"] - df["stec_pred"]).abs()
    try:
        unc_bin = pd.qcut(df["pred_total_unc"], q=20, duplicates="drop")
    except ValueError:
        logger.warning(
            "pred_total_unc has too little spread to bin - skipping uncertainty_calibration_binned"
        )
        return []

    frame = pd.DataFrame(
        {
            "unc_bin": unc_bin,
            "pred_total_unc": df["pred_total_unc"].to_numpy(dtype=float),
            "abs_residual": abs_residual.to_numpy(dtype=float),
        }
    )
    bin_stats = (
        frame.groupby("unc_bin", observed=True)
        .agg(
            mean_predicted_unc=("pred_total_unc", "mean"),
            mean_observed_error=("abs_residual", "mean"),
            count=("abs_residual", "size"),
        )
        .reset_index(drop=True)
    )
    if len(bin_stats) < 2:
        logger.warning(
            "fewer than 2 calibration bins - skipping uncertainty_calibration_binned"
        )
        return []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    ax1.scatter(
        bin_stats["mean_predicted_unc"],
        bin_stats["mean_observed_error"],
        s=100,
        alpha=0.7,
        c="blue",
    )
    max_val = float(
        max(
            bin_stats["mean_predicted_unc"].max(),
            bin_stats["mean_observed_error"].max(),
        )
    )
    ax1.plot(
        [0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect calibration"
    )
    ax1.set_xlabel("Mean predicted uncertainty [TECU]")
    ax1.set_ylabel("Mean observed error [TECU]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    correlation, p_value = pearsonr(
        bin_stats["mean_predicted_unc"], bin_stats["mean_observed_error"]
    )

    ax2.bar(range(len(bin_stats)), bin_stats["count"], alpha=0.7, color="skyblue")
    ax2.set_xlabel("Uncertainty bin")
    ax2.set_ylabel("Sample count")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Uncertainty calibration analysis (r={correlation:.3f}, p={p_value:.2e})"
    )
    bin_stats = bin_stats.assign(correlation=correlation, p_value=p_value)
    return [
        _save(
            fig,
            "uncertainty_calibration_binned.png",
            "uncertainty",
            output_dir,
            bin_stats,
        )
    ]


def fig_uncertainty_calibration(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    """Predicted uncertainty vs. observed absolute error, raw scatter (<=10,000 points).

    Ported from `src/viz/uncertainty.py::plot_uncertainty_calibration`. Correlation is
    computed over the full frame, matching the source; only the rendered scatter is
    subsampled.
    """
    if "pred_total_unc" not in df.columns:
        logger.warning("no pred_total_unc - skipping uncertainty_calibration_scatter")
        return []

    abs_residual = (df["true_stec"] - df["stec_pred"]).abs()
    correlation, _ = pearsonr(df["pred_total_unc"], abs_residual)

    frame = pd.DataFrame(
        {
            "pred_total_unc": df["pred_total_unc"].to_numpy(dtype=float),
            "abs_residual": abs_residual.to_numpy(dtype=float),
        }
    )
    sample = frame.sample(n=10_000, random_state=42) if len(frame) > 10_000 else frame

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    ax.scatter(sample["pred_total_unc"], sample["abs_residual"], alpha=0.3, s=5)
    max_val = float(max(frame["pred_total_unc"].max(), frame["abs_residual"].max()))
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect calibration")
    ax.set_xlabel("Predicted uncertainty [TECU]")
    ax.set_ylabel("Observed error [TECU]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Uncertainty calibration scatter (r={correlation:.3f})")

    data = sample.assign(correlation=correlation).reset_index(drop=True)
    return [
        _save(
            fig, "uncertainty_calibration_scatter.png", "uncertainty", output_dir, data
        )
    ]


_UNCERTAINTY_TYPES = {
    "total": "pred_total_unc",
    "epistemic": "pred_epistemic_unc",
    "aleatoric": "pred_aleatoric_unc",
}


def fig_coverage_probability(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    """Observed vs. expected-Gaussian coverage fraction, 0-3 sigma, total/epistemic/
    aleatoric uncertainty.

    Ported from `src/viz/uncertainty.py::plot_coverage_probability`. `df` needs
    `true_stec`, `stec_pred` and at least one of the three uncertainty columns.
    """
    residual = (df["true_stec"] - df["stec_pred"]).to_numpy(dtype=float)
    sigma_levels = np.linspace(0, 3, 31)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    data: dict[str, np.ndarray | list[float]] = {"sigma": sigma_levels}
    plotted_any = False
    for label, column in _UNCERTAINTY_TYPES.items():
        if column not in df.columns or df[column].isna().all():
            continue
        unc = df[column].to_numpy(dtype=float)
        observed = [
            float(np.mean(np.abs(residual) <= sigma * unc)) for sigma in sigma_levels
        ]
        ax.plot(
            sigma_levels,
            observed,
            marker=".",
            linestyle="-",
            label=f"Observed ({label})",
        )
        data[f"observed_{label}"] = observed
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        logger.warning(
            "no uncertainty column present - skipping uncertainty_coverage_probability"
        )
        return []

    expected = [float(norm.cdf(s) - norm.cdf(-s)) for s in sigma_levels]
    ax.plot(sigma_levels, expected, "r--", label="Expected (Gaussian)", linewidth=2)
    data["expected_gaussian"] = expected

    ax.set_xlabel("Predicted uncertainty interval [sigma]")
    ax.set_ylabel("Observed coverage probability")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.set_title("Uncertainty coverage probability (reliability diagram)")

    return [
        _save(
            fig,
            "uncertainty_coverage_probability.png",
            "uncertainty",
            output_dir,
            pd.DataFrame(data),
        )
    ]


def fig_sigma_coverage_comparison(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    """1/2/3-sigma coverage percentage, bar chart, expected vs. total/epistemic/aleatoric.

    Ported from `src/viz/uncertainty.py::plot_sigma_coverage_comparison`.
    """
    if "pred_total_unc" not in df.columns:
        logger.warning("no pred_total_unc - skipping sigma_coverage_comparison")
        return []

    abs_residual = (df["true_stec"] - df["stec_pred"]).abs().to_numpy(dtype=float)
    n = len(abs_residual)

    def coverage(unc: np.ndarray) -> list[float]:
        return [float(np.sum(abs_residual <= k * unc) / n * 100) for k in (1, 2, 3)]

    series: dict[str, list[float]] = {"Expected (perfect)": [68.27, 95.45, 99.73]}
    colors = {"Expected (perfect)": "#7f7f7f"}
    for label, column, color in (
        ("Total", "pred_total_unc", "navy"),
        ("Epistemic (model)", "pred_epistemic_unc", "darkred"),
        ("Aleatoric (data noise)", "pred_aleatoric_unc", "darkgreen"),
    ):
        if column in df.columns and not df[column].isna().all():
            series[label] = coverage(df[column].to_numpy(dtype=float))
            colors[label] = color

    sigma_labels = ["1sigma", "2sigma", "3sigma"]
    x = np.arange(3)
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for i, (label, values) in enumerate(series.items()):
        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=label,
            alpha=0.8,
            color=colors[label],
            edgecolor="black",
            linewidth=1.5,
        )
        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
            )  # fmt: skip
    ax.set_xlabel("Sigma level")
    ax.set_ylabel("Coverage [%]")
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(sigma_labels)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_title("Interval coverage analysis (1sigma, 2sigma, 3sigma)")

    data = (
        pd.DataFrame(series, index=sigma_labels)
        .reset_index()
        .rename(columns={"index": "sigma_level"})
    )
    return [
        _save(fig, "sigma_coverage_comparison.png", "uncertainty", output_dir, data)
    ]


def fig_uncertainty_distributions(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    """Histograms of total/epistemic/aleatoric uncertainty, overlaid, with mean lines.

    Ported from `src/viz/uncertainty.py::plot_uncertainty_distributions` - see the
    module docstring for the `uncertainty_distributions.png` filename collision this
    port deliberately does not reproduce (only this one of the two colliding source
    functions is ported here).
    """
    if "pred_total_unc" not in df.columns:
        logger.warning("no pred_total_unc - skipping uncertainty_distributions")
        return []

    fig, ax = plt.subplots(figsize=(12, 8))
    rows = []
    for label, column, color in (
        ("Total", "pred_total_unc", "navy"),
        ("Epistemic (model)", "pred_epistemic_unc", "darkred"),
        ("Aleatoric (data noise)", "pred_aleatoric_unc", "darkgreen"),
    ):
        if column not in df.columns or df[column].isna().all():
            continue
        values = df[column].to_numpy(dtype=float)
        counts, edges, _ = ax.hist(
            values,
            bins=50,
            alpha=0.7,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        mean_val = float(values.mean())
        ax.axvline(
            mean_val,
            color=color,
            linestyle="--",
            linewidth=3,
            alpha=0.9,
            label=f"{label} mean: {mean_val:.3f} TECU",
        )
        centers = (edges[:-1] + edges[1:]) / 2
        rows.extend(
            {
                "component": label,
                "bin_left": float(bin_left),
                "bin_right": float(bin_right),
                "bin_center": float(bin_center),
                "count": int(bin_count),
                "mean": mean_val,
            }
            for bin_left, bin_right, bin_center, bin_count in zip(
                edges[:-1], edges[1:], centers, counts
            )
        )

    if not rows:
        plt.close(fig)
        logger.warning(
            "no uncertainty column present - skipping uncertainty_distributions"
        )
        return []

    ax.set_xlabel("Uncertainty [TECU]")
    ax.set_ylabel("Frequency")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Distribution of uncertainty components")

    return [
        _save(
            fig,
            "uncertainty_distributions.png",
            "uncertainty",
            output_dir,
            pd.DataFrame(rows),
        )
    ]


# --------------------------------------------------------------------------
# Registry and entry point
# --------------------------------------------------------------------------


def _build_spatial_figures(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    records = fig_spatial_error_map(df, output_dir)
    records += fig_spatial_error_map_by_local_time(df, output_dir)
    return records


def _build_az_el_figures(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    records = fig_az_el_heatmap(df, output_dir, metric="residual")
    records += fig_az_el_heatmap(df, output_dir, metric="mae")
    return records


def _build_feature_axis_figures(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    records: list[dict] = []
    for axis in _FEATURE_AXES:
        records += fig_residuals_vs_feature(df, axis, output_dir)
    return records


def _build_uncertainty_figures(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    records = fig_uncertainty_calibration_binned(df, output_dir)
    records += fig_uncertainty_calibration(df, output_dir)
    records += fig_coverage_probability(df, output_dir)
    records += fig_sigma_coverage_comparison(df, output_dir)
    records += fig_uncertainty_distributions(df, output_dir)
    return records


def _build_remaining_figures(df: pd.DataFrame, output_dir: Path) -> list[dict]:
    records = fig_prediction_scatter(df, output_dir)
    records += fig_residuals_vs_date(df, output_dir)
    records += fig_histogram_of_residuals(df, output_dir)
    records += fig_residuals_vs_solar_indices(df, output_dir)
    return records


# Priority order per the task's scope-management guidance: spatial maps, az/el heatmaps,
# feature-axis boxplots, uncertainty diagnostics, everything else.
FIGURE_BUILDERS = (
    _build_spatial_figures,
    _build_az_el_figures,
    _build_feature_axis_figures,
    _build_uncertainty_figures,
    _build_remaining_figures,
)


def build_all(observations: pd.DataFrame, output_dir: Path) -> list[dict]:
    """Derive local time once, then run every registered builder against the one shared
    frame - all figures in this module read the same per-observation cache."""
    configure_plotting()
    df = add_local_time(observations)
    records: list[dict] = []
    for build in FIGURE_BUILDERS:
        records.extend(build(df, output_dir))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=dto.DEFAULT_STORE_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=dto.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild the observation cache even if one already exists at --cache-dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cache_path = args.cache_dir / "observations.parquet"
    if args.rebuild_cache or not cache_path.exists():
        logger.info(f"building the observation cache at {args.cache_dir}")
        observations = dto.build(args.store_root, args.cache_dir)
    else:
        logger.info(f"reusing the existing cache at {cache_path}")
        observations = pd.read_parquet(cache_path)

    records = build_all(observations, args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "diagnostic_figures_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    logger.info(
        f"wrote {len(records)} figure(s) to {args.output_dir} (manifest: {manifest_path})"
    )


if __name__ == "__main__":
    main()
