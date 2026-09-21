"""Figures for `stec.analysis.positioning_geography` (owner request, 2026-08-28).

Diagnostic output, not a manuscript or revision-response figure set - lives in its own
`plots/positioning_geography/` tree, the same convention `positioning_diagnostics.py`
uses and for the same reason (see that module's docstring). Reads the CSVs
`stec.analysis.positioning_geography` writes to
`multiday_results/analyses/positioning_geography/rebuilt/`; does not recompute anything
itself.

Colour rule: `stec.viz.style.APPROACH_COLORS` only where a figure compares the four
methods against each other (the latitude-stratification line plots) - an approach
colour must only ever mean that approach. Everywhere else - error magnitude on the
station maps, the STEC-minus-GIM difference, and the recovered/original population
split - uses a palette outside `APPROACH_COLORS`: a sequential colormap (`viridis`) for
plain error magnitude, and a diverging grey-white-red colormap built from
`stec.viz.style.CONDITION_COLORS` (the repo's own "not an approach" pair, already used
by `positioning_diagnostics.fig_per_station_diff` for the identical
better/worse-than-GIM contrast) for anything centred at zero or splitting
recovered-vs-original.

Every figure is written twice via `_save` (titled working copy + `_notitle` manuscript
copy), and `_save` always writes the plotted values as a CSV alongside the PNGs.

Usage::

    python -m stec.viz.positioning_geography --output_dir plots/positioning_geography
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.positioning_geography import LATITUDE_BINS
from ..config import paths
from .revision_figures import _grouped_bars, analysis_dir
from .style import APPROACH_COLORS, CONDITION_COLORS, FIGSIZE_WIDE, configure_plotting

import matplotlib.pyplot as plt  # noqa: E402  (style.py sets the Agg backend on import)
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

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
FIGSIZE_TWO_MAPS = (26, 9)
FIGSIZE_TWO_PANEL = (14, 12)

# Sequential magnitude colormap for plain error maps - not one of the four approach
# hues, and a continuous colormap reads visually distinct from a discrete approach
# swatch regardless.
MAGNITUDE_CMAP = "viridis"

# Diverging colormap for anything centred at zero, built from the repo's own
# non-approach condition colours (grey = better/baseline, red = worse/contrast) -
# the same two colours `positioning_diagnostics.fig_per_station_diff` already uses for
# this exact better/worse-than-GIM contrast, so a reader sees one consistent encoding
# for "STEC worse than GIM" across both figure sets.
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "stec_minus_gim_diff",
    [CONDITION_COLORS["baseline"], "#ffffff", CONDITION_COLORS["contrast"]],
)

_LAT_KIND_LABEL = {
    "geographic": "Geographic latitude",
    "geomagnetic": "Geomagnetic latitude",
}
_LAT_KIND_COLUMN = {"geographic": "lat", "geomagnetic": "sm_lat"}


def _bin_order(bins: tuple[float, ...] = LATITUDE_BINS) -> list[str]:
    """String form of every latitude bin, in latitude order - matches exactly how
    `pandas.cut(..., include_lowest=True)` labels (and therefore how `to_csv`/
    `read_csv` round-trips) the interval, including the small epsilon it subtracts
    from the very first edge."""
    categories = pd.cut([0], bins=bins, include_lowest=True).categories
    return [str(c) for c in categories]


def _marker_sizes(
    counts: pd.Series, min_size: float = 15.0, scale: float = 45.0
) -> np.ndarray:
    """sqrt-scaled marker size from a station-day (or N) count, the same formula
    `positioning_diagnostics.fig_per_station_diff` uses, so marker size means the same
    thing in both figure sets: a 3-day station must not read like a 200-day one."""
    counts = counts.astype(float)
    denom = counts.max() if counts.max() > 0 else 1.0
    return min_size + scale * np.sqrt(counts / denom)


def _save(
    fig: plt.Figure,
    name: str,
    source: str,
    output_dir: Path,
    provenance: str,
    data: pd.DataFrame | None = None,
) -> None:
    """Mirrors `positioning_diagnostics._save` / `revision_figures._save` exactly;
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


def _map_axis(ax) -> None:
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="#f2f2f2", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=0)
    ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.6, zorder=1)
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels, gl.right_labels = False, False
    ax.set_global()


# --------------------------------------------------------------------------
# 1a. Station map - Direct STEC error magnitude (mean, median)
# --------------------------------------------------------------------------


def fig_station_map_direct_stec(
    df: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """Two panels (mean, median): every station's Direct STEC 3D positioning error
    across 2024, coloured by error magnitude (`viridis`, not an approach colour),
    marker size scaled to `station_days` so a 3-day station is not misread as a
    200-day one."""
    import cartopy.crs as ccrs

    fig, axes = plt.subplots(
        1, 2, figsize=FIGSIZE_TWO_MAPS, subplot_kw={"projection": ccrs.PlateCarree()}
    )
    sizes = _marker_sizes(df["station_days"])
    for ax, column, label in zip(
        axes, ["mean_error_3d_m", "median_error_3d_m"], ["Mean", "Median"]
    ):
        _map_axis(ax)
        sc = ax.scatter(
            df["lon"], df["lat"], c=df[column], s=sizes, cmap=MAGNITUDE_CMAP,
            edgecolor="black", linewidth=0.4, zorder=3, transform=ccrs.PlateCarree(),
        )  # fmt: skip
        fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7,
                     label=f"{label} 3D error [m]")  # fmt: skip
        ax.set_title(f"{label} (N={len(df)} stations)")
    fig.suptitle("Direct STEC positioning error by station, 2024 (iono weighting)")
    _save(
        fig, "station_map_direct_stec", "positioning", output_dir, provenance,
        df[["station", "lat", "lon", "sm_lat", "station_days", "mean_error_3d_m", "median_error_3d_m"]],
    )  # fmt: skip


# --------------------------------------------------------------------------
# 1b. Station map - Direct STEC minus IGS GIM
# --------------------------------------------------------------------------


def fig_station_map_diff(df: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """Two panels (mean, median): Direct STEC minus IGS GIM 3D error per station,
    diverging colour scale centred at zero (grey = STEC better, red = STEC worse) so
    losing stations are immediately visible. Marker size scaled to
    `stec_station_days`."""
    import cartopy.crs as ccrs

    plotted = df.dropna(subset=["diff_mean_m"]).copy()
    missing = df[df["diff_mean_m"].isna()]["station"].tolist()
    sizes = _marker_sizes(plotted["stec_station_days"])

    fig, axes = plt.subplots(
        1, 2, figsize=FIGSIZE_TWO_MAPS, subplot_kw={"projection": ccrs.PlateCarree()}
    )
    for ax, column, label in zip(
        axes, ["diff_mean_m", "diff_median_m"], ["Mean", "Median"]
    ):
        _map_axis(ax)
        bound = float(np.nanmax(np.abs(plotted[column]))) or 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
        sc = ax.scatter(
            plotted["lon"], plotted["lat"], c=plotted[column], s=sizes, cmap=DIFF_CMAP,
            norm=norm, edgecolor="black", linewidth=0.4, zorder=3, transform=ccrs.PlateCarree(),
        )  # fmt: skip
        fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7,
                     label=f"{label} Direct STEC minus IGS GIM, 3D error [m]\n(positive = Direct STEC worse)")  # fmt: skip
        ax.set_title(f"{label} (N={len(plotted)} stations)")
    fig.suptitle(
        "Direct STEC vs IGS GIM by station, 2024 (iono weighting)"
        + (f" - {len(missing)} station(s) with no Direct STEC solution not shown" if missing else "")
    )  # fmt: skip
    _save(
        fig, "station_map_diff", "positioning", output_dir, provenance,
        plotted[["station", "lat", "lon", "sm_lat", "stec_station_days", "gim_station_days",
                 "stec_mean_m", "gim_mean_m", "diff_mean_m", "diff_median_m"]],
    )  # fmt: skip


# --------------------------------------------------------------------------
# 2a. Latitude stratification - per-method error
# --------------------------------------------------------------------------


def fig_latitude_stratification(
    df: pd.DataFrame, lat_kind: str, output_dir: Path, provenance: str
) -> None:
    """Two panels (mean top, median bottom): 3D error per method against latitude bin,
    marker size scaled to N. One call for geographic, one for geomagnetic - same axes
    layout, so the two are visually comparable bin-for-bin."""
    order = _bin_order()
    df = (
        df.set_index(["lat_bin", "Method"])
        .reindex(
            pd.MultiIndex.from_product(
                [order, METHOD_ORDER], names=["lat_bin", "Method"]
            )
        )
        .reset_index()
    )
    x = np.arange(len(order))

    fig, (ax_mean, ax_median) = plt.subplots(
        2, 1, figsize=FIGSIZE_TWO_PANEL, sharex=True
    )
    for method in METHOD_ORDER:
        sub = df[df["Method"] == method].set_index("lat_bin").reindex(order)
        color = APPROACH_COLORS[method]
        marker = _METHOD_MARKERS[method]
        sizes = _marker_sizes(sub["n"].fillna(0))
        ax_mean.plot(x, sub["mean"], color=color, linewidth=1, zorder=2, label=method)
        ax_mean.scatter(x, sub["mean"], color=color, marker=marker, s=sizes, zorder=3)
        ax_median.plot(
            x, sub["median"], color=color, linewidth=1, zorder=2, label=method
        )
        ax_median.scatter(
            x, sub["median"], color=color, marker=marker, s=sizes, zorder=3
        )

    for ax, ylabel in (
        (ax_mean, "Mean 3D error [m]"),
        (ax_median, "Median 3D error [m]"),
    ):
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_ylabel(ylabel)
    ax_median.set_xticks(x)
    ax_median.set_xticklabels(order, rotation=45, ha="right")
    ax_median.set_xlabel(f"{_LAT_KIND_LABEL[lat_kind]} bin [deg]")
    ax_mean.legend(loc="upper center", ncol=2)
    fig.suptitle(f"Positioning 3D error by {lat_kind} latitude (iono weighting)")
    _save(
        fig, f"latitude_stratification_{lat_kind}", "positioning", output_dir, provenance,
        df[["lat_bin", "Method", "mean", "median", "n"]],
    )  # fmt: skip


# --------------------------------------------------------------------------
# 2b. Latitude stratification - Direct STEC minus IGS GIM
# --------------------------------------------------------------------------


def fig_latitude_stratification_diff(
    df: pd.DataFrame, lat_kind: str, output_dir: Path, provenance: str
) -> None:
    """One row of two bar panels (mean, median): the paired Direct-STEC-minus-GIM
    difference per latitude bin, bars coloured red where STEC is worse and grey where
    it is better - the same convention `positioning_diagnostics.fig_per_station_diff`
    uses. N is annotated above each bar."""
    order = _bin_order()
    df = df.set_index("lat_bin").reindex(order).reset_index()
    x = np.arange(len(order))

    fig, (ax_mean, ax_median) = plt.subplots(1, 2, figsize=(22, 8))
    for ax, column, label in zip(
        (ax_mean, ax_median), ("diff_mean_m", "diff_median_m"), ("Mean", "Median")
    ):
        colors = [
            CONDITION_COLORS["contrast"] if pd.notna(v) and v > 0 else CONDITION_COLORS["baseline"]
            for v in df[column]
        ]  # fmt: skip
        ax.bar(x, df[column], color=colors, zorder=3)
        ax.axhline(0, color="black", linewidth=1.0)
        for xi, (v, n) in enumerate(zip(df[column], df["n"])):
            if pd.notna(v):
                ax.text(xi, v, f"n={int(n)}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=10)  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=45, ha="right")
        ax.set_ylabel(
            f"{label} Direct STEC minus IGS GIM, 3D error [m]\n(positive = Direct STEC worse)"
        )
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.suptitle(f"Direct STEC vs IGS GIM by {lat_kind} latitude (iono weighting)")
    _save(
        fig, f"latitude_stratification_diff_{lat_kind}", "positioning", output_dir, provenance,
        df[["lat_bin", "n", "diff_mean_m", "mean_improvement_pct", "diff_median_m", "median_improvement_pct"]],
    )  # fmt: skip


# --------------------------------------------------------------------------
# 3a. Recovered-station-day concentration by latitude
# --------------------------------------------------------------------------


def fig_population_by_latitude(
    df: pd.DataFrame, lat_kind: str, output_dir: Path, provenance: str
) -> None:
    """Share of station-days that are geometry-only recovered (vs original STEC-DB
    coverage), per latitude bin - answers "are the recovered station-days concentrated
    at particular latitudes" directly. Single series, so it takes the "contrast"
    condition colour rather than an approach one."""
    order = _bin_order()
    df = df.set_index("lat_bin").reindex(order).reset_index()
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.bar(x, df["pct_recovered"], color=CONDITION_COLORS["contrast"], zorder=3)
    for xi, (pct, total) in enumerate(zip(df["pct_recovered"], df["total"])):
        if pd.notna(pct):
            ax.text(xi, pct, f"N={int(total)}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylabel("Recovered (geometry-only) station-days [%]")
    ax.set_xlabel(f"{_LAT_KIND_LABEL[lat_kind]} bin [deg]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_title(f"Recovered station-day share by {lat_kind} latitude")
    _save(
        fig, f"population_by_latitude_{lat_kind}", "positioning", output_dir, provenance,
        df[["lat_bin", "original", "recovered", "total", "pct_recovered"]],
    )  # fmt: skip


# --------------------------------------------------------------------------
# 3b. Population split crossed with latitude
# --------------------------------------------------------------------------


def fig_population_latitude_cross(
    df: pd.DataFrame, lat_kind: str, output_dir: Path, provenance: str
) -> None:
    """Grouped bars: the Direct-STEC-minus-GIM mean difference per latitude bin, split
    into original and recovered station-days - shows directly whether the recovered
    population's underperformance survives within a fixed latitude bin, or whether it
    only shows up because recovered days are concentrated at latitudes that are already
    harder for everyone."""
    order = _bin_order()
    pivot_mean = df.pivot(
        index="lat_bin", columns="population", values="diff_mean_m"
    ).reindex(order)
    pivot_median = df.pivot(
        index="lat_bin", columns="population", values="diff_median_m"
    ).reindex(order)
    populations = [p for p in ("original", "recovered") if p in pivot_mean.columns]
    colors = [CONDITION_COLORS["baseline"], CONDITION_COLORS["contrast"]][
        : len(populations)
    ]

    fig, (ax_mean, ax_median) = plt.subplots(1, 2, figsize=(24, 8))
    plotted_mean = _grouped_bars(
        ax_mean, order, populations, {p: pivot_mean[p].values for p in populations}, colors,
        "Mean Direct STEC minus IGS GIM, 3D error [m]",
    )  # fmt: skip
    plotted_median = _grouped_bars(
        ax_median, order, populations, {p: pivot_median[p].values for p in populations}, colors,
        "Median Direct STEC minus IGS GIM, 3D error [m]",
    )  # fmt: skip
    for ax in (ax_mean, ax_median):
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticklabels(order, rotation=45, ha="right")
    ax_mean.legend(loc="best")
    ax_mean.set_title("Mean")
    ax_median.set_title("Median")
    fig.suptitle(
        f"Direct STEC vs IGS GIM by {lat_kind} latitude, original vs recovered"
    )
    plotted_mean["statistic"] = "mean"
    plotted_median["statistic"] = "median"
    _save(
        fig, f"population_latitude_cross_{lat_kind}", "positioning", output_dir, provenance,
        pd.concat([plotted_mean, plotted_median], ignore_index=True),
    )  # fmt: skip


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _build_figures(args: argparse.Namespace, output_dir: Path) -> None:
    source = analysis_dir(args.results_dir, "positioning_geography")
    if not source.is_dir():
        logger.warning(
            f"{source} not found - run stec/analysis/positioning_geography.py first"
        )
        return
    prov = f"{source} (stec.analysis.positioning_geography), SF-PPP 2024 test period, iono weighting"

    direct_stec_path = source / "station_map_direct_stec.csv"
    if direct_stec_path.exists():
        fig_station_map_direct_stec(pd.read_csv(direct_stec_path), output_dir, prov)

    diff_path = source / "station_map_diff.csv"
    if diff_path.exists():
        fig_station_map_diff(pd.read_csv(diff_path), output_dir, prov)

    for lat_kind in ("geographic", "geomagnetic"):
        strat_path = source / f"latitude_stratification_{lat_kind}.csv"
        if strat_path.exists():
            fig_latitude_stratification(
                pd.read_csv(strat_path), lat_kind, output_dir, prov
            )

        strat_diff_path = source / f"latitude_stratification_diff_{lat_kind}.csv"
        if strat_diff_path.exists():
            fig_latitude_stratification_diff(
                pd.read_csv(strat_diff_path), lat_kind, output_dir, prov
            )

        population_path = (
            source / f"population_concentration_by_latitude_{lat_kind}.csv"
        )
        if population_path.exists():
            fig_population_by_latitude(
                pd.read_csv(population_path), lat_kind, output_dir, prov
            )

        cross_path = source / f"population_latitude_diff_{lat_kind}.csv"
        if cross_path.exists():
            fig_population_latitude_cross(
                pd.read_csv(cross_path), lat_kind, output_dir, prov
            )


FIGURE_BUILDERS = (_build_figures,)


def build_all(args: argparse.Namespace) -> None:
    configure_plotting()
    for build in FIGURE_BUILDERS:
        build(args, args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("plots/positioning_geography")
    )
    parser.add_argument("--results_dir", type=Path, default=paths.RESULTS_ROOT)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    build_all(args)


if __name__ == "__main__":
    main()
