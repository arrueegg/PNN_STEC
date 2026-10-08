"""Two diagnostic maps of the positioning test stations (preview, not a manuscript figure).

1. problem_stations: where and why individual stations are difficult - tail cases above
   10 m (Direct STEC and IGS GIM), weak satellite geometry, remaining single-constellation
   days, and station-days without a model solution.
2. station_performance: median improvement of Direct STEC over IGS GIM + Mapping per
   station, both with their own uncertainty weighting, on the common set.

All inputs are read from the positioning stage outputs, so the maps describe exactly the
population behind the manuscript numbers.

Usage::

    python scripts/plot_station_maps.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
ANALYSES = REPO / "multiday_results" / "analyses"
OUTPUT_DIR = REPO / "plots" / "preview" / "station_maps"

TAIL_THRESHOLD_M = 10.0
LOW_SATELLITE_THRESHOLD = 9.0
MIN_SINGLE_CONSTELLATION_DAYS = 10
MIN_MISSING_SOLUTION_DAYS = 10


def load_station_table() -> pd.DataFrame:
    """One row per station with coordinates, performance and problem indicators."""
    daily = pd.read_csv(
        ANALYSES / "positioning_distributions/rebuilt/common_set_daily_rows.csv"
    )
    wide = daily.pivot_table(
        index=["station", "doy"], columns="method", values="error_3d_rms"
    ).reset_index()
    wide["log_ratio"] = np.log(wide["STEC_iono"] / wide["gim_iono"])

    per_station = wide.groupby("station").agg(
        station_days=("doy", "size"),
        median_stec=("STEC_iono", "median"),
        median_gim=("gim_iono", "median"),
        median_log_ratio=("log_ratio", "median"),
        stec_tail=("STEC_iono", lambda v: int((v > TAIL_THRESHOLD_M).sum())),
        gim_tail=("gim_iono", lambda v: int((v > TAIL_THRESHOLD_M).sum())),
    )
    # Median of the per-day ratio, so a station's improvement is not dominated by its worst days.
    per_station["improvement_pct"] = (1 - np.exp(per_station["median_log_ratio"])) * 100

    summary = pd.read_csv(
        ANALYSES / "positioning_coverage/rebuilt/multiday_summary.csv"
    )
    gim_rows = summary[summary["method"] == "gim_iono"]
    per_station["median_satellites"] = gim_rows.groupby("station")["mean_nsat"].median()

    flags = pd.read_csv(
        ANALYSES / "constellation_coverage/rebuilt/station_day_flags.csv"
    )
    per_station["single_constellation_days"] = (
        flags[flags["single_constellation"]].groupby("station").size()
    )

    coverage = pd.read_csv(ANALYSES / "positioning_coverage/rebuilt/coverage.csv")
    missing = coverage[coverage["cause"] != "solved by all methods"]
    missing_counts = missing.groupby("station").size().rename("missing_solution_days")
    per_station = per_station.join(missing_counts, how="outer")

    coords = pd.read_csv(
        ANALYSES / "positioning_geography/rebuilt/station_map_direct_stec.csv"
    ).set_index("station")[["lat", "lon"]]
    per_station = per_station.join(coords, how="left")
    per_station = per_station.fillna(
        {"single_constellation_days": 0, "missing_solution_days": 0}
    )
    return per_station.reset_index()


def base_axes(fig: plt.Figure, position: int = 111) -> plt.Axes:
    ax = fig.add_subplot(position, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f2f2f2")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#888888")
    return ax


# Thresholds that decide which reason a flagged station is attributed to, checked in order.
FAR_FROM_TRAINING_KM = 500.0
SMALL_IMPROVEMENT_PCT = 10.0
REASONS = [
    ("isolated", "#d62728", "Far from training stations (> 500 km)"),
    ("geometry", "#6a3d9a", "Weak geometry (median < 9 satellites)"),
    ("hard_days", "#ff7f0e", "Few extreme days, otherwise clear gain"),
    ("unclear", "#7f7f7f", "Small gain, no clear single cause"),
]


def attribute_reason(row: pd.Series) -> str | None:
    """Return the most likely reason a station is difficult, or None if it is not flagged."""
    flagged = (
        row["improvement_pct"] < SMALL_IMPROVEMENT_PCT
        or row["stec_tail"] > 0
        or row["gim_tail"] > 0
        or row["median_satellites"] < LOW_SATELLITE_THRESHOLD
    )
    if not flagged:
        return None
    if row["distance_km"] > FAR_FROM_TRAINING_KM:
        return "isolated"
    if row["median_satellites"] < LOW_SATELLITE_THRESHOLD:
        return "geometry"
    if row["stec_tail"] > 0 or row["gim_tail"] > 0:
        return "hard_days"
    return "unclear"


def plot_problem_stations(stations: pd.DataFrame) -> Path:
    distances = pd.read_csv(ANALYSES / "station_independence/rebuilt/per_station.csv")[
        ["station", "distance_km", "nearest_train_station"]
    ]
    located = stations.merge(distances, on="station", how="left").dropna(
        subset=["lat", "lon"]
    )
    located["reason"] = located.apply(attribute_reason, axis=1)
    flagged = located.dropna(subset=["reason"]).copy()
    reason_order = {key: i for i, (key, _, _) in enumerate(REASONS)}
    flagged["order"] = flagged["reason"].map(reason_order)
    flagged = flagged.sort_values(["order", "improvement_pct"]).reset_index(drop=True)
    flagged["number"] = np.arange(1, len(flagged) + 1)

    fig = plt.figure(figsize=(14, 13))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.22)
    ax = fig.add_subplot(grid[0], projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f2f2f2")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#888888")
    transform = ccrs.PlateCarree()

    others = located[located["reason"].isna()]
    ax.scatter(
        others["lon"],
        others["lat"],
        s=25,
        color="#c8c8c8",
        edgecolors="white",
        linewidths=0.5,
        transform=transform,
        zorder=2,
        label="Other test stations (no problem)",
    )
    colors = {key: color for key, color, _ in REASONS}
    for key, color, label in REASONS:
        subset = flagged[flagged["reason"] == key]
        ax.scatter(
            subset["lon"],
            subset["lat"],
            s=260,
            color=color,
            edgecolors="black",
            linewidths=0.8,
            transform=transform,
            zorder=4,
            label=label,
        )
    for _, row in flagged.iterrows():
        ax.text(
            row["lon"],
            row["lat"],
            str(row["number"]),
            fontsize=8,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            transform=transform,
            zorder=5,
        )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=9,
        title="Most likely reason (number = row in table)",
        title_fontsize=9,
    )
    ax.set_title(
        "Test stations where Direct STEC positioning gains little or has extreme days"
    )

    table_ax = fig.add_subplot(grid[1])
    table_ax.axis("off")
    header = [
        "#",
        "Station",
        "Station-days",
        "Gain over\nGIM [%]",
        "Days > 10 m\nDirect / GIM",
        "Median\nsatellites",
        "Nearest training\nstation [km]",
        "Most likely reason",
    ]
    cells = [
        [
            row["number"],
            row["station"],
            int(row["station_days"]),
            f"{row['improvement_pct'] + 0.0:+.0f}".replace("-0", "0"),
            f"{row['stec_tail']:.0f} / {row['gim_tail']:.0f}",
            f"{row['median_satellites']:.1f}",
            f"{row['distance_km']:.0f} ({row['nearest_train_station']})",
            dict((k, lab) for k, _, lab in REASONS)[row["reason"]],
        ]
        for _, row in flagged.iterrows()
    ]
    table = table_ax.table(
        cellText=cells,
        colLabels=header,
        loc="upper center",
        cellLoc="center",
        colWidths=[0.04, 0.08, 0.09, 0.08, 0.11, 0.08, 0.14, 0.38],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row_index, col_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_height(cell.get_height() * 1.6)
        elif col_index == 0:
            cell.set_facecolor(colors[flagged.loc[row_index - 1, "reason"]])
            cell.set_text_props(color="white", fontweight="bold")
    table_ax.text(
        0.0,
        -0.02,
        "Flagged: median gain over IGS GIM + Mapping below 10%, any day above 10 m, or fewer "
        "than 9 satellites. Gain = median of the daily 3D RMSE ratio, uncertainty weighting, "
        "common set.\nReasons are checked in the order of the legend; distance to the nearest "
        "training station from the station-independence analysis.",
        fontsize=8,
        va="top",
        transform=table_ax.transAxes,
    )
    path = OUTPUT_DIR / "problem_stations.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_station_performance(stations: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(14, 7.5))
    ax = base_axes(fig)
    located = stations.dropna(subset=["lat", "lon", "improvement_pct"])
    limit = float(np.ceil(np.nanmax(np.abs(located["improvement_pct"])) / 10) * 10)
    sizes = 20 + 0.6 * located["station_days"]
    points = ax.scatter(
        located["lon"],
        located["lat"],
        c=located["improvement_pct"],
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
        s=sizes,
        edgecolors="black",
        linewidths=0.5,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    for _, row in located.iterrows():
        ax.text(
            row["lon"] + 2.5,
            row["lat"] - 3.5,
            row["station"],
            fontsize=6,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
    bar = fig.colorbar(points, ax=ax, orientation="horizontal", shrink=0.6, pad=0.04)
    bar.set_label("Median daily improvement of Direct STEC over IGS GIM + Mapping [%]")
    ax.set_title(
        "Per-station positioning performance (common set, uncertainty weighting; "
        "marker size = station-days)"
    )
    path = OUTPUT_DIR / "station_performance.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stations = load_station_table()
    table_path = OUTPUT_DIR / "station_table.csv"
    stations.to_csv(table_path, index=False)
    logger.info(f"Wrote {table_path} ({len(stations)} stations)")
    logger.info(f"Wrote {plot_problem_stations(stations)}")
    logger.info(f"Wrote {plot_station_performance(stations)}")


if __name__ == "__main__":
    main()
