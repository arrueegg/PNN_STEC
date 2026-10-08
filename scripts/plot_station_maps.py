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


def plot_problem_stations(stations: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(14, 7.5))
    ax = base_axes(fig)
    located = stations.dropna(subset=["lat", "lon"])
    transform = ccrs.PlateCarree()

    ax.scatter(
        located["lon"],
        located["lat"],
        s=18,
        color="#bbbbbb",
        transform=transform,
        label="Test station (no flag)",
        zorder=2,
    )
    categories = [
        (
            located["median_satellites"] < LOW_SATELLITE_THRESHOLD,
            dict(
                marker="s", s=170, facecolors="none", edgecolors="#6a3d9a", linewidths=2
            ),
            f"Weak geometry (median < {LOW_SATELLITE_THRESHOLD:.0f} satellites)",
        ),
        (
            located["single_constellation_days"] >= MIN_SINGLE_CONSTELLATION_DAYS,
            dict(
                marker="D",
                s=120,
                facecolors="none",
                edgecolors="#1f78b4",
                linewidths=1.8,
            ),
            f"Corrections cover fewer satellites than GIM on >= {MIN_SINGLE_CONSTELLATION_DAYS} days",
        ),
        (
            located["missing_solution_days"] >= MIN_MISSING_SOLUTION_DAYS,
            dict(
                marker="^",
                s=150,
                facecolors="none",
                edgecolors="#ff7f00",
                linewidths=1.8,
            ),
            f"No solution for some methods on >= {MIN_MISSING_SOLUTION_DAYS} days",
        ),
    ]
    for mask, style, label in categories:
        subset = located[mask]
        ax.scatter(
            subset["lon"],
            subset["lat"],
            transform=transform,
            label=label,
            zorder=4,
            **style,
        )

    tail = located[located["stec_tail"] > 0]
    ax.scatter(
        tail["lon"],
        tail["lat"],
        s=40 + 40 * tail["stec_tail"],
        color="#e31a1c",
        alpha=0.6,
        transform=transform,
        zorder=3,
        label=f"Direct STEC days > {TAIL_THRESHOLD_M:.0f} m (size = count)",
    )
    gim_tail = located[located["gim_tail"] > 0]
    ax.scatter(
        gim_tail["lon"],
        gim_tail["lat"],
        s=40 + 40 * gim_tail["gim_tail"],
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        linestyle="--",
        transform=transform,
        zorder=5,
        label=f"IGS GIM days > {TAIL_THRESHOLD_M:.0f} m (size = count)",
    )

    # Label only the stations that explain the tail or underperform, to keep the map legible.
    flagged = located[
        (located["stec_tail"] > 0)
        | (located["gim_tail"] > 0)
        | (located["median_satellites"] < LOW_SATELLITE_THRESHOLD)
        | (located["improvement_pct"] < 0)
    ]
    for _, row in flagged.iterrows():
        text = (
            f"{row['station']}: {row['stec_tail']:.0f}/{row['gim_tail']:.0f} >10 m, "
            f"{row['median_satellites']:.1f} sats, {row['improvement_pct']:+.0f}%"
        )
        ax.text(row["lon"] + 3, row["lat"] + 2, text, fontsize=7, transform=transform,
                zorder=6, bbox=dict(facecolor="white", alpha=0.7, linewidth=0, pad=1))

    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("Positioning test stations with problematic station-days")
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
