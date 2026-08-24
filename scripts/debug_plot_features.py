#!/usr/bin/env python3
"""Diagnostic feature plots for infer_from_log.py pipeline verification.

Usage:
    python debug_plot_features.py \
        --config config/config.yaml \
        --data_file data/Poland_positioning/WROC00POL_R_20201370000_01D_30S_MO_101000_ss5g_RAW_F_Pnn.log \
        [--output_dir plots_debug] \
        [--stec_file predictions.stec]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_scripts_dir = str(Path(__file__).parent)
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Re-use feature preparation from the inference script
from infer_from_log import read_log_file, prepare_features  # noqa: E402
from stec.config.config_parser import load_config
from stec.data.feature_registry import initialize_feature_registry


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_file", required=True)
    p.add_argument("--output_dir", default="plots_debug")
    p.add_argument("--ele_cutoff", type=float, default=5.0)
    p.add_argument(
        "--stec_file",
        default=None,
        help="Path to .stec output file for prediction sanity-check plots",
    )
    return p.parse_args()


def sat_colors(prns):
    """Assign a consistent color per unique satellite PRN."""
    unique = sorted(set(prns))
    cmap = cm.get_cmap("tab20", len(unique))
    return {prn: cmap(i) for i, prn in enumerate(unique)}, unique


def sod_to_hhmm(sod_arr):
    """Convert seconds-of-day to HH:MM label array."""
    return [f"{int(s) // 3600:02d}:{(int(s) % 3600) // 60:02d}" for s in sod_arr]


def plot_sky(df, out_dir):
    """Polar sky plot: elevation (radial) vs azimuth, colored by PRN."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "polar"})
    colors, unique_prns = sat_colors(df["PRN"])

    for prn in unique_prns:
        sub = df[df["PRN"] == prn]
        # polar: theta = azimuth in radians, r = 90 - elevation (zenith = 0)
        theta = np.deg2rad(sub["satazi"].values)
        r = 90.0 - sub["satele"].values
        ax.plot(theta, r, ".", markersize=2, color=colors[prn], label=prn)
        # mark start
        ax.plot(theta[0], r[0], "o", markersize=5, color=colors[prn])

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 30, 60, 90])
    ax.set_yticklabels(["90°", "60°", "30°", "0°"])
    ax.set_title("Sky plot (elevation vs azimuth)", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(out_dir / "01_sky_plot.png", dpi=120)
    plt.close(fig)
    print("  Saved: 01_sky_plot.png")


def plot_ipp_map(df, out_dir):
    """World map with IPP scatter and station location."""
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    ax.set_global()

    colors, unique_prns = sat_colors(df["PRN"])
    for prn in unique_prns:
        sub = df[df["PRN"] == prn]
        ax.scatter(
            sub["lon_ipp"],
            sub["lat_ipp"],
            s=1,
            color=colors[prn],
            transform=ccrs.PlateCarree(),
            label=prn,
            zorder=3,
        )

    # Station marker (single point — same for all rows)
    lat_sta = df["lat_sta"].iloc[0]
    lon_sta = df["lon_sta"].iloc[0]
    ax.plot(
        lon_sta,
        lat_sta,
        "r*",
        markersize=12,
        transform=ccrs.PlateCarree(),
        zorder=5,
        label="Station",
    )

    ax.set_title("IPP positions (geographic)")
    ax.legend(loc="lower left", fontsize=6, markerscale=4, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "02_ipp_map_geo.png", dpi=120)
    plt.close(fig)
    print("  Saved: 02_ipp_map_geo.png")


def plot_ipp_sm_map(df, out_dir):
    """Solar magnetic coordinate scatter for IPP and station."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors, unique_prns = sat_colors(df["PRN"])

    for ax, (lat_col, lon_col, title) in zip(
        axes,
        [
            ("lat_ipp", "lon_ipp", "IPP geographic"),
            ("sm_lat_ipp", "sm_lon_ipp", "IPP solar magnetic"),
        ],
    ):
        for prn in unique_prns:
            sub = df[df["PRN"] == prn]
            ax.scatter(sub[lon_col], sub[lat_col], s=2, color=colors[prn], label=prn)
        # Station
        lat_key = lat_col.replace("ipp", "sta")
        lon_key = lon_col.replace("ipp", "sta")
        ax.scatter(
            df[lon_key].iloc[0],
            df[lat_key].iloc[0],
            s=120,
            marker="*",
            color="red",
            zorder=5,
            label="Station",
        )
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)

    axes[0].legend(fontsize=6, markerscale=4, ncol=3)
    fig.suptitle("Geographic vs Solar Magnetic IPP coordinates", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "03_ipp_geo_vs_sm.png", dpi=120)
    plt.close(fig)
    print("  Saved: 03_ipp_geo_vs_sm.png")


def plot_elevation_arcs(df, out_dir):
    """Elevation angle vs time for each satellite."""
    fig, ax = plt.subplots(figsize=(14, 5))
    colors, unique_prns = sat_colors(df["PRN"])

    for prn in unique_prns:
        sub = df[df["PRN"] == prn].sort_values("sod")
        ax.plot(
            sub["sod"] / 3600, sub["satele"], color=colors[prn], label=prn, linewidth=1
        )

    ax.axhline(5, color="k", linestyle="--", linewidth=0.8, label="Cutoff 5°")
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("Elevation (°)")
    ax.set_title("Satellite elevation arcs over the day")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 95)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / "04_elevation_arcs.png", dpi=120)
    plt.close(fig)
    print("  Saved: 04_elevation_arcs.png")


def plot_temporal_features(df, out_dir):
    """Local time hours, sod, and doy sanity check."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors, unique_prns = sat_colors(df["PRN"])

    # local_time_hours vs sod
    for prn in unique_prns:
        sub = df[df["PRN"] == prn].sort_values("sod")
        axes[0].plot(
            sub["sod"] / 3600,
            sub["local_time_hours"],
            ".",
            markersize=2,
            color=colors[prn],
        )
    axes[0].set_xlabel("SOD / 3600 (UTC hour)")
    axes[0].set_ylabel("local_time_hours")
    axes[0].set_title("Local time hours vs UTC")
    axes[0].grid(True, alpha=0.3)

    # IPP longitude vs local time — should be roughly sod/3600 shifted by lon/15
    for prn in unique_prns:
        sub = df[df["PRN"] == prn].sort_values("sod")
        expected = (sub["sod"] / 3600 + sub["lon_ipp"] / 15) % 24
        axes[1].plot(
            sub["local_time_hours"], expected, ".", markersize=2, color=colors[prn]
        )
    axes[1].plot([0, 24], [0, 24], "k--", linewidth=1, label="1:1")
    axes[1].set_xlabel("local_time_hours (computed)")
    axes[1].set_ylabel("sod/3600 + lon_ipp/15 (expected)")
    axes[1].set_title("Local time consistency check")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Histogram of sod coverage
    axes[2].hist(
        df["sod"] / 3600, bins=48, color="steelblue", edgecolor="white", linewidth=0.3
    )
    axes[2].set_xlabel("UTC hour")
    axes[2].set_ylabel("Observation count")
    axes[2].set_title("Temporal coverage (obs per 30-min bin)")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Temporal feature verification", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "05_temporal_features.png", dpi=120)
    plt.close(fig)
    print("  Saved: 05_temporal_features.png")


def plot_station_coords(df, out_dir):
    """Station geographic vs solar magnetic coordinates (should be constant over day)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, col, label in zip(
        axes.flat,
        ["lat_sta", "lon_sta", "sm_lat_sta", "sm_lon_sta"],
        ["lat_sta (°)", "lon_sta (°)", "sm_lat_sta (°)", "sm_lon_sta (°)"],
    ):
        ax.plot(
            df["sod"] / 3600, df[col], ".", markersize=1, alpha=0.3, color="steelblue"
        )
        ax.set_xlabel("UTC hour")
        ax.set_ylabel(label)
        ax.set_title(f"{col} over day")
        # Compute spread
        spread = df[col].max() - df[col].min()
        ax.set_title(f"{col}  (spread: {spread:.4f}°)")
        ax.grid(True, alpha=0.3)
        # Add mean line
        mean_val = df[col].mean()
        ax.axhline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"mean={mean_val:.2f}°",
        )
        ax.legend(fontsize=8)

    fig.suptitle(
        "Station coordinates (should be near-constant if single station)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out_dir / "06_station_coords.png", dpi=120)
    plt.close(fig)
    print("  Saved: 06_station_coords.png")


def plot_swi_features(df, out_dir, config):
    """SWI features over the day (constant per hour, step function)."""
    from stec.data.feature_registry import FeatureType

    swi_features = config["feature_registry"].get_features_by_type(FeatureType.SWI)
    if not swi_features:
        print("  No SWI features — skipping SWI plot")
        return

    n = len(swi_features)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)

    for i, feat in enumerate(swi_features):
        ax = axes[i // cols][i % cols]
        ax.plot(df["sod"] / 3600, df[feat], ".", markersize=2, color="darkorange")
        ax.set_xlabel("UTC hour")
        ax.set_ylabel(feat)
        ax.set_title(feat)
        ax.grid(True, alpha=0.3)

    # Hide any unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Space Weather Index features over the day", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "07_swi_features.png", dpi=120)
    plt.close(fig)
    print("  Saved: 07_swi_features.png")


def plot_ipp_arcs(df, out_dir):
    """IPP lat/lon as time series per satellite."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors, unique_prns = sat_colors(df["PRN"])

    for ax, col in zip(axes.flat, ["lat_ipp", "lon_ipp", "sm_lat_ipp", "sm_lon_ipp"]):
        for prn in unique_prns:
            sub = df[df["PRN"] == prn].sort_values("sod")
            ax.plot(
                sub["sod"] / 3600, sub[col], color=colors[prn], linewidth=0.8, label=prn
            )
        ax.set_xlabel("UTC hour")
        ax.set_ylabel(f"{col} (°)")
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    axes[0][0].legend(fontsize=6, ncol=3)
    fig.suptitle("IPP coordinate arcs over the day", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "08_ipp_arcs.png", dpi=120)
    plt.close(fig)
    print("  Saved: 08_ipp_arcs.png")


def load_stec_file(stec_path: str) -> "pd.DataFrame":
    """Load a .stec prediction file into a DataFrame with sod column added.

    Handles both 9-column (no GIM) and 10-column (with GIM) formats.
    """
    import pandas as pd

    # Peek at first line to detect column count and presence of a text header
    with open(stec_path) as fh:
        first_line = fh.readline()
    n_cols = len(first_line.split(";"))
    has_header = not first_line.split(";")[0].strip().lstrip("-").isdigit()

    base_names = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "PRN",
        "stec_pred",
        "stec_unc",
    ]
    names = base_names + (["gim_stec"] if n_cols >= 10 else [])

    if has_header:
        df = pd.read_csv(stec_path, sep=";", header=0)
        df = df.rename(
            columns={
                "STEC": "stec_pred",
                "Uncertainty": "stec_unc",
                "GIM_STEC": "gim_stec",
            }
        )
    else:
        df = pd.read_csv(stec_path, sep=";", header=None, names=names)

    df["sod"] = df["hour"] * 3600 + df["minute"] * 60 + df["second"]
    return df


def plot_stec_arcs(stec_df, out_dir):
    """STEC prediction + ±1σ uncertainty band per satellite arc."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors, unique_prns = sat_colors(stec_df["PRN"])

    for prn in unique_prns:
        sub = stec_df[stec_df["PRN"] == prn].sort_values("sod")
        t = sub["sod"] / 3600
        mu = sub["stec_pred"].values
        sigma = sub["stec_unc"].values
        c = colors[prn]
        axes[0].plot(t, mu, color=c, linewidth=0.8, label=prn)
        axes[0].fill_between(t, mu - sigma, mu + sigma, color=c, alpha=0.15)
        axes[1].plot(t, sigma, color=c, linewidth=0.8)

    axes[0].set_ylabel("STEC prediction (TECU)")
    axes[0].set_title("STEC arcs with ±1σ uncertainty")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=6, ncol=5)

    axes[1].set_xlabel("UTC hour")
    axes[1].set_ylabel("Uncertainty σ (TECU)")
    axes[1].set_title("Prediction uncertainty per arc")
    axes[1].set_xlim(0, 24)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "09_stec_arcs.png", dpi=120)
    plt.close(fig)
    print("  Saved: 09_stec_arcs.png")


def plot_stec_vs_elevation(feat_df, stec_df, out_dir):
    """STEC and uncertainty vs elevation — lower elevation should give higher STEC."""

    # Merge on PRN + sod (round sod to nearest 30 s to handle small differences)
    feat = feat_df[["PRN", "sod", "satele"]].copy()
    feat["sod_key"] = (feat["sod"] / 30).round().astype(int)
    pred = stec_df[["PRN", "sod", "stec_pred", "stec_unc"]].copy()
    pred["sod_key"] = (pred["sod"] / 30).round().astype(int)
    merged = feat.merge(
        pred[["PRN", "sod_key", "stec_pred", "stec_unc"]],
        on=["PRN", "sod_key"],
        how="inner",
    )

    if merged.empty:
        print(
            "  WARNING: could not merge feature and stec dataframes — skipping elevation plot"
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        merged["satele"], merged["stec_pred"], s=1, alpha=0.2, color="steelblue"
    )
    axes[0].set_xlabel("Elevation (°)")
    axes[0].set_ylabel("STEC (TECU)")
    axes[0].set_title(
        "STEC vs elevation\n(lower elevation → longer path → higher STEC)"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        merged["satele"], merged["stec_unc"], s=1, alpha=0.2, color="darkorange"
    )
    axes[1].set_xlabel("Elevation (°)")
    axes[1].set_ylabel("Uncertainty σ (TECU)")
    axes[1].set_title(
        "Uncertainty vs elevation\n(expected: higher uncertainty at low elevation)"
    )
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "10_stec_vs_elevation.png", dpi=120)
    plt.close(fig)
    print("  Saved: 10_stec_vs_elevation.png")


def plot_stec_distribution(stec_df, out_dir):
    """Histogram of STEC predictions and uncertainty distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        stec_df["stec_pred"],
        bins=80,
        color="steelblue",
        edgecolor="white",
        linewidth=0.3,
    )
    axes[0].set_xlabel("STEC prediction (TECU)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(
        f"STEC distribution  (median={stec_df['stec_pred'].median():.1f}, "
        f"max={stec_df['stec_pred'].max():.1f} TECU)"
    )
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].hist(
        stec_df["stec_unc"],
        bins=80,
        color="darkorange",
        edgecolor="white",
        linewidth=0.3,
    )
    axes[1].set_xlabel("Uncertainty σ (TECU)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(
        f"Uncertainty distribution  (median={stec_df['stec_unc'].median():.2f} TECU)"
    )
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("STEC prediction and uncertainty distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "11_stec_distribution.png", dpi=120)
    plt.close(fig)
    print("  Saved: 11_stec_distribution.png")


def plot_stec_vs_gim(stec_df, out_dir):
    """Three-panel comparison of model STEC vs IGS GIM STEC.

    Panel 1: time series overlay per satellite (model solid, GIM dashed)
    Panel 2: scatter model vs GIM with 1:1 line and RMSE/bias annotation
    Panel 3: residuals (model - GIM) vs time, coloured by satellite
    """
    if "gim_stec" not in stec_df.columns:
        print("  No gim_stec column in .stec file — skipping GIM comparison plot")
        return

    valid = stec_df.dropna(subset=["gim_stec"])
    if valid.empty:
        print("  All GIM values are NaN — skipping GIM comparison plot")
        return

    colors, unique_prns = sat_colors(stec_df["PRN"])

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    ax_ts = fig.add_subplot(gs[0, :])  # full-width time series
    ax_sc = fig.add_subplot(gs[1, 0])  # scatter
    ax_res = fig.add_subplot(gs[1, 1])  # residuals

    # --- Time series ---
    for prn in unique_prns:
        sub = stec_df[stec_df["PRN"] == prn].sort_values("sod")
        t = sub["sod"] / 3600
        c = colors[prn]
        ax_ts.plot(t, sub["stec_pred"], color=c, linewidth=0.9, label=prn)
        ax_ts.plot(
            t, sub["gim_stec"], color=c, linewidth=0.9, linestyle="--", alpha=0.6
        )

    # Proxy artists for the legend entries (solid=model, dashed=GIM)
    from matplotlib.lines import Line2D

    ax_ts.legend(
        handles=[
            Line2D([0], [0], color="k", linewidth=1.2, label="Model"),
            Line2D(
                [0],
                [0],
                color="k",
                linewidth=1.2,
                linestyle="--",
                alpha=0.6,
                label="IGS GIM",
            ),
        ],
        loc="upper right",
        fontsize=9,
    )
    ax_ts.set_xlabel("UTC hour")
    ax_ts.set_ylabel("STEC (TECU)")
    ax_ts.set_title("Model (solid) vs IGS GIM (dashed) STEC arcs")
    ax_ts.set_xlim(0, 24)
    ax_ts.grid(True, alpha=0.3)

    # --- Scatter ---
    pred = valid["stec_pred"].values
    gim = valid["gim_stec"].values
    residuals = pred - gim
    rmse = float(np.sqrt(np.mean(residuals**2)))
    bias = float(np.mean(residuals))
    mae = float(np.mean(np.abs(residuals)))

    ax_sc.scatter(gim, pred, s=1, alpha=0.15, color="steelblue")
    lim = [min(gim.min(), pred.min()) * 0.95, max(gim.max(), pred.max()) * 1.05]
    ax_sc.plot(lim, lim, "k--", linewidth=1, label="1:1")
    ax_sc.set_xlim(lim)
    ax_sc.set_ylim(lim)
    ax_sc.set_xlabel("IGS GIM STEC (TECU)")
    ax_sc.set_ylabel("Model STEC (TECU)")
    ax_sc.set_title(
        f"Scatter: Model vs GIM\nRMSE={rmse:.2f}  Bias={bias:+.2f}  MAE={mae:.2f} TECU"
    )
    ax_sc.legend(fontsize=8)
    ax_sc.grid(True, alpha=0.3)

    # --- Residuals vs time ---
    for prn in unique_prns:
        sub = valid[valid["PRN"] == prn].sort_values("sod")
        ax_res.plot(
            sub["sod"] / 3600,
            sub["stec_pred"].values - sub["gim_stec"].values,
            color=colors[prn],
            linewidth=0.8,
        )
    ax_res.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax_res.set_xlabel("UTC hour")
    ax_res.set_ylabel("Model − GIM (TECU)")
    ax_res.set_title("Residuals (Model − GIM) per arc")
    ax_res.set_xlim(0, 24)
    ax_res.grid(True, alpha=0.3)

    fig.suptitle("Model STEC vs IGS GIM STEC", fontsize=13)
    fig.savefig(out_dir / "12_stec_vs_gim.png", dpi=120)
    plt.close(fig)
    print("  Saved: 12_stec_vs_gim.png")


def main():
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    if "target" not in config:
        config["target"] = "stec"
    config["device"] = "cpu"
    initialize_feature_registry(config)

    logger.info(f"Reading {args.data_file}")
    df_raw = read_log_file(args.data_file)
    df = prepare_features(df_raw, config, args.ele_cutoff, logger)

    logger.info(f"Generating diagnostic plots → {out_dir}/")
    print(
        f"\nDataset summary: {len(df)} observations, {df['PRN'].nunique()} satellites"
    )
    print(f"  Year={df['year'].iloc[0]}, DOY={df['doy'].iloc[0]}")
    print(
        f"  Station: lat={df['lat_sta'].iloc[0]:.4f}°, lon={df['lon_sta'].iloc[0]:.4f}°"
    )
    print(f"  SOD range: {df['sod'].min():.0f}–{df['sod'].max():.0f} s")
    print()

    plot_sky(df, out_dir)
    plot_ipp_map(df, out_dir)
    plot_ipp_sm_map(df, out_dir)
    plot_elevation_arcs(df, out_dir)
    plot_temporal_features(df, out_dir)
    plot_station_coords(df, out_dir)
    plot_swi_features(df, out_dir, config)
    plot_ipp_arcs(df, out_dir)

    if args.stec_file:
        logger.info(f"Loading STEC predictions from {args.stec_file}")
        stec_df = load_stec_file(args.stec_file)
        print(
            f"\nSTEC file summary: {len(stec_df)} predictions, "
            f"range [{stec_df['stec_pred'].min():.1f}, {stec_df['stec_pred'].max():.1f}] TECU"
        )
        plot_stec_arcs(stec_df, out_dir)
        plot_stec_vs_elevation(df, stec_df, out_dir)
        plot_stec_distribution(stec_df, out_dir)
        plot_stec_vs_gim(stec_df, out_dir)

    print(f"\nAll plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
