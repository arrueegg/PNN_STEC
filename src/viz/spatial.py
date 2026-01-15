"""
Spatial and geographic visualization functions.

This module handles geographic plots, spatial error maps, and coordinate-based visualizations.
"""

import pandas as pd
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .base import (
    save_plot,
)


def plot_spatial_error_map(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Create spatial map of model errors using geographic coordinates.

    Args:
        df: DataFrame with 'lat_ipp', 'lon_ipp', 'target_stec', 'pred_stec' columns
        output_dir: Directory to save plot
    """
    df = df.copy()
    df["mae"] = np.abs(df["target_stec"] - df["pred_stec"])
    df["residual"] = df["target_stec"] - df["pred_stec"]

    # Create spatial bins
    lat_bins = np.linspace(-90, 90, 37)  # 5-degree bins
    lon_bins = np.linspace(-180, 180, 73)  # 5-degree bins

    df["lat_bin"] = pd.cut(df["lat_ipp"], bins=lat_bins, include_lowest=True)
    df["lon_bin"] = pd.cut(df["lon_ipp"], bins=lon_bins, include_lowest=True)

    # Calculate statistics for each bin
    spatial_stats = (
        df.groupby(["lat_bin", "lon_bin"])
        .agg({"mae": ["mean", "count"], "residual": "mean"})
        .reset_index()
    )

    spatial_stats.columns = ["lat_bin", "lon_bin", "mae_mean", "count", "residual_mean"]

    # Filter bins with sufficient data
    spatial_stats = spatial_stats[spatial_stats["count"] >= 10]

    # Get bin centers
    spatial_stats["lat_center"] = spatial_stats["lat_bin"].apply(lambda x: x.mid)
    spatial_stats["lon_center"] = spatial_stats["lon_bin"].apply(lambda x: x.mid)

    # Create standalone plots instead of subplots

    # 1. MAE map - standalone
    fig1, ax1 = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    scatter1 = ax1.scatter(
        spatial_stats["lon_center"],
        spatial_stats["lat_center"],
        c=spatial_stats["mae_mean"],
        s=50,
        cmap="viridis",
        transform=ccrs.PlateCarree(),
        alpha=0.8,
    )

    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS)
    ax1.set_global()
    ax1.set_title("Mean Absolute Error by Location", fontweight="bold", pad=20)

    # Add colorbar
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar1 = plt.colorbar(scatter1, cax=cax1)
    cbar1.set_label("MAE [TECU]", fontweight="bold")

    plt.tight_layout()
    save_plot(fig1, "spatial_error_map_mae.png", output_dir)
    plt.close(fig1)

    # 2. Residual map - standalone
    fig2, ax2 = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    # Center colormap at zero for residuals
    vmax = max(
        abs(spatial_stats["residual_mean"].min()),
        abs(spatial_stats["residual_mean"].max()),
    )
    scatter2 = ax2.scatter(
        spatial_stats["lon_center"],
        spatial_stats["lat_center"],
        c=spatial_stats["residual_mean"],
        s=50,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        alpha=0.8,
    )

    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS)
    ax2.set_global()
    ax2.set_title("Mean Residual by Location", fontweight="bold", pad=20)

    # Add colorbar
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar2 = plt.colorbar(scatter2, cax=cax2)
    cbar2.set_label("Residual [TECU]", fontweight="bold")

    plt.tight_layout()
    save_plot(fig2, "spatial_error_map_residual.png", output_dir)
    plt.close(fig2)

    # 3. Sample count map - standalone
    fig3, ax3 = plt.subplots(
        figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    scatter3 = ax3.scatter(
        spatial_stats["lon_center"],
        spatial_stats["lat_center"],
        c=spatial_stats["count"],
        s=50,
        cmap="plasma",
        transform=ccrs.PlateCarree(),
        alpha=0.8,
    )

    ax3.add_feature(cfeature.COASTLINE)
    ax3.add_feature(cfeature.BORDERS)
    ax3.set_global()
    ax3.set_title("Sample Count by Location", fontweight="bold", pad=20)

    # Add colorbar
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar3 = plt.colorbar(scatter3, cax=cax3)
    cbar3.set_label("Count", fontweight="bold")

    plt.tight_layout()
    save_plot(fig3, "spatial_error_map_count.png", output_dir)
    plt.close(fig3)

    # 4. Save statistics to text file
    stats_text = f"""Spatial Error Statistics

Total Grid Cells: {len(spatial_stats)}

MAE Statistics:
Mean: {spatial_stats['mae_mean'].mean():.4f} TECU
Std:  {spatial_stats['mae_mean'].std():.4f} TECU
Min:  {spatial_stats['mae_mean'].min():.4f} TECU
Max:  {spatial_stats['mae_mean'].max():.4f} TECU

Residual Statistics:
Mean: {spatial_stats['residual_mean'].mean():.4f} TECU
Std:  {spatial_stats['residual_mean'].std():.4f} TECU
Min:  {spatial_stats['residual_mean'].min():.4f} TECU
Max:  {spatial_stats['residual_mean'].max():.4f} TECU

Sample Count:
Total: {spatial_stats['count'].sum():,}
Mean per cell: {spatial_stats['count'].mean():.0f}
"""

    # Save metrics to text file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "spatial_error_map_metrics.txt"), "w") as f:
        f.write(stats_text)


def plot_spatial_error_map_by_local_time(
    df: pd.DataFrame, output_dir: str = "plots"
) -> None:
    """Create spatial error maps split by local time periods."""
    df = df.copy()
    df["mae"] = np.abs(df["target_stec"] - df["pred_stec"])

    # Define time periods
    time_periods = {
        "Dawn (04-08h)": (4, 8),
        "Day (08-16h)": (8, 16),
        "Dusk (16-20h)": (16, 20),
        "Night (20-04h)": [(20, 24), (0, 4)],
    }

    fig = plt.figure(figsize=(20, 16))

    for i, (period_name, time_range) in enumerate(time_periods.items(), 1):
        # Filter data for time period
        if isinstance(time_range[0], tuple):
            # Handle night period (spans midnight)
            mask = (
                (df["time"] >= time_range[0][0]) & (df["time"] < time_range[0][1])
            ) | ((df["time"] >= time_range[1][0]) & (df["time"] < time_range[1][1]))
        else:
            mask = (df["time"] >= time_range[0]) & (df["time"] < time_range[1])

        df_period = df[mask]

        if len(df_period) == 0:
            continue

        # Create spatial bins
        lat_bins = np.linspace(-90, 90, 19)  # 10-degree bins
        lon_bins = np.linspace(-180, 180, 37)  # 10-degree bins

        df_period["lat_bin"] = pd.cut(
            df_period["lat_ipp"], bins=lat_bins, include_lowest=True
        )
        df_period["lon_bin"] = pd.cut(
            df_period["lon_ipp"], bins=lon_bins, include_lowest=True
        )

        # Calculate MAE for each bin
        spatial_mae = (
            df_period.groupby(["lat_bin", "lon_bin"])["mae"].mean().reset_index()
        )
        spatial_mae = spatial_mae.dropna()

        if len(spatial_mae) == 0:
            continue

        # Get bin centers
        spatial_mae["lat_center"] = spatial_mae["lat_bin"].apply(lambda x: x.mid)
        spatial_mae["lon_center"] = spatial_mae["lon_bin"].apply(lambda x: x.mid)

        # Create subplot
        ax = fig.add_subplot(2, 2, i, projection=ccrs.PlateCarree())

        scatter = ax.scatter(
            spatial_mae["lon_center"],
            spatial_mae["lat_center"],
            c=spatial_mae["mae"],
            s=80,
            cmap="viridis",
            transform=ccrs.PlateCarree(),
            alpha=0.8,
            vmin=0,
            vmax=df["mae"].quantile(0.95),
        )

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_global()
        ax.set_title(
            f"{period_name}\n(N={len(df_period):,})", fontweight="bold", pad=20
        )

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label("MAE [TECU]", fontweight="bold")

    plt.suptitle(
        "Spatial Error Distribution by Local Time", fontsize=24, fontweight="bold"
    )
    plt.tight_layout()
    save_plot(fig, "spatial_error_by_local_time.png", output_dir)


def plot_solar_magnetic_ipp_error_map(
    df: pd.DataFrame, output_dir: str = "plots"
) -> None:
    """Create error map in solar magnetic coordinates."""
    df = df.copy()
    df["mae"] = np.abs(df["target_stec"] - df["pred_stec"])
    df["residual"] = df["target_stec"] - df["pred_stec"]

    # Check if solar magnetic coordinates exist
    if "sm_lat_ipp" not in df.columns or "sm_lon_ipp" not in df.columns:
        logger = logging.getLogger(__name__)
        logger.warning("Solar magnetic coordinates not found. Skipping this plot.")
        return

    # Create spatial bins for solar magnetic coordinates
    sm_lat_bins = np.linspace(-90, 90, 37)  # 5-degree bins
    sm_lon_bins = np.linspace(-180, 180, 73)  # 5-degree bins

    df["sm_lat_bin"] = pd.cut(df["sm_lat_ipp"], bins=sm_lat_bins, include_lowest=True)
    df["sm_lon_bin"] = pd.cut(df["sm_lon_ipp"], bins=sm_lon_bins, include_lowest=True)

    # Calculate statistics for each bin
    sm_stats = (
        df.groupby(["sm_lat_bin", "sm_lon_bin"])
        .agg({"mae": ["mean", "count"], "residual": "mean"})
        .reset_index()
    )

    sm_stats.columns = [
        "sm_lat_bin",
        "sm_lon_bin",
        "mae_mean",
        "count",
        "residual_mean",
    ]

    # Filter bins with sufficient data
    sm_stats = sm_stats[sm_stats["count"] >= 10]

    # Get bin centers
    sm_stats["sm_lat_center"] = sm_stats["sm_lat_bin"].apply(lambda x: x.mid)
    sm_stats["sm_lon_center"] = sm_stats["sm_lon_bin"].apply(lambda x: x.mid)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # MAE map
    scatter1 = ax1.scatter(
        sm_stats["sm_lon_center"],
        sm_stats["sm_lat_center"],
        c=sm_stats["mae_mean"],
        s=50,
        cmap="viridis",
        alpha=0.8,
    )
    ax1.set_xlabel("Solar Magnetic Longitude [°]", fontweight="bold")
    ax1.set_ylabel("Solar Magnetic Latitude [°]", fontweight="bold")
    ax1.set_title("MAE in Solar Magnetic Coordinates", fontweight="bold", pad=20)
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label("MAE [TECU]", fontweight="bold")

    # Residual map
    vmax = max(
        abs(sm_stats["residual_mean"].min()), abs(sm_stats["residual_mean"].max())
    )
    scatter2 = ax2.scatter(
        sm_stats["sm_lon_center"],
        sm_stats["sm_lat_center"],
        c=sm_stats["residual_mean"],
        s=50,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        alpha=0.8,
    )
    ax2.set_xlabel("Solar Magnetic Longitude [°]", fontweight="bold")
    ax2.set_ylabel("Solar Magnetic Latitude [°]", fontweight="bold")
    ax2.set_title("Residuals in Solar Magnetic Coordinates", fontweight="bold", pad=20)
    ax2.grid(True, alpha=0.3)

    # Add colorbar
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label("Residual [TECU]", fontweight="bold")

    plt.tight_layout()
    save_plot(fig, "solar_magnetic_error_map.png", output_dir)


def plot_box_by_lat(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Creates a two-panel plot showing RMSE/MAE and residual boxplots by solar magnetic latitude.

    Args:
        df: DataFrame with test results containing sm_lat_ipp, target_stec, pred_stec
        output_dir: Directory to save the plot
    """
    from analysis.metrics import calc_rmse

    df = df.copy()
    df["error"] = df["target_stec"] - df["pred_stec"]
    df["ae"] = np.abs(df["error"])

    # Use sm_lat_ipp as the magnetic latitude column
    lat_col = "sm_lat_ipp"
    if lat_col not in df.columns:
        return  # Skip if column doesn't exist

    
    # Set shared style
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
    plt.rcParams['figure.dpi'] = 300
    colors = sns.color_palette("colorblind")

    bins = np.arange(-90, 91, 10)
    df["lat_bin"] = pd.cut(df[lat_col], bins=bins, include_lowest=True, right=False)

    # Ensure all bins are present, even if empty
    all_bins = pd.IntervalIndex.from_breaks(bins, closed="left")

    rmse_lat = (
        df.groupby("lat_bin", observed=False)["error"]
        .apply(calc_rmse)
        .reindex(all_bins)
        .reset_index()
    )
    mae_lat = (
        df.groupby("lat_bin", observed=False)["ae"]
        .mean()
        .reindex(all_bins)
        .reset_index()
    )

    grouped = (
        df.groupby("lat_bin", observed=False)["error"].apply(list).reindex(all_bins)
    )
    box_data = [
        grouped[bin] if len(grouped[bin]) > 0 else [np.nan] for bin in grouped.index
    ]

    bin_centers = [(interval.left + interval.right) / 2 for interval in grouped.index]

    fig, ax = plt.subplots(figsize=(12, 7))

    # 1. Boxplot of residuals (Background)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, zorder=1, alpha=0.8)
    bp = ax.boxplot(
        box_data,
        positions=bin_centers,
        widths=5,
        showfliers=False,
        zorder=2,
        patch_artist=True,
        notch=False,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(colors[2])
        patch.set_alpha(0.5)  # Slightly more transparent to see grid
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_linewidth(1.5)
            if element == 'medians':
                item.set_color('black')

    # 2. RMSE and MAE Lines (Foreground)
    ax.plot(bin_centers, rmse_lat["error"], marker="o", label="RMSE", color=colors[0], linewidth=3, markersize=8, zorder=3)
    ax.plot(bin_centers, mae_lat["ae"], marker="s", label="MAE", color=colors[1], linewidth=3, markersize=8, zorder=3)

    # Styling
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{b}" for b in bins], rotation=45)
    ax.set_ylim([-30, 30])
    ax.set_xlabel("Solar Magnetic Latitude [°]")
    ax.set_ylabel("Residual / Error (TECU)")
    
    # Legend
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.title("Performance by Solar Magnetic Latitude", fontweight="bold", y=1.02)
    plt.tight_layout()
    save_plot(fig, "mLat_summary.png", output_dir)
    plt.close(fig)
