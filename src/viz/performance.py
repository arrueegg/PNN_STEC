"""
Model performance visualization functions.

This module handles scatter plots, prediction quality plots, and model evaluation visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .base import (
    FIGSIZE_WIDE,
    FIGSIZE_SQUARE,
    get_scientific_label,
    save_plot,
)


def plot_prediction_scatter(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Create individual scatter plots of predictions vs true values.

    Args:
        df: DataFrame with 'target_stec' and 'pred_stec' columns
        output_dir: Directory to save plot
    """
    # Extract data
    y_true = df["target_stec"].values
    y_pred = df["pred_stec"].values

    # Calculate metrics
    r2_score(y_true, y_pred)
    corr, p_value = pearsonr(y_true, y_pred)
    np.sqrt(np.mean((y_true - y_pred) ** 2))
    np.mean(np.abs(y_true - y_pred))
    np.mean(y_pred - y_true)

    # 1. Main scatter plot
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    h = ax.hist2d(y_true, y_pred, bins=50, cmap="Blues", norm=LogNorm())

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r-",
        linewidth=3,
        alpha=0.8,
        label="Perfect Prediction",
    )

    ax.set_xlabel("True STEC [TECU]", fontweight="bold")
    ax.set_ylabel("Predicted STEC [TECU]", fontweight="bold")
    ax.set_title("Prediction Analysis: Predicted vs Observed STEC", fontweight="bold", pad=20)
    ax.legend(framealpha=0.9)

    # Add colorbar
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label("Count (log scale)", fontweight="bold")

    plt.tight_layout()
    save_plot(fig, "prediction_scatter.png", output_dir)
    plt.close(fig)

    plt.close(fig)


def plot_az_el_heatmap(
    df: pd.DataFrame, output_dir: str = "plots", metric: str = "residual"
) -> None:
    """
    Create azimuth-elevation heatmap for specified metric.

    Args:
        df: DataFrame with 'satazi', 'satele', and target columns
        output_dir: Directory to save plot
        metric: Metric to plot ('residual', 'mae', 'pred_stec', etc.)
    """
    df = df.copy()

    # Calculate metric if needed
    if metric == "residual":
        df[metric] = df["target_stec"] - df["pred_stec"]
    elif metric == "mae":
        df[metric] = np.abs(df["target_stec"] - df["pred_stec"])
    elif metric == "abs_residual":
        df[metric] = np.abs(df["target_stec"] - df["pred_stec"])

    # Create bins
    az_bins = np.linspace(0, 360, 37)  # 10-degree bins
    el_bins = np.linspace(5, 90, 18)  # 5-degree bins

    # Bin the data
    df["az_bin"] = pd.cut(df["satazi"], bins=az_bins, include_lowest=True)
    df["el_bin"] = pd.cut(df["satele"], bins=el_bins, include_lowest=True)

    # Calculate mean metric for each bin
    pivot_table = df.groupby(["el_bin", "az_bin"])[metric].mean().unstack()

    # Create heatmap
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Choose colormap based on metric
    if metric in ["residual"]:
        cmap = "RdBu_r"
        center = 0
    else:
        cmap = "viridis"
        center = None

    im = ax.imshow(
        pivot_table.values,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        interpolation="bilinear",
    )

    if center is not None:
        # Center colormap at zero for residuals
        vmax = max(abs(pivot_table.min().min()), abs(pivot_table.max().max()))
        im.set_clim(-vmax, vmax)

    # Set labels
    ax.set_xlabel("Azimuth [°]", fontweight="bold")
    ax.set_ylabel("Elevation [°]", fontweight="bold")

    # Set tick labels
    az_ticks = np.arange(0, len(pivot_table.columns), 6)
    el_ticks = np.arange(0, len(pivot_table.index), 3)

    ax.set_xticks(az_ticks)
    ax.set_yticks(el_ticks)
    ax.set_xticklabels([f"{int(az_bins[i]):d}" for i in az_ticks])
    ax.set_yticklabels([f"{int(el_bins[i]):d}" for i in el_ticks])

    # Title and colorbar
    metric_label = get_scientific_label(metric)
    if "residual" in metric.lower():
        title = "Residual Analysis: Azimuth & Elevation"
    else:
        title = f"Spatial Analysis: {metric_label} by Azimuth & Elevation"
        
    ax.set_title(title, fontweight="bold", pad=20)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_label, fontweight="bold")

    plt.tight_layout()
    save_plot(fig, f"{metric}_azimuth_elevation_heatmap.png", output_dir)


def plot_residuals_vs_date(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """Plot residuals vs date with temporal trends using monthly bins and combined metrics."""
    df = df.copy()
    df["residual"] = df["target_stec"] - df["pred_stec"]
    df["abs_residual"] = np.abs(df["residual"])

    # Convert datetime if needed
    if "datetime" not in df.columns and "year" in df.columns and "doy" in df.columns:
        df["datetime"] = pd.to_datetime(df["year"], format="%Y") + pd.to_timedelta(
            df["doy"] - 1, unit="D"
        )

    # Create monthly bins for better visualization
    df["month"] = df["datetime"].dt.to_period("M")

    # Group by month and calculate statistics
    monthly_stats = (
        df.groupby("month")
        .agg(
            {
                "residual": ["mean", "std", "count"],
                "abs_residual": "mean",
                "target_stec": "mean",
                "pred_stec": "mean",
            }
        )
        .reset_index()
    )

    # Flatten column names
    monthly_stats.columns = [
        "month",
        "mean_residual",
        "std_residual",
        "count",
        "mae",
        "mean_target",
        "mean_pred",
    ]

    # Convert month periods to datetime for plotting
    monthly_stats["date"] = monthly_stats["month"].dt.start_time

    # Calculate RMSE for each month
    rmse_list = []
    for month in monthly_stats["month"]:
        month_data = df[df["month"] == month]
        if len(month_data) > 0:
            rmse = np.sqrt(
                np.mean((month_data["target_stec"] - month_data["pred_stec"]) ** 2)
            )
            rmse_list.append(rmse)
        else:
            rmse_list.append(np.nan)

    monthly_stats["rmse"] = rmse_list

    # Create plot with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=FIGSIZE_WIDE, sharex=True)

    # 1. Boxplot of residuals instead of mean residuals
    months_for_boxplot = []
    residuals_for_boxplot = []
    month_positions = []

    for i, month in enumerate(monthly_stats["month"]):
        month_data = df[df["month"] == month]
        if len(month_data) >= 5:  # Only include months with sufficient data
            months_for_boxplot.append(month)
            residuals_for_boxplot.append(month_data["residual"].values)
            month_positions.append(i)

    if residuals_for_boxplot:
        ax1.boxplot(
            residuals_for_boxplot,
            positions=month_positions,
            widths=0.8,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", edgecolor="black", alpha=0.7),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="midnightblue", linewidth=2),
        )

    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.7, linewidth=2)
    ax1.set_ylabel("Residual [TECU]", fontweight="bold")
    ax1.set_title("Residual Analysis: Temporal Evolution", fontweight="bold", pad=20)
    ax1.grid(True, alpha=0.3)

    # 2. Combined MAE and RMSE as line plots
    ax2.plot(
        range(len(monthly_stats)),
        monthly_stats["mae"],
        color="green",
        marker="o",
        linewidth=2,
        markersize=6,
        label="MAE",
    )
    ax2.plot(
        range(len(monthly_stats)),
        monthly_stats["rmse"],
        color="orange",
        marker="s",
        linewidth=2,
        markersize=6,
        label="RMSE",
    )
    ax2.set_ylabel("Error [TECU]", fontweight="bold")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # 3. Sample count as bars
    ax3.bar(
        range(len(monthly_stats)),
        monthly_stats["count"],
        color="purple",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    ax3.set_ylabel("Sample Count", fontweight="bold")
    ax3.set_xlabel("Month", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Format x-axis with date labels (every few months to avoid crowding)
    n_ticks = min(12, len(monthly_stats))  # Maximum 12 ticks for months
    tick_indices = np.linspace(0, len(monthly_stats) - 1, n_ticks, dtype=int)
    tick_labels = [
        monthly_stats.iloc[i]["date"].strftime("%Y-%m") for i in tick_indices
    ]

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(tick_indices)
        ax.set_xlim(-0.5, len(monthly_stats) - 0.5)

    ax3.set_xticklabels(tick_labels, rotation=45)

    plt.tight_layout()
    save_plot(fig, "residuals_vs_date.png", output_dir)


def plot_prediction_density(df: pd.DataFrame, output_dir: str = "plots", max_limit: float = None) -> None:
    """
    Create standalone prediction density plot using hexagonal binning.

    Args:
        df: DataFrame with 'target_stec' and 'pred_stec' columns
        output_dir: Directory to save plot
        max_limit: Optional maximum value for axes (TECU). If None, uses data max.
    """
    # Extract data
    y_true = df["target_stec"].values
    y_pred = df["pred_stec"].values

    # Calculate metrics
    r_value, _ = pearsonr(y_true, y_pred)
    r2_value = r2_score(y_true, y_pred)

    # Set proper axis limits: min=0, equal max for both axes
    min_val = 0  # Always start from 0
    if max_limit is not None:
        max_val = max_limit
    else:
        max_val = max(y_true.max(), y_pred.max())

    # Hexagon density plot with enhanced visuals
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    # Create hexbin plot with even finer gridsize and BuGn colormap
    hb = ax.hexbin(
        y_true,
        y_pred,
        gridsize=100,
        cmap="BuGn",
        mincnt=1,
        extent=[min_val, max_val, min_val, max_val],
        norm=LogNorm(),
    )

    # Perfect prediction line
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r-",
        linewidth=3,
        label="Perfect Prediction",
        alpha=0.9,
    )

    # Set equal axis limits
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel("True STEC [TECU]", fontweight="bold")
    ax.set_ylabel("Predicted STEC [TECU]", fontweight="bold")
    ax.set_title("Prediction Analysis: Predicted vs Observed STEC (Density)", fontweight="bold", pad=25)

    # Add colorbar with log scale
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(hb, cax=cax)
    cbar.set_label("Count", fontweight="bold", rotation=270, labelpad=35)
    cbar.ax.tick_params(labelsize=16)

    # Add legend with metrics
    legend_elements = [
        plt.Line2D([0], [0], color="red", linewidth=3, label="Perfect Prediction"),
        plt.Line2D([0], [0], color="none", label=f"Pearson r = {r_value:.3f}"),
        plt.Line2D([0], [0], color="none", label=f"R² = {r2_value:.3f}"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    filename = "prediction_density_limited.png" if max_limit is not None else "prediction_density.png"
    save_plot(fig, filename, output_dir)

    # Save a version without the legend
    if ax.get_legend():
        ax.get_legend().remove()
    filename_no_legend = filename.replace(".png", "_no_legend.png")
    save_plot(fig, filename_no_legend, output_dir)

    plt.close(fig)
