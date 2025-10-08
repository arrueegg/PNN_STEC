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
    r2 = r2_score(y_true, y_pred)
    corr, p_value = pearsonr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_pred - y_true)

    # 1. Main scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
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
    ax.set_title("Predictions vs True Values", fontweight="bold", pad=20)
    ax.legend()

    # Add colorbar
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label("Count (log scale)", fontweight="bold")

    plt.tight_layout()
    save_plot(fig, "prediction_scatter.png", output_dir)
    plt.close(fig)

    # 2. 2D histogram version
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist2d(y_true, y_pred, bins=100, cmap="viridis")
    ax.plot([min_val, max_val], [min_val, max_val], "r-", linewidth=2, alpha=0.8)
    ax.set_xlabel("True STEC [TECU]", fontweight="bold")
    ax.set_ylabel("Predicted STEC [TECU]", fontweight="bold")
    ax.set_title("Prediction Quality (2D Histogram)", fontweight="bold", pad=20)
    plt.tight_layout()
    save_plot(fig, "prediction_hist2d.png", output_dir)
    plt.close(fig)

    # 3. Density plot version
    fig, ax = plt.subplots(figsize=(10, 8))
    # Use hexbin for density
    hb = ax.hexbin(y_true, y_pred, gridsize=50, cmap='Blues', mincnt=1)
    ax.plot([min_val, max_val], [min_val, max_val], "r-", linewidth=2, alpha=0.8)
    ax.set_xlabel("True STEC [TECU]", fontweight="bold")
    ax.set_ylabel("Predicted STEC [TECU]", fontweight="bold")
    ax.set_title("Prediction Density", fontweight="bold", pad=20)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Count", fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "prediction_density.png", output_dir)
    plt.close(fig)

    # 4. Residuals vs predictions
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.3, s=1)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted STEC [TECU]", fontweight="bold")
    ax.set_ylabel("Residuals [TECU]", fontweight="bold")
    ax.set_title("Residuals vs Predictions", fontweight="bold", pad=20)
    plt.tight_layout()
    save_plot(fig, "residuals_vs_predictions.png", output_dir)
    plt.close(fig)

    # 5. Residuals histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        residuals, bins=50, density=True, alpha=0.7, color="skyblue", edgecolor="black"
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.axvline(
        np.mean(residuals),
        color="orange",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {np.mean(residuals):.3f}",
    )
    ax.set_xlabel("Residuals [TECU]", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Residual Distribution", fontweight="bold", pad=20)
    ax.legend()
    plt.tight_layout()
    save_plot(fig, "residual_histogram.png", output_dir)
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
    ax.set_title(f"{metric_label} vs Azimuth/Elevation", fontweight="bold", pad=20)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_label, fontweight="bold")

    plt.tight_layout()
    save_plot(fig, f"{metric}_azimuth_elevation_heatmap.png", output_dir)


def plot_residuals_vs_date(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """Plot residuals vs date with temporal trends."""
    df = df.copy()
    df["residual"] = df["target_stec"] - df["pred_stec"]
    df["abs_residual"] = np.abs(df["residual"])

    # Convert datetime if needed
    if "datetime" not in df.columns and "year" in df.columns and "doy" in df.columns:

        df["datetime"] = pd.to_datetime(df["year"], format="%Y") + pd.to_timedelta(
            df["doy"] - 1, unit="D"
        )

    # Group by date and calculate statistics
    daily_stats = (
        df.groupby(df["datetime"].dt.date)
        .agg({"residual": ["mean", "std", "count"], "abs_residual": "mean"})
        .reset_index()
    )

    daily_stats.columns = ["date", "mean_residual", "std_residual", "count", "mae"]

    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Mean residuals
    ax1.plot(daily_stats["date"], daily_stats["mean_residual"], "b-", linewidth=2)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax1.set_ylabel("Mean Residual [TECU]", fontweight="bold")
    ax1.set_title("Daily Residual Statistics", fontweight="bold", pad=20)
    ax1.grid(True, alpha=0.3)

    # MAE
    ax2.plot(daily_stats["date"], daily_stats["mae"], "g-", linewidth=2)
    ax2.set_ylabel("MAE [TECU]", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Sample count
    ax3.plot(daily_stats["date"], daily_stats["count"], "orange", linewidth=2)
    ax3.set_ylabel("Sample Count", fontweight="bold")
    ax3.set_xlabel("Date", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(fig, "residuals_vs_date.png", output_dir)


def plot_prediction_density(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Create standalone prediction density plot using 2D histogram with color mapping.
    
    Args:
        df: DataFrame with 'target_stec' and 'pred_stec' columns
        output_dir: Directory to save plot
    """
    # Extract data
    y_true = df["target_stec"].values
    y_pred = df["pred_stec"].values
    
    # Calculate metrics
    r_value, _ = pearsonr(y_true, y_pred)
    r2_value = r2_score(y_true, y_pred)
    
    # Density scatter plot with enhanced visuals
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    
    # Create density plot
    h = ax.hist2d(y_true, y_pred, bins=100, cmap='plasma', norm=LogNorm(), alpha=0.8)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=3, 
            label='Perfect Prediction', alpha=0.9)
    
    ax.set_xlabel('True STEC [TECU]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Predicted STEC [TECU]', fontsize=16, fontweight='bold')
    ax.set_title('Prediction Density Analysis', fontsize=20, fontweight='bold', pad=25)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(h[3], cax=cax)
    cbar.set_label('Density', fontweight='bold', rotation=270, labelpad=35)
    cbar.ax.tick_params(labelsize=16)
    
    # Add legend with metrics
    legend_elements = [
        plt.Line2D([0], [0], color='red', linewidth=3, label='Perfect Prediction'),
        plt.Line2D([0], [0], color='none', label=f'Pearson r = {r_value:.3f}'),
        plt.Line2D([0], [0], color='none', label=f'R² = {r2_value:.3f}')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    save_plot(fig, "prediction_density.png", output_dir)
