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

from .base import (
    FIGSIZE_WIDE,
    get_scientific_label,
    save_plot,
)


def plot_prediction_scatter(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Create comprehensive scatter plot of predictions vs true values.

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

    # Create plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Main scatter plot
    h = ax1.hist2d(y_true, y_pred, bins=50, cmap="Blues", norm=LogNorm())

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r-",
        linewidth=3,
        alpha=0.8,
        label="Perfect Prediction",
    )

    ax1.set_xlabel("True STEC [TECU]", fontweight="bold")
    ax1.set_ylabel("Predicted STEC [TECU]", fontweight="bold")
    ax1.set_title("Predictions vs True Values", fontweight="bold", pad=20)
    ax1.legend()

    # Add colorbar
    cbar = fig.colorbar(h[3], ax=ax1)
    cbar.set_label("Count (log scale)", fontweight="bold")

    # 2. Residuals vs predictions
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.3, s=1)
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Predicted STEC [TECU]", fontweight="bold")
    ax2.set_ylabel("Residuals [TECU]", fontweight="bold")
    ax2.set_title("Residuals vs Predictions", fontweight="bold", pad=20)

    # 3. Residuals histogram
    ax3.hist(
        residuals, bins=50, density=True, alpha=0.7, color="skyblue", edgecolor="black"
    )
    ax3.axvline(0, color="red", linestyle="--", linewidth=2)
    ax3.axvline(
        np.mean(residuals),
        color="orange",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {np.mean(residuals):.3f}",
    )
    ax3.set_xlabel("Residuals [TECU]", fontweight="bold")
    ax3.set_ylabel("Density", fontweight="bold")
    ax3.set_title("Residual Distribution", fontweight="bold", pad=20)
    ax3.legend()

    # 4. Metrics text
    ax4.axis("off")
    metrics_text = f"""
    Model Performance Metrics
    
    R² Score: {r2:.4f}
    Correlation: {corr:.4f} (p={p_value:.2e})
    RMSE: {rmse:.4f} TECU
    MAE: {mae:.4f} TECU
    Bias: {bias:.4f} TECU
    
    Data Points: {len(y_true):,}
    
    True STEC Range: {y_true.min():.2f} - {y_true.max():.2f} TECU
    Pred STEC Range: {y_pred.min():.2f} - {y_pred.max():.2f} TECU
    """
    ax4.text(
        0.1,
        0.9,
        metrics_text,
        transform=ax4.transAxes,
        fontsize=16,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    save_plot(fig, "prediction_scatter_comprehensive.png", output_dir)


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
