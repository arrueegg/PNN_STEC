"""
Distribution and statistical visualization functions.

This module handles boxplots, histograms, and other statistical distribution plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from datetime import datetime, timedelta

from .base import (
    FIGSIZE_HISTOGRAM,
    get_scientific_label,
    save_plot,
)


def plot_binned_boxplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bins: int = 20,
    output_dir: str = "plots",
    bin_range_dict: Optional[Dict[str, tuple]] = None,
) -> None:
    """
    Plot boxplot of y_col values grouped by binned x_col intervals.

    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        bins: Number of bins for x_col
        output_dir: Directory to save plot
        bin_range_dict: Optional dict with (min, max) ranges for features
    """
    df = df.copy()

    # Determine bin edges
    if bin_range_dict and x_col in bin_range_dict:
        min_val, max_val = bin_range_dict[x_col]
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        df["x_bin"] = pd.cut(df[x_col], bins=bin_edges, include_lowest=True)
    else:
        df["x_bin"] = pd.cut(df[x_col], bins=bins)

    # Group data by bins
    grouped = df.groupby("x_bin", observed=True)[y_col].apply(list)
    box_data = [grouped[bin] for bin in grouped.index]
    x_labels = [f"{(b.left):.0f}–{b.right:.0f}" for b in grouped.index]

    # Create plot
    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)

    # Add zero reference line
    ax.axhline(y=0, color="red", linestyle="-", linewidth=2, zorder=1, alpha=0.8)

    # Create boxplot with styling
    bp = ax.boxplot(
        box_data,
        labels=x_labels,
        showfliers=False,
        zorder=2,
        patch_artist=True,
        notch=False,
    )

    # Style the boxplot
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_linewidth(2)

    ax.tick_params(axis="x", rotation=45, labelsize=18)

    # Set labels
    x_label = get_scientific_label(x_col)
    y_label = get_scientific_label(y_col)

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(f"{y_label} vs {x_label}", fontweight="bold", pad=20)

    plt.tight_layout()
    filename = f"{y_col}_vs_{x_col}_boxplot.png"
    save_plot(fig, filename, output_dir)


def plot_binned_boxplot_clipped(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bins: int = 20,
    output_dir: str = "plots",
    bin_range_dict: Optional[Dict[str, tuple]] = None,
    x_limits: Optional[tuple] = None,
    y_limits: Optional[tuple] = None,
    suffix: str = "_clipped",
) -> None:
    """
    Plot boxplot with axis clipping to focus on specific data ranges.

    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        bins: Number of bins for x_col
        output_dir: Directory to save plot
        bin_range_dict: Optional dict with (min, max) ranges for features
        x_limits: Optional (min, max) limits for x-axis
        y_limits: Optional (min, max) limits for y-axis
        suffix: Suffix to add to filename
    """
    df = df.copy()

    # Determine bin edges
    if bin_range_dict and x_col in bin_range_dict:
        min_val, max_val = bin_range_dict[x_col]
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        df["x_bin"] = pd.cut(df[x_col], bins=bin_edges, include_lowest=True)
    else:
        df["x_bin"] = pd.cut(df[x_col], bins=bins)

    # Group data by bins
    grouped = df.groupby("x_bin", observed=True)[y_col].apply(list)
    box_data = [grouped[bin] for bin in grouped.index]
    x_labels = [f"{(b.left):.0f}–{b.right:.0f}" for b in grouped.index]

    # Create plot
    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)

    # Add zero reference line
    ax.axhline(y=0, color="red", linestyle="-", linewidth=2, zorder=1, alpha=0.8)

    # Create boxplot with styling
    bp = ax.boxplot(
        box_data,
        labels=x_labels,
        showfliers=False,
        zorder=2,
        patch_artist=True,
        notch=False,
    )

    # Style the boxplot
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_linewidth(2)

    ax.tick_params(axis="x", rotation=45, labelsize=18)

    # Apply axis limits if specified
    if x_limits:
        ax.set_xlim(x_limits)
    if y_limits:
        ax.set_ylim(y_limits)

    # Set labels
    x_label = get_scientific_label(x_col)
    y_label = get_scientific_label(y_col)

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(f"{y_label} vs {x_label}", fontweight="bold", pad=20)

    plt.tight_layout()
    filename = f"{y_col}_vs_{x_col}_boxplot{suffix}.png"
    save_plot(fig, filename, output_dir)


def plot_histogram_of_residuals(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """Plot histogram of model residuals."""
    residuals = df["target_stec"] - df["pred_stec"]

    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)

    # Create histogram
    n, bins, patches = ax.hist(
        residuals, bins=50, density=True, alpha=0.7, color="skyblue", edgecolor="black"
    )

    # Add statistics text
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax.axvline(
        mean_res,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_res:.3f}",
    )
    ax.axvline(0, color="black", linestyle="-", linewidth=2, alpha=0.8)

    # Labels and title
    ax.set_xlabel("Residuals [TECU]", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Distribution of Residuals", fontweight="bold", pad=20)
    ax.legend()

    # Add statistics text box
    textstr = f"Mean: {mean_res:.3f}\nStd: {std_res:.3f}\nN: {len(residuals)}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    save_plot(fig, "residuals_histogram.png", output_dir)


def plot_mae_vs_doy(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """Plot Mean Absolute Error vs Day of Year."""
    df = df.copy()
    df["mae"] = np.abs(df["target_stec"] - df["pred_stec"])
    plot_binned_boxplot(df, "doy", "mae", bins=24, output_dir=output_dir)


def plot_residuals_vs_feature(
    df: pd.DataFrame,
    feature: str,
    num_bins: int = 24,
    output_dir: str = "plots",
    bin_range_dict: Optional[Dict[str, tuple]] = None,
) -> None:
    """Plot residuals vs any feature with proper scientific formatting."""
    df = df.copy()
    df["residual"] = df["target_stec"] - df["pred_stec"]
    plot_binned_boxplot(
        df,
        feature,
        "residual",
        bins=num_bins,
        output_dir=output_dir,
        bin_range_dict=bin_range_dict,
    )


def plot_residuals_vs_feature_clipped(
    df: pd.DataFrame,
    feature: str,
    num_bins: int = 24,
    output_dir: str = "plots",
    bin_range_dict: Optional[Dict[str, tuple]] = None,
    x_limits: Optional[tuple] = None,
    y_limits: Optional[tuple] = None,
) -> None:
    """Plot residuals vs feature with axis clipping."""
    df = df.copy()
    df["residual"] = df["target_stec"] - df["pred_stec"]
    plot_binned_boxplot_clipped(
        df,
        feature,
        "residual",
        bins=num_bins,
        output_dir=output_dir,
        bin_range_dict=bin_range_dict,
        x_limits=x_limits,
        y_limits=y_limits,
    )


def plot_box_by_date(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Creates a two-panel plot showing RMSE/MAE and residual boxplots over time.
    
    Args:
        df: DataFrame with test results containing year, doy, target_stec, pred_stec
        output_dir: Directory to save the plot
    """
    from analysis.metrics import calc_rmse
    
    df = df.copy()
    df['error'] = df['target_stec'] - df['pred_stec']
    df['ae'] = np.abs(df['error'])

    # Create datetime from year and doy
    def create_date(row):
        try:
            year = int(row['year'])
            doy = int(row['doy'])
            date = datetime(year, 1, 1) + timedelta(days=doy - 1)
            return date
        except:
            return None
    
    df['date'] = df.apply(create_date, axis=1)
    df = df.dropna(subset=['date'])
    df['year_month'] = df['date'].dt.to_period('M').astype(str)

    order = sorted(df["year_month"].unique())
    pos = np.arange(len(order))
    
    # Calculate metrics, ensuring order is maintained
    rmse_lat = df.groupby("year_month")['error'].apply(calc_rmse).reindex(order).reset_index()
    mae_lat = df.groupby("year_month")["ae"].mean().reindex(order).reset_index()

    grouped = df.groupby('year_month')['error'].apply(list).reindex(order)
    box_data = [grouped[date] for date in order]

    fig, axs = plt.subplots(2, 1, figsize=(16, 12), sharex=True, gridspec_kw={'hspace': 0.1})
    fig.align_ylabels()

    # Top panel: RMSE and MAE
    axs[0].plot(pos, rmse_lat['error'], marker='o', label="RMSE")
    axs[0].plot(pos, mae_lat['ae'], marker='o', label="MAE")
    axs[0].legend(loc="upper right", fontsize=14, framealpha=0.9)
    axs[0].set_ylim(bottom=0)
    axs[0].set_ylabel("RMSE/MAE [TECU]")
    axs[0].set_title('Monthly Performance Metrics', fontweight='bold', pad=20)

    # Bottom panel: Boxplot of residuals
    axs[1].axhline(y=0, color='red', linestyle='-', linewidth=1.0, zorder=1, alpha=0.8)
    bp = axs[1].boxplot(box_data, widths=0.5, positions=pos,
                        showfliers=False, zorder=2, patch_artist=True, notch=False)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_linewidth(1.2)

    axs[1].set_xticks(pos)
    axs[1].set_xticklabels(order, rotation=45, ha='right')
    axs[1].set_ylim([-30, 30])
    axs[1].set_xlabel('Year-Month')
    axs[1].set_ylabel("Residual [TECU]")
    
    axs[0].grid(True, alpha=0.3)
    axs[1].grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_plot(fig, 'year_month_summary.png', output_dir)
    plt.close(fig)
