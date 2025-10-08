"""
Uncertainty analysis and calibration visualization functions.

This module handles uncertainty calibration plots, coverage probability,
and uncertainty analysis visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from .base import (
    FIGSIZE_SQUARE,
    FIGSIZE_WIDE,
    save_plot,
)


def plot_uncertainty_calibration_binned(
    df: pd.DataFrame, output_dir: str = "plots"
) -> None:
    """
    Plot uncertainty calibration using binned approach.

    Args:
        df: DataFrame with uncertainty columns and predictions
        output_dir: Directory to save plot
    """
    if "pred_total_unc" not in df.columns:
        print(
            "Warning: No uncertainty columns found. Skipping uncertainty calibration plot."
        )
        return

    df = df.copy()
    df["abs_residual"] = np.abs(df["target_stec"] - df["pred_stec"])

    # Create uncertainty bins
    n_bins = 20
    df["unc_bin"] = pd.qcut(df["pred_total_unc"], q=n_bins, duplicates="drop")

    # Calculate statistics per bin
    bin_stats = (
        df.groupby("unc_bin", observed=True)
        .agg({"pred_total_unc": "mean", "abs_residual": "mean", "target_stec": "count"})
        .reset_index()
    )

    bin_stats.columns = [
        "unc_bin",
        "mean_predicted_unc",
        "mean_observed_error",
        "count",
    ]

    # Create calibration plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Main calibration plot
    ax1.scatter(
        bin_stats["mean_predicted_unc"],
        bin_stats["mean_observed_error"],
        s=100,
        alpha=0.7,
        c="blue",
    )

    # Perfect calibration line
    max_val = max(
        bin_stats["mean_predicted_unc"].max(), bin_stats["mean_observed_error"].max()
    )
    ax1.plot(
        [0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Calibration"
    )

    ax1.set_xlabel("Mean Predicted Uncertainty [TECU]", fontweight="bold")
    ax1.set_ylabel("Mean Observed Error [TECU]", fontweight="bold")
    ax1.set_title("Uncertainty Calibration", fontweight="bold", pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Calculate calibration metrics
    corr, p_val = pearsonr(
        bin_stats["mean_predicted_unc"], bin_stats["mean_observed_error"]
    )

    # Bin count plot
    ax2.bar(range(len(bin_stats)), bin_stats["count"], alpha=0.7, color="skyblue")
    ax2.set_xlabel("Uncertainty Bin", fontweight="bold")
    ax2.set_ylabel("Sample Count", fontweight="bold")
    ax2.set_title("Samples per Uncertainty Bin", fontweight="bold", pad=20)
    ax2.grid(True, alpha=0.3)

    # Add correlation text
    fig.suptitle(
        f"Uncertainty Calibration (r={corr:.3f}, p={p_val:.2e})",
        fontsize=20,
        fontweight="bold",
    )

    plt.tight_layout()
    save_plot(fig, "uncertainty_calibration_binned.png", output_dir)


def plot_coverage_probability(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Plot coverage probability analysis for uncertainty estimates.

    Args:
        df: DataFrame with uncertainty columns and predictions
        output_dir: Directory to save plot
    """
    if "pred_total_unc" not in df.columns:
        print(
            "Warning: No uncertainty columns found. Skipping coverage probability plot."
        )
        return

    df = df.copy()
    df["abs_residual"] = np.abs(df["target_stec"] - df["pred_stec"])

    # Calculate coverage for different confidence levels
    confidence_levels = np.linspace(0.1, 0.99, 50)
    coverage_ratios = []

    for conf_level in confidence_levels:
        # Z-score for confidence level (assuming normal distribution)
        z_score = np.abs(
            np.percentile(
                np.random.normal(0, 1, 10000),
                [(1 - conf_level) / 2 * 100, (1 + conf_level) / 2 * 100],
            )
        )
        z_score = z_score[1]  # Take upper bound

        # Count samples within confidence interval
        within_interval = df["abs_residual"] <= (z_score * df["pred_total_unc"])
        coverage_ratio = within_interval.mean()
        coverage_ratios.append(coverage_ratio)

    # Create plot
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    ax.plot(
        confidence_levels, coverage_ratios, "b-", linewidth=3, label="Observed Coverage"
    )
    ax.plot(
        confidence_levels,
        confidence_levels,
        "r--",
        linewidth=2,
        label="Perfect Calibration",
    )

    ax.set_xlabel("Confidence Level", fontweight="bold")
    ax.set_ylabel("Coverage Probability", fontweight="bold")
    ax.set_title("Coverage Probability Analysis", fontweight="bold", pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_diff = np.mean(np.abs(np.array(coverage_ratios) - confidence_levels))
    ax.text(
        0.05,
        0.95,
        f"Mean |Difference|: {mean_diff:.4f}",
        transform=ax.transAxes,
        fontsize=14,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    save_plot(fig, "coverage_probability.png", output_dir)


def plot_binned_uncertainty_analysis(
    df: pd.DataFrame, output_dir: str = "plots"
) -> None:
    """
    Individual uncertainty analysis plots with multiple binning approaches.

    Args:
        df: DataFrame with uncertainty columns and predictions
        output_dir: Directory to save plot
    """
    if "pred_total_unc" not in df.columns:
        print("Warning: No uncertainty columns found. Skipping uncertainty analysis.")
        return

    df = df.copy()
    df["abs_residual"] = np.abs(df["target_stec"] - df["pred_stec"])

    # Create different bin types
    n_bins = 15

    # Uncertainty bins (quantile-based)
    df["unc_bin"] = pd.qcut(df["pred_total_unc"], q=n_bins, duplicates="drop")

    # Prediction bins
    df["pred_bin"] = pd.qcut(df["pred_stec"], q=n_bins, duplicates="drop")

    # True value bins
    df["true_bin"] = pd.qcut(df["target_stec"], q=n_bins, duplicates="drop")

    # 1. Uncertainty vs Observed Error (by uncertainty bins)
    unc_stats = (
        df.groupby("unc_bin", observed=True)
        .agg({"pred_total_unc": "mean", "abs_residual": "mean", "target_stec": "count"})
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        unc_stats["pred_total_unc"], unc_stats["abs_residual"], s=100, alpha=0.7
    )

    # Perfect calibration line
    max_val = max(unc_stats["pred_total_unc"].max(), unc_stats["abs_residual"].max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Uncertainty [TECU]", fontweight="bold")
    ax.set_ylabel("Mean Observed Error [TECU]", fontweight="bold")
    ax.set_title("Uncertainty Calibration", fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    save_plot(fig, "calibration_plot.png", output_dir)
    plt.close(fig)

    # 2. Uncertainty vs Prediction Value
    pred_stats = (
        df.groupby("pred_bin", observed=True)
        .agg({"pred_stec": "mean", "pred_total_unc": "mean", "abs_residual": "mean"})
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        pred_stats["pred_stec"],
        pred_stats["pred_total_unc"],
        s=100,
        alpha=0.7,
        color="green",
    )
    ax.set_xlabel("Mean Predicted STEC [TECU]", fontweight="bold")
    ax.set_ylabel("Mean Predicted Uncertainty [TECU]", fontweight="bold")
    ax.set_title("Uncertainty vs Prediction", fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "uncertainty_vs_prediction.png", output_dir)
    plt.close(fig)

    # 3. Error vs Prediction Value
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        pred_stats["pred_stec"],
        pred_stats["abs_residual"],
        s=100,
        alpha=0.7,
        color="orange",
    )
    ax.set_xlabel("Mean Predicted STEC [TECU]", fontweight="bold")
    ax.set_ylabel("Mean Observed Error [TECU]", fontweight="bold")
    ax.set_title("Error vs Prediction", fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "error_vs_prediction.png", output_dir)
    plt.close(fig)

    # 4. Uncertainty vs True Value
    true_stats = (
        df.groupby("true_bin", observed=True)
        .agg({"target_stec": "mean", "pred_total_unc": "mean", "abs_residual": "mean"})
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        true_stats["target_stec"],
        true_stats["pred_total_unc"],
        s=100,
        alpha=0.7,
        color="purple",
    )
    ax.set_xlabel("Mean True STEC [TECU]", fontweight="bold")
    ax.set_ylabel("Mean Predicted Uncertainty [TECU]", fontweight="bold")
    ax.set_title("Uncertainty vs True Value", fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "uncertainty_vs_true_value.png", output_dir)
    plt.close(fig)

    # 5. Error vs True Value
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        true_stats["target_stec"],
        true_stats["abs_residual"],
        s=100,
        alpha=0.7,
        color="red",
    )
    ax.set_xlabel("Mean True STEC [TECU]", fontweight="bold")
    ax.set_ylabel("Mean Observed Error [TECU]", fontweight="bold")
    ax.set_title("Error vs True Value", fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "error_vs_true_value.png", output_dir)
    plt.close(fig)

    # 6. Uncertainty distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        df["pred_total_unc"], bins=50, alpha=0.7, color="skyblue", density=True
    )
    ax.axvline(
        df["pred_total_unc"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["pred_total_unc"].mean():.3f}',
    )
    ax.set_xlabel("Predicted Uncertainty [TECU]", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Uncertainty Distribution", fontweight="bold", pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "uncertainty_distributions.png", output_dir)
    plt.close(fig)


def plot_uncertainty_calibration(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Simple uncertainty calibration plot.

    Args:
        df: DataFrame with uncertainty columns and predictions
        output_dir: Directory to save plot
    """
    if "pred_total_unc" not in df.columns:
        print(
            "Warning: No uncertainty columns found. Skipping uncertainty calibration."
        )
        return

    df = df.copy()
    df["abs_residual"] = np.abs(df["target_stec"] - df["pred_stec"])

    # Create scatter plot
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    # Sample data if too large
    if len(df) > 10000:
        df_sample = df.sample(n=10000, random_state=42)
    else:
        df_sample = df

    ax.scatter(df_sample["pred_total_unc"], df_sample["abs_residual"], alpha=0.3, s=5)

    # Perfect calibration line
    max_val = max(df["pred_total_unc"].max(), df["abs_residual"].max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Calibration")

    # Calculate correlation
    corr, p_val = pearsonr(df["pred_total_unc"], df["abs_residual"])

    ax.set_xlabel("Predicted Uncertainty [TECU]", fontweight="bold")
    ax.set_ylabel("Observed Error [TECU]", fontweight="bold")
    ax.set_title(f"Uncertainty Calibration (r={corr:.3f})", fontweight="bold", pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "uncertainty_calibration_scatter.png", output_dir)
