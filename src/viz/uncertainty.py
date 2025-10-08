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


def plot_coverage_probability(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Plot the coverage probability for total, epistemic, and aleatoric uncertainty.
    This shows how often the true value falls within the predicted uncertainty interval.
    
    Args:
        df: DataFrame with uncertainty columns and predictions
        output_dir: Directory to save plot
    """
    df = df.copy()
    residuals = df['target_stec'] - df['pred_stec']

    uncertainty_types = {
        'Total': 'pred_total_unc',
        'Epistemic': 'pred_epistemic_unc',
        'Aleatoric': 'pred_aleatoric_unc'
    }

    coverage_levels = np.linspace(0, 3, 31)  # From 0 to 3 sigma
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)

    for label, col in uncertainty_types.items():
        if col not in df.columns or df[col].isnull().all():
            continue

        observed_coverage = []
        for sigma in coverage_levels:
            is_covered = np.abs(residuals) <= (sigma * df[col])
            coverage_fraction = is_covered.mean()
            observed_coverage.append(coverage_fraction)
        
        ax.plot(coverage_levels, observed_coverage, marker='.', linestyle='-', label=f'Observed ({label})')

    # Expected coverage for a Gaussian distribution
    from scipy.stats import norm
    expected_coverage = [norm.cdf(s) - norm.cdf(-s) for s in coverage_levels]
    ax.plot(coverage_levels, expected_coverage, 'r--', label='Expected (Gaussian)', linewidth=2)

    ax.set_xlabel('Predicted Uncertainty Interval (Number of Sigmas)', fontweight='bold')
    ax.set_ylabel('Observed Coverage Probability', fontweight='bold')
    ax.set_title('Uncertainty Coverage Probability', fontweight='bold', pad=20)
    ax.legend(fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 3.5, 0.5))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    plt.tight_layout()
    save_plot(fig, "uncertainty_coverage_probability.png", output_dir)


def plot_sigma_coverage_comparison(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Create sigma interval coverage analysis comparing different uncertainty types.
    Shows how well uncertainty estimates match expected coverage for 1σ, 2σ, and 3σ intervals.
    
    Args:
        df: DataFrame with uncertainty columns and predictions
        output_dir: Directory to save plot
    """
    # Calculate absolute residuals
    abs_residuals = np.abs(df['target_stec'] - df['pred_stec'])
    
    # Extract uncertainties
    total_unc = df['pred_total_unc'].values
    epistemic_unc = df['pred_epistemic_unc'].values if 'pred_epistemic_unc' in df.columns else None
    aleatoric_unc = df['pred_aleatoric_unc'].values if 'pred_aleatoric_unc' in df.columns else None
    
    def calculate_sigma_coverage(uncertainties, residuals):
        """Calculate coverage for 1σ, 2σ, and 3σ intervals"""
        n_total = len(residuals)
        
        # Count observations within each sigma interval
        within_1sigma = np.sum(residuals <= uncertainties)
        within_2sigma = np.sum(residuals <= 2 * uncertainties)
        within_3sigma = np.sum(residuals <= 3 * uncertainties)
        
        # Calculate percentages
        pct_1sigma = (within_1sigma / n_total) * 100
        pct_2sigma = (within_2sigma / n_total) * 100
        pct_3sigma = (within_3sigma / n_total) * 100
        
        return [pct_1sigma, pct_2sigma, pct_3sigma]
    
    # Calculate coverage for all uncertainty types
    total_coverage = calculate_sigma_coverage(total_unc, abs_residuals)
    epistemic_coverage = calculate_sigma_coverage(epistemic_unc, abs_residuals) if epistemic_unc is not None else None
    aleatoric_coverage = calculate_sigma_coverage(aleatoric_unc, abs_residuals) if aleatoric_unc is not None else None
    
    # Define colors for consistency
    colors = {'total': 'navy', 'epistemic': 'darkred', 'aleatoric': 'darkgreen', 'expected': 'gray'}
    
    # Coverage comparison with clear interpretation
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sigma_levels = ['1σ', '2σ', '3σ']
    expected_values = [68.27, 95.45, 99.73]
    
    x = np.arange(len(sigma_levels))
    width = 0.2
    
    # Plot with clear legend and interpretation
    bars1 = ax.bar(x + 0.0*width, expected_values, width, label='Expected (Perfect)', 
                   alpha=0.8, color=colors['expected'], edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + 1.0*width, total_coverage, width, label='Total', 
                   alpha=0.8, color=colors['total'], edgecolor='black', linewidth=1.5)
    
    if epistemic_coverage is not None:
        bars3 = ax.bar(x + 2.0*width, epistemic_coverage, width, label='Epistemic (Model)', 
                       alpha=0.8, color=colors['epistemic'], edgecolor='black', linewidth=1.5)
    
    if aleatoric_coverage is not None:
        bars4 = ax.bar(x + 3.0*width, aleatoric_coverage, width, label='Aleatoric (Data Noise)', 
                       alpha=0.8, color=colors['aleatoric'], edgecolor='black', linewidth=1.5)
    
    # Add horizontal reference lines - aligned with each sigma level
    for i, exp_val in enumerate(expected_values):
        # Calculate proper x-position for each sigma level
        x_center = i + 1.5*width  # Center of the bar group
        x_left = i - 0.5 + width*0.5  # Left edge of bar group
        x_right = i + 0.5 + width*3.5  # Right edge of bar group
        
        ax.axhline(y=exp_val, xmin=x_left/len(sigma_levels), xmax=x_right/len(sigma_levels), 
                   color='red', linestyle='--', alpha=0.7, linewidth=3)
    
    ax.set_xlabel('Sigma Levels', fontsize=16, fontweight='bold')
    ax.set_ylabel('Coverage [%]', fontsize=16, fontweight='bold')
    ax.set_title('Uncertainty Coverage Analysis\nCloser to Expected = Better Calibrated', 
                fontsize=20, fontweight='bold', pad=25)
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(sigma_levels, fontsize=18, fontweight='bold')
    ax.legend(fontsize=18, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    autolabel(bars1, expected_values)
    autolabel(bars2, total_coverage)
    if epistemic_coverage is not None:
        autolabel(bars3, epistemic_coverage)
    if aleatoric_coverage is not None:
        autolabel(bars4, aleatoric_coverage)
    
    plt.tight_layout()
    save_plot(fig, "sigma_coverage_comparison.png", output_dir)


def plot_uncertainty_distributions(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Plot uncertainty distributions with clear interpretation of different uncertainty types.
    
    Args:
        df: DataFrame with uncertainty columns
        output_dir: Directory to save plot
    """
    # Extract uncertainties
    total_unc = df['pred_total_unc'].values
    epistemic_unc = df['pred_epistemic_unc'].values if 'pred_epistemic_unc' in df.columns else None
    aleatoric_unc = df['pred_aleatoric_unc'].values if 'pred_aleatoric_unc' in df.columns else None
    
    # Define colors for consistency
    colors = {'total': 'navy', 'epistemic': 'darkred', 'aleatoric': 'darkgreen'}
    
    # Uncertainty distributions with clear interpretation
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create histogram with better styling
    n1, bins1, patches1 = ax.hist(total_unc, bins=50, alpha=0.7, label='Total', 
                                  color=colors['total'], edgecolor='black', linewidth=0.5)
    
    if epistemic_unc is not None:
        n2, bins2, patches2 = ax.hist(epistemic_unc, bins=50, alpha=0.7, label='Epistemic (Model)', 
                                      color=colors['epistemic'], edgecolor='black', linewidth=0.5)
        # Add mean line
        ax.axvline(epistemic_unc.mean(), color=colors['epistemic'], linestyle='--', linewidth=3, alpha=0.9,
                   label=f'Epistemic mean: {epistemic_unc.mean():.3f} TECU')
    
    if aleatoric_unc is not None:
        n3, bins3, patches3 = ax.hist(aleatoric_unc, bins=50, alpha=0.7, label='Aleatoric (Data Noise)', 
                                      color=colors['aleatoric'], edgecolor='black', linewidth=0.5)
        # Add mean line
        ax.axvline(aleatoric_unc.mean(), color=colors['aleatoric'], linestyle='--', linewidth=3, alpha=0.9,
                   label=f'Aleatoric mean: {aleatoric_unc.mean():.3f} TECU')
    
    # Add mean line for total
    ax.axvline(total_unc.mean(), color=colors['total'], linestyle='--', linewidth=3, alpha=0.9, 
               label=f'Total mean: {total_unc.mean():.3f} TECU')
    
    ax.set_xlabel('Uncertainty [TECU]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax.set_title('Uncertainty Distributions', fontsize=20, fontweight='bold', pad=25)
    ax.legend(fontsize=18, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, "uncertainty_distributions.png", output_dir)


def plot_binned_uncertainty_error_analysis(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Creates a comprehensive binned uncertainty analysis plot.
    Shows the distribution of absolute error against total uncertainty, along with
    mean model and data uncertainties using boxplots and line plots.
    
    Args:
        df: DataFrame with uncertainty columns and predictions
        output_dir: Directory to save plot
    """
    if "pred_total_unc" not in df.columns:
        print("Warning: No uncertainty columns found. Skipping binned uncertainty error analysis.")
        return
        
    df = df.copy()
    df['abs_error'] = (df['target_stec'] - df['pred_stec']).abs()
    
    # Create proper bins based on uncertainty VALUE ranges (not sample counts)
    max_unc = df['pred_total_unc'].quantile(0.95)  # Use 95th percentile to avoid outliers
    n_bins = min(30, max(10, len(df) // 300))  # Adaptive number of bins
    
    # Create equal-width bins from 0 to max_unc
    bin_edges = np.linspace(0, max_unc, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Assign each point to a bin
    df['unc_bin'] = pd.cut(df['pred_total_unc'], bins=bin_edges, include_lowest=True, labels=False)
    
    # Filter out points beyond our max_unc threshold
    df = df.dropna(subset=['unc_bin'])

    # Calculate Metrics per Bin
    valid_groups = []
    plot_positions = []
    mean_abs_error = []
    mean_model_unc = []
    mean_data_unc = []
    mean_total_unc = []
    box_data = []
    
    for bin_idx in range(len(bin_centers)):
        bin_data = df[df['unc_bin'] == bin_idx]
        if len(bin_data) >= 5:  # Only bins with at least 5 points
            valid_groups.append(bin_idx)
            plot_positions.append(bin_centers[bin_idx])
            mean_abs_error.append(bin_data['abs_error'].mean())
            
            # Handle uncertainty columns that might not exist
            if 'pred_epistemic_unc' in bin_data.columns:
                mean_model_unc.append(bin_data['pred_epistemic_unc'].mean())
            else:
                mean_model_unc.append(0)
                
            if 'pred_aleatoric_unc' in bin_data.columns:
                mean_data_unc.append(bin_data['pred_aleatoric_unc'].mean())
            else:
                mean_data_unc.append(0)
                
            mean_total_unc.append(bin_data['pred_total_unc'].mean())
            box_data.append(bin_data['abs_error'].values)
    
    if not valid_groups:
        print("Warning: No bins with sufficient data for uncertainty analysis")
        return

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # 1. Boxplots for Absolute Error
    if box_data:
        bp = ax.boxplot(box_data, positions=plot_positions, 
                        widths=min(0.8, (max_unc / n_bins) * 0.8),  # Scale width to bin size
                        showfliers=False, patch_artist=True,
                        boxprops=dict(facecolor='white', edgecolor='black'),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'),
                        medianprops=dict(color='orange'))

        # 2. Line plots for mean values
        ax.plot(plot_positions, mean_abs_error, color='orange', marker='o', linestyle='-', 
                label='Mean Absolute Error', zorder=20, linewidth=2, markersize=6)
        ax.plot(plot_positions, mean_total_unc, color='red', marker='s', linestyle='-', 
                label='Mean Total Uncertainty', zorder=20, linewidth=2, markersize=6)
        
        # Only plot epistemic/aleatoric if they exist and have non-zero values
        if max(mean_model_unc) > 0:
            ax.plot(plot_positions, mean_model_unc, color='black', marker='^', linestyle='-', 
                    label='Mean Epistemic Uncertainty', zorder=20, linewidth=2, markersize=6)
        
        if max(mean_data_unc) > 0:
            ax.plot(plot_positions, mean_data_unc, color='blue', marker='v', linestyle='-', 
                    label='Mean Aleatoric Uncertainty', zorder=20, linewidth=2, markersize=6)

    # 3. Perfect Calibration Line
    ax.plot([0, max_unc], [0, max_unc], 'r--', label='Perfect Calibration', zorder=15, linewidth=2)

    # Formatting
    ax.set_xlabel('Total Uncertainty [TECU]', fontweight='bold', fontsize=14)
    ax.set_ylabel('Values [TECU]', fontweight='bold', fontsize=14)
    ax.set_title('Binned Uncertainty & Error Analysis', fontweight='bold', pad=20, fontsize=16)
    
    # Improve x-axis tick formatting
    if plot_positions:
        x_max = max(plot_positions) * 1.1
        
        # Set reasonable number of ticks based on the range
        if x_max <= 10:
            x_ticks = np.arange(0, x_max + 1, 1)
        elif x_max <= 20:
            x_ticks = np.arange(0, x_max + 2, 2)
        elif x_max <= 50:
            x_ticks = np.arange(0, x_max + 5, 5)
        else:
            x_ticks = np.arange(0, x_max + 10, 10)
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{int(tick)}' for tick in x_ticks], fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
    
    ax.legend(fontsize=12, framealpha=0.9, loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Set axis limits to show all data properly
    if plot_positions:
        x_max = max(plot_positions) * 1.1
        y_max = max(max(mean_abs_error), max(mean_total_unc)) * 1.2
        
        # Also consider the boxplot data for y-axis
        if box_data:
            max_box_val = max([np.percentile(data, 95) for data in box_data])
            y_max = max(y_max, max_box_val * 1.1)
        
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
    else:
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)

    plt.tight_layout()
    save_plot(fig, "binned_uncertainty_error_analysis.png", output_dir)
