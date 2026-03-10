"""
Publication-ready evaluation plots for STEC model comparison.

Generates comprehensive, high-quality figures suitable for publication.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
from scipy import stats


MODEL_COLORS = {
    'Direct STEC Model': '#1f77b4',
    'Pretrained STEC': '#9467bd',
    'VTEC + Mapping': '#ff7f0e',
    'IGS GIM': '#2ca02c',
}


def get_model_color(model_name: str) -> str:
    """Return a stable color for each model name."""
    return MODEL_COLORS.get(model_name, '#7f7f7f')


def set_publication_style():
    """Set matplotlib style for publication-ready figures."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'figure.titlesize': 20,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def create_scatter_plots(test_df: pd.DataFrame, models: Dict, output_dir: Path):
    """Create scatter plots comparing predictions vs ground truth."""
    ground_truth = test_df['true_stec'].values
    elevations = test_df['satele'].values
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6.5))
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, predictions) in enumerate(models.items()):
        ax = axes[i]
        
        # Handle NaNs for GIM
        if 'GIM' in model_name and predictions is not None:
            mask = ~np.isnan(predictions)
            gt_plot = ground_truth[mask]
            pred_plot = predictions[mask]
            elev_plot = elevations[mask]
        else:
            gt_plot = ground_truth
            pred_plot = predictions
            elev_plot = elevations
        
        # Density scatter
        scatter = ax.hexbin(gt_plot, pred_plot, gridsize=50, cmap='YlOrRd', mincnt=1, alpha=0.8)
        
        # Perfect prediction line
        lims = [min(gt_plot.min(), pred_plot.min()), 
                max(gt_plot.max(), pred_plot.max())]
        ax.plot(lims, lims, 'k--', lw=3, label='Perfect', alpha=0.7, zorder=10)
        
        # Calculate metrics for annotation
        rmse = np.sqrt(np.mean((pred_plot - gt_plot) ** 2))
        r2 = 1 - np.sum((pred_plot - gt_plot) ** 2) / np.sum((gt_plot - gt_plot.mean()) ** 2)
        bias = np.mean(pred_plot - gt_plot)
        
        # Add metrics text box
        textstr = f'RMSE: {rmse:.2f} TECU\nR²: {r2:.3f}\nBias: {bias:.2f} TECU'
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel('True STEC (TECU)', fontweight='bold')
        ax.set_ylabel('Predicted STEC (TECU)', fontweight='bold')
        ax.set_title(model_name, fontweight='bold', pad=15)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.set_aspect('equal', adjustable='box')
        
        # Colorbar for last subplot
        if i == n_models - 1:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Number of observations', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_comparison.png')
    plt.close()


def create_residual_plots(test_df: pd.DataFrame, models: Dict, output_dir: Path):
    """Create residual analysis plots."""
    ground_truth = test_df['true_stec'].values
    elevations = test_df['satele'].values
    
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(7*n_models, 12))
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (model_name, predictions) in enumerate(models.items()):
        # Handle NaNs
        if 'GIM' in model_name and predictions is not None:
            mask = ~np.isnan(predictions)
            gt = ground_truth[mask]
            pred = predictions[mask]
            elev = elevations[mask]
        else:
            gt = ground_truth
            pred = predictions
            elev = elevations
        
        residuals = pred - gt
        
        # Residuals vs predicted values
        ax = axes[0, i]
        ax.hexbin(pred, residuals, gridsize=50, cmap='Blues', mincnt=1)
        ax.axhline(0, color='red', linestyle='--', lw=2.5, alpha=0.8)
        ax.set_xlabel('Predicted STEC (TECU)', fontweight='bold')
        ax.set_ylabel('Residuals (TECU)', fontweight='bold')
        ax.set_title(f'{model_name} - Residuals vs Predicted', fontweight='bold', pad=15)
        
        # Q-Q plot
        ax = axes[1, i]
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'{model_name} - Q-Q Plot', fontweight='bold', pad=15)
        ax.set_xlabel('Theoretical Quantiles', fontweight='bold')
        ax.set_ylabel('Sample Quantiles', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residual_analysis.png')
    plt.close()


def create_error_distribution(test_df: pd.DataFrame, models: Dict, output_dir: Path):
    """Create error distribution comparison."""
    ground_truth = test_df['true_stec'].values
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    ax = axes[0]
    for model_name, predictions in models.items():
        if 'GIM' in model_name and predictions is not None:
            mask = ~np.isnan(predictions)
            errors = predictions[mask] - ground_truth[mask]
        else:
            errors = predictions - ground_truth
        
        ax.hist(errors, bins=100, alpha=0.6, label=model_name, density=True, 
               color=get_model_color(model_name), edgecolor='black', linewidth=0.5)
    
    ax.axvline(0, color='red', linestyle='--', lw=3, alpha=0.8, label='Zero error')
    ax.set_xlabel('Prediction Error (TECU)', fontweight='bold')
    ax.set_ylabel('Probability Density', fontweight='bold')
    ax.set_title('Error Distribution', fontweight='bold', pad=15)
    ax.legend(framealpha=0.9)
    ax.set_xlim(-30, 30)
    
    # Cumulative distribution
    ax = axes[1]
    for model_name, predictions in models.items():
        if 'GIM' in model_name and predictions is not None:
            mask = ~np.isnan(predictions)
            errors = np.abs(predictions[mask] - ground_truth[mask])
        else:
            errors = np.abs(predictions - ground_truth)
        
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        ax.plot(sorted_errors, cumulative, lw=3, label=model_name, color=get_model_color(model_name))
    
    ax.set_xlabel('Absolute Error (TECU)', fontweight='bold')
    ax.set_ylabel('Cumulative Percentage (%)', fontweight='bold')
    ax.set_title('Cumulative Error Distribution', fontweight='bold', pad=15)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 25)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png')
    plt.close()


def create_elevation_analysis(test_df: pd.DataFrame, models: Dict, output_dir: Path):
    """Create elevation-stratified analysis."""
    ground_truth = test_df['true_stec'].values
    elevations = test_df['satele'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    elev_bins = np.arange(5, 91, 5)
    bin_centers = (elev_bins[:-1] + elev_bins[1:]) / 2
    
    # RMSE vs Elevation
    ax = axes[0, 0]
    for model_name, predictions in models.items():
        rmse_bins = []
        for j in range(len(elev_bins) - 1):
            if 'GIM' in model_name and predictions is not None:
                mask = (~np.isnan(predictions)) & (elevations >= elev_bins[j]) & (elevations < elev_bins[j+1])
            else:
                mask = (elevations >= elev_bins[j]) & (elevations < elev_bins[j+1])
            
            if mask.sum() > 10:
                errors = predictions[mask] - ground_truth[mask]
                rmse_bins.append(np.sqrt(np.mean(errors ** 2)))
            else:
                rmse_bins.append(np.nan)
        
        ax.plot(bin_centers, rmse_bins, 'o-', label=model_name, lw=2.5, 
                   markersize=8, color=get_model_color(model_name))
    
    ax.set_xlabel('Elevation Angle (°)', fontweight='bold')
    ax.set_ylabel('RMSE (TECU)', fontweight='bold')
    ax.set_title('RMSE vs Elevation', fontweight='bold', pad=15)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # MAE vs Elevation
    ax = axes[0, 1]
    for model_name, predictions in models.items():
        mae_bins = []
        for j in range(len(elev_bins) - 1):
            if 'GIM' in model_name and predictions is not None:
                mask = (~np.isnan(predictions)) & (elevations >= elev_bins[j]) & (elevations < elev_bins[j+1])
            else:
                mask = (elevations >= elev_bins[j]) & (elevations < elev_bins[j+1])
            
            if mask.sum() > 10:
                errors = np.abs(predictions[mask] - ground_truth[mask])
                mae_bins.append(np.mean(errors))
            else:
                mae_bins.append(np.nan)
        
        ax.plot(bin_centers, mae_bins, 'o-', label=model_name, lw=2.5, 
                   markersize=8, color=get_model_color(model_name))
    
    ax.set_xlabel('Elevation Angle (°)', fontweight='bold')
    ax.set_ylabel('MAE (TECU)', fontweight='bold')
    ax.set_title('MAE vs Elevation', fontweight='bold', pad=15)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Sample count vs Elevation
    ax = axes[1, 0]
    counts = []
    for j in range(len(elev_bins) - 1):
        mask = (elevations >= elev_bins[j]) & (elevations < elev_bins[j+1])
        counts.append(mask.sum())
    
    ax.bar(bin_centers, counts, width=4, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1)
    ax.set_xlabel('Elevation Angle (°)', fontweight='bold')
    ax.set_ylabel('Number of Observations', fontweight='bold')
    ax.set_title('Data Distribution by Elevation', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Relative improvement
    ax = axes[1, 1]
    if len(models) >= 2:
        model_names = list(models.keys())
        baseline_name = [m for m in model_names if 'GIM' in m or 'VTEC' in m][0] if any('GIM' in m or 'VTEC' in m for m in model_names) else model_names[1]
        stec_name = 'Direct STEC Model' if 'Direct STEC Model' in model_names else [m for m in model_names if 'STEC' in m and 'GIM' not in m][0]
        
        baseline_rmse = []
        stec_rmse = []
        for j in range(len(elev_bins) - 1):
            if 'GIM' in baseline_name:
                mask_baseline = (~np.isnan(models[baseline_name])) & (elevations >= elev_bins[j]) & (elevations < elev_bins[j+1])
            else:
                mask_baseline = (elevations >= elev_bins[j]) & (elevations < elev_bins[j+1])
            
            mask_stec = (elevations >= elev_bins[j]) & (elevations < elev_bins[j+1])
            
            if mask_baseline.sum() > 10 and mask_stec.sum() > 10:
                baseline_errors = models[baseline_name][mask_baseline] - ground_truth[mask_baseline]
                stec_errors = models[stec_name][mask_stec] - ground_truth[mask_stec]
                baseline_rmse.append(np.sqrt(np.mean(baseline_errors ** 2)))
                stec_rmse.append(np.sqrt(np.mean(stec_errors ** 2)))
            else:
                baseline_rmse.append(np.nan)
                stec_rmse.append(np.nan)
        
        improvement = 100 * (np.array(baseline_rmse) - np.array(stec_rmse)) / np.array(baseline_rmse)
        colors_bar = ['green' if x > 0 else 'red' for x in improvement]
        ax.bar(bin_centers, improvement, width=4, alpha=0.7, color=colors_bar, edgecolor='black', linewidth=1)
        ax.axhline(0, color='black', linestyle='--', lw=2)
        ax.set_xlabel('Elevation Angle (°)', fontweight='bold')
        ax.set_ylabel('RMSE Improvement (%)', fontweight='bold')
        ax.set_title(f'Direct STEC Improvement over {baseline_name}', fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'Multiple models required\nfor improvement analysis', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'elevation_analysis.png')
    plt.close()


def create_comparison_summary(test_df: pd.DataFrame, models: Dict, metrics: Dict, output_dir: Path):
    """Create summary comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    model_names = list(models.keys())
    colors = [get_model_color(name) for name in model_names]
    
    # RMSE comparison
    ax = axes[0]
    rmse_values = [metrics[name]['rmse'] for name in model_names]
    bars = ax.bar(range(len(model_names)), rmse_values, color=colors[:len(model_names)], 
                  alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel('RMSE (TECU)', fontweight='bold')
    ax.set_title('Root Mean Square Error', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # R² comparison
    ax = axes[1]
    r2_values = [metrics[name]['r2'] for name in model_names]
    bars = ax.bar(range(len(model_names)), r2_values, color=colors[:len(model_names)], 
                  alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_title('Coefficient of Determination', fontweight='bold', pad=15)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Bias comparison
    ax = axes[2]
    bias_values = [metrics[name]['bias'] for name in model_names]
    colors_bias = ['green' if abs(b) < 2 else 'orange' if abs(b) < 5 else 'red' for b in bias_values]
    bars = ax.bar(range(len(model_names)), bias_values, color=colors_bias, 
                  alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel('Bias (TECU)', fontweight='bold')
    ax.set_title('Mean Bias', fontweight='bold', pad=15)
    ax.axhline(0, color='black', linestyle='--', lw=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.2 if height > 0 else -0.5),
                f'{height:.2f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.png')
    plt.close()


def generate_all_plots(test_df: pd.DataFrame, stec_col: str, vtec_col: Optional[str], 
                       gim_col: Optional[str], metrics: Dict, output_dir: Path, logger,
                       pretrain_col: Optional[str] = None):
    """Generate all publication-ready plots."""
    set_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare model dictionary
    models = {'Direct STEC Model': test_df[stec_col].values}
    
    # Add pretrained model if available (as the second item for consistent plotting order)
    if pretrain_col and pretrain_col in test_df.columns:
        models['Pretrained STEC'] = test_df[pretrain_col].values
        
    if vtec_col and vtec_col in test_df.columns:
        models['VTEC + Mapping'] = test_df[vtec_col].values
    if gim_col and gim_col in test_df.columns:
        models['IGS GIM'] = test_df[gim_col].values
    
    logger.info(f"📊 Generating {5} publication-ready figures...")
    
    create_scatter_plots(test_df, models, output_dir)
    logger.info("  ✓ Scatter plots created")
    
    create_residual_plots(test_df, models, output_dir)
    logger.info("  ✓ Residual analysis created")
    
    create_error_distribution(test_df, models, output_dir)
    logger.info("  ✓ Error distributions created")
    
    create_elevation_analysis(test_df, models, output_dir)
    logger.info("  ✓ Elevation analysis created")
    
    create_comparison_summary(test_df, models, metrics, output_dir)
    logger.info("  ✓ Metrics summary created")
    
    logger.info(f"✅ All plots saved to {output_dir}")
