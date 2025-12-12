#!/usr/bin/env python3
"""
Fair Comparison: STEC Model vs VTEC Model with Mapping Function

This script provides a rigorous comparison between:
1. Direct STEC Modeling: Neural network predicts STEC directly from all features
2. Classical VTEC Approach: Neural network predicts VTEC, then maps to STEC using geometry

Why This Comparison is Fair:
- Both models use the same neural network architecture (e.g., BNN_NLL)
- Both models trained on the same data (different targets: STEC vs VTEC)
- Both models evaluated on the same held-out test set
- VTEC model uses standard mapping functions (MSLM or SLM)
- Identical evaluation metrics applied to both approaches

This demonstrates the value of direct STEC modeling vs classical VTEC+mapping approach.

Usage:
    python src/compare_stec_vtec.py \\
        --stec_experiment "Pretrain_STEC_BNN_NLL_h256_l4_..." \\
        --vtec_experiment "Pretrain_VTEC_BNN_NLL_h256_l4_..." \\
        --mapping_function MSLM \\
        --output_dir "comparisons/stec_vs_vtec"
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_parser import parse_config
from utils.feature_registry import initialize_feature_registry
from training.base_trainer import BaseTrainer
from data_loader import get_test_data_loader
from model.model import get_model
from evaluation.gim_mapper import MappingFunction


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def load_experiment_config(experiment_folder: str) -> Dict:
    """Load configuration from a trained experiment."""
    experiment_dir = Path(experiment_folder)
    if not experiment_dir.is_absolute():
        if not str(experiment_folder).startswith("experiments/"):
            experiment_dir = Path("experiments") / experiment_folder
        else:
            experiment_dir = Path(experiment_folder)
    
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config, experiment_dir


def find_best_checkpoint(experiment_dir: Path) -> Path:
    """Find the best model checkpoint in an experiment directory."""
    # Check for .pt files first
    checkpoint_files = list(experiment_dir.glob("*.pt"))
    
    # Prioritize best_model.pt
    best_model = experiment_dir / "best_model.pt"
    if best_model.exists():
        return best_model
    
    # Fall back to model.pt
    model_pt = experiment_dir / "model.pt"
    if model_pt.exists():
        return model_pt
    
    # Check in model/ subdirectory for .pth files (finetune convention)
    model_dir = experiment_dir / "model"
    if model_dir.exists():
        pth_files = list(model_dir.glob("*.pth"))
        if pth_files:
            # Prioritize files with 'best' or 'finetune' in name
            for pth in pth_files:
                if 'best' in pth.name.lower() or 'finetune' in pth.name.lower():
                    return pth
            return pth_files[0]  # Return first if no priority match
    
    # Check for .pth files in root
    pth_files = list(experiment_dir.glob("*.pth"))
    if pth_files:
        return pth_files[0]
    
    # Otherwise, find latest .pt checkpoint
    if checkpoint_files:
        return max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    
    raise FileNotFoundError(f"No model checkpoint found in {experiment_dir}")


def load_model_from_checkpoint(config: Dict, checkpoint_path: Path, logger) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    # Initialize feature registry
    feature_registry = initialize_feature_registry(config)
    config["feature_registry"] = feature_registry
    
    # Initialize CollateWithSH to set output_indices in the feature registry
    # This is required before creating models that use FeatureSplitter
    from data_loader.collation import CollateWithSH
    collate_fn = CollateWithSH(config)
    
    # Create model
    model = get_model(config).to(config["device"])
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config["device"], weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    logger.info(f"✅ Loaded {config['model']['model_type']} from {checkpoint_path.name}")
    return model, feature_registry


def run_stec_inference(
    model: torch.nn.Module,
    test_loader,
    config: Dict,
    logger,
    num_mc_samples: int = 100
) -> pd.DataFrame:
    """Run inference with STEC model."""
    logger.info("🧠 Running STEC model inference...")
    
    trainer = BaseTrainer(config, logger)
    
    # Determine if Bayesian
    model_type = config["model"]["model_type"]
    is_bayesian = "BNN" in model_type or "Bayesian" in model_type or "FactorizedSTEC" in model_type
    samples = num_mc_samples if is_bayesian else 1
        
    # Run Bayesian inference
    bayesian_results, test_df = trainer.bayesian_inference_total_uncertainty(
        model, test_loader, num_samples=samples
    )
    
    logger.info(f"✅ STEC inference completed: {len(test_df):,} predictions")
    return test_df


def run_vtec_inference(
    model: torch.nn.Module,
    test_loader,
    config: Dict,
    logger,
    num_mc_samples: int = 100
) -> pd.DataFrame:
    """Run inference with VTEC model."""
    logger.info("🧠 Running VTEC model inference...")
    
    trainer = BaseTrainer(config, logger)
    
    # Determine if Bayesian
    model_type = config["model"]["model_type"]
    is_bayesian = "BNN" in model_type or "Bayesian" in model_type
    samples = num_mc_samples if is_bayesian else 1
        
    # Run Bayesian inference
    bayesian_results, test_df = trainer.bayesian_inference_total_uncertainty(
        model, test_loader, num_samples=samples
    )
    
    logger.info(f"✅ VTEC inference completed: {len(test_df):,} predictions")
    return test_df


def apply_mapping_function(
    vtec_df: pd.DataFrame,
    mapping_type: str,
    logger
) -> pd.DataFrame:
    """
    Apply mapping function to convert VTEC predictions to STEC.
    
    Args:
        vtec_df: DataFrame with VTEC predictions and elevation angles
        mapping_type: 'SLM' or 'MSLM'
        logger: Logger instance
        
    Returns:
        DataFrame with additional 'pred_stec_mapped' column
    """
    logger.info(f"📐 Applying {mapping_type} mapping function to VTEC predictions...")
    
    # Initialize mapping function
    mapper = MappingFunction(mapping_type=mapping_type)
    
    # Get elevations (in degrees) and convert to radians
    elevations_rad = np.radians(vtec_df['satele'].values)
    
    # Get VTEC predictions
    vtec_pred = vtec_df['pred_mean'].values
    
    # Compute mapping factor
    mapping_factors = np.array([mapper.get_mapping_factor(el) for el in elevations_rad])
    
    # Convert VTEC to STEC
    stec_mapped = vtec_pred * mapping_factors
    
    # Also propagate uncertainty (variance scales with mapping factor squared)
    if 'pred_var' in vtec_df.columns:
        vtec_var = vtec_df['pred_var'].values
        stec_var_mapped = vtec_var * (mapping_factors ** 2)
        vtec_df['pred_var_mapped'] = stec_var_mapped
    
    vtec_df['pred_stec_mapped'] = stec_mapped
    vtec_df['mapping_factor'] = mapping_factors
    
    logger.info(f"✅ Mapping applied. Mean mapping factor: {mapping_factors.mean():.3f}")
    return vtec_df


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    errors = predictions - ground_truth
    
    metrics = {
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mae': np.mean(np.abs(errors)),
        'bias': np.mean(errors),
        'std': np.std(errors),
        'r2': 1 - np.sum(errors ** 2) / np.sum((ground_truth - ground_truth.mean()) ** 2)
    }
    
    return metrics


def compare_models(
    stec_df: pd.DataFrame,
    vtec_df: pd.DataFrame,
    logger
) -> Dict[str, Dict[str, float]]:
    """
    Compare STEC model vs VTEC+mapping model.
    
    Both dataframes must have aligned indices and 'true_stec' ground truth.
    """
    logger.info("📊 Computing comparison metrics...")
    
    # Get ground truth STEC
    ground_truth = stec_df['true_stec'].values
    
    # Get predictions
    stec_pred = stec_df['pred_mean'].values
    vtec_mapped_pred = vtec_df['pred_stec_mapped'].values
    
    # Compute metrics for both approaches
    stec_metrics = compute_metrics(stec_pred, ground_truth)
    vtec_metrics = compute_metrics(vtec_mapped_pred, ground_truth)
    
    # Compute improvement
    improvement = {
        'rmse_improvement_percent': 100 * (vtec_metrics['rmse'] - stec_metrics['rmse']) / vtec_metrics['rmse'],
        'mae_improvement_percent': 100 * (vtec_metrics['mae'] - stec_metrics['mae']) / vtec_metrics['mae'],
        'bias_difference': stec_metrics['bias'] - vtec_metrics['bias']
    }
    
    logger.info("\n" + "="*60)
    logger.info("COMPARISON RESULTS")
    logger.info("="*60)
    logger.info(f"\nDirect STEC Model:")
    logger.info(f"  RMSE: {stec_metrics['rmse']:.4f} TECU")
    logger.info(f"  MAE:  {stec_metrics['mae']:.4f} TECU")
    logger.info(f"  Bias: {stec_metrics['bias']:.4f} TECU")
    logger.info(f"  R²:   {stec_metrics['r2']:.4f}")
    
    logger.info(f"\nVTEC + Mapping Model:")
    logger.info(f"  RMSE: {vtec_metrics['rmse']:.4f} TECU")
    logger.info(f"  MAE:  {vtec_metrics['mae']:.4f} TECU")
    logger.info(f"  Bias: {vtec_metrics['bias']:.4f} TECU")
    logger.info(f"  R²:   {vtec_metrics['r2']:.4f}")
    
    logger.info(f"\nImprovement (STEC over VTEC+Mapping):")
    logger.info(f"  RMSE: {improvement['rmse_improvement_percent']:.2f}%")
    logger.info(f"  MAE:  {improvement['mae_improvement_percent']:.2f}%")
    logger.info("="*60 + "\n")
    
    return {
        'stec_model': stec_metrics,
        'vtec_mapped_model': vtec_metrics,
        'improvement': improvement
    }


def create_comparison_plots(
    stec_df: pd.DataFrame,
    vtec_df: pd.DataFrame,
    output_dir: Path,
    logger
):
    """Create visualization plots comparing both approaches."""
    logger.info("📈 Creating comparison plots...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data
    ground_truth = stec_df['true_stec'].values
    stec_pred = stec_df['pred_mean'].values
    vtec_mapped_pred = vtec_df['pred_stec_mapped'].values
    elevations = stec_df['satele'].values
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Scatter plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # STEC model
    ax = axes[0]
    ax.scatter(ground_truth, stec_pred, alpha=0.3, s=1, c=elevations, cmap='viridis')
    ax.plot([ground_truth.min(), ground_truth.max()], 
            [ground_truth.min(), ground_truth.max()], 
            'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('True STEC (TECU)', fontsize=12)
    ax.set_ylabel('Predicted STEC (TECU)', fontsize=12)
    ax.set_title('Direct STEC Model', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # VTEC + Mapping
    ax = axes[1]
    scatter = ax.scatter(ground_truth, vtec_mapped_pred, alpha=0.3, s=1, c=elevations, cmap='viridis')
    ax.plot([ground_truth.min(), ground_truth.max()], 
            [ground_truth.min(), ground_truth.max()], 
            'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('True STEC (TECU)', fontsize=12)
    ax.set_ylabel('Predicted STEC (TECU)', fontsize=12)
    ax.set_title('VTEC + Mapping Function', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('Elevation (degrees)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Error distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    stec_errors = stec_pred - ground_truth
    vtec_errors = vtec_mapped_pred - ground_truth
    
    # Error histograms
    ax = axes[0]
    ax.hist(stec_errors, bins=100, alpha=0.6, label='Direct STEC', density=True, color='blue')
    ax.hist(vtec_errors, bins=100, alpha=0.6, label='VTEC + Mapping', density=True, color='orange')
    ax.axvline(0, color='red', linestyle='--', lw=2, label='Zero error')
    ax.set_xlabel('Prediction Error (TECU)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error vs elevation
    ax = axes[1]
    ax.scatter(elevations, np.abs(stec_errors), alpha=0.3, s=1, label='Direct STEC', color='blue')
    ax.scatter(elevations, np.abs(vtec_errors), alpha=0.3, s=1, label='VTEC + Mapping', color='orange')
    ax.set_xlabel('Elevation (degrees)', fontsize=12)
    ax.set_ylabel('Absolute Error (TECU)', fontsize=12)
    ax.set_title('Error vs Elevation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Elevation-binned metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Create elevation bins
    elev_bins = np.arange(5, 91, 5)
    bin_centers = (elev_bins[:-1] + elev_bins[1:]) / 2
    
    stec_rmse_bins = []
    vtec_rmse_bins = []
    stec_bias_bins = []
    vtec_bias_bins = []
    
    for i in range(len(elev_bins) - 1):
        mask = (elevations >= elev_bins[i]) & (elevations < elev_bins[i+1])
        if mask.sum() > 0:
            stec_rmse_bins.append(np.sqrt(np.mean(stec_errors[mask] ** 2)))
            vtec_rmse_bins.append(np.sqrt(np.mean(vtec_errors[mask] ** 2)))
            stec_bias_bins.append(np.mean(stec_errors[mask]))
            vtec_bias_bins.append(np.mean(vtec_errors[mask]))
        else:
            stec_rmse_bins.append(np.nan)
            vtec_rmse_bins.append(np.nan)
            stec_bias_bins.append(np.nan)
            vtec_bias_bins.append(np.nan)
    
    # RMSE vs elevation
    ax = axes[0]
    ax.plot(bin_centers, stec_rmse_bins, 'o-', label='Direct STEC', linewidth=2, markersize=6)
    ax.plot(bin_centers, vtec_rmse_bins, 's-', label='VTEC + Mapping', linewidth=2, markersize=6)
    ax.set_xlabel('Elevation (degrees)', fontsize=12)
    ax.set_ylabel('RMSE (TECU)', fontsize=12)
    ax.set_title('RMSE vs Elevation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bias vs elevation
    ax = axes[1]
    ax.plot(bin_centers, stec_bias_bins, 'o-', label='Direct STEC', linewidth=2, markersize=6)
    ax.plot(bin_centers, vtec_bias_bins, 's-', label='VTEC + Mapping', linewidth=2, markersize=6)
    ax.axhline(0, color='red', linestyle='--', lw=1)
    ax.set_xlabel('Elevation (degrees)', fontsize=12)
    ax.set_ylabel('Bias (TECU)', fontsize=12)
    ax.set_title('Bias vs Elevation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Improvement percentage
    ax = axes[2]
    improvement_percent = 100 * (np.array(vtec_rmse_bins) - np.array(stec_rmse_bins)) / np.array(vtec_rmse_bins)
    ax.bar(bin_centers, improvement_percent, width=4, alpha=0.7, color='green')
    ax.axhline(0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('Elevation (degrees)', fontsize=12)
    ax.set_ylabel('RMSE Improvement (%)', fontsize=12)
    ax.set_title('Direct STEC Improvement over VTEC+Mapping', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'elevation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Plots saved to {output_dir}")


def save_comparison_results(
    metrics: Dict,
    stec_df: pd.DataFrame,
    vtec_df: pd.DataFrame,
    output_dir: Path,
    args,
    logger
):
    """Save comparison results to files."""
    logger.info("💾 Saving results...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics summary
    summary_path = output_dir / 'comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("STEC vs VTEC+Mapping Comparison Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"STEC Experiment: {args.stec_experiment}\n")
        f.write(f"VTEC Experiment: {args.vtec_experiment}\n")
        f.write(f"Mapping Function: {args.mapping_function}\n")
        f.write(f"Test Samples: {len(stec_df):,}\n\n")
        
        f.write("Direct STEC Model:\n")
        for key, val in metrics['stec_model'].items():
            f.write(f"  {key}: {val:.6f}\n")
        
        f.write("\nVTEC + Mapping Model:\n")
        for key, val in metrics['vtec_mapped_model'].items():
            f.write(f"  {key}: {val:.6f}\n")
        
        f.write("\nImprovement (STEC over VTEC+Mapping):\n")
        for key, val in metrics['improvement'].items():
            f.write(f"  {key}: {val:.6f}\n")
    
    # Save detailed predictions CSV
    comparison_df = pd.DataFrame({
        'true_stec': stec_df['true_stec'].values,
        'stec_pred': stec_df['pred_mean'].values,
        'vtec_mapped_pred': vtec_df['pred_stec_mapped'].values,
        'elevation': stec_df['satele'].values,
        'stec_error': stec_df['pred_mean'].values - stec_df['true_stec'].values,
        'vtec_mapped_error': vtec_df['pred_stec_mapped'].values - stec_df['true_stec'].values,
    })
    
    csv_path = output_dir / 'detailed_predictions.csv'
    comparison_df.to_csv(csv_path, index=False)
    
    logger.info(f"✅ Results saved to {output_dir}")


def main():
    """Main comparison workflow."""
    parser = argparse.ArgumentParser(description="Compare STEC vs VTEC+Mapping models")
    parser.add_argument("--stec_experiment", type=str, required=True,
                       help="Path to STEC model experiment folder")
    parser.add_argument("--vtec_experiment", type=str, required=True,
                       help="Path to VTEC model experiment folder")
    parser.add_argument("--mapping_function", type=str, default="MSLM",
                       choices=["SLM", "MSLM"],
                       help="Mapping function to use (default: MSLM)")
    parser.add_argument("--output_dir", type=str, default="comparisons/stec_vs_vtec",
                       help="Output directory for results")
    parser.add_argument("--num_mc_samples", type=int, default=100,
                       help="Number of inference samples for Bayesian models")
    parser.add_argument("--test_size", type=int, default=None,
                       help="Number of test samples (None = use config default)")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("STEC vs VTEC+Mapping Fair Comparison")
    logger.info("="*60)
    
    # Load configurations
    stec_config, stec_dir = load_experiment_config(args.stec_experiment)
    vtec_config, vtec_dir = load_experiment_config(args.vtec_experiment)
    
    # Verify targets
    if stec_config.get('target', 'stec').lower() != 'stec':
        raise ValueError(f"STEC experiment must have target='stec', got {stec_config.get('target')}")
    if vtec_config.get('target', 'vtec').lower() != 'vtec':
        raise ValueError(f"VTEC experiment must have target='vtec', got {vtec_config.get('target')}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stec_config["device"] = device
    vtec_config["device"] = device
    logger.info(f"Using device: {device}")
    
    # Override test size if specified
    if args.test_size:
        stec_config['data']['test_size'] = args.test_size
        vtec_config['data']['test_size'] = args.test_size
    
    # Load models
    stec_checkpoint = find_best_checkpoint(stec_dir)
    vtec_checkpoint = find_best_checkpoint(vtec_dir)
    
    stec_model, stec_registry = load_model_from_checkpoint(stec_config, stec_checkpoint, logger)
    vtec_model, vtec_registry = load_model_from_checkpoint(vtec_config, vtec_checkpoint, logger)
    
    # Get test data loaders (both use STEC data, but VTEC model extracts VTEC target)
    logger.info("📦 Loading test data...")
    stec_test_loader = get_test_data_loader(stec_config, logger)
    vtec_test_loader = get_test_data_loader(vtec_config, logger)
    
    # Run inference
    stec_df = run_stec_inference(stec_model, stec_test_loader, stec_config, logger, args.num_mc_samples)
    vtec_df = run_vtec_inference(vtec_model, vtec_test_loader, vtec_config, logger, args.num_mc_samples)
    
    # Apply mapping function to VTEC predictions
    vtec_df = apply_mapping_function(vtec_df, args.mapping_function, logger)
    
    # Ensure we're comparing the same samples (should be aligned by default)
    if len(stec_df) != len(vtec_df):
        logger.warning(f"Sample count mismatch: STEC={len(stec_df)}, VTEC={len(vtec_df)}")
        min_len = min(len(stec_df), len(vtec_df))
        stec_df = stec_df.iloc[:min_len].reset_index(drop=True)
        vtec_df = vtec_df.iloc[:min_len].reset_index(drop=True)
    
    # Compare models
    metrics = compare_models(stec_df, vtec_df, logger)
    
    # Create visualizations
    output_dir = Path(args.output_dir)
    create_comparison_plots(stec_df, vtec_df, output_dir, logger)
    
    # Save results
    save_comparison_results(metrics, stec_df, vtec_df, output_dir, args, logger)
    
    logger.info("="*60)
    logger.info("✅ Comparison completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
