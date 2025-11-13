#!/usr/bin/env python3
"""
Differential STEC (dSTEC) Evaluation Tool

This standalone script evaluates STEC model predictions vs IGS GIM products using
the differential STEC (dSTEC) metric. dSTEC compares measurements at different 
elevations during a satellite pass, removing common-mode errors.

Key Concept:
- For each satellite pass, identify the maximum elevation point as reference
- Compute dSTEC = STEC(t) - STEC(t_max) for observations with elevation 
  difference > threshold (default 20°)
- Compare model dSTEC vs GIM dSTEC vs ground truth dSTEC
- Compute RMSE, MAE, and relative error metrics

Configuration:
All parameters are set directly in this file (not in config.yaml)
See EVALUATION_CONFIG below for all configurable options.
"""

import os
import sys
import logging
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import traceback
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_parser import parse_config, compute_exp_name
from utils.feature_registry import initialize_feature_registry
from training.base_trainer import BaseTrainer
from data_loader import get_test_data_loader
from model.model import get_model
from evaluation.gim_mapper import MappingFunction

# ===========================
# EVALUATION CONFIGURATION
# ===========================
EVALUATION_CONFIG = {
    # Experiment settings
    "experiment_folder": None,  # Set via CLI arg, e.g., "Pretrain_STEC_BNN_NLL_..."
    
    # dSTEC computation parameters
    "dstec": {
        "min_samples_per_pass": 10,  # Minimum observations per satellite pass
        "elevation_diff_threshold": 20.0,  # Degrees - minimum elevation difference from max
        "use_mask": True,  # Apply elevation difference mask
    },
    
    # Mapping function for VTEC->STEC conversion
    "mapping_function": {
        "type": "MSLM",  # Modified Single Layer Model (from evaluation.gim_mapper)
        # MSLM uses height=506.7km and alpha=0.9782 (built into MappingFunction class)
    },
    
    # GIM data settings
    "gim": {
        "enabled": True,  # Compare against GIM
        "gim_path": "/path/to/gim/data",  # UPDATE THIS PATH
        "ac": "cod",  # Analysis center (cod, jpl, esa, etc.)
    },
    
    # Output settings
    "output": {
        "save_per_pass_csv": True,  # Save detailed per-pass CSV files
        "generate_plots": True,  # Generate comparison plots
        "save_summary_stats": True,  # Save summary statistics
    },
    
    # Model inference settings
    "inference": {
        "num_bayesian_samples": 100,  # For BNN uncertainty quantification
        "batch_size": None,  # Use config default if None
        "test_size_limit": None,  # Limit test samples (None = use all)
    },
}


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def find_experiment_directory(experiment_name: str) -> Optional[Path]:
    """Find experiment directory."""
    base_dir = Path("experiments")
    if not base_dir.exists():
        return None
    
    exp_path = base_dir / experiment_name
    return exp_path if exp_path.exists() else None


def find_model_checkpoint(experiment_dir: Path) -> Optional[Path]:
    """Find model checkpoint in experiment directory."""
    model_dir = experiment_dir / "model"
    if not model_dir.exists():
        return None
    
    pth_files = list(model_dir.glob("*.pth"))
    return pth_files[0] if pth_files else None


def load_experiment_config(experiment_dir: Path) -> Dict:
    """Load experiment configuration."""
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_model_inference(
    config: Dict,
    experiment_dir: Path,
    model_path: Path,
    logger: logging.Logger,
    eval_config: Dict,
) -> pd.DataFrame:
    """
    Run model inference on test set.
    
    Returns:
        DataFrame with predictions and metadata
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Running Model Inference")
    logger.info("=" * 80)
    
    # Initialize feature registry
    feature_registry = initialize_feature_registry(config)
    config["feature_registry"] = feature_registry
    
    # Override test_size if specified
    if eval_config["inference"]["test_size_limit"] is not None:
        config["data"]["test_size"] = eval_config["inference"]["test_size_limit"]
    
    # Get test dataloader
    test_loader = get_test_data_loader(config, logger)
    logger.info(f"  Test samples: {len(test_loader.dataset):,}")
    
    # Create trainer
    trainer = BaseTrainer(config, logger)
    
    # Load model
    model = get_model(config).to(config["device"])
    checkpoint = torch.load(model_path, map_location=config["device"], weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    logger.info(f"  Model type: {config['model']['model_type']}")
    logger.info(f"  Device: {config['device']}")
    
    # Determine number of Bayesian samples
    is_bnn = "BNN" in config["model"]["model_type"]
    num_samples = eval_config["inference"]["num_bayesian_samples"] if is_bnn else 1
    
    logger.info(f"  Bayesian samples: {num_samples}")
    
    # Run Bayesian inference
    _, test_df = trainer.bayesian_inference_total_uncertainty(
        model, test_loader, num_samples=num_samples
    )
    
    logger.info(f"  ✓ Inference complete: {len(test_df):,} predictions")
    
    return test_df


def create_satellite_pass_id(df: pd.DataFrame) -> pd.Series:
    """
    Create unique satellite pass IDs for individual satellite passes.
    
    Each pass is uniquely identified by:
    - Station identifier
    - Satellite ID
    - Slip cycle (phase continuity indicator)
    - Date (year + day of year)
    
    This ensures that passes on different days are treated separately.
    
    Args:
        df: DataFrame with station, satellite, and temporal information
        
    Returns:
        Series of pass IDs
        
    Raises:
        ValueError: If required fields are missing
    """
    # Check for required fields
    required_fields = ["station", "sat", "slipc", "year", "doy"]
    missing_fields = [f for f in required_fields if f not in df.columns]
    
    if missing_fields:
        raise ValueError(
            f"Missing required fields for pass identification: {missing_fields}\n"
            f"Required fields: station, sat, slipc, year, doy"
        )
    
    # Handle station field (may be bytes or string)
    if df["station"].dtype == object:
        # Try to decode if bytes
        station = df["station"].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x)
        )
    else:
        station = df["station"].astype(str)
    
    # Handle satellite ID (may be string or numeric)
    sat = df["sat"].astype(str)
    
    # Handle slip cycle
    slipc = df["slipc"].astype(int).astype(str)
    
    # Handle date
    year = df["year"].astype(int).astype(str)
    doy = df["doy"].astype(int).apply(lambda x: f"{x:03d}")  # Zero-padded 3-digit DOY
    
    # Create composite ID: STA{station}_SAT{sat}_YYYYDDD_SLIP{slipc}
    # This uniquely identifies each satellite pass on each day
    pass_id = "STA" + station + "_SAT" + sat + "_" + year + doy + "_SLIP" + slipc
    
    return pass_id


def compute_dstec_for_pass(
    group: pd.DataFrame,
    eval_config: Dict,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """
    Compute differential STEC for a single satellite pass.
    
    Args:
        group: DataFrame containing observations for one satellite pass
        eval_config: Evaluation configuration
        logger: Logger instance
        
    Returns:
        DataFrame with dSTEC results, or None if insufficient data
    """
    n_samples = len(group)
    pass_id = group.name  # Groupby sets the group name
    
    # Check minimum sample requirement
    min_samples = eval_config["dstec"]["min_samples_per_pass"]
    if n_samples < min_samples:
        return None
    
    # Sort by time (seconds of day) to ensure temporal ordering
    # This is critical for identifying the actual pass maximum elevation
    group = group.sort_values("sod").reset_index(drop=True)
    
    # Find maximum elevation index
    idx_max = group["satele"].idxmax()
    
    # Get elevation difference from maximum
    elev_diff = group["satele"] - group.loc[idx_max, "satele"]
    
    # Create mask for independent measurements
    # (elevation difference > threshold ensures independence)
    elev_threshold = eval_config["dstec"]["elevation_diff_threshold"]
    if eval_config["dstec"]["use_mask"]:
        mask = elev_diff < -elev_threshold  # Only use points well below max elevation
    else:
        mask = np.ones(len(group), dtype=bool)
    
    # Compute mapping function using existing MappingFunction class
    mapping_func = MappingFunction(mapping_type='MSLM')
    
    # Convert elevation to radians for mapping function
    elev_rad = np.deg2rad(group["satele"].values)
    elev_max_rad = np.deg2rad(group.loc[idx_max, "satele"])
    
    # Get mapping factors
    facion = np.array([mapping_func.get_mapping_factor(e) for e in elev_rad])
    facion_max = mapping_func.get_mapping_factor(elev_max_rad)
    
    # Compute differential STEC from geometry-free phase (ground truth)
    dstec_truth = group["gfphase"].values - group.loc[idx_max, "gfphase"]
    
    # Compute differential STEC from model predictions
    dstec_model = (
        group["pred_stec"].values * facion - 
        group.loc[idx_max, "pred_stec"] * facion_max
    )
    
    # Compute variance propagation for model uncertainty
    dstec_model_var = (
        group["pred_total_unc"].values**2 * facion**2 + 
        group.loc[idx_max, "pred_total_unc"]**2 * facion_max**2
    )
    
    # Compute errors
    dstec_model_error = dstec_model - dstec_truth
    
    # Build result dataframe
    result = pd.DataFrame({
        "pass_id": pass_id,
        "lat_sta": group["lat_sta"].values,
        "lon_sta": group["lon_sta"].values,
        "sod": group["sod"].values,
        "satele": group["satele"].values,
        "satele_max": group.loc[idx_max, "satele"],
        "elev_diff": elev_diff.values,
        "mask": mask,
        "facion": facion,
        "dstec_truth": dstec_truth,
        "dstec_model": dstec_model,
        "dstec_model_var": dstec_model_var,
        "dstec_model_error": dstec_model_error,
    })
    
    return result


def compute_pass_statistics(
    dstec_df: pd.DataFrame,
    eval_config: Dict,
) -> pd.DataFrame:
    """
    Compute statistics for each satellite pass.
    
    Args:
        dstec_df: DataFrame with dSTEC results
        eval_config: Evaluation configuration
        
    Returns:
        DataFrame with per-pass statistics
    """
    
    def compute_stats(group):
        """Compute statistics for one pass."""
        # Apply mask if enabled
        if eval_config["dstec"]["use_mask"]:
            group = group[group["mask"]]
        
        if len(group) == 0:
            return pd.Series({
                "n_samples": 0,
                "n_masked": 0,
                "dstec_rms": np.nan,
                "model_rmse": np.nan,
                "model_mae": np.nan,
                "model_re": np.nan,
            })
        
        # Compute dSTEC RMS (variability of truth)
        dstec_rms = np.sqrt(np.nanmean(group["dstec_truth"]**2))
        
        # Compute model metrics
        model_rmse = np.sqrt(np.nanmean(group["dstec_model_error"]**2))
        model_mae = np.nanmean(np.abs(group["dstec_model_error"]))
        model_re = model_rmse / dstec_rms if dstec_rms > 0 else np.nan
        
        return pd.Series({
            "n_samples": len(group),
            "n_masked": np.sum(group["mask"]),
            "dstec_rms": dstec_rms,
            "model_rmse": model_rmse,
            "model_mae": model_mae,
            "model_re": model_re,
        })
    
    stats = dstec_df.groupby("pass_id").apply(compute_stats).reset_index()
    
    # Remove passes with no valid masked samples
    stats = stats[stats["n_masked"] > 0].reset_index(drop=True)
    
    return stats


def save_results(
    dstec_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
    eval_config: Dict,
    logger: logging.Logger,
):
    """Save dSTEC evaluation results."""
    logger.info("=" * 80)
    logger.info("STEP 3: Saving Results")
    logger.info("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-pass CSV files if requested
    if eval_config["output"]["save_per_pass_csv"]:
        per_pass_dir = output_dir / "per_pass"
        per_pass_dir.mkdir(exist_ok=True)
        
        n_passes = 0
        for pass_id, group in dstec_df.groupby("pass_id"):
            # Save only passes with sufficient masked samples
            if np.sum(group["mask"]) >= 5:
                csv_path = per_pass_dir / f"{pass_id}.csv"
                group.to_csv(csv_path, index=False, float_format="%.4f")
                n_passes += 1
        
        logger.info(f"  ✓ Saved {n_passes} per-pass CSV files to {per_pass_dir}")
    
    # Save summary statistics
    if eval_config["output"]["save_summary_stats"]:
        stats_path = output_dir / "pass_statistics.csv"
        stats_df.to_csv(stats_path, index=False, float_format="%.4f")
        logger.info(f"  ✓ Saved pass statistics to {stats_path}")
        
        # Save overall summary
        summary_path = output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("dSTEC Evaluation Summary\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total satellite passes: {len(stats_df)}\n")
            f.write(f"Total observations: {len(dstec_df)}\n")
            f.write(f"Masked observations: {np.sum(dstec_df['mask'])}\n\n")
            
            f.write("Model Performance (across all passes):\n")
            f.write(f"  Mean RMSE: {stats_df['model_rmse'].mean():.4f} TECU\n")
            f.write(f"  Median RMSE: {stats_df['model_rmse'].median():.4f} TECU\n")
            f.write(f"  Mean MAE: {stats_df['model_mae'].mean():.4f} TECU\n")
            f.write(f"  Median MAE: {stats_df['model_mae'].median():.4f} TECU\n")
            f.write(f"  Mean Relative Error: {stats_df['model_re'].mean():.4f}\n")
            f.write(f"  Median Relative Error: {stats_df['model_re'].median():.4f}\n")
        
        logger.info(f"  ✓ Saved summary to {summary_path}")


def generate_plots(
    dstec_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
):
    """Generate comparison plots."""
    logger.info("=" * 80)
    logger.info("STEP 4: Generating Plots")
    logger.info("=" * 80)
    
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150
    
    # Filter to masked samples for plotting
    masked_df = dstec_df[dstec_df["mask"]].copy()
    
    # Plot 1: dSTEC scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(
        masked_df["dstec_truth"],
        masked_df["dstec_model"],
        alpha=0.3,
        s=1,
        label="Model",
    )
    
    # 1:1 line
    lim = max(abs(masked_df["dstec_truth"].max()), abs(masked_df["dstec_model"].max()))
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label="1:1")
    
    ax.set_xlabel("dSTEC Truth (TECU)")
    ax.set_ylabel("dSTEC Model (TECU)")
    ax.set_title("Differential STEC: Model vs Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "dstec_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved dSTEC scatter plot")
    
    # Plot 2: Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error histogram
    axes[0].hist(masked_df["dstec_model_error"], bins=100, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0].set_xlabel("dSTEC Error (TECU)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error vs elevation
    axes[1].scatter(
        masked_df["satele"],
        masked_df["dstec_model_error"],
        alpha=0.3,
        s=1,
    )
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel("Elevation (degrees)")
    axes[1].set_ylabel("dSTEC Error (TECU)")
    axes[1].set_title("Error vs Elevation")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "error_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved error analysis plots")
    
    # Plot 3: Per-pass statistics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RMSE distribution
    axes[0, 0].hist(stats_df["model_rmse"], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel("RMSE (TECU)")
    axes[0, 0].set_ylabel("Number of Passes")
    axes[0, 0].set_title("RMSE Distribution Across Passes")
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE distribution
    axes[0, 1].hist(stats_df["model_mae"], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel("MAE (TECU)")
    axes[0, 1].set_ylabel("Number of Passes")
    axes[0, 1].set_title("MAE Distribution Across Passes")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Relative error distribution
    axes[1, 0].hist(
        stats_df["model_re"][stats_df["model_re"] < 2],  # Cap at 2 for visibility
        bins=50,
        alpha=0.7,
        edgecolor='black',
    )
    axes[1, 0].set_xlabel("Relative Error")
    axes[1, 0].set_ylabel("Number of Passes")
    axes[1, 0].set_title("Relative Error Distribution (<2.0)")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sample count distribution
    axes[1, 1].hist(stats_df["n_masked"], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel("Number of Masked Samples")
    axes[1, 1].set_ylabel("Number of Passes")
    axes[1, 1].set_title("Samples Per Pass Distribution")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "pass_statistics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved pass statistics plots")
    logger.info(f"  All plots saved to {plots_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="dSTEC Evaluation: Compare STEC model predictions vs IGS GIMs"
    )
    parser.add_argument(
        "experiment_folder",
        type=str,
        help="Experiment folder name (e.g., 'Pretrain_STEC_BNN_NLL_...')",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation",
    )
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.verbose)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Update config
    EVALUATION_CONFIG["experiment_folder"] = args.experiment_folder
    if args.no_plots:
        EVALUATION_CONFIG["output"]["generate_plots"] = False
    
    try:
        logger.info("=" * 80)
        logger.info("dSTEC EVALUATION TOOL")
        logger.info("=" * 80)
        logger.info(f"Experiment: {args.experiment_folder}")
        
        # Find experiment
        exp_dir = find_experiment_directory(args.experiment_folder)
        if exp_dir is None:
            logger.error(f"Experiment not found: {args.experiment_folder}")
            return 1
        
        # Find model checkpoint
        model_path = find_model_checkpoint(exp_dir)
        if model_path is None:
            logger.error(f"No model checkpoint found in {exp_dir}")
            return 1
        
        logger.info(f"Model: {model_path}")
        
        # Load config
        config = load_experiment_config(exp_dir)
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Run model inference
        test_df = run_model_inference(
            config, exp_dir, model_path, logger, EVALUATION_CONFIG
        )
        
        # Create satellite pass IDs
        logger.info("=" * 80)
        logger.info("STEP 2: Computing dSTEC for Satellite Passes")
        logger.info("=" * 80)
        
        test_df["pass_id"] = create_satellite_pass_id(test_df)
        n_passes = test_df["pass_id"].nunique()
        logger.info(f"  Identified {n_passes} unique satellite passes")
        
        # Compute dSTEC for each pass
        dstec_results = []
        for pass_id, group in tqdm(
            test_df.groupby("pass_id"),
            desc="  Processing passes",
            total=n_passes,
        ):
            group.name = pass_id  # Set name for compute_dstec_for_pass
            result = compute_dstec_for_pass(group, EVALUATION_CONFIG, logger)
            if result is not None:
                dstec_results.append(result)
        
        if len(dstec_results) == 0:
            logger.error("No valid satellite passes found!")
            return 1
        
        # Combine results
        dstec_df = pd.concat(dstec_results, ignore_index=True)
        logger.info(f"  ✓ Computed dSTEC for {len(dstec_results)} passes")
        logger.info(f"  Total observations: {len(dstec_df):,}")
        logger.info(f"  Masked observations: {np.sum(dstec_df['mask']):,}")
        
        # Compute per-pass statistics
        stats_df = compute_pass_statistics(dstec_df, EVALUATION_CONFIG)
        logger.info(f"  ✓ Computed statistics for {len(stats_df)} passes")
        
        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("Summary Statistics")
        logger.info("=" * 80)
        logger.info(f"Mean RMSE: {stats_df['model_rmse'].mean():.4f} TECU")
        logger.info(f"Median RMSE: {stats_df['model_rmse'].median():.4f} TECU")
        logger.info(f"Mean MAE: {stats_df['model_mae'].mean():.4f} TECU")
        logger.info(f"Median MAE: {stats_df['model_mae'].median():.4f} TECU")
        logger.info(f"Mean Relative Error: {stats_df['model_re'].mean():.4f}")
        logger.info(f"Median Relative Error: {stats_df['model_re'].median():.4f}")
        
        # Save results
        output_dir = exp_dir / "dstec_evaluation"
        save_results(dstec_df, stats_df, output_dir, EVALUATION_CONFIG, logger)
        
        # Generate plots
        if EVALUATION_CONFIG["output"]["generate_plots"]:
            generate_plots(dstec_df, stats_df, output_dir, logger)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ EVALUATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ EVALUATION FAILED: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
