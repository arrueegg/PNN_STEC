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
from evaluation.gim_mapper import MappingFunction, GIMMapper
from datetime import timedelta

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
        "gim_path": "/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex",
        "ac": "IGS",  # Analysis center (cod, jpl, esa, etc.)
    },
    
    # Output settings
    "output": {
        "save_per_pass_csv": False,  # Save detailed per-pass CSV files (disabled - not needed)
        "generate_plots": True,  # Generate comparison plots
        "save_summary_stats": True,  # Save summary statistics
    },
    
    # Plot settings
    "plots": {
        "use_percentile_limits": True,  # Use percentile-based axis limits (robust to outliers)
        "percentile_lower": 1,  # Lower percentile for axis limits (1%)
        "percentile_upper": 99,  # Upper percentile for axis limits (99%)
        "error_percentile": 99.5,  # Percentile for error histogram limits
    },
    
    # Debug settings
    "debug": {
        "single_station": "ALGO",  # Restrict to single station for faster debugging
                                  # Examples: "ALGO", "BRUS", "NYA1", etc.
                                  # Set to None to process all stations
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
    
    # Enable metadata loading for dSTEC evaluation
    config["return_metadata"] = True
    config["metadata_fields"] = ["station", "sat", "slipc", "gfphase"]
    
    # Override test_size if specified
    if eval_config["inference"]["test_size_limit"] is not None:
        config["data"]["test_size"] = eval_config["inference"]["test_size_limit"]
    
    # Get test dataloader
    test_loader = get_test_data_loader(config, logger)
    logger.info(f"  Test samples: {len(test_loader.dataset):,}")
    logger.info(f"  Metadata fields enabled: {config['metadata_fields']}")
    
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
    
    # Debug: Filter to single station if requested
    if eval_config.get("debug", {}).get("single_station") is not None:
        debug_station = eval_config["debug"]["single_station"]
        
        # Handle bytes vs string station names
        if test_df["station"].dtype == object:
            # Try to decode if bytes
            test_df["station_str"] = test_df["station"].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else str(x)
            )
            mask = test_df["station_str"] == debug_station
        else:
            mask = test_df["station"].astype(str) == debug_station
        
        original_len = len(test_df)
        test_df = test_df[mask].copy()
        logger.info(f"  🐛 DEBUG: Filtered to station '{debug_station}': {len(test_df):,}/{original_len:,} samples")
        
        if len(test_df) == 0:
            raise ValueError(f"No samples found for station '{debug_station}'")
    
    # Verify metadata fields are present
    required_metadata = ["station", "sat", "slipc", "gfphase"]
    missing_fields = [f for f in required_metadata if f not in test_df.columns]
    
    if missing_fields:
        raise ValueError(
            f"Missing required metadata fields: {missing_fields}\n"
            f"Available columns: {test_df.columns.tolist()}"
        )
    
    logger.info(f"  ✓ Metadata fields present: {', '.join(required_metadata)}")
    
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
    gim_mapper: Optional[GIMMapper] = None,
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
    
    # NOTE: Model already predicts STEC (slant TEC), so NO mapping function needed!
    # Only apply mapping for GIM (which provides VTEC that needs conversion to STEC)
    
    # Compute differential STEC from geometry-free phase (ground truth)
    # Ground truth is also already in slant coordinates
    dstec_truth = group["gfphase"].values - group.loc[idx_max, "gfphase"]
    
    # Compute differential STEC from model predictions
    # Model already outputs STEC, so simple difference is correct
    dstec_model = group["pred_stec"].values - group.loc[idx_max, "pred_stec"]
    
    # Compute variance propagation for model uncertainty
    # Simple addition since measurements are at different times
    dstec_model_var = (
        group["pred_total_unc"].values**2 + 
        group.loc[idx_max, "pred_total_unc"]**2
    )
    
    # Compute errors
    dstec_model_error = dstec_model - dstec_truth
    
    # If GIM comparison is enabled and a mapper was provided, compute GIM-derived STEC
    stec_gim = None
    dstec_gim = None
    if eval_config.get("gim", {}).get("enabled", False) and gim_mapper is not None:
        try:
            # Build observation arrays for mapper
            # Prefer IPP coordinates if available, otherwise fall back to station coords
            if "lat_ipp" in group.columns and "lon_ipp" in group.columns:
                ipp_lat = group["lat_ipp"].values
                ipp_lon = group["lon_ipp"].values
            else:
                ipp_lat = group["lat_sta"].values
                ipp_lon = group["lon_sta"].values

            # Use elevations in degrees
            elevations = group["satele"].values

            # Build times as datetimes (use year, doy, sod)
            years = group["year"].astype(int).values
            doys = group["doy"].astype(int).values
            sods = group["sod"].astype(float).values
            times = []
            for y, d, s in zip(years, doys, sods):
                day0 = datetime(y, 1, 1)
                dt = day0 + timedelta(days=int(d) - 1, seconds=float(s))
                times.append(dt)

            # Ensure GIM data is loaded for the pass day (mapper may already have data)
            # We attempt to (re)load for the first time in this pass if necessary
            try:
                gim_mapper.load_gim_data(eval_config.get("gim", {}).get("gim_path", "."), times[0])
            except Exception:
                # If loading fails, mapper may already be loaded or files missing; proceed and let mapper handle it
                pass

            # Use mapper to get STEC values (map_vtec_to_stec accepts sods/lat/lon/elev arrays)
            stec_gim = gim_mapper.map_vtec_to_stec(
                sods=np.array(sods),
                ipp_lat=np.array(ipp_lat),
                ipp_lon=np.array(ipp_lon),
                elevations=np.array(elevations),
            )

            # Compute differential STEC for GIM (subtract STEC at pass max)
            if stec_gim is not None and len(stec_gim) == len(group):
                stec_gim = np.array(stec_gim)
                # If the STEC at idx_max is NaN, dstec_gim will be NaN for all
                dstec_gim = stec_gim - stec_gim[idx_max]
            else:
                stec_gim = np.full(len(group), np.nan)
                dstec_gim = np.full(len(group), np.nan)

        except Exception as e:
            logger.debug(f"GIM mapping failed for pass {pass_id}: {e}")
            stec_gim = np.full(len(group), np.nan)
            dstec_gim = np.full(len(group), np.nan)

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
        "dstec_truth": dstec_truth,
        "dstec_model": dstec_model,
        "dstec_model_var": dstec_model_var,
        "dstec_model_error": dstec_model_error,
    })
    
    # Attach GIM-derived columns if computed
    if stec_gim is not None:
        result["stec_gim"] = stec_gim
        result["dstec_gim"] = dstec_gim

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
        # Store original sample count and masked count BEFORE filtering
        n_total_samples = len(group)
        n_masked_samples = np.sum(group["mask"]) if "mask" in group.columns else 0
        
        # Apply mask if enabled
        if eval_config["dstec"]["use_mask"]:
            group = group[group["mask"]]
        
        if len(group) == 0:
            return pd.Series({
                "n_samples": n_total_samples,
                "n_masked": n_masked_samples,
                "dstec_rms": np.nan,
                "model_rmse": np.nan,
                "model_mae": np.nan,
                "model_re": np.nan,
                "gim_rmse": np.nan,
                "gim_mae": np.nan,
                "gim_re": np.nan,
            })
        
        # Compute dSTEC RMS (variability of truth)
        dstec_rms = np.sqrt(np.nanmean(group["dstec_truth"]**2))
        
        # Compute model metrics
        model_rmse = np.sqrt(np.nanmean(group["dstec_model_error"]**2))
        model_mae = np.nanmean(np.abs(group["dstec_model_error"]))
        model_re = model_rmse / dstec_rms if dstec_rms > 0 else np.nan
        
        # Compute GIM metrics if available
        gim_rmse = np.nan
        gim_mae = np.nan
        gim_re = np.nan
        if "dstec_gim" in group.columns:
            gim_error = group["dstec_gim"] - group["dstec_truth"]
            # Only compute if we have valid GIM data
            valid_gim = ~np.isnan(gim_error)
            if np.sum(valid_gim) > 0:
                gim_rmse = np.sqrt(np.nanmean(gim_error**2))
                gim_mae = np.nanmean(np.abs(gim_error))
                gim_re = gim_rmse / dstec_rms if dstec_rms > 0 else np.nan
        
        return pd.Series({
            "n_samples": n_total_samples,
            "n_masked": n_masked_samples,
            "dstec_rms": dstec_rms,
            "model_rmse": model_rmse,
            "model_mae": model_mae,
            "model_re": model_re,
            "gim_rmse": gim_rmse,
            "gim_mae": gim_mae,
            "gim_re": gim_re,
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
        
        # Compute correlation metrics for summary
        from scipy.stats import pearsonr
        masked_df = dstec_df[dstec_df["mask"]].copy()
        pearson_r_model, _ = pearsonr(masked_df["dstec_truth"], masked_df["dstec_model"])
        r_squared_model = pearson_r_model ** 2
        
        # Check if GIM data is available
        has_gim = "dstec_gim" in masked_df.columns and masked_df["dstec_gim"].notna().any()
        if has_gim:
            valid_gim_mask = masked_df["dstec_gim"].notna()
            gim_df = masked_df[valid_gim_mask]
            if len(gim_df) > 0:
                pearson_r_gim, _ = pearsonr(gim_df["dstec_truth"], gim_df["dstec_gim"])
                r_squared_gim = pearson_r_gim ** 2
            else:
                has_gim = False
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("dSTEC Evaluation Summary\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total satellite passes: {len(stats_df)}\n")
            f.write(f"Total observations: {len(dstec_df)}\n")
            f.write(f"Masked observations: {np.sum(dstec_df['mask'])}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("MODEL Performance (across all passes):\n")
            f.write("=" * 80 + "\n")
            f.write(f"  Mean RMSE: {stats_df['model_rmse'].mean():.4f} TECU\n")
            f.write(f"  Median RMSE: {stats_df['model_rmse'].median():.4f} TECU\n")
            f.write(f"  Mean MAE: {stats_df['model_mae'].mean():.4f} TECU\n")
            f.write(f"  Median MAE: {stats_df['model_mae'].median():.4f} TECU\n")
            f.write(f"  Mean Relative Error: {stats_df['model_re'].mean():.4f}\n")
            f.write(f"  Median Relative Error: {stats_df['model_re'].median():.4f}\n")
            f.write(f"  Pearson R: {pearson_r_model:.6f}\n")
            f.write(f"  R²: {r_squared_model:.6f}\n\n")
            
            if has_gim:
                # Filter stats to only passes with valid GIM data
                valid_gim_stats = stats_df[~stats_df['gim_rmse'].isna()]
                f.write("=" * 80 + "\n")
                f.write(f"GIM Performance (across {len(valid_gim_stats)} passes with valid GIM data):\n")
                f.write("=" * 80 + "\n")
                f.write(f"  Mean RMSE: {valid_gim_stats['gim_rmse'].mean():.4f} TECU\n")
                f.write(f"  Median RMSE: {valid_gim_stats['gim_rmse'].median():.4f} TECU\n")
                f.write(f"  Mean MAE: {valid_gim_stats['gim_mae'].mean():.4f} TECU\n")
                f.write(f"  Median MAE: {valid_gim_stats['gim_mae'].median():.4f} TECU\n")
                f.write(f"  Mean Relative Error: {valid_gim_stats['gim_re'].mean():.4f}\n")
                f.write(f"  Median Relative Error: {valid_gim_stats['gim_re'].median():.4f}\n")
                f.write(f"  Pearson R: {pearson_r_gim:.6f}\n")
                f.write(f"  R²: {r_squared_gim:.6f}\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("MODEL vs GIM Comparison:\n")
                f.write("=" * 80 + "\n")
                # Compare on same passes where both have data
                comparison_stats = valid_gim_stats
                model_better_rmse = (comparison_stats['model_rmse'] < comparison_stats['gim_rmse']).sum()
                model_better_mae = (comparison_stats['model_mae'] < comparison_stats['gim_mae']).sum()
                f.write(f"  Passes where Model RMSE < GIM RMSE: {model_better_rmse}/{len(comparison_stats)} ({100*model_better_rmse/len(comparison_stats):.1f}%)\n")
                f.write(f"  Passes where Model MAE < GIM MAE: {model_better_mae}/{len(comparison_stats)} ({100*model_better_mae/len(comparison_stats):.1f}%)\n")
                f.write(f"  Mean RMSE improvement: {(valid_gim_stats['gim_rmse'].mean() - valid_gim_stats['model_rmse'].mean()):.4f} TECU\n")
                f.write(f"  Mean MAE improvement: {(valid_gim_stats['gim_mae'].mean() - valid_gim_stats['model_mae'].mean()):.4f} TECU\n")
        
        logger.info(f"  ✓ Saved summary to {summary_path}")


def generate_plots(
    dstec_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
    eval_config: Dict,
):
    """Generate comparison plots with robust handling of outliers."""
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
    
    # Get percentile-based limits for robust plotting
    use_percentiles = eval_config.get("plots", {}).get("use_percentile_limits", True)
    p_lower = eval_config.get("plots", {}).get("percentile_lower", 1)
    p_upper = eval_config.get("plots", {}).get("percentile_upper", 99)
    p_error = eval_config.get("plots", {}).get("error_percentile", 99.5)
    
    # Plot 1: dSTEC scatter plot (robust to outliers)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if use_percentiles:
        # Use percentile-based limits to exclude outliers
        truth_limits = np.percentile(masked_df["dstec_truth"], [p_lower, p_upper])
        model_limits = np.percentile(masked_df["dstec_model"], [p_lower, p_upper])
        
        # Filter to percentile range for plotting
        plot_mask = (
            (masked_df["dstec_truth"] >= truth_limits[0]) &
            (masked_df["dstec_truth"] <= truth_limits[1]) &
            (masked_df["dstec_model"] >= model_limits[0]) &
            (masked_df["dstec_model"] <= model_limits[1])
        )
        plot_df = masked_df[plot_mask]
        
        # Asymmetric limits based on actual data range
        xlim = truth_limits
        ylim = model_limits
        
        n_outliers = len(masked_df) - len(plot_df)
        logger.info(f"  Scatter plot: Using {p_lower}-{p_upper}% percentile range, excluding {n_outliers:,} outliers")
    else:
        plot_df = masked_df
        xlim = [masked_df["dstec_truth"].min(), masked_df["dstec_truth"].max()]
        ylim = [masked_df["dstec_model"].min(), masked_df["dstec_model"].max()]
    
    # Compute correlation metrics
    from scipy.stats import pearsonr
    pearson_r, p_value = pearsonr(plot_df["dstec_truth"], plot_df["dstec_model"])
    r_squared = pearson_r ** 2
    
    ax.scatter(
        plot_df["dstec_truth"],
        plot_df["dstec_model"],
        alpha=0.3,
        s=1,
        label=f"Model (n={len(plot_df):,})",
    )
    
    # 1:1 line spanning the data range
    line_min = max(xlim[0], ylim[0])
    line_max = min(xlim[1], ylim[1])
    ax.plot([line_min, line_max], [line_min, line_max], 'k--', alpha=0.5, linewidth=2, label="1:1")
    
    # Add correlation metrics to title
    ax.set_xlabel("dSTEC Truth (TECU)", fontsize=12)
    ax.set_ylabel("dSTEC Model (TECU)", fontsize=12)
    ax.set_title(f"Differential STEC: Model vs Truth\nPearson R = {pearson_r:.4f}, R² = {r_squared:.4f}", fontsize=13)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "dstec_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved dSTEC scatter plot (R={pearson_r:.4f}, R²={r_squared:.4f})")
    
    # Plot 2: Error distribution (robust to outliers)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error histogram with percentile-based limits
    if use_percentiles:
        error_limits = np.percentile(masked_df["dstec_model_error"], [100-p_error, p_error])
        error_mask = (
            (masked_df["dstec_model_error"] >= error_limits[0]) &
            (masked_df["dstec_model_error"] <= error_limits[1])
        )
        error_plot_df = masked_df[error_mask]
        n_error_outliers = len(masked_df) - len(error_plot_df)
        logger.info(f"  Error histogram: Using ±{p_error}% percentile range, excluding {n_error_outliers:,} outliers")
    else:
        error_plot_df = masked_df
    
    axes[0].hist(error_plot_df["dstec_model_error"], bins=100, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0].set_xlabel("dSTEC Error (TECU)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Error Distribution\n(±{p_error}th percentile range)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error vs elevation (with percentile-based y-limits)
    if use_percentiles:
        elev_error_plot_df = error_plot_df
    else:
        elev_error_plot_df = masked_df
    
    axes[1].scatter(
        elev_error_plot_df["satele"],
        elev_error_plot_df["dstec_model_error"],
        alpha=0.3,
        s=1,
    )
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel("Elevation (degrees)")
    axes[1].set_ylabel("dSTEC Error (TECU)")
    axes[1].set_title("Error vs Elevation")
    axes[1].grid(True, alpha=0.3)
    
    if use_percentiles:
        axes[1].set_ylim(error_limits[0], error_limits[1])
    
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
    
    # Plot 4: Density plots (2D histograms with colormaps)
    logger.info("  Generating density visualizations...")
    
    # Compute correlation metrics for density plots
    from scipy.stats import pearsonr
    pearson_r_full, _ = pearsonr(masked_df["dstec_truth"], masked_df["dstec_model"])
    r_squared_full = pearson_r_full ** 2
    
    # 4a: dSTEC density plot (hexbin)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Use percentile limits - asymmetric based on actual data
    if use_percentiles:
        truth_limits = np.percentile(masked_df["dstec_truth"], [p_lower, p_upper])
        model_limits = np.percentile(masked_df["dstec_model"], [p_lower, p_upper])
        xlim = truth_limits
        ylim = model_limits
    else:
        xlim = [masked_df["dstec_truth"].min(), masked_df["dstec_truth"].max()]
        ylim = [masked_df["dstec_model"].min(), masked_df["dstec_model"].max()]
    
    # Hexbin plot
    hb = axes[0].hexbin(
        masked_df["dstec_truth"],
        masked_df["dstec_model"],
        gridsize=50,
        cmap='viridis',
        mincnt=1,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        bins='log',
    )
    # 1:1 line spanning the overlap range
    line_min = max(xlim[0], ylim[0])
    line_max = min(xlim[1], ylim[1])
    axes[0].plot([line_min, line_max], [line_min, line_max], 'r--', alpha=0.7, linewidth=2, 
                 label=f"1:1 (R={pearson_r_full:.3f})")
    axes[0].set_xlabel("dSTEC Truth (TECU)", fontsize=12)
    axes[0].set_ylabel("dSTEC Model (TECU)", fontsize=12)
    axes[0].set_title(f"Model vs Truth - Hexbin Density\nR² = {r_squared_full:.4f}", fontsize=13)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(hb, ax=axes[0], label='log10(count)')
    
    # 2D histogram with colormap
    h, xedges, yedges = np.histogram2d(
        masked_df["dstec_truth"],
        masked_df["dstec_model"],
        bins=100,
        range=[xlim, ylim]
    )
    
    im = axes[1].imshow(
        h.T,
        origin='lower',
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap='plasma',
        aspect='auto',
        norm=plt.matplotlib.colors.LogNorm(vmin=max(1, h[h>0].min()), vmax=h.max()),
        interpolation='bilinear'
    )
    axes[1].plot([line_min, line_max], [line_min, line_max], 'r--', alpha=0.7, linewidth=2, 
                 label=f"1:1 (R={pearson_r_full:.3f})")
    axes[1].set_xlabel("dSTEC Truth (TECU)", fontsize=12)
    axes[1].set_ylabel("dSTEC Model (TECU)", fontsize=12)
    axes[1].set_title(f"Model vs Truth - 2D Histogram\nR² = {r_squared_full:.4f}", fontsize=13)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3, color='white', linewidth=0.5)
    plt.colorbar(im, ax=axes[1], label='log10(count)')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "dstec_density.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved dSTEC density plots (R={pearson_r_full:.4f}, R²={r_squared_full:.4f})")
    
    # 4b: Error density by elevation
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Use percentile limits for error
    if use_percentiles:
        error_limits = np.percentile(masked_df["dstec_model_error"], [100-p_error, p_error])
    else:
        error_limits = [masked_df["dstec_model_error"].min(), masked_df["dstec_model_error"].max()]
    
    # Hexbin: Error vs Elevation
    hb = axes[0].hexbin(
        masked_df["satele"],
        masked_df["dstec_model_error"],
        gridsize=50,
        cmap='RdYlBu_r',
        mincnt=1,
        extent=[0, 90, error_limits[0], error_limits[1]],
        bins='log',
    )
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero error')
    axes[0].set_xlabel("Satellite Elevation (degrees)", fontsize=12)
    axes[0].set_ylabel("dSTEC Error (TECU)", fontsize=12)
    axes[0].set_title("Error vs Elevation - Hexbin Density (log scale)", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(hb, ax=axes[0], label='log10(count)')
    
    # 2D histogram: Error vs Elevation
    h, xedges, yedges = np.histogram2d(
        masked_df["satele"],
        masked_df["dstec_model_error"],
        bins=[50, 100],
        range=[[0, 90], error_limits]
    )
    
    im = axes[1].imshow(
        h.T,
        origin='lower',
        extent=[0, 90, error_limits[0], error_limits[1]],
        cmap='RdYlBu_r',
        aspect='auto',
        norm=plt.matplotlib.colors.LogNorm(vmin=max(1, h[h>0].min()), vmax=h.max()),
        interpolation='bilinear'
    )
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero error')
    axes[1].set_xlabel("Satellite Elevation (degrees)", fontsize=12)
    axes[1].set_ylabel("dSTEC Error (TECU)", fontsize=12)
    axes[1].set_title("Error vs Elevation - 2D Histogram (log scale)", fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, color='white', linewidth=0.5)
    plt.colorbar(im, ax=axes[1], label='log10(count)')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "error_elevation_density.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved error vs elevation density plots")
    
    # 4c: Residual analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Q-Q plot for error normality check
    from scipy import stats
    stats.probplot(masked_df["dstec_model_error"], dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title("Q-Q Plot: Error Normality Check", fontsize=13)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Error vs predicted value (heteroscedasticity check)
    if use_percentiles:
        pred_limits = np.percentile(masked_df["dstec_model"], [p_lower, p_upper])
        het_mask = (
            (masked_df["dstec_model"] >= pred_limits[0]) &
            (masked_df["dstec_model"] <= pred_limits[1]) &
            (masked_df["dstec_model_error"] >= error_limits[0]) &
            (masked_df["dstec_model_error"] <= error_limits[1])
        )
        het_df = masked_df[het_mask]
    else:
        het_df = masked_df
    
    hb = axes[0, 1].hexbin(
        het_df["dstec_model"],
        het_df["dstec_model_error"],
        gridsize=50,
        cmap='coolwarm',
        mincnt=1,
        bins='log',
    )
    axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel("Predicted dSTEC (TECU)", fontsize=12)
    axes[0, 1].set_ylabel("Error (TECU)", fontsize=12)
    axes[0, 1].set_title("Residuals vs Fitted (Heteroscedasticity Check)", fontsize=13)
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(hb, ax=axes[0, 1], label='log10(count)')
    
    # Error distribution by time of day
    masked_df_copy = masked_df.copy()
    masked_df_copy["hour"] = (masked_df_copy["sod"] / 3600.0).astype(int)
    
    h, xedges, yedges = np.histogram2d(
        masked_df_copy["hour"],
        masked_df_copy["dstec_model_error"],
        bins=[24, 100],
        range=[[0, 24], error_limits]
    )
    
    im = axes[1, 0].imshow(
        h.T,
        origin='lower',
        extent=[0, 24, error_limits[0], error_limits[1]],
        cmap='RdYlBu_r',
        aspect='auto',
        norm=plt.matplotlib.colors.LogNorm(vmin=max(1, h[h>0].min()), vmax=h.max()),
        interpolation='bilinear'
    )
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel("Hour of Day (UTC)", fontsize=12)
    axes[1, 0].set_ylabel("dSTEC Error (TECU)", fontsize=12)
    axes[1, 0].set_title("Error vs Time of Day", fontsize=13)
    axes[1, 0].grid(True, alpha=0.3, color='white', linewidth=0.5)
    plt.colorbar(im, ax=axes[1, 0], label='log10(count)')
    
    # Absolute error vs elevation (to see variance pattern)
    masked_df_copy["abs_error"] = np.abs(masked_df_copy["dstec_model_error"])
    
    # Binned statistics for smoother visualization
    from scipy.stats import binned_statistic
    elev_bins = np.linspace(0, 90, 50)
    bin_means, bin_edges, _ = binned_statistic(
        masked_df_copy["satele"],
        masked_df_copy["abs_error"],
        statistic='mean',
        bins=elev_bins
    )
    bin_medians, _, _ = binned_statistic(
        masked_df_copy["satele"],
        masked_df_copy["abs_error"],
        statistic='median',
        bins=elev_bins
    )
    bin_std, _, _ = binned_statistic(
        masked_df_copy["satele"],
        masked_df_copy["abs_error"],
        statistic='std',
        bins=elev_bins
    )
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    axes[1, 1].plot(bin_centers, bin_means, 'b-', linewidth=2, label='Mean |Error|')
    axes[1, 1].plot(bin_centers, bin_medians, 'g-', linewidth=2, label='Median |Error|')
    axes[1, 1].fill_between(
        bin_centers,
        bin_means - bin_std,
        bin_means + bin_std,
        alpha=0.3,
        color='blue',
        label='±1 Std'
    )
    axes[1, 1].set_xlabel("Satellite Elevation (degrees)", fontsize=12)
    axes[1, 1].set_ylabel("Absolute Error (TECU)", fontsize=12)
    axes[1, 1].set_title("Error Statistics vs Elevation", fontsize=13)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "residual_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved residual diagnostic plots")
    
    # 4d: GIM comparison plots (if GIM data available)
    has_gim = "dstec_gim" in masked_df.columns and masked_df["dstec_gim"].notna().any()
    
    if has_gim:
        logger.info("  Generating GIM comparison plots...")
        
        # Create GIM error column
        masked_df["dstec_gim_error"] = masked_df["dstec_gim"] - masked_df["dstec_truth"]
        
        # Filter out NaN values for GIM
        gim_mask = masked_df["dstec_gim"].notna() & masked_df["dstec_gim_error"].notna()
        gim_df = masked_df[gim_mask].copy()
        
        if len(gim_df) > 0:
            # Compute GIM correlation metrics
            pearson_r_gim = np.corrcoef(gim_df["dstec_truth"], gim_df["dstec_gim"])[0, 1]
            r_squared_gim = pearson_r_gim ** 2
            
            # GIM density plots (matching model plots)
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # Hexbin: GIM vs Truth
            if use_percentiles:
                gim_truth_limits = np.percentile(gim_df["dstec_truth"], [p_lower, p_upper])
                gim_pred_limits = np.percentile(gim_df["dstec_gim"], [p_lower, p_upper])
                gim_mask_perc = (
                    (gim_df["dstec_truth"] >= gim_truth_limits[0]) &
                    (gim_df["dstec_truth"] <= gim_truth_limits[1]) &
                    (gim_df["dstec_gim"] >= gim_pred_limits[0]) &
                    (gim_df["dstec_gim"] <= gim_pred_limits[1])
                )
                gim_plot_df = gim_df[gim_mask_perc]
                xlim = gim_truth_limits
                ylim = gim_pred_limits
            else:
                gim_plot_df = gim_df
                xlim = [gim_df["dstec_truth"].min(), gim_df["dstec_truth"].max()]
                ylim = [gim_df["dstec_gim"].min(), gim_df["dstec_gim"].max()]
            
            line_min = min(xlim[0], ylim[0])
            line_max = max(xlim[1], ylim[1])
            
            hb = axes[0].hexbin(
                gim_plot_df["dstec_truth"],
                gim_plot_df["dstec_gim"],
                gridsize=50,
                cmap='viridis',
                mincnt=1,
                extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                bins='log',
            )
            axes[0].plot([line_min, line_max], [line_min, line_max], 'r--', alpha=0.7, linewidth=2,
                        label=f"1:1 (R={pearson_r_gim:.3f})")
            axes[0].set_xlabel("dSTEC Truth (TECU)", fontsize=12)
            axes[0].set_ylabel("dSTEC GIM (TECU)", fontsize=12)
            axes[0].set_title(f"GIM vs Truth - Hexbin Density (log scale)\nR² = {r_squared_gim:.4f}", fontsize=13)
            axes[0].legend(loc='best')
            axes[0].grid(True, alpha=0.3)
            plt.colorbar(hb, ax=axes[0], label='log10(count)')
            
            # 2D histogram: GIM vs Truth
            h, xedges, yedges = np.histogram2d(
                gim_plot_df["dstec_truth"],
                gim_plot_df["dstec_gim"],
                bins=[100, 100],
                range=[xlim, ylim]
            )
            
            im = axes[1].imshow(
                h.T,
                origin='lower',
                extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                cmap='viridis',
                aspect='auto',
                norm=plt.matplotlib.colors.LogNorm(vmin=max(1, h[h>0].min()), vmax=h.max()),
                interpolation='bilinear'
            )
            axes[1].plot([line_min, line_max], [line_min, line_max], 'r--', alpha=0.7, linewidth=2,
                        label=f"1:1 (R={pearson_r_gim:.3f})")
            axes[1].set_xlabel("dSTEC Truth (TECU)", fontsize=12)
            axes[1].set_ylabel("dSTEC GIM (TECU)", fontsize=12)
            axes[1].set_title(f"GIM vs Truth - 2D Histogram\nR² = {r_squared_gim:.4f}", fontsize=13)
            axes[1].legend(loc='best')
            axes[1].grid(True, alpha=0.3, color='white', linewidth=0.5)
            plt.colorbar(im, ax=axes[1], label='log10(count)')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "dstec_gim_density.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  ✓ Saved GIM density plots (R={pearson_r_gim:.4f}, R²={r_squared_gim:.4f})")
            
            # GIM error vs elevation
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            if use_percentiles:
                gim_error_limits = np.percentile(gim_df["dstec_gim_error"], [100-p_error, p_error])
            else:
                gim_error_limits = [gim_df["dstec_gim_error"].min(), gim_df["dstec_gim_error"].max()]
            
            # Hexbin: GIM Error vs Elevation
            hb = axes[0].hexbin(
                gim_df["satele"],
                gim_df["dstec_gim_error"],
                gridsize=50,
                cmap='RdYlBu_r',
                mincnt=1,
                extent=[0, 90, gim_error_limits[0], gim_error_limits[1]],
                bins='log',
            )
            axes[0].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero error')
            axes[0].set_xlabel("Satellite Elevation (degrees)", fontsize=12)
            axes[0].set_ylabel("dSTEC GIM Error (TECU)", fontsize=12)
            axes[0].set_title("GIM Error vs Elevation - Hexbin Density", fontsize=13)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            plt.colorbar(hb, ax=axes[0], label='log10(count)')
            
            # 2D histogram: GIM Error vs Elevation
            h, xedges, yedges = np.histogram2d(
                gim_df["satele"],
                gim_df["dstec_gim_error"],
                bins=[50, 100],
                range=[[0, 90], gim_error_limits]
            )
            
            im = axes[1].imshow(
                h.T,
                origin='lower',
                extent=[0, 90, gim_error_limits[0], gim_error_limits[1]],
                cmap='RdYlBu_r',
                aspect='auto',
                norm=plt.matplotlib.colors.LogNorm(vmin=max(1, h[h>0].min()), vmax=h.max()),
                interpolation='bilinear'
            )
            axes[1].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero error')
            axes[1].set_xlabel("Satellite Elevation (degrees)", fontsize=12)
            axes[1].set_ylabel("dSTEC GIM Error (TECU)", fontsize=12)
            axes[1].set_title("GIM Error vs Elevation - 2D Histogram", fontsize=13)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, color='white', linewidth=0.5)
            plt.colorbar(im, ax=axes[1], label='log10(count)')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "gim_error_elevation_density.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  ✓ Saved GIM error vs elevation density plots")
            
            # Model vs GIM direct comparison
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            
            # Compare absolute errors
            model_gim_df = masked_df[gim_mask].copy()
            model_gim_df["abs_model_error"] = np.abs(model_gim_df["dstec_model_error"])
            model_gim_df["abs_gim_error"] = np.abs(model_gim_df["dstec_gim_error"])
            
            if use_percentiles:
                abs_error_limits = np.percentile(
                    np.concatenate([model_gim_df["abs_model_error"], model_gim_df["abs_gim_error"]]),
                    [0, 99.5]
                )
            else:
                abs_error_limits = [0, max(model_gim_df["abs_model_error"].max(), 
                                          model_gim_df["abs_gim_error"].max())]
            
            # Hexbin: |Model Error| vs |GIM Error|
            hb = axes[0].hexbin(
                model_gim_df["abs_gim_error"],
                model_gim_df["abs_model_error"],
                gridsize=50,
                cmap='plasma',
                mincnt=1,
                extent=[abs_error_limits[0], abs_error_limits[1], 
                       abs_error_limits[0], abs_error_limits[1]],
                bins='log',
            )
            axes[0].plot([abs_error_limits[0], abs_error_limits[1]], 
                        [abs_error_limits[0], abs_error_limits[1]], 
                        'r--', alpha=0.7, linewidth=2, label='Equal error')
            axes[0].set_xlabel("|GIM Error| (TECU)", fontsize=12)
            axes[0].set_ylabel("|Model Error| (TECU)", fontsize=12)
            axes[0].set_title("Model vs GIM: Absolute Error Comparison\n(Below line = Model better)", fontsize=13)
            axes[0].legend(loc='best')
            axes[0].grid(True, alpha=0.3)
            plt.colorbar(hb, ax=axes[0], label='log10(count)')
            
            # Error improvement distribution
            model_gim_df["error_improvement"] = model_gim_df["abs_gim_error"] - model_gim_df["abs_model_error"]
            
            axes[1].hist(model_gim_df["error_improvement"], bins=100, color='teal', alpha=0.7, edgecolor='black')
            axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='No improvement')
            axes[1].axvline(model_gim_df["error_improvement"].mean(), color='orange', 
                           linestyle='-', linewidth=2, label=f'Mean = {model_gim_df["error_improvement"].mean():.3f}')
            axes[1].set_xlabel("Error Improvement (|GIM Error| - |Model Error|) (TECU)", fontsize=12)
            axes[1].set_ylabel("Frequency", fontsize=12)
            axes[1].set_title("Model Improvement over GIM\n(Positive = Model better)", fontsize=13)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "model_vs_gim_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  ✓ Saved Model vs GIM comparison plots")
            
            # 4e: GIM analysis plots (matching model analysis plots)
            logger.info("  Generating GIM analysis plots...")
            
            # GIM Scatter Plot (matching Plot 1 for model)
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            if use_percentiles:
                gim_truth_limits = np.percentile(gim_df["dstec_truth"], [p_lower, p_upper])
                gim_pred_limits = np.percentile(gim_df["dstec_gim"], [p_lower, p_upper])
                gim_plot_mask = (
                    (gim_df["dstec_truth"] >= gim_truth_limits[0]) &
                    (gim_df["dstec_truth"] <= gim_truth_limits[1]) &
                    (gim_df["dstec_gim"] >= gim_pred_limits[0]) &
                    (gim_df["dstec_gim"] <= gim_pred_limits[1])
                )
                gim_scatter_df = gim_df[gim_plot_mask]
                xlim_gim = gim_truth_limits
                ylim_gim = gim_pred_limits
                n_gim_outliers = len(gim_df) - len(gim_scatter_df)
            else:
                gim_scatter_df = gim_df
                xlim_gim = [gim_df["dstec_truth"].min(), gim_df["dstec_truth"].max()]
                ylim_gim = [gim_df["dstec_gim"].min(), gim_df["dstec_gim"].max()]
                n_gim_outliers = 0
            
            # Compute GIM correlation
            pearson_r_gim_scatter, _ = pearsonr(gim_scatter_df["dstec_truth"], gim_scatter_df["dstec_gim"])
            r_squared_gim_scatter = pearson_r_gim_scatter ** 2
            
            ax.scatter(
                gim_scatter_df["dstec_truth"],
                gim_scatter_df["dstec_gim"],
                alpha=0.3,
                s=1,
                color='green',
                label=f"GIM (n={len(gim_scatter_df):,})",
            )
            
            line_min_gim = max(xlim_gim[0], ylim_gim[0])
            line_max_gim = min(xlim_gim[1], ylim_gim[1])
            ax.plot([line_min_gim, line_max_gim], [line_min_gim, line_max_gim], 'k--', 
                   alpha=0.5, linewidth=2, label="1:1")
            
            ax.set_xlabel("dSTEC Truth (TECU)", fontsize=12)
            ax.set_ylabel("dSTEC GIM (TECU)", fontsize=12)
            ax.set_title(f"Differential STEC: GIM vs Truth\nPearson R = {pearson_r_gim_scatter:.4f}, R² = {r_squared_gim_scatter:.4f}", fontsize=13)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(xlim_gim)
            ax.set_ylim(ylim_gim)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "dstec_gim_scatter.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  ✓ Saved GIM scatter plot (R={pearson_r_gim_scatter:.4f}, R²={r_squared_gim_scatter:.4f})")
            
            # GIM Error Analysis (matching Plot 2 for model)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            if use_percentiles:
                gim_error_limits = np.percentile(gim_df["dstec_gim_error"], [100-p_error, p_error])
                gim_error_mask = (
                    (gim_df["dstec_gim_error"] >= gim_error_limits[0]) &
                    (gim_df["dstec_gim_error"] <= gim_error_limits[1])
                )
                gim_error_plot_df = gim_df[gim_error_mask]
                n_gim_error_outliers = len(gim_df) - len(gim_error_plot_df)
            else:
                gim_error_plot_df = gim_df
                gim_error_limits = [gim_df["dstec_gim_error"].min(), gim_df["dstec_gim_error"].max()]
            
            # Error histogram
            axes[0].hist(gim_error_plot_df["dstec_gim_error"], bins=100, alpha=0.7, 
                        color='green', edgecolor='black')
            axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
            axes[0].set_xlabel("dSTEC GIM Error (TECU)")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title(f"GIM Error Distribution\n(±{p_error}th percentile range)")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Error vs elevation
            axes[1].scatter(
                gim_error_plot_df["satele"],
                gim_error_plot_df["dstec_gim_error"],
                alpha=0.3,
                s=1,
                color='green',
            )
            axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
            axes[1].set_xlabel("Elevation (degrees)")
            axes[1].set_ylabel("dSTEC GIM Error (TECU)")
            axes[1].set_title("GIM Error vs Elevation")
            axes[1].grid(True, alpha=0.3)
            
            if use_percentiles:
                axes[1].set_ylim(gim_error_limits[0], gim_error_limits[1])
            
            plt.tight_layout()
            plt.savefig(plots_dir / "gim_error_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  ✓ Saved GIM error analysis plots")
            
            # GIM Pass Statistics (matching Plot 3 for model)
            # Filter stats for passes with valid GIM data
            gim_stats_df = stats_df[~stats_df["gim_rmse"].isna()].copy()
            
            if len(gim_stats_df) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # GIM RMSE distribution
                axes[0, 0].hist(gim_stats_df["gim_rmse"], bins=50, alpha=0.7, 
                               color='green', edgecolor='black')
                axes[0, 0].set_xlabel("RMSE (TECU)")
                axes[0, 0].set_ylabel("Number of Passes")
                axes[0, 0].set_title("GIM RMSE Distribution Across Passes")
                axes[0, 0].grid(True, alpha=0.3)
                
                # GIM MAE distribution
                axes[0, 1].hist(gim_stats_df["gim_mae"], bins=50, alpha=0.7, 
                               color='green', edgecolor='black')
                axes[0, 1].set_xlabel("MAE (TECU)")
                axes[0, 1].set_ylabel("Number of Passes")
                axes[0, 1].set_title("GIM MAE Distribution Across Passes")
                axes[0, 1].grid(True, alpha=0.3)
                
                # GIM Relative error distribution
                gim_re_valid = gim_stats_df["gim_re"][gim_stats_df["gim_re"] < 2]
                axes[1, 0].hist(gim_re_valid, bins=50, alpha=0.7, 
                               color='green', edgecolor='black')
                axes[1, 0].set_xlabel("Relative Error")
                axes[1, 0].set_ylabel("Number of Passes")
                axes[1, 0].set_title("GIM Relative Error Distribution (<2.0)")
                axes[1, 0].grid(True, alpha=0.3)
                
                # Overlay comparison: Model vs GIM RMSE
                axes[1, 1].hist(stats_df["model_rmse"], bins=50, alpha=0.5, 
                               color='blue', edgecolor='black', label='Model')
                axes[1, 1].hist(gim_stats_df["gim_rmse"], bins=50, alpha=0.5, 
                               color='green', edgecolor='black', label='GIM')
                axes[1, 1].set_xlabel("RMSE (TECU)")
                axes[1, 1].set_ylabel("Number of Passes")
                axes[1, 1].set_title("RMSE Comparison: Model vs GIM")
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(plots_dir / "gim_pass_statistics.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"  ✓ Saved GIM pass statistics plots")
            
            # GIM Residual Diagnostics (matching Plot 4c for model)
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Q-Q plot for GIM error normality
            from scipy import stats
            stats.probplot(gim_df["dstec_gim_error"], dist="norm", plot=axes[0, 0])
            axes[0, 0].set_title("GIM Q-Q Plot: Error Normality Check", fontsize=13)
            axes[0, 0].grid(True, alpha=0.3)
            
            # GIM Error vs predicted value (heteroscedasticity)
            if use_percentiles:
                gim_pred_limits_het = np.percentile(gim_df["dstec_gim"], [p_lower, p_upper])
                gim_het_mask = (
                    (gim_df["dstec_gim"] >= gim_pred_limits_het[0]) &
                    (gim_df["dstec_gim"] <= gim_pred_limits_het[1]) &
                    (gim_df["dstec_gim_error"] >= gim_error_limits[0]) &
                    (gim_df["dstec_gim_error"] <= gim_error_limits[1])
                )
                gim_het_df = gim_df[gim_het_mask]
            else:
                gim_het_df = gim_df
            
            hb_gim = axes[0, 1].hexbin(
                gim_het_df["dstec_gim"],
                gim_het_df["dstec_gim_error"],
                gridsize=50,
                cmap='Greens',
                mincnt=1,
                bins='log',
            )
            axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            axes[0, 1].set_xlabel("Predicted dSTEC GIM (TECU)", fontsize=12)
            axes[0, 1].set_ylabel("GIM Error (TECU)", fontsize=12)
            axes[0, 1].set_title("GIM Residuals vs Fitted (Heteroscedasticity Check)", fontsize=13)
            axes[0, 1].grid(True, alpha=0.3)
            plt.colorbar(hb_gim, ax=axes[0, 1], label='log10(count)')
            
            # GIM Error by time of day
            gim_df_copy = gim_df.copy()
            gim_df_copy["hour"] = (gim_df_copy["sod"] / 3600.0).astype(int)
            
            h_gim, xedges_gim, yedges_gim = np.histogram2d(
                gim_df_copy["hour"],
                gim_df_copy["dstec_gim_error"],
                bins=[24, 100],
                range=[[0, 24], gim_error_limits]
            )
            
            im_gim = axes[1, 0].imshow(
                h_gim.T,
                origin='lower',
                extent=[0, 24, gim_error_limits[0], gim_error_limits[1]],
                cmap='RdYlGn_r',
                aspect='auto',
                norm=plt.matplotlib.colors.LogNorm(vmin=max(1, h_gim[h_gim>0].min()), vmax=h_gim.max()),
                interpolation='bilinear'
            )
            axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
            axes[1, 0].set_xlabel("Hour of Day (UTC)", fontsize=12)
            axes[1, 0].set_ylabel("dSTEC GIM Error (TECU)", fontsize=12)
            axes[1, 0].set_title("GIM Error vs Time of Day", fontsize=13)
            axes[1, 0].grid(True, alpha=0.3, color='white', linewidth=0.5)
            plt.colorbar(im_gim, ax=axes[1, 0], label='log10(count)')
            
            # GIM Absolute error vs elevation
            gim_df_copy["abs_error"] = np.abs(gim_df_copy["dstec_gim_error"])
            
            from scipy.stats import binned_statistic
            elev_bins = np.linspace(0, 90, 50)
            gim_bin_means, gim_bin_edges, _ = binned_statistic(
                gim_df_copy["satele"],
                gim_df_copy["abs_error"],
                statistic='mean',
                bins=elev_bins
            )
            gim_bin_medians, _, _ = binned_statistic(
                gim_df_copy["satele"],
                gim_df_copy["abs_error"],
                statistic='median',
                bins=elev_bins
            )
            gim_bin_std, _, _ = binned_statistic(
                gim_df_copy["satele"],
                gim_df_copy["abs_error"],
                statistic='std',
                bins=elev_bins
            )
            
            gim_bin_centers = (gim_bin_edges[:-1] + gim_bin_edges[1:]) / 2
            
            axes[1, 1].plot(gim_bin_centers, gim_bin_means, 'g-', linewidth=2, label='Mean |Error|')
            axes[1, 1].plot(gim_bin_centers, gim_bin_medians, 'lime', linewidth=2, label='Median |Error|')
            axes[1, 1].fill_between(
                gim_bin_centers,
                gim_bin_means - gim_bin_std,
                gim_bin_means + gim_bin_std,
                alpha=0.3,
                color='green',
                label='±1 Std'
            )
            axes[1, 1].set_xlabel("Satellite Elevation (degrees)", fontsize=12)
            axes[1, 1].set_ylabel("Absolute GIM Error (TECU)", fontsize=12)
            axes[1, 1].set_title("GIM Error Statistics vs Elevation", fontsize=13)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "gim_residual_diagnostics.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  ✓ Saved GIM residual diagnostic plots")
    
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
        
        # Check pass size distribution for debugging
        pass_sizes = test_df.groupby("pass_id").size()
        logger.info(f"  Pass size statistics:")
        logger.info(f"    Mean: {pass_sizes.mean():.2f} samples/pass")
        logger.info(f"    Median: {pass_sizes.median():.0f} samples/pass")
        logger.info(f"    Min: {pass_sizes.min()} | Max: {pass_sizes.max()}")
        logger.info(f"    Passes with ≥10 samples: {(pass_sizes >= 10).sum()}")
        
        # Prepare GIM mapper if requested
        gim_mapper = None
        if EVALUATION_CONFIG.get("gim", {}).get("enabled", False):
            try:
                gim_ac = EVALUATION_CONFIG.get("gim", {}).get("ac", "IGS")
                map_type = EVALUATION_CONFIG.get("mapping_function", {}).get("type", "MSLM")
                gim_mapper = GIMMapper(mapping_type=map_type, gim_type=gim_ac)

                # Attempt to pre-load GIM data using the first observation date (if available)
                if len(test_df) > 0 and "year" in test_df.columns and "doy" in test_df.columns:
                    first_row = test_df.iloc[0]
                    first_date = datetime(int(first_row["year"]), 1, 1) + timedelta(days=int(first_row["doy"]) - 1)
                    try:
                        gim_mapper.load_gim_data(EVALUATION_CONFIG.get("gim", {}).get("gim_path", "."), first_date)
                        logger.info("  ✓ Loaded initial GIM data for comparison")
                    except Exception as e:
                        logger.warning(f"Could not pre-load GIM data: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize GIM mapper: {e}")

        # Compute dSTEC for each pass
        dstec_results = []
        for pass_id, group in tqdm(
            test_df.groupby("pass_id"),
            desc="  Processing passes",
            total=n_passes,
        ):
            group.name = pass_id  # Set name for compute_dstec_for_pass
            result = compute_dstec_for_pass(group, EVALUATION_CONFIG, logger, gim_mapper=gim_mapper)
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
            generate_plots(dstec_df, stats_df, output_dir, logger, EVALUATION_CONFIG)
        
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
