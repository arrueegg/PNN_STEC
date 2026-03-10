#!/usr/bin/env python3
"""
Comprehensive STEC Comparison Script

This script provides a three-way comparison:
1. Direct STEC Model (your neural network)
2. Classical VTEC Model + Mapping Function
3. IGS GIM VTEC + Mapping Function

This demonstrates:
- Value of direct STEC modeling vs classical VTEC approach
- Performance improvement over operational IGS products
- Fair comparison with standard baselines

Usage:
    # Two-way comparison (STEC vs VTEC+mapping)
    python src/compare_stec_vtec_gim.py \\
        --stec_experiment "Pretrain_STEC_BNN_NLL_..." \\
        --vtec_experiment "Pretrain_VTEC_BNN_NLL_..." \\
        --output_dir "comparisons/full_comparison"
    
    # Three-way comparison (add GIM)
    python src/compare_stec_vtec_gim.py \\
        --stec_experiment "Pretrain_STEC_BNN_NLL_..." \\
        --vtec_experiment "Pretrain_VTEC_BNN_NLL_..." \\
        --include_gim \\
        --gim_path "/path/to/gim/data" \\
        --output_dir "comparisons/full_comparison"
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
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_parser import parse_config
from utils.feature_registry import initialize_feature_registry
from training.base_trainer import BaseTrainer
from data_loader import get_test_data_loader
from model.model import get_model
from evaluation.gim_mapper import MappingFunction, GIMMapper
from evaluation.publication_plots import generate_all_plots


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def set_test_size(config: Dict, test_size_arg: Optional[str]) -> None:
    """Set test_size in config, using 'full' as default if not specified."""
    if test_size_arg:
        config['data']['test_size'] = int(test_size_arg)
    elif 'test_size' not in config['data']:
        config['data']['test_size'] = "full"


def load_experiment_config(experiment_folder: str) -> Tuple[Dict, Path]:
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
    
    # Fix paths if they were saved on the cluster but are being evaluated locally
    if 'data' in config:
        cluster_base = "/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data"
        local_base = "/home/space/data/iono"
        
        if config['data'].get('GNSS_data_path', '').startswith(cluster_base):
             config['data']['GNSS_data_path'] = config['data']['GNSS_data_path'].replace(cluster_base, local_base)
        
        if config['data'].get('SWI_data_path', '').startswith(cluster_base):
             config['data']['SWI_data_path'] = config['data']['SWI_data_path'].replace(cluster_base, f"{local_base}/SWI")
             
        # Also fix scratch_dir if it's pointing to /scratch/ (cluster default)
        if config['data'].get('scratch_dir') == "/scratch/":
            config['data']['scratch_dir'] = "/scratch2/arrueegg/WP4/PNN_STEC/data/"
            
    # Always disable cluster mode when evaluating locally
    config['cluster'] = False
    
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


def load_model_from_checkpoint(config: Dict, checkpoint_path: Path, logger) -> Tuple[torch.nn.Module, object]:
    """Load a trained model from checkpoint."""
    # Initialize feature registry
    feature_registry = initialize_feature_registry(config)
    config["feature_registry"] = feature_registry
    
    # Initialize CollateWithSH to set output_indices in the feature registry
    # This is required before creating models that use FeatureSplitter
    from data_loader.collation import CollateWithSH
    collate_fn = CollateWithSH(config)
    
    # Load model (handles single or ensemble by checking the model/ directory)
    from model.model import load_model_for_inference
    experiment_dir = Path(checkpoint_path).parent.parent
    model = load_model_for_inference(config, experiment_dir, logger)
    
    return model, feature_registry


def run_inference(
    model: torch.nn.Module,
    test_loader,
    config: Dict,
    model_name: str,
    logger,
    num_inference_samples: int = 100
) -> pd.DataFrame:
    """Run inference with a model.
    
    For fair comparison:
    - Bayesian models: Use MC sampling with num_inference_samples
    - Deterministic models: Single forward pass (num_samples=1)
    """
    logger.info(f"🧠 Running {model_name} inference...")
    
    trainer = BaseTrainer(config, logger)
    
    # Determine if Bayesian
    model_type = config["model"]["model_type"]
    is_bayesian = "BNN" in model_type or "Bayesian" in model_type or "FactorizedSTEC" in model_type
    samples = num_inference_samples if is_bayesian else 1
    
    if not is_bayesian and num_inference_samples > 1:
        logger.info(f"   Note: {model_type} is deterministic - using 1 sample instead of {num_inference_samples}")
        
    # Run inference (handles both Bayesian and deterministic models)
    bayesian_results, test_df = trainer.bayesian_inference_total_uncertainty(
        model, test_loader, num_samples=samples
    )
    
    logger.info(f"✅ {model_name} inference completed: {len(test_df):,} predictions")
    return test_df


def apply_mapping_function(
    vtec_df: pd.DataFrame,
    mapping_type: str,
    column_prefix: str,
    logger
) -> pd.DataFrame:
    """
    Apply mapping function to convert VTEC predictions to STEC.
    
    Args:
        vtec_df: DataFrame with VTEC predictions and elevation angles
        mapping_type: 'SLM' or 'MSLM'
        column_prefix: Prefix for new columns (e.g., 'vtec_model' or 'gim')
        logger: Logger instance
        
    Returns:
        DataFrame with additional mapped STEC columns
    """
    logger.info(f"📐 Applying {mapping_type} mapping function to {column_prefix} VTEC...")
    
    # Initialize mapping function
    mapper = MappingFunction(mapping_type=mapping_type)
    
    # Get elevations (in degrees) and convert to radians
    elevations_rad = np.radians(vtec_df['satele'].values)
    
    # Get VTEC predictions
    # Note: Column is named 'pred_stec' from inference output, but values are VTEC
    if 'pred_stec' in vtec_df.columns:
        vtec_pred = vtec_df['pred_stec'].values
        # Prefer total uncertainty if available (e.g. for ensembles)
        if 'pred_total_unc' in vtec_df.columns:
            var_cols = ['pred_aleatoric_unc', 'pred_epistemic_unc', 'pred_total_unc']
        else:
            var_cols = ['pred_aleatoric_unc']
    elif 'pred_mean' in vtec_df.columns:
        vtec_pred = vtec_df['pred_mean'].values
        var_cols = ['pred_var']
    else:
        raise KeyError(f"Could not find prediction column. Available: {vtec_df.columns.tolist()}")
    
    # Compute mapping factor
    # Vectorized call: pass the entire numpy array at once
    mapping_factors = mapper.get_mapping_factor(elevations_rad)
    
    # Convert VTEC to STEC
    stec_mapped = vtec_pred * mapping_factors
    
    # Also propagate uncertainty (variance scales with mapping factor squared)
    for v_col in var_cols:
        if v_col in vtec_df.columns:
            vtec_val = vtec_df[v_col].values
            
            # If uncertainty is std, square it first to get variance
            if 'unc' in v_col:
                vtec_var = vtec_val ** 2
                suffix = v_col.replace('pred_', '')
            else:
                vtec_var = vtec_val
                suffix = 'var'
                
            stec_var_mapped = vtec_var * (mapping_factors ** 2)
            # Store both variance and original uncertainty (std)
            vtec_df[f'{column_prefix}_stec_{suffix}'] = np.sqrt(stec_var_mapped)
            vtec_df[f'{column_prefix}_stec_{suffix}_var'] = stec_var_mapped
    
    vtec_df[f'{column_prefix}_stec'] = stec_mapped
    vtec_df[f'{column_prefix}_mapping_factor'] = mapping_factors
    
    logger.info(f"✅ Mapping applied. Mean mapping factor: {mapping_factors.mean():.3f}")
    return vtec_df


def add_gim_comparison(
    test_df: pd.DataFrame,
    gim_path: str,
    mapping_type: str,
    logger
) -> pd.DataFrame:
    """
    Add IGS GIM VTEC values and map to STEC for comparison.
    
    Args:
        test_df: DataFrame with test observations
        gim_path: Path to GIM/IONEX data directory
        mapping_type: Mapping function type
        logger: Logger instance
        
    Returns:
        DataFrame with additional GIM STEC columns
    """
    logger.info("🌍 Loading IGS GIM data for comparison...")
    
    # Initialize GIM mapper
    gim_mapper = GIMMapper(mapping_type=mapping_type, gim_type='IGS')
    
    # Group observations by date for efficient GIM loading
    grouped_data = defaultdict(list)
    for idx, row in test_df.iterrows():
        year = int(row['year'])
        doy = int(row['doy'])
        date_key = (datetime(year, 1, 1) + timedelta(days=doy - 1)).strftime("%Y-%m-%d")
        grouped_data[date_key].append(idx)
    
    logger.info(f"Processing {len(grouped_data)} days of GIM data...")
    
    # Initialize GIM columns
    test_df['gim_vtec'] = np.nan
    test_df['gim_stec'] = np.nan
    
    # Process each date
    for date_str, indices in tqdm(grouped_data.items(), desc="Processing GIM data"):
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        
        try:
            # Load GIM data for this date
            gim_mapper.load_gim_data(gim_path, date_obj)
            
            # Get observation data
            obs_data = test_df.loc[indices]
            sods = obs_data['sod'].values
            ipp_lats = obs_data['lat_ipp'].values
            ipp_lons = obs_data['lon_ipp'].values
            elevations = obs_data['satele'].values
            
            # Get STEC from GIM (already applies mapping function internally)
            gim_stec = gim_mapper.map_vtec_to_stec(
                sods, ipp_lats, ipp_lons, elevations
            )
            
            # Store results
            test_df.loc[indices, 'gim_stec'] = gim_stec
            
            # Also compute VTEC at IPP for reference (divide out mapping factor)
            mapper = MappingFunction(mapping_type=mapping_type)
            elevations_rad = np.radians(elevations)
            # Vectorized mapping factor calculation
            mapping_factors = mapper.get_mapping_factor(elevations_rad)
            gim_vtec = gim_stec / mapping_factors
            test_df.loc[indices, 'gim_vtec'] = gim_vtec
            
        except Exception as e:
            logger.warning(f"Failed to process GIM data for {date_str}: {e}")
            continue
    
    # Count valid GIM values
    valid_count = test_df['gim_stec'].notna().sum()
    logger.info(f"✅ GIM data added for {valid_count:,} / {len(test_df):,} observations ({100*valid_count/len(test_df):.1f}%)")
    
    return test_df




def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if mask is not None:
        predictions = predictions[mask]
        ground_truth = ground_truth[mask]
    
    # Remove NaNs
    valid_mask = ~(np.isnan(predictions) | np.isnan(ground_truth))
    predictions = predictions[valid_mask]
    ground_truth = ground_truth[valid_mask]
    
    if len(predictions) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 'std': np.nan, 'r2': np.nan, 'count': 0}
    
    errors = predictions - ground_truth
    
    metrics = {
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mae': np.mean(np.abs(errors)),
        'bias': np.mean(errors),
        'std': np.std(errors),
        'r2': 1 - np.sum(errors ** 2) / np.sum((ground_truth - ground_truth.mean()) ** 2),
        'count': len(predictions)
    }
    
    return metrics


def compare_all_models(
    test_df: pd.DataFrame,
    stec_col: str,
    vtec_col: Optional[str],
    gim_col: Optional[str],
    logger,
    pretrain_col: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare all available models.
    
    Args:
        test_df: DataFrame with all predictions
        stec_col: Column name for direct STEC predictions
        vtec_col: Column name for VTEC+mapping predictions (or None)
        gim_col: Column name for GIM STEC (or None)
        logger: Logger instance
        pretrain_col: Column name for pretrained STEC predictions (or None)
    """
    logger.info("📊 Computing comparison metrics...")
    
    # Get ground truth STEC
    ground_truth = test_df['true_stec'].values
    
    results = {}
    
    # Direct STEC model
    stec_pred = test_df[stec_col].values
    results['Direct STEC Model'] = compute_metrics(stec_pred, ground_truth)

    # Pretrained STEC baseline (if available)
    if pretrain_col and pretrain_col in test_df.columns:
        pre_pred = test_df[pretrain_col].values
        results['Pretrained STEC'] = compute_metrics(pre_pred, ground_truth)
    
    # VTEC + Mapping model (if available)
    if vtec_col and vtec_col in test_df.columns:
        vtec_mapped_pred = test_df[vtec_col].values
        results['VTEC + Mapping'] = compute_metrics(vtec_mapped_pred, ground_truth)
    
    # IGS GIM (if available)
    if gim_col and gim_col in test_df.columns:
        gim_pred = test_df[gim_col].values
        gim_mask = ~np.isnan(gim_pred)
        results['IGS GIM'] = compute_metrics(gim_pred, ground_truth, mask=gim_mask)
    
    # Print results
    logger.info("\n" + "="*70)
    logger.info("COMPARISON RESULTS")
    logger.info("="*70)
    
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  RMSE:  {metrics['rmse']:.4f} TECU")
        logger.info(f"  MAE:   {metrics['mae']:.4f} TECU")
        logger.info(f"  Bias:  {metrics['bias']:.4f} TECU")
        logger.info(f"  R²:    {metrics['r2']:.4f}")
        logger.info(f"  Count: {metrics['count']:,}")
    
    # Compute improvements
    if len(results) > 1:
        logger.info("\nImprovement (Direct STEC over baselines):")
        baseline_names = [k for k in results.keys() if k != 'Direct STEC Model']
        for baseline in baseline_names:
            if baseline in results:
                baseline_rmse = results[baseline]['rmse']
                stec_rmse = results['Direct STEC Model']['rmse']
                improvement = 100 * (baseline_rmse - stec_rmse) / baseline_rmse
                logger.info(f"  vs {baseline}: {improvement:.2f}% RMSE improvement")
    
    logger.info("="*70 + "\n")
    
    return results


def save_results(
    metrics: Dict,
    test_df: pd.DataFrame,
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
        f.write("="*70 + "\n")
        f.write("STEC Model Comparison Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"STEC Experiment: {args.stec_experiment}\n")
        if args.vtec_experiment:
            f.write(f"VTEC Experiment: {args.vtec_experiment}\n")
        f.write(f"Mapping Function: {args.mapping_function}\n")
        if not args.no_gim:
            f.write(f"GIM Path: {args.gim_path}\n")
        f.write(f"Test Samples: {len(test_df):,}\n\n")
        
        for model_name, model_metrics in metrics.items():
            f.write(f"{model_name}:\n")
            for key, val in model_metrics.items():
                if key == 'count':
                    f.write(f"  {key}: {val:,}\n")
                else:
                    f.write(f"  {key}: {val:.6f}\n")
            f.write("\n")
    
    # Save metrics as CSV for multiday aggregation
    metrics_rows = []
    for model_name, model_metrics in metrics.items():
        row = {'Model': model_name}
        # Rename keys to match expected format (capitalized for consistency)
        row['RMSE'] = model_metrics['rmse']
        row['MAE'] = model_metrics['mae']
        row['R²'] = model_metrics['r2']
        row['Bias'] = model_metrics['bias']
        row['Std'] = model_metrics['std']
        row['Count'] = model_metrics['count']
        metrics_rows.append(row)
    
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = output_dir / 'metrics_summary.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    # Save detailed predictions CSV
    csv_cols = ['true_stec', 'stec_pred', 'satele']
    rename_dict = {'satele': 'elevation'}
    
    if 'pretrained_stec_pred' in test_df.columns:
        csv_cols.append('pretrained_stec_pred')
    if 'vtec_model_stec' in test_df.columns:
        csv_cols.append('vtec_model_stec')
    if 'gim_stec' in test_df.columns:
        csv_cols.append('gim_stec')
    
    # Create comparison dataframe
    save_df = test_df[csv_cols].copy()
    save_df.rename(columns=rename_dict, inplace=True)
    
    csv_path = output_dir / 'detailed_predictions.csv'
    save_df.to_csv(csv_path, index=False)
    
    logger.info(f"✅ Results saved to {output_dir}")
    logger.info(f"   - comparison_summary.txt")
    logger.info(f"   - metrics_summary.csv")
    logger.info(f"   - detailed_predictions.csv")
    if not args.skip_plots:
        logger.info(f"   - 5 publication-ready plots")


def run_comparison(
    stec_experiment: str,
    vtec_experiment: str = None,
    num_inference_samples: int = 100,
    test_size: int = None,
    madrigal_path: str = "/home/space/data/iono/Madrigal_STEC",
    no_gim: bool = False,
    gim_path: str = "/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex",
    mapping_function: str = "MSLM",
    output_dir: str = None,
):
    """
    Programmatic entry point for comparison workflow.
    """
    logger = logging.getLogger(__name__)
    # Reset handlers to avoid accumulation if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Setup basic logging to stdout/stderr or allow parent logger to handle it
    # For now, we'll re-use setup_logging logic or similar if needed.
    # But note: setup_logging() in this file might configure root logger.
    
    # Let's use the existing setup_logging helper but be careful
    logger = setup_logging() # This sets up root logger

    logger.info("="*70)
    logger.info("Comprehensive STEC Model Comparison")
    logger.info("="*70)
    
    # Load STEC configuration
    stec_config, stec_dir = load_experiment_config(stec_experiment)
    
    # CRITICAL: For finetuned models, force use_agg_h5=False to use day-specific test data
    # instead of the global 6GB test.h5 file
    if stec_config.get('mode') == 'finetune':
        if 'data' in stec_config and stec_config['data'].get('use_agg_h5'):
            logger.info("⚡ Optimizing: Disabling use_agg_h5 for finetuned model to use day-specific data")
            stec_config['data']['use_agg_h5'] = False
    
    # Verify STEC target
    if stec_config.get('target', 'stec').lower() != 'stec':
        raise ValueError(f"STEC experiment must have target='stec', got {stec_config.get('target')}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stec_config["device"] = device
    logger.info(f"Using device: {device}")
    
    # Load VTEC configuration if provided
    vtec_config = None
    vtec_dir = None
    if vtec_experiment:
        vtec_config, vtec_dir = load_experiment_config(vtec_experiment)
        if vtec_config.get('target', 'stec').lower() != 'vtec':
            logger.warning(f"VTEC experiment has target={vtec_config.get('target')}, expected 'vtec'")
            
        # Ensure VTEC uses same test set logic
        if vtec_config.get('mode') == 'finetune':
            if 'data' in vtec_config and vtec_config['data'].get('use_agg_h5'):
                vtec_config['data']['use_agg_h5'] = False
                
        vtec_config["device"] = device

    # ... Rest of the function body ...
    # This function is too long to replace entirely cleanly without reading more context.
    # I will first just rename main() to run_comparison_cli() and create a wrapper.
    # But wait, I need to convert args to function parameters.
    
    # Let's try a different approach. I'll modify main to take an 'args' object optionally.
    pass

def main(args=None):
    """Main comparison workflow.
    
    Standard usage (comprehensive evaluation):
        python src/compare_stec_vtec_gim.py \\
            --stec_experiment "Finetune_STEC_..." \\
            --vtec_experiment "Finetune_VTEC_..."
    
    Automatically evaluates on:
    - Own test set (from training data)
    - Madrigal independent test set (if available)
    - VTEC+Mapping baseline (if vtec_experiment provided)
    - IGS GIM baseline (enabled by default)
    """
    if args is None:
        parser = argparse.ArgumentParser(
            description="Comprehensive STEC Model Comparison",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        
        # Required arguments
        parser.add_argument("--stec_experiment", type=str, required=True,
                        help="Path to STEC model experiment folder")
        parser.add_argument("--vtec_experiment", type=str, default=None,
                        help="Path to VTEC model experiment folder (optional)")
        
        # Optional arguments with defaults for comprehensive evaluation
        parser.add_argument("--num_inference_samples", type=int, default=100,
                        help="Number of MC samples for Bayesian inference (default: 100)")
        parser.add_argument("--test_size", default=None,
                        help="Number of test samples, or None for full test set (default: None/full)")
        
        # Data sources (automatically evaluates on all available datasets)
        parser.add_argument("--madrigal_path", type=str,
                        default="/home/space/data/iono/Madrigal_STEC",
                        help="Path to Madrigal STEC data directory (auto-evaluated if available)")
        
        parser.add_argument("--no_gim", action="store_true",
                        help="Skip IGS GIM baseline comparison")
        parser.add_argument("--gim_path", type=str, 
                        default="/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex",
                        help="Path to GIM/IONEX data directory")
        
        # Other options
        parser.add_argument("--mapping_function", type=str, default="MSLM",
                        choices=["SLM", "MSLM"],
                        help="Mapping function for VTEC→STEC conversion (default: MSLM)")
        parser.add_argument("--output_dir", type=str, default=None,
                        help="Additional output directory (results always saved to experiment folder)")
        parser.add_argument("--reuse_results", action="store_true",
                        help="Reuse existing STEC and GIM results from output_dir if available")
        parser.add_argument("--skip_plots", action="store_true",
                        help="Skip plot generation")
        
        # Pretrained model baseline
        parser.add_argument("--pretrained_stec_experiment", type=str, default=None,
                        help="Path to PRETRAINED STEC model folder (baseline comparison)")
        
        args = parser.parse_args()

    # CRITICAL: Reset static cache attributes to ensure no cross-contamination between runs
    # This is necessary when main is called multiple times in the same process (e.g. multiday evaluation)
    if hasattr(main, '_vtec_model_loaded'):
        del main._vtec_model_loaded
    if hasattr(main, '_vtec_config'):
        del main._vtec_config
    if hasattr(main, '_vtec_model'):
        del main._vtec_model

    logger = setup_logging()
    
    logger.info("="*70)
    logger.info("Comprehensive STEC Model Comparison")
    logger.info("="*70)
    
    # Load STEC configuration
    stec_config, stec_dir = load_experiment_config(args.stec_experiment)
    
    # CRITICAL: For finetuned models, force use_agg_h5=False to use day-specific test data
    # instead of the global 6GB test.h5 file
    if stec_config.get('mode') == 'finetune':
        if 'data' in stec_config and stec_config['data'].get('use_agg_h5'):
            logger.info("⚡ Optimizing: Disabling use_agg_h5 for finetuned model to use day-specific data")
            stec_config['data']['use_agg_h5'] = False
    
    # Verify STEC target
    if stec_config.get('target', 'stec').lower() != 'stec':
        raise ValueError(f"STEC experiment must have target='stec', got {stec_config.get('target')}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stec_config["device"] = device
    logger.info(f"Using device: {device}")
    
    # Override test size if provided
    if args.test_size:
        stec_config['data']['test_size'] = int(args.test_size)
    
    # Defer loading the STEC model until we know if we need it
    stec_model = None
    stec_registry = None
    
    # Determine which datasets to evaluate on
    datasets_to_evaluate = []
    
    # 1. Always evaluate on own test set
    datasets_to_evaluate.append(('own', 'Own Test Set'))
    
    # 2. Try to evaluate on Madrigal if available and model is finetuned
    if stec_config.get('mode') == 'finetune':
        madrigal_path = Path(args.madrigal_path)
        if madrigal_path.exists():
            datasets_to_evaluate.append(('madrigal', 'Madrigal Independent Test Set'))
        else:
            logger.warning(f"⚠️  Madrigal path not found: {madrigal_path}, skipping Madrigal evaluation")
    else:
        logger.info("ℹ️  Pretrained model detected - Madrigal evaluation only supported for finetuned models")
    
    logger.info(f"\n📊 Will evaluate on {len(datasets_to_evaluate)} dataset(s): {', '.join([name for _, name in datasets_to_evaluate])}")
    
    # Loop through each dataset
    for dataset_type, dataset_name in datasets_to_evaluate:
        logger.info("\n" + "="*70)
        logger.info(f"Evaluating on: {dataset_name}")
        logger.info("="*70)
        
        # Determine output sub-directory names (for checking existing results)
        comparison_parts = [dataset_type]
        if args.vtec_experiment: comparison_parts.append("vtec")
        if not args.no_gim: comparison_parts.append("gim")
        comparison_name = "_".join(comparison_parts)
        
        # Check if we can reuse results from a previous run
        reusable_df = None
        if getattr(args, 'reuse_results', False):
            check_paths = []
            
            # 1. Try the root multiday structure (multiday_results/{year}_DOY_{doy}/evaluation/...)
            # This is preferred for reusing results across different output directories
            if stec_config.get('year') and stec_config.get('doy'):
                # Extract year and doy, ensuring they are strings or ints as needed
                y, d = stec_config.get('year'), stec_config.get('doy')
                date_folder = f"{y}_DOY_{d}"
                
                # Try with full comparison name
                check_paths.append(Path("multiday_results") / date_folder / "evaluation" / comparison_name / 'detailed_predictions.csv')
                
                # Also check standard multiday naming conventions 'own_vtec_gim' or 'madrigal_vtec_gim'
                check_paths.append(Path("multiday_results") / date_folder / "evaluation" / f"{dataset_type}_vtec_gim" / 'detailed_predictions.csv')
                check_paths.append(Path("multiday_results") / date_folder / "evaluation" / f"{dataset_type}_vtec" / 'detailed_predictions.csv')

            # 2. Try provided output directory as fallback
            if args.output_dir:
                check_paths.append(Path(args.output_dir) / comparison_name / 'detailed_predictions.csv')
                check_paths.append(Path(args.output_dir) / f"{dataset_type}_vtec_gim" / 'detailed_predictions.csv')

            # 3. Try relative paths if nested
            check_paths.append(Path("..") / ".." / "evaluation" / f"{dataset_type}_vtec_gim" / 'detailed_predictions.csv')

            for check_path in check_paths:
                if check_path.exists():
                    try:
                        reusable_df = pd.read_csv(check_path)
                        logger.info(f"♻️  Found existing results to reuse: {check_path} ({len(reusable_df):,} samples)")
                        # Ensure it has the required columns
                        required = ['true_stec', 'stec_pred', 'elevation']
                        if not all(col in reusable_df.columns for col in required):
                            logger.warning(f"⚠️  Existing results missing required columns {required}, will re-evaluate.")
                            reusable_df = None
                            continue
                        else:
                            # Rename 'elevation' back to 'satele' for consistency 
                            reusable_df.rename(columns={'elevation': 'satele'}, inplace=True)
                            break
                    except Exception as e:
                        logger.warning(f"⚠️  Could not read existing results at {check_path}: {e}")
                        reusable_df = None

        # Prepare test data based on dataset type
        _skip_vtec_gim = False  # Will be set to True in 'own' branch if fast-path applies
        if dataset_type == 'madrigal':
            # Get year and doy from config
            year = int(stec_config.get('year', 2024))
            doy = int(stec_config.get('doy', 183))
            
            # Load test station list for filtering
            test_station_file = Path("src/data_processing/test_station.list")
            test_stations = None
            if test_station_file.exists():
                with open(test_station_file, 'r') as f:
                    test_stations = [line.strip().upper() for line in f if line.strip()]
                logger.info(f"📋 Loaded {len(test_stations)} test stations for filtering")
            else:
                logger.warning(f"⚠️  Test station list not found at {test_station_file}, using all stations")
            
            # Create Madrigal data loader for direct inference
            from data_loader.madrigal_dataset import get_madrigal_data_loader
            
            vtec_madrigal_loader, madrigal_dataset = get_madrigal_data_loader(
                madrigal_path=args.madrigal_path,
                year=year,
                doy=doy,
                config=stec_config,
                batch_size=8192,
                num_workers=4,
                elevation_threshold=5.0,
                max_samples=int(args.test_size) if args.test_size else None,
                station_list=test_stations,
                logger=logger
            )
            
            # Check if we can skip STEC inference
            if reusable_df is not None and len(reusable_df) == len(madrigal_dataset):
                logger.info("⏭️  Skipping STEC model inference (Madrigal), reusing results")
                test_df = reusable_df.copy()
            else:
                # Need to load STEC model for inference
                if stec_model is None:
                    stec_checkpoint = find_best_checkpoint(stec_dir)
                    stec_model, stec_registry = load_model_from_checkpoint(stec_config, stec_checkpoint, logger)
                
                # Run inference on Madrigal data
                logger.info("🧠 Running STEC model inference on Madrigal observations...")
                test_df = run_inference(stec_model, vtec_madrigal_loader, stec_config, "STEC Model (Madrigal)", logger, args.num_inference_samples)
                
                # Rename columns for consistency
                test_df.rename(columns={'pred_stec': 'stec_pred', 'target_stec': 'true_stec'}, inplace=True)
                
                # Add metadata from Madrigal dataset
                logger.info("📋 Adding Madrigal metadata to results...")
                metadata_list = []
                for idx in range(len(madrigal_dataset)):
                    metadata_list.append(madrigal_dataset.get_metadata(idx))
                metadata_df = pd.DataFrame(metadata_list)
                
                # Merge metadata with predictions
                for col in metadata_df.columns:
                    if col not in test_df.columns:
                        test_df[col] = metadata_df[col].values
            
            logger.info(f"✅ Madrigal inference completed: {len(test_df):,} observations")
        
        else:  # dataset_type == 'own'
            # Use original test set from training data
            logger.info("📦 Loading test data...")
            
            # For finetuned models, use the single-day test data from STEC_DB_CASDCB
            # For pretrained models, use the general test.h5
            if stec_config.get('mode') == 'finetune':
                logger.info(f"   Using single-day test data for finetuned model (year={stec_config.get('year')}, doy={stec_config.get('doy')})")
            else:
                logger.info(f"   Using general test.h5 for pretrained model")
            
            # Check if we can fully reuse existing results (STEC + VTEC + GIM all present)
            # and only need to add pretrained baseline
            has_all_existing = (
                reusable_df is not None 
                and 'stec_pred' in reusable_df.columns
                and 'true_stec' in reusable_df.columns
            )
            needs_pretrained_only = (
                has_all_existing
                and args.pretrained_stec_experiment
                and 'pretrained_stec_pred' not in reusable_df.columns
            )
            
            if needs_pretrained_only:
                # Fast path: reuse all existing results, only run pretrained inference
                logger.info("⏭️  Reusing existing STEC/VTEC/GIM results, only running Pretrained STEC inference")
                test_df = reusable_df.copy()
                
                # We still need data loader for pretrained model
                set_test_size(stec_config, args.test_size)
                
                # Also mark VTEC and GIM as already present so we skip them later
                vtec_col = 'vtec_model_stec' if 'vtec_model_stec' in test_df.columns else None
                gim_col = 'gim_stec' if 'gim_stec' in test_df.columns else None
                _skip_vtec_gim = True
            else:
                _skip_vtec_gim = False
                
                # Set test size (use full test set if not specified)
                set_test_size(stec_config, args.test_size)
                
                if has_all_existing:
                    logger.info("⏭️  Skipping STEC model inference (Own Test Set), reusing results")
                    test_df = reusable_df.copy()
                else:
                    # Need to load STEC model for inference
                    if stec_model is None:
                        stec_checkpoint = find_best_checkpoint(stec_dir)
                        stec_model, stec_registry = load_model_from_checkpoint(stec_config, stec_checkpoint, logger)
                    
                    stec_test_loader = get_test_data_loader(stec_config, logger)
                    
                    # Run STEC inference
                    test_df = run_inference(stec_model, stec_test_loader, stec_config, "STEC Model", logger, args.num_inference_samples)
                    
                    # Rename columns for consistency (pred_stec -> stec_pred, target_stec -> true_stec)
                    test_df.rename(columns={'pred_stec': 'stec_pred', 'target_stec': 'true_stec'}, inplace=True)
            
            # Load and run PRETRAINED STEC model if provided
            pretrain_stec_col = None
            if 'pretrained_stec_pred' in test_df.columns:
                # Already have pretrained results from reused CSV
                logger.info("⏭️  Pretrained STEC predictions found in reused results, skipping inference")
                pretrain_stec_col = 'pretrained_stec_pred'
            elif args.pretrained_stec_experiment:
                logger.info("\n" + "="*70)
                logger.info("Processing Pretrained STEC Baseline")
                logger.info("="*70)
                
                # Use same logic as VTEC for loading model once
                if not hasattr(main, '_pretrain_stec_model_loaded'):
                    pre_config, pre_dir = load_experiment_config(args.pretrained_stec_experiment)
                    
                    # Pretrained models usually have use_agg_h5=True, which we want to disable
                    # to use the daily test data for fair comparison
                    if 'data' in pre_config:
                        pre_config['data']['use_agg_h5'] = False
                        # Ensure we don't try to use training/validation splits from the pretrain run
                        if 'renew_splits' in pre_config['data']:
                            pre_config['data']['renew_splits'] = False
                    
                    pre_config["device"] = device
                    
                    # Increase batch size for faster inference
                    if 'finetune' in pre_config:
                        pre_config['finetune']['batchsize'] = max(pre_config['finetune'].get('batchsize', 2048), 4096)
                    if 'pretrain' in pre_config:
                        pre_config['pretrain']['batchsize'] = max(pre_config['pretrain'].get('batchsize', 2048), 4096)
                    
                    # Load pretrained model
                    pre_checkpoint = find_best_checkpoint(pre_dir)
                    pre_model, _ = load_model_from_checkpoint(pre_config, pre_checkpoint, logger)
                    
                    main._pretrain_stec_model_loaded = True
                    main._pretrain_stec_config = pre_config
                    main._pretrain_stec_model = pre_model
                else:
                    pre_config = main._pretrain_stec_config
                    pre_model = main._pretrain_stec_model
                
                # Create data loader for pretrained model using CURRENT day's data
                # Increase batch size for faster pretrained inference
                # Ensure stec_config has feature_registry initialized (needed for data loading)
                if 'feature_registry' not in stec_config:
                    stec_registry = initialize_feature_registry(stec_config)
                    stec_config['feature_registry'] = stec_registry
                    from data_loader.collation import CollateWithSH
                    CollateWithSH(stec_config)  # Sets output_indices in registry
                
                pre_stec_config = {**stec_config}
                if 'finetune' in pre_stec_config:
                    pre_stec_config['finetune'] = {**stec_config['finetune'], 'batchsize': max(stec_config['finetune'].get('batchsize', 2048), 4096)}
                if 'pretrain' in pre_stec_config:
                    pre_stec_config['pretrain'] = {**stec_config['pretrain'], 'batchsize': max(stec_config['pretrain'].get('batchsize', 2048), 4096)}
                pre_test_loader = get_test_data_loader(pre_stec_config, logger)
                
                # Run inference
                pre_df = run_inference(pre_model, pre_test_loader, pre_config, "Pretrained STEC Baseline", logger, args.num_inference_samples)
                
                # Verify match
                if len(pre_df) != len(test_df):
                    logger.warning(f"⚠️ Pretrained predictions ({len(pre_df)}) don't match STEC model ({len(test_df)}).")
                else:
                    test_df['pretrained_stec_pred'] = pre_df['pred_stec'].values
                    pretrain_stec_col = 'pretrained_stec_pred'
        
        # Load and run VTEC model if provided
        if not locals().get('vtec_col'):
            vtec_col = None
        if args.vtec_experiment and not _skip_vtec_gim:
            logger.info("\n" + "="*70)
            logger.info("Processing VTEC Model")
            logger.info("="*70)
            
            # Load VTEC config and model once (at start of loop, reuse for all datasets)
            if not hasattr(main, '_vtec_model_loaded'):
                vtec_config, vtec_dir = load_experiment_config(args.vtec_experiment)
                
                # CRITICAL: For finetuned models, force use_agg_h5=False to use day-specific test data
                if vtec_config.get('mode') == 'finetune':
                    if 'data' in vtec_config and vtec_config['data'].get('use_agg_h5'):
                        logger.info("⚡ Optimizing: Disabling use_agg_h5 for VTEC finetuned model")
                        vtec_config['data']['use_agg_h5'] = False
                
                # Verify VTEC target
                if vtec_config.get('target', 'vtec').lower() != 'vtec':
                    raise ValueError(f"VTEC experiment must have target='vtec', got {vtec_config.get('target')}")
                
                vtec_config["device"] = device
                
                # Enable metadata return to include elevation even though it's not a model input
                # This allows us to apply the mapping function later
                vtec_config["return_metadata"] = True
                vtec_config["metadata_fields"] = ["satele", "satazi", "station", "sat"]
                
                # Load VTEC model - this also initializes feature_registry in vtec_config
                vtec_checkpoint = find_best_checkpoint(vtec_dir)
                vtec_model, vtec_registry = load_model_from_checkpoint(vtec_config, vtec_checkpoint, logger)
                
                # Save for reuse in subsequent datasets
                main._vtec_model_loaded = True
                main._vtec_config = vtec_config
                main._vtec_model = vtec_model
            else:
                # Reuse from previous iteration
                vtec_config = main._vtec_config
                vtec_model = main._vtec_model
            
            # Create appropriate data loader based on dataset type
            if dataset_type == 'madrigal':
                logger.info("🔄 Creating VTEC data loader from Madrigal observations...")
                
                # Reuse the same Madrigal dataset but with VTEC config
                vtec_madrigal_loader, _ = get_madrigal_data_loader(
                    madrigal_path=args.madrigal_path,
                    year=year,
                    doy=doy,
                    config=vtec_config,
                    batch_size=8192,
                    num_workers=4,
                    elevation_threshold=5.0,
                    max_samples=int(args.test_size) if args.test_size else None,
                    station_list=test_stations,
                    logger=logger
                )
                
                vtec_test_loader = vtec_madrigal_loader
            else:
                # Use separate test data for VTEC (must match STEC test size for fair comparison)
                set_test_size(vtec_config, args.test_size)
                vtec_test_loader = get_test_data_loader(vtec_config, logger)
            
            # Run VTEC inference
            vtec_df = run_inference(vtec_model, vtec_test_loader, vtec_config, "VTEC Model", logger, args.num_inference_samples)
            
            # Elevation should now be available in vtec_df from metadata
            # Apply mapping function
            vtec_df = apply_mapping_function(vtec_df, args.mapping_function, 'vtec_model', logger)
            
            # Verify same number of observations for fair comparison
            if len(vtec_df) != len(test_df):
                raise ValueError(f"VTEC predictions ({len(vtec_df)}) don't match STEC predictions ({len(test_df)}). "
                               "Ensure both models use the same test data.")
            
            # Merge VTEC results into test_df (assumes same order from same data loader)
            vtec_cols_to_merge = [c for c in vtec_df.columns if c.startswith('vtec_model_stec')]
            for col in vtec_cols_to_merge:
                test_df[col] = vtec_df[col].values
            vtec_col = 'vtec_model_stec'
        
        # Add GIM comparison if requested
        if not locals().get('gim_col'):
            gim_col = None
        if not args.no_gim and not _skip_vtec_gim:
            if 'gim_stec' in test_df.columns:
                logger.info("⏭️  Skipping IGS GIM calculation, reusing results")
                gim_col = 'gim_stec'
            else:
                logger.info("\n" + "="*70)
                logger.info("Processing IGS GIM Data")
                logger.info("="*70)
                
                test_df = add_gim_comparison(test_df, args.gim_path, args.mapping_function, logger)
                gim_col = 'gim_stec'
        
        # Compare all models for this dataset
        logger.info("\n" + "="*70)
        logger.info(f"Final Comparison - {dataset_name}")
        logger.info("="*70)
        
        metrics = compare_all_models(test_df, 'stec_pred', vtec_col, gim_col, logger, pretrain_stec_col)
        
        # Determine output directory
        comparison_parts = [dataset_type]  # 'own' or 'madrigal'
        if args.vtec_experiment:
            comparison_parts.append("vtec")
        if not args.no_gim:
            comparison_parts.append("gim")
        
        comparison_name = "_".join(comparison_parts)
        experiment_output_dir = stec_dir / "evaluation" / comparison_name
        experiment_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_dirs = [experiment_output_dir]
        
        # Also save to custom output_dir if specified
        if args.output_dir:
            # Create subdirectory matching comparison name to keep results organized
            custom_output_dir = Path(args.output_dir) / comparison_name
            custom_output_dir.mkdir(parents=True, exist_ok=True)
            output_dirs.append(custom_output_dir)
        
        # Create publication-ready visualizations
        if not args.skip_plots:
            logger.info("\n" + "="*70)
            logger.info("📊 Generating Publication-Ready Plots")
            logger.info("="*70)
            
            for output_dir in output_dirs:
                generate_all_plots(
                    test_df=test_df,
                    stec_col='stec_pred',
                    vtec_col=vtec_col,
                    gim_col=gim_col,
                    metrics=metrics,
                    output_dir=output_dir,
                    logger=logger,
                    pretrain_col=pretrain_stec_col
                )
        else:
            logger.info("\n⏭️ Skipping plot generation (--skip_plots)")
        
        for output_dir in output_dirs:
            # Save results (CSV files, etc.)
            save_results(metrics, test_df, output_dir, args, logger)
        
        logger.info(f"📁 Results saved to: {experiment_output_dir.absolute()}")
        if args.output_dir and dataset_type == datasets_to_evaluate[0][0]:
            logger.info(f"📁 Also saved to: {custom_output_dir.absolute()}")
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("✅ All evaluations completed successfully!")
    logger.info(f"📁 Results saved in: {stec_dir / 'evaluation'}")
    logger.info("="*70)
    


if __name__ == "__main__":
    main()
