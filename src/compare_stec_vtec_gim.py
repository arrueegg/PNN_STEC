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
    
    # Create model
    model = get_model(config).to(config["device"])
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=config["device"], weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    logger.info(f"✅ Loaded {config['model']['model_type']} from {checkpoint_path.name}")
    return model, feature_registry


def run_inference(
    model: torch.nn.Module,
    test_loader,
    config: Dict,
    model_name: str,
    logger,
    num_inference_samples: int = 100
) -> pd.DataFrame:
    """Run inference with a model."""
    logger.info(f"🧠 Running {model_name} inference...")
    
    trainer = BaseTrainer(config, logger)
    
    # Determine if Bayesian
    model_type = config["model"]["model_type"]
    is_bayesian = "BNN" in model_type or "Bayesian" in model_type or "FactorizedSTEC" in model_type
    samples = num_inference_samples if is_bayesian else 1
        
    # Run Bayesian inference
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
    vtec_pred = vtec_df['pred_mean'].values
    
    # Compute mapping factor
    mapping_factors = np.array([mapper.get_mapping_factor(el) for el in elevations_rad])
    
    # Convert VTEC to STEC
    stec_mapped = vtec_pred * mapping_factors
    
    # Also propagate uncertainty (variance scales with mapping factor squared)
    if 'pred_var' in vtec_df.columns:
        vtec_var = vtec_df['pred_var'].values
        stec_var_mapped = vtec_var * (mapping_factors ** 2)
        vtec_df[f'{column_prefix}_stec_var'] = stec_var_mapped
    
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
            mapping_factors = np.array([mapper.get_mapping_factor(el) for el in elevations_rad])
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
    logger
) -> Dict[str, Dict[str, float]]:
    """
    Compare all available models.
    
    Args:
        test_df: DataFrame with all predictions
        stec_col: Column name for direct STEC predictions
        vtec_col: Column name for VTEC+mapping predictions (or None)
        gim_col: Column name for GIM STEC (or None)
        logger: Logger instance
    """
    logger.info("📊 Computing comparison metrics...")
    
    # Get ground truth STEC
    ground_truth = test_df['true_stec'].values
    
    results = {}
    
    # Direct STEC model
    stec_pred = test_df[stec_col].values
    results['Direct STEC Model'] = compute_metrics(stec_pred, ground_truth)
    
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
        if args.include_gim:
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
    
    # Save detailed predictions CSV
    csv_cols = ['true_stec', 'stec_pred', 'satele']
    rename_dict = {'satele': 'elevation'}
    
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


def main():
    """Main comparison workflow."""
    parser = argparse.ArgumentParser(description="Compare STEC model(s) vs VTEC+Mapping and/or GIM")
    parser.add_argument("--stec_experiment", type=str, required=True,
                       help="Path to STEC model experiment folder")
    parser.add_argument("--vtec_experiment", type=str, default=None,
                       help="Path to VTEC model experiment folder (optional)")
    parser.add_argument("--include_gim", action="store_true",
                       help="Include IGS GIM comparison")
    parser.add_argument("--gim_path", type=str, 
                       default="/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex",
                       help="Path to GIM/IONEX data directory")
    parser.add_argument("--use_madrigal", action="store_true",
                       help="Use Madrigal STEC as independent ground truth")
    parser.add_argument("--madrigal_path", type=str,
                       default="/home/space/data/iono/Madrigal_STEC",
                       help="Path to Madrigal STEC data directory")
    parser.add_argument("--mapping_function", type=str, default="MSLM",
                       choices=["SLM", "MSLM"],
                       help="Mapping function to use (default: MSLM)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results (default: inside experiment folder)")
    parser.add_argument("--num_inference_samples", type=int, default=100,
                       help="Number of inference samples for Bayesian models")
    parser.add_argument("--test_size", default=None,
                       help="Number of test samples (for faster testing, use smaller value)")
    
    args = parser.parse_args()
    logger = setup_logging()
    
    logger.info("="*70)
    logger.info("Comprehensive STEC Model Comparison")
    logger.info("="*70)
    
    # Load STEC configuration
    stec_config, stec_dir = load_experiment_config(args.stec_experiment)
    
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
    
    # Load STEC model
    stec_checkpoint = find_best_checkpoint(stec_dir)
    stec_model, stec_registry = load_model_from_checkpoint(stec_config, stec_checkpoint, logger)
    
    # Decide test data source: Madrigal (independent) or original test set
    if args.use_madrigal:
        logger.info("\n" + "="*70)
        logger.info("Using Madrigal STEC as Independent Test Set")
        logger.info("="*70)
        
        # Get year and doy from config
        year = stec_config.get('year', 2024)
        doy = stec_config.get('doy', 183)
        if stec_config.get('mode') == 'finetune':
            year = int(year)
            doy = int(doy)
        else:
            # For pretrained models, need to infer from test data
            logger.warning("Pretrained model detected - using first observation's date")
            raise NotImplementedError("Madrigal comparison currently only supported for finetuned models")
        
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
        
        madrigal_loader, madrigal_dataset = get_madrigal_data_loader(
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
        
        # Run inference on Madrigal data
        logger.info("🧠 Running STEC model inference on Madrigal observations...")
        test_df = run_inference(stec_model, madrigal_loader, stec_config, "STEC Model (Madrigal)", logger, args.num_inference_samples)
        
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
    
    else:
        # Use original test set from training data
        logger.info("📦 Loading test data...")
        
        # For finetuned models, use the single-day test data from STEC_DB_CASDCB
        # For pretrained models, use the general test.h5
        if stec_config.get('mode') == 'finetune':
            logger.info(f"   Using single-day test data for finetuned model (year={stec_config.get('year')}, doy={stec_config.get('doy')})")
        else:
            logger.info(f"   Using general test.h5 for pretrained model")
        
        # Override test_size if specified in arguments, otherwise use config default
        if args.test_size:
            stec_config['data']['test_size'] = args.test_size
        elif 'test_size' not in stec_config['data']:
            # Set default if not specified
            stec_config['data']['test_size'] = 100000
        
        stec_test_loader = get_test_data_loader(stec_config, logger)
        
        # Run STEC inference
        test_df = run_inference(stec_model, stec_test_loader, stec_config, "STEC Model", logger, args.num_inference_samples)
        
        # Rename columns for consistency (pred_stec -> stec_pred, target_stec -> true_stec)
        test_df.rename(columns={'pred_stec': 'stec_pred', 'target_stec': 'true_stec'}, inplace=True)
    
    # Load and run VTEC model if provided
    vtec_col = None
    if args.vtec_experiment:
        logger.info("\n" + "="*70)
        logger.info("Processing VTEC Model")
        logger.info("="*70)
        
        vtec_config, vtec_dir = load_experiment_config(args.vtec_experiment)
        
        # Verify VTEC target
        if vtec_config.get('target', 'vtec').lower() != 'vtec':
            raise ValueError(f"VTEC experiment must have target='vtec', got {vtec_config.get('target')}")
        
        vtec_config["device"] = device
        vtec_config['data']['test_size'] = args.test_size
        
        # Load VTEC model
        vtec_checkpoint = find_best_checkpoint(vtec_dir)
        vtec_model, vtec_registry = load_model_from_checkpoint(vtec_config, vtec_checkpoint, logger)
        
        # Get VTEC test data loader
        vtec_test_loader = get_test_data_loader(vtec_config, logger)
        
        # Run VTEC inference
        vtec_df = run_inference(vtec_model, vtec_test_loader, vtec_config, "VTEC Model", logger, args.num_inference_samples)
        
        # Apply mapping function
        vtec_df = apply_mapping_function(vtec_df, args.mapping_function, 'vtec_model', logger)
        
        # Merge VTEC results into test_df
        test_df['vtec_model_stec'] = vtec_df['vtec_model_stec'].values
        vtec_col = 'vtec_model_stec'
    
    # Add GIM comparison if requested
    gim_col = None
    if args.include_gim:
        logger.info("\n" + "="*70)
        logger.info("Processing IGS GIM Data")
        logger.info("="*70)
        
        test_df = add_gim_comparison(test_df, args.gim_path, args.mapping_function, logger)
        gim_col = 'gim_stec'
    
    # Compare all models
    logger.info("\n" + "="*70)
    logger.info("Final Comparison")
    if args.use_madrigal:
        logger.info("(Using Madrigal STEC as independent ground truth)")
    logger.info("="*70)
    
    metrics = compare_all_models(test_df, 'stec_pred', vtec_col, gim_col, logger)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Save in experiment folder by default
        comparison_type = "Madrigal" if args.use_madrigal else "Normal"
        output_dir = stec_dir / "evaluation" / f"{comparison_type.lower()}_comparison"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create publication-ready visualizations
    logger.info("\n" + "="*70)
    logger.info("📊 Generating Publication-Ready Plots")
    logger.info("="*70)
    
    generate_all_plots(
        test_df=test_df,
        stec_col='stec_pred',
        vtec_col=vtec_col,
        gim_col=gim_col,
        metrics=metrics,
        output_dir=output_dir,
        logger=logger
    )
    
    # Save results
    save_results(metrics, test_df, output_dir, args, logger)
    
    logger.info("\n" + "="*70)
    logger.info("✅ Comparison completed successfully!")
    logger.info(f"📁 Results saved to: {output_dir.absolute()}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
