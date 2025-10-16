#!/usr/bin/env python3
"""
STEC Evaluation Script - Model vs GIM Comparison

This script reuses inference_testset.py patterns to:
1. Load config and experiment 
2. Run Bayesian inference on test set
3. Group test results by day
4. Load GIM VTEC for each day and map to STEC
5. Compare model STEC vs GIM STEC vs ground truth

Strategy: Reuse as much existing code as possible from inference_testset.py
"""

import torch
import os
import sys
import logging
import traceback
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List

# Add src to path for imports  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Reuse inference_testset patterns
from utils.config_parser import parse_config, compute_exp_name
from utils.feature_registry import initialize_feature_registry
from training.base_trainer import BaseTrainer
from data_loader import get_test_data_loader
from model.model import get_model
from inference_testset import find_experiment_directory, find_model_checkpoint
from evaluation.gim_mapper import GIMMapper


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def load_eval_config():
    """Load evaluation config similar to parse_config()."""
    config_path = Path(__file__).parent.parent / "config" / "config_eval.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_experiment_config(experiment_folder):
    """Load experiment config from trained model."""
    experiment_dir = Path(experiment_folder)
    if not experiment_dir.is_absolute():
        # Check if it already starts with "experiments/"
        if not str(experiment_folder).startswith("experiments/"):
            experiment_dir = Path("experiments") / experiment_folder
        else:
            experiment_dir = Path(experiment_folder)
    
    # Load config.yaml from experiment directory
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_model_inference(config, experiment_dir, model_path, logger):
    """Run Bayesian inference following inference_testset.py patterns."""
    
    # Initialize feature registry (same as inference_testset.py)
    feature_registry = initialize_feature_registry(config)
    config["feature_registry"] = feature_registry
    
    # Get test dataloader (same as inference_testset.py) 
    config['data']['test_size'] = 10_000
    test_loader = get_test_data_loader(config, logger)
    
    # Create trainer (same as inference_testset.py)
    trainer = BaseTrainer(config, logger)
    
    # Load model (same as inference_testset.py)
    model = get_model(config).to(config["device"])
    checkpoint = torch.load(model_path, map_location=config["device"], weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"✅ Model loaded: {config['model']['model_type']}")
    
    # Run Bayesian inference (same as inference_testset.py)
    logger.info("🧠 Running Bayesian inference...")
    bayesian_results, test_df = trainer.bayesian_inference_total_uncertainty(
        model,
        test_loader,
        num_samples=100 if "BNN" in config["model"]["model_type"] else 1,
    )
    
    logger.info(f"✅ Model inference completed: {len(test_df):,} predictions")
    return test_df


def group_observations_by_date(test_df, logger):
    """Group test observations by observation date for efficient GIM loading."""
    
    # Extract dates from the test dataframe
    # Assuming we have temporal features that can be converted to dates
    logger.info("📅 Grouping observations by date...")
    
    # Group by date - need to extract date from features or add it to dataframe
    # For now, create a simple grouping strategy
    grouped_data = defaultdict(list)
    
    # Add observation indices to track which rows belong to which group
    for idx, row in test_df.iterrows():
        # This is a placeholder - in reality we need proper date extraction
        # from the denormalized temporal features
        date_key = "2023-06-15"  # Placeholder date
        grouped_data[date_key].append(idx)
    
    logger.info(f"✅ Grouped into {len(grouped_data)} days")
    return grouped_data


def process_gim_for_date(date_str: str, observation_indices: List[int], test_df: pd.DataFrame, 
                        gim_mapper: GIMMapper, logger) -> Dict[str, np.ndarray]:
    """Load GIM data for a specific date and compute STEC for observations."""
    
    logger.info(f"🌍 Processing GIM for {date_str}: {len(observation_indices)} observations")
    
    # Extract coordinates and times for this date's observations
    obs_data = test_df.iloc[observation_indices]
    
    # Extract coordinates - need proper coordinate extraction here
    # This is placeholder - need to implement proper coordinate denormalization
    times = [datetime(2023, 6, 15, 12, 0, 0)] * len(observation_indices)  # Placeholder
    ipp_lat = np.random.uniform(-90, 90, len(observation_indices))  # Placeholder  
    ipp_lon = np.random.uniform(-180, 180, len(observation_indices))  # Placeholder
    elevations = np.random.uniform(5, 90, len(observation_indices))  # Placeholder
    
    # Load GIM data for this date (use time range covering the whole day)
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    start_time = date_obj
    end_time = date_obj + timedelta(days=1)
    
    try:
        gim_mapper.load_gim_data(gim_mapper.gim_path, (start_time, end_time))
        
        # Compute STEC from VTEC for all observations 
        gim_stec = gim_mapper.map_vtec_to_stec(
            np.array(times),
            np.array(ipp_lat), 
            np.array(ipp_lon),
            np.array(elevations)
        )
        
        # Check for valid results
        valid_mask = ~np.isnan(gim_stec)
        n_valid = valid_mask.sum()
        
        logger.info(f"✅ Processed {n_valid}/{len(observation_indices)} valid GIM predictions")
        
        return {
            'stec': gim_stec,
            'success': valid_mask
        }
        
    except Exception as e:
        logger.warning(f"❌ Failed to process GIM data for {date_str}: {e}")
        return {
            'stec': np.full(len(observation_indices), np.nan),
            'success': np.zeros(len(observation_indices), dtype=bool)
        }


def run_evaluation(eval_config):
    """Main evaluation function following inference_testset.py structure."""
    logger = setup_logging()
    
    # Extract config sections
    experiment_folder = eval_config['stec_evaluation']['experiment_folder']
    gim_path = eval_config['stec_evaluation']['gim_path']
    
    logger.info(f"📋 Using experiment: {Path(experiment_folder).name}")
    logger.info(f"🌍 GIM path: {gim_path}")
    
    # Load experiment config (same as inference_testset.py)
    config = load_experiment_config(experiment_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    
    # Find experiment directory and model (same as inference_testset.py)
    experiment_name = Path(experiment_folder).name
    experiment_dir = find_experiment_directory(experiment_name)
    if experiment_dir is None:
        raise FileNotFoundError(f"Experiment not found: {experiment_name}")
    
    model_path = find_model_checkpoint(experiment_dir)
    logger.info(f"✅ Found model: {model_path}")
    
    # Run model inference (reusing inference_testset.py patterns)
    logger.info("🚀 Starting STEC evaluation")
    test_df = run_model_inference(config, experiment_dir, model_path, logger)
    
    # Group by dates for efficient GIM processing
    date_groups = group_observations_by_date(test_df, logger)
    
    # Initialize GIM mapper
    gim_mapper = GIMMapper()
    gim_mapper.gim_path = gim_path
    
    # Process each date group
    all_gim_results = {}
    logger.info(f"📊 Processing {len(date_groups)} date groups...")
    
    for date_str, obs_indices in tqdm(date_groups.items(), desc="Processing dates"):
        gim_results = process_gim_for_date(date_str, obs_indices, test_df, gim_mapper, logger)
        all_gim_results[date_str] = gim_results
    
    # Combine results back into test_df
    gim_stec_values = np.full(len(test_df), np.nan)
    gim_success_flags = np.zeros(len(test_df), dtype=bool)
    
    for date_str, obs_indices in date_groups.items():
        gim_results = all_gim_results[date_str]
        for i, obs_idx in enumerate(obs_indices):
            gim_stec_values[obs_idx] = gim_results['stec'][i]
            gim_success_flags[obs_idx] = gim_results['success'][i]
    
    # Add GIM results to dataframe
    test_df['gim_stec'] = gim_stec_values
    test_df['gim_success'] = gim_success_flags
    
    # Calculate comparison metrics
    valid_mask = gim_success_flags & ~np.isnan(test_df['pred_stec']) & ~np.isnan(test_df['target_stec'])
    n_valid = valid_mask.sum()
    n_total = len(test_df)
    
    if n_valid > 0:
        model_stec = test_df.loc[valid_mask, 'pred_stec'].values
        gim_stec = test_df.loc[valid_mask, 'gim_stec'].values
        truth_stec = test_df.loc[valid_mask, 'target_stec'].values
        
        # Calculate RMSEs
        model_vs_truth_rmse = np.sqrt(np.mean((model_stec - truth_stec)**2))
        gim_vs_truth_rmse = np.sqrt(np.mean((gim_stec - truth_stec)**2))
        model_vs_gim_rmse = np.sqrt(np.mean((model_stec - gim_stec)**2))
        
        results = {
            'n_total': n_total,
            'n_valid': n_valid,
            'validity_rate': n_valid / n_total,
            'model_vs_truth': {'rmse': model_vs_truth_rmse},
            'gim_vs_truth': {'rmse': gim_vs_truth_rmse},
            'model_vs_gim': {'rmse': model_vs_gim_rmse},
            'test_df': test_df
        }
    else:
        results = {
            'n_total': n_total,
            'n_valid': 0,
            'validity_rate': 0.0,
            'error': 'No valid comparisons possible'
        }
    
    return results


def main():
    """Main entry point."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    try:
        eval_config = load_eval_config()
        results = run_evaluation(eval_config)
        
        logger = logging.getLogger(__name__)
        logger.info("✅ EVALUATION COMPLETED!")
        logger.info(f"📊 Samples processed: {results['n_total']:,}")
        logger.info(f"✓ Valid predictions: {results['n_valid']:,} ({results['validity_rate']*100:.1f}%)")
        
        if 'error' not in results:
            logger.info(f"📈 Model vs Truth RMSE: {results['model_vs_truth']['rmse']:.4f}")
            logger.info(f"🌍 GIM vs Truth RMSE: {results['gim_vs_truth']['rmse']:.4f}")
            logger.info(f"🔄 Model vs GIM RMSE: {results['model_vs_gim']['rmse']:.4f}")

        return 0

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ EVALUATION FAILED: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
