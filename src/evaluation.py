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
from evaluation.utils import save_results_csv
from evaluation.plotter import create_stec_plots
from evaluation.madrigal_loader import find_madrigal_file, extract_stec_for_date


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
    
    # Group by date using 'year' and 'doy' columns
    grouped_data = defaultdict(list)
    
    # Ensure 'year' and 'doy' columns exist in the dataframe
    if 'year' not in test_df.columns or 'doy' not in test_df.columns:
        raise ValueError("The test dataframe must contain 'year' and 'doy' columns for date grouping.")
    
    # Add observation indices to track which rows belong to which group
    for idx, row in test_df.iterrows():
        # Construct the date from 'year' and 'doy' columns
        year = int(row['year'])
        doy = int(row['doy'])
        date_key = (datetime(year, 1, 1) + timedelta(days=doy - 1)).strftime("%Y-%m-%d")
        grouped_data[date_key].append(idx)
    
    logger.info(f"✅ Grouped into {len(grouped_data)} days")
    return grouped_data


def process_gim_for_date(date_str: str, observation_indices: List[int], test_df: pd.DataFrame, 
                        gim_mapper: GIMMapper, gim_path: str, logger) -> Dict[str, np.ndarray]:
    """Load GIM data for a specific date and compute STEC for observations."""
        
    # Extract coordinates and times for this date's observations
    obs_data = test_df.iloc[observation_indices]
    
    # Extract coordinates and times
    ipp_lat = obs_data['lat_ipp'].values
    ipp_lon = obs_data['lon_ipp'].values
    elevations = obs_data['satele'].values
    sod = obs_data['sod'].values
    
    # Load GIM data for this date (use time range covering the whole day)
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    
    try:
        # Load GIM data for this specific date
        gim_mapper.load_gim_data(gim_path, date_obj)
        
        # Compute STEC from VTEC for all observations 
        gim_stec = gim_mapper.map_vtec_to_stec(
            np.array(sod),
            np.array(ipp_lat), 
            np.array(ipp_lon),
            np.array(elevations)
        )
        
        # Check for valid results
        valid_mask = ~np.isnan(gim_stec)                
        return {
            'gim_stec': gim_stec,
            'success': valid_mask,
            'indices': observation_indices
        }
        
    except Exception as e:
        logger.warning(f"  ❌ GIM processing failed for {date_str}: {e}")
        return {
            'gim_stec': np.full(len(observation_indices), np.nan),
            'success': np.zeros(len(observation_indices), dtype=bool),
            'indices': observation_indices
        }


def run_evaluation(eval_config):
    """Main evaluation workflow mirroring inference_testset.py structure."""
    
    logger = setup_logging()
    logger.info("🚀 STARTING STEC EVALUATION - Model vs GIM Comparison")
    
    # Load experiment configuration
    experiment_folder = eval_config['stec_evaluation']['experiment_folder']
    experiment_dir = Path(experiment_folder)
    config = load_experiment_config(experiment_dir)
    
    # Update device settings
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"💻 Using device: {config['device']}")
    
    # Find model checkpoint
    model_path = find_model_checkpoint(experiment_dir)
    logger.info(f"📂 Experiment: {experiment_dir}")
    logger.info(f"🎯 Model: {model_path}")
    
    # Decide which dataset to evaluate: 'testset' (original) or 'madrigal'
    dataset_choice = eval_config['stec_evaluation'].get('dataset', 'testset')
    logger.info(f"🗂️  Evaluation dataset: {dataset_choice}")

    # Run model inference (we need model preds in both modes)
    test_df = run_model_inference(config, experiment_dir, model_path, logger)

    if dataset_choice == 'testset':
        # Group observations by date for efficient GIM loading
        grouped_observations = group_observations_by_date(test_df, logger)
        
        # Initialize GIM mapper
        gim_path = eval_config['stec_evaluation'].get('gim_path', '/scratch2/arrueegg/WP4/data/GIM')
        gim_mapper = GIMMapper(gim_path)
        logger.info(f"🌍 Initialized GIM mapper: {gim_path}")
        
        # Process GIM data for each date
        logger.info(f"🔄 Processing GIM data for {len(grouped_observations)} dates...")
        
        gim_stec_values = np.full(len(test_df), np.nan)
        gim_success_flags = np.zeros(len(test_df), dtype=bool)

        # Prepare madrigal columns if requested
        use_madrigal = bool(eval_config['stec_evaluation'].get('use_madrigal', False))
        if use_madrigal:
            madrigal_path = eval_config['stec_evaluation'].get('madrigal_path')
            test_df['madrigal_stec'] = np.nan
            test_df['madrigal_success'] = False
            if not madrigal_path:
                logger.warning("Madrigal path not configured; will skip per-day Madrigal lookup")
        
        for date_str, observation_indices in tqdm(grouped_observations.items(), desc="Processing GIM dates"):
            gim_result = process_gim_for_date(date_str, observation_indices, test_df, gim_mapper, gim_path, logger)
            
            # Store results back in test_df indices
            for i, idx in enumerate(observation_indices):
                gim_stec_values[idx] = gim_result['gim_stec'][i]
                gim_success_flags[idx] = gim_result['success'][i]

            # Also attach Madrigal STEC per-day when requested
            if use_madrigal and madrigal_path:
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    h5file = find_madrigal_file(madrigal_path, date_obj)
                    if h5file is not None:
                        obs_df = test_df.iloc[observation_indices]
                        stec_vals, success = extract_stec_for_date(h5file, obs_df)
                        for i, idx in enumerate(observation_indices):
                            test_df.at[idx, 'madrigal_stec'] = stec_vals[i]
                            test_df.at[idx, 'madrigal_success'] = bool(success[i])
                    else:
                        logger.debug("No Madrigal file for %s", date_str)
                except Exception as e:
                    logger.warning(f"Failed Madrigal lookup for {date_str}: {e}")

        # Add GIM results to test dataframe
        test_df['gim_stec'] = gim_stec_values
        test_df['gim_success'] = gim_success_flags

    else:
        logger.warning(f"Unknown dataset choice '{dataset_choice}'; creating placeholder columns")
        # Create placeholder columns for consistency
        test_df['gim_stec'] = np.nan
        test_df['gim_success'] = False
    
    # Create output directory inside experiment folder
    eval_results_dir = experiment_dir / "eval_results"
    
    # Save CSV results and generate plots using lightweight utilities
    save_results_csv(test_df, eval_results_dir)
    
    # Run comprehensive STEC analysis
    if eval_config.get('enhanced_analysis', {}).get('enabled', True):
        logger.info("🔬 Running comprehensive STEC analysis...")
        create_stec_plots(test_df, eval_results_dir, logger)
    
    return test_df


def main():
    """Main entry point."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    try:
        eval_config = load_eval_config()
        results_df = run_evaluation(eval_config)

        logger = logging.getLogger(__name__)
        
        # Get the experiment directory to print stats in correct location
        experiment_folder = eval_config['stec_evaluation']['experiment_folder']
        experiment_dir = Path(experiment_folder)
        if not str(experiment_folder).startswith("experiments/"):
            experiment_dir = Path("experiments") / experiment_folder
        eval_results_dir = experiment_dir / "eval_results"
        
        # Print summary statistics using lightweight utilities
        from evaluation.utils import print_and_save_statistics
        print_and_save_statistics(results_df, eval_results_dir)

        logger.info("✅ EVALUATION COMPLETED!")
        return 0

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ EVALUATION FAILED: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    main()