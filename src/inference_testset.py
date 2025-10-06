#!/usr/bin/env python3
"""
Config-based Inference Script for PNN_STEC Project

This script performs inference on pre-trained neural network models without requiring 
any training. It automatically loads the trained model that matches the current 
configuration in config/config.yaml and runs comprehensive inference analysis.

Key Features:
- Uses existing config parser to generate exact experiment names (compute_exp_name)
- Leverages BaseTrainer class for consistent inference methods
- Supports both Bayesian (BNN) and standard (MLP) neural network models
- Performs proper uncertainty quantification for Bayesian models (100 samples)
- Generates comprehensive plots and uncertainty analysis
- Saves results to CSV files and summary reports
- Memory efficient (releases unused data loaders)

Usage:
    python src/inference.py

Requirements:
- Model must be already trained and saved in experiments/ directory
- config/config.yaml must match an existing experiment
- Uses exact name matching - no approximations

Output:
- CSV file with predictions and uncertainties
- Summary text file with metrics
- Comprehensive plots and uncertainty analysis
- All saved to: experiments/<experiment_name>/

The script will error if no matching trained model is found, ensuring you only
run inference on properly trained models.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import glob
from tqdm import tqdm
import logging
from datetime import datetime, timedelta

# Add the parent directory to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_parser import parse_config, compute_exp_name
from utils.data import get_test_data_loader
from utils.feature_registry import initialize_feature_registry
from utils.metrics import calculate_metrics
from utils.plot import plot_test_metrics
from utils.base_trainer import BaseTrainer
from model.model import get_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def split_test_data_by_date(test_df):
    """
    Split test dataframe into interpolation and extrapolation subsets.
    Simple rule: May 2024 and later = extrapolation, everything before = interpolation.
    
    Args:
        test_df: Test dataframe with 'year' and 'doy' columns
    
    Returns:
        tuple: (interpolation_df, extrapolation_df, split_info)
    """
    if 'year' not in test_df.columns or 'doy' not in test_df.columns:
        logger.warning("Year or DOY columns not found in test data. Cannot split by date.")
        return test_df, pd.DataFrame(), {}
    
    # Create datetime from year and doy
    def create_date(row):
        try:
            year = int(row['year'])
            doy = int(row['doy'])
            date = datetime(year, 1, 1) + timedelta(days=doy - 1)
            return date
        except:
            return None
    
    test_df = test_df.copy()
    test_df['date'] = test_df.apply(create_date, axis=1)
    test_df = test_df.dropna(subset=['date'])
    
    # Extract year-month periods for comparison
    test_df['year_month'] = test_df['date'].dt.to_period('M')
    
    # Simple split: May 2024 and later = extrapolation, everything before = interpolation
    cutoff_period = pd.Period('2024-05')
    
    interpolation_mask = test_df['year_month'] < cutoff_period
    extrapolation_mask = test_df['year_month'] >= cutoff_period
    
    interpolation_df = test_df[interpolation_mask].copy().reset_index(drop=True)
    extrapolation_df = test_df[extrapolation_mask].copy().reset_index(drop=True)
    
    # Create split information summary
    interpolation_months = sorted(interpolation_df['year_month'].unique()) if len(interpolation_df) > 0 else []
    extrapolation_months = sorted(extrapolation_df['year_month'].unique()) if len(extrapolation_df) > 0 else []
    
    split_info = {
        'total_samples': len(test_df),
        'interpolation_samples': len(interpolation_df),
        'extrapolation_samples': len(extrapolation_df),
        'interpolation_months': [str(m) for m in interpolation_months],
        'extrapolation_months': [str(m) for m in extrapolation_months],
        'interpolation_percentage': (len(interpolation_df) / len(test_df)) * 100 if len(test_df) > 0 else 0,
        'extrapolation_percentage': (len(extrapolation_df) / len(test_df)) * 100 if len(test_df) > 0 else 0,
        'cutoff_date': str(cutoff_period)
    }
    
    return interpolation_df, extrapolation_df, split_info

def save_temporal_split_metrics(interpolation_df, extrapolation_df, split_info, experiment_dir):
    """
    Calculate and save metrics for interpolation/extrapolation splits.
    
    Args:
        interpolation_df: Test data before May 2024 (interpolation)
        extrapolation_df: Test data May 2024 and later (extrapolation)
        split_info: Dictionary with split information
        experiment_dir: Experiment directory path
    """
    # Create test_metrics subdirectories
    interpolation_dir = os.path.join(experiment_dir, 'interpolation')
    extrapolation_dir = os.path.join(experiment_dir, 'extrapolation')

    os.makedirs(interpolation_dir, exist_ok=True)
    os.makedirs(extrapolation_dir, exist_ok=True)
    
    # Calculate metrics for each subset
    metrics_summary = {}
    
    if len(interpolation_df) > 0:
        # Convert dataframe to tensors for metrics calculation
        interpolation_predictions = torch.stack([
            torch.tensor(interpolation_df['pred_stec'].values, dtype=torch.float32),
            torch.tensor(interpolation_df['pred_total_unc'].values, dtype=torch.float32)
        ], dim=1)
        interpolation_targets = torch.tensor(interpolation_df['target_stec'].values, dtype=torch.float32)
        
        interpolation_metrics = calculate_metrics(interpolation_predictions, interpolation_targets, prefix="interpolation")
        metrics_summary['interpolation'] = interpolation_metrics
        
        # Save interpolation period summary
        interpolation_summary_path = os.path.join(interpolation_dir, 'metrics_summary.txt')
        with open(interpolation_summary_path, 'w') as f:
            f.write("METRICS FOR INTERPOLATION (BEFORE MAY 2024)\n")
            f.write("=" * 60 + "\n\n")
            f.write("This includes test months before May 2024.\n")
            f.write("These are months within or close to the training period.\n\n")
            f.write(f"Cutoff date: {split_info['cutoff_date']}\n")
            f.write(f"Number of samples: {len(interpolation_df):,}\n")
            f.write(f"Percentage of total test data: {split_info['interpolation_percentage']:.1f}%\n")
            f.write(f"Months included: {', '.join(split_info['interpolation_months'])}\n\n")
            f.write("METRICS:\n")
            f.write("-" * 20 + "\n")
            for k, v in interpolation_metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        
        mae_value = interpolation_metrics.get('interpolation_MAE', 'N/A')
        mae_str = f"{mae_value:.4f}" if isinstance(mae_value, (int, float)) else str(mae_value)
        logger.info(f"Interpolation - Samples: {len(interpolation_df):,}, MAE: {mae_str}")
    
    if len(extrapolation_df) > 0:
        # Convert dataframe to tensors for metrics calculation  
        extrapolation_predictions = torch.stack([
            torch.tensor(extrapolation_df['pred_stec'].values, dtype=torch.float32),
            torch.tensor(extrapolation_df['pred_total_unc'].values, dtype=torch.float32)
        ], dim=1)
        extrapolation_targets = torch.tensor(extrapolation_df['target_stec'].values, dtype=torch.float32)
        
        extrapolation_metrics = calculate_metrics(extrapolation_predictions, extrapolation_targets, prefix="extrapolation")
        metrics_summary['extrapolation'] = extrapolation_metrics
        
        # Save extrapolation period summary
        extrapolation_summary_path = os.path.join(extrapolation_dir, 'metrics_summary.txt')
        with open(extrapolation_summary_path, 'w') as f:
            f.write("METRICS FOR EXTRAPOLATION (MAY 2024 AND LATER)\n")
            f.write("=" * 60 + "\n\n")
            f.write("This includes test months from May 2024 onwards.\n")
            f.write("These are true forecasting/extrapolation months.\n\n")
            f.write(f"Cutoff date: {split_info['cutoff_date']}\n")
            f.write(f"Number of samples: {len(extrapolation_df):,}\n")
            f.write(f"Percentage of total test data: {split_info['extrapolation_percentage']:.1f}%\n")
            f.write(f"Months included: {', '.join(split_info['extrapolation_months'])}\n\n")
            f.write("METRICS:\n")
            f.write("-" * 20 + "\n")
            for k, v in extrapolation_metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        
        mae_value = extrapolation_metrics.get('extrapolation_MAE', 'N/A')
        mae_str = f"{mae_value:.4f}" if isinstance(mae_value, (int, float)) else str(mae_value)
        logger.info(f"Extrapolation - Samples: {len(extrapolation_df):,}, MAE: {mae_str}")
    
    # Save combined temporal split summary
    split_summary_path = os.path.join(experiment_dir, 'test_metrics', 'temporal_split_summary.txt')
    with open(split_summary_path, 'w') as f:
        f.write("TEMPORAL SPLIT ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Split cutoff: {split_info['cutoff_date']}\n")
        f.write(f"Total test samples: {split_info['total_samples']:,}\n\n")
        
        f.write("INTERPOLATION (BEFORE MAY 2024):\n")
        f.write("-" * 35 + "\n")
        f.write(f"Samples: {split_info['interpolation_samples']:,} ({split_info['interpolation_percentage']:.1f}%)\n")
        f.write(f"Months: {', '.join(split_info['interpolation_months'])}\n")
        if 'interpolation' in metrics_summary:
            f.write("Key Metrics:\n")
            for k, v in metrics_summary['interpolation'].items():
                if any(metric in k.lower() for metric in ['mae', 'mse', 'rmse']):
                    f.write(f"  {k}: {v:.4f}\n")
        f.write("\n")
        
        f.write("EXTRAPOLATION (MAY 2024 AND LATER):\n")
        f.write("-" * 38 + "\n")
        f.write(f"Samples: {split_info['extrapolation_samples']:,} ({split_info['extrapolation_percentage']:.1f}%)\n")
        f.write(f"Months: {', '.join(split_info['extrapolation_months'])}\n")
        if 'extrapolation' in metrics_summary:
            f.write("Key Metrics:\n")
            for k, v in metrics_summary['extrapolation'].items():
                if any(metric in k.lower() for metric in ['mae', 'mse', 'rmse']):
                    f.write(f"  {k}: {v:.4f}\n")
        f.write("\n")
        
        # Performance comparison if both subsets exist
        if 'interpolation' in metrics_summary and 'extrapolation' in metrics_summary:
            f.write("PERFORMANCE COMPARISON:\n")
            f.write("-" * 25 + "\n")
            for metric in ['MAE', 'MSE', 'RMSE']:
                interpolation_key = f"interpolation_{metric}"
                extrapolation_key = f"extrapolation_{metric}"
                if interpolation_key in metrics_summary['interpolation'] and extrapolation_key in metrics_summary['extrapolation']:
                    interpolation_val = metrics_summary['interpolation'][interpolation_key]
                    extrapolation_val = metrics_summary['extrapolation'][extrapolation_key]
                    diff = extrapolation_val - interpolation_val
                    pct_change = (diff / interpolation_val) * 100 if interpolation_val != 0 else 0
                    f.write(f"{metric}:\n")
                    f.write(f"  Interpolation: {interpolation_val:.4f}\n")
                    f.write(f"  Extrapolation: {extrapolation_val:.4f}\n")
                    f.write(f"  Difference: {diff:+.4f} ({pct_change:+.1f}%)\n\n")
    
    return metrics_summary

def find_experiment_directory(experiment_name, base_dir='experiments'):
    """Find the experiment directory that matches the generated name exactly."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Experiments directory not found: {base_dir}")
    
    # Only look for exact match
    exact_path = os.path.join(base_dir, experiment_name)
    if os.path.exists(exact_path):
        return exact_path
    
    # No exact match found
    return None

def find_model_checkpoint(experiment_dir, config):
    """Find the model checkpoint in the experiment directory."""
    model_dir = os.path.join(experiment_dir, 'model')
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Look for checkpoint files
    mode = config['mode']
    model_type = config['model']['model_type']
    
    # Try different naming patterns
    patterns = [
        f"{mode}_{model_type}_seed*.pth",
        f"{mode}_*.pth",
        f"*{model_type}*.pth",
        "*.pth"
    ]
    
    for pattern in patterns:
        pth_files = glob.glob(os.path.join(model_dir, pattern))
        if pth_files:
            checkpoint_path = pth_files[0]  # Take the first match
            return checkpoint_path
    
    raise FileNotFoundError(f"No model checkpoint found in {model_dir}")

def run_inference_pipeline(config, experiment_dir, checkpoint_path):
    """Run the complete inference pipeline using BaseTrainer."""
    logger.info(f"Running inference...")
    logger.info("💾 Loading test data only (optimized for inference)")
    
    # Setup device
    device = config['device']
    
    # Initialize feature registry
    feature_registry = initialize_feature_registry(config)
    config['feature_registry'] = feature_registry
    
    # Create BaseTrainer instance (reuse existing inference methods)
    trainer = BaseTrainer(config, logger)
    
    # Load model
    model = get_model(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data only (more efficient for inference)
    config['data']['use_all_test_samples'] = True
    config['pretrain']['batchsize'] = 4096
    if not 'BNN' in config['model']['model_type']:
        config['pretrain']['batchsize'] = 4096 * 8  # Larger batch size for non-BNN models
        config['pretrain']['num_workers'] = 16
    test_loader = get_test_data_loader(config, logger)

    # Run inference using BaseTrainer's methods
    model_type = config['model']['model_type']
    is_bayesian = 'BNN' in model_type
    
    # Check if inference results already exist
    results_path = os.path.join(experiment_dir, 'inference_results.csv')
    
    if os.path.exists(results_path):
        logger.info(f"🔄 Found existing inference results: {results_path}")
        logger.info(f"📂 Loading pre-computed results instead of running inference...")
        
        try:
            # Load existing results
            test_df = pd.read_csv(results_path)
            logger.info(f"✅ Loaded DataFrame with shape: {test_df.shape}")
            
            # Extract the required data for metrics calculation
            bayesian_results = {
                'mean': torch.tensor(test_df['pred_stec'].values, dtype=torch.float32),
                'epistemic_std': torch.tensor(test_df['pred_epistemic_unc'].values, dtype=torch.float32),
                'aleatoric_std': torch.tensor(test_df['pred_aleatoric_unc'].values, dtype=torch.float32),
                'total_std': torch.tensor(test_df['pred_total_unc'].values, dtype=torch.float32),
                'targets': torch.tensor(test_df['target_stec'].values, dtype=torch.float32)
            }
            
            # Extract outputs and targets for metrics calculation
            test_outputs = torch.stack([
                bayesian_results['mean'],
                bayesian_results['total_std']
            ], dim=1)
            test_targets = bayesian_results['targets']
            
            logger.info(f"🚀 Skipped inference - using cached results with {len(test_targets):,} samples")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load existing results: {e}")
            logger.info(f"🔄 Falling back to running inference...")
            
            # Fall back to running inference
            run_inference = True
    else:
        logger.info(f"🚀 No existing results found, running inference...")
        run_inference = True
    
    # Run inference if needed
    if 'run_inference' in locals() and run_inference:
        # Use Bayesian inference for both BNN and non-BNN models
        # For non-BNN models, use num_samples=1 to get the same feature extraction
        num_samples = 100 if is_bayesian else 1
        
        # Always use the full, optimized Bayesian inference to ensure all input variables
        # are preserved for detailed analysis and plotting.
        dataset_size = len(test_loader.dataset)
        if config['data'].get('use_all_test_samples', False):
            logger.info(f"📊 Using FULL inference for complete analysis ({dataset_size:,} samples)")

        bayesian_results, test_df = trainer.bayesian_inference_total_uncertainty(
            model, test_loader, num_samples=num_samples
        )
        
        # Extract outputs and targets for metrics calculation
        test_outputs = torch.stack([
            bayesian_results['mean'],
            bayesian_results['total_std']
        ], dim=1)
        test_targets = bayesian_results['targets']
        
        # Save the complete results DataFrame (only when we just ran inference)
        try:
            if test_df is not None and not test_df.empty:
                test_df.to_csv(results_path, index=False)
                logger.info(f"💾 Saved complete inference results: {results_path}")
                logger.info(f"📊 DataFrame shape: {test_df.shape} (rows, columns)")
            else:
                logger.warning("⚠️ No DataFrame data to save (empty or None)")
        except Exception as e:
            logger.warning(f"❌ Failed to save inference results DataFrame: {e}")
    else:
        # We loaded from cache, targets already extracted above
        pass
    
    # Calculate metrics
    metrics = calculate_metrics(test_outputs, test_targets, prefix="test")
    
    # Use the same output directory structure as during training
    # Set config['output_dir'] to match training behavior
    config['output_dir'] = experiment_dir
    
    # Save summary
    summary_path = os.path.join(experiment_dir, 'inference_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"CONFIG-BASED INFERENCE SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Experiment: {os.path.basename(experiment_dir)}\n")
        f.write(f"Model type: {config['model']['model_type']}\n")
        f.write(f"Checkpoint: {os.path.basename(checkpoint_path)}\n")
        f.write(f"Number of samples: {len(test_targets)}\n\n")
        f.write("METRICS:\n")
        f.write("-"*20 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    # Create test_metrics directory for plots (same as during training)
    # Note: plot_test_metrics() will automatically append 'test_metrics' to the path,
    # so we pass the base experiment directory
    
    # Generate plots using existing plot functions
    try:
        # plot_test_metrics automatically creates test_metrics subdirectory
        # and includes comprehensive uncertainty analysis for Bayesian models
        plot_test_metrics(test_df, output_dir=experiment_dir, 
                         feature_registry=config.get('feature_registry'))
        
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")
    
    # NEW: Temporal split analysis
    try:
        logger.info("Performing temporal split analysis...")
        
        # Simple split: May 2024 and later = extrapolation, everything before = interpolation
        interpolation_df, extrapolation_df, split_info = split_test_data_by_date(test_df)
        
        # Calculate and save metrics for each subset
        temporal_metrics = save_temporal_split_metrics(interpolation_df, extrapolation_df, split_info, experiment_dir)
        
        # Generate separate plots for each subset if they have sufficient data
        if len(interpolation_df) > 1000:  # Minimum threshold for meaningful plots
            try:
                # plot_test_metrics automatically appends 'test_metrics', so we pass the base directory
                interpolation_base_dir = os.path.join(experiment_dir, 'interpolation')
                plot_test_metrics(interpolation_df, output_dir=interpolation_base_dir, 
                                feature_registry=config.get('feature_registry'))
                logger.info(f"Generated plots for interpolation data")
            except Exception as e:
                logger.warning(f"Could not generate plots for interpolation: {e}")
        
        if len(extrapolation_df) > 1000:  # Minimum threshold for meaningful plots
            try:
                # plot_test_metrics automatically appends 'test_metrics', so we pass the base directory
                extrapolation_base_dir = os.path.join(experiment_dir, 'extrapolation')
                plot_test_metrics(extrapolation_df, output_dir=extrapolation_base_dir,
                                feature_registry=config.get('feature_registry'))
                logger.info(f"Generated plots for extrapolation data")
            except Exception as e:
                logger.warning(f"Could not generate plots for extrapolation: {e}")
        
        # Log summary of temporal split
        logger.info("Temporal split analysis completed:")
        logger.info(f"  Total samples: {split_info['total_samples']:,}")
        logger.info(f"  Interpolation: {split_info['interpolation_samples']:,} ({split_info['interpolation_percentage']:.1f}%)")
        logger.info(f"  Extrapolation: {split_info['extrapolation_samples']:,} ({split_info['extrapolation_percentage']:.1f}%)")
            
    except Exception as e:
        logger.warning(f"Temporal split analysis failed: {e}")
    
    return metrics, test_df

def main():
    """Main function."""
    logger.info("Starting config-based inference...")
    
    # Setup
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        # Load config using the existing parser (respecting command line args)
        config = parse_config()
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config['device'] = device
        
        logger.info(f"Config: {config['mode']} | {config['model']['model_type']} | Device: {device}")
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    try:
        # Generate expected experiment name using existing function
        experiment_name = compute_exp_name(config)
        logger.info(f"Looking for experiment: {experiment_name}")
        
        # Find experiment directory
        experiment_dir = find_experiment_directory(experiment_name)
        
        if experiment_dir is None:
            logger.error(f"❌ EXPERIMENT NOT FOUND: {experiment_name}")
            logger.error(f"Available experiments:")
            experiments_dir = 'experiments'
            if os.path.exists(experiments_dir):
                for exp in os.listdir(experiments_dir):
                    if os.path.isdir(os.path.join(experiments_dir, exp)):
                        logger.error(f"  - {exp}")
            logger.error(f"Please train the model first or check your config.yaml settings.")
            return 1
        
        # Find model checkpoint
        checkpoint_path = find_model_checkpoint(experiment_dir, config)
        
        # Run inference
        metrics, test_df = run_inference_pipeline(config, experiment_dir, checkpoint_path)
        
        logger.info(f"✅ INFERENCE COMPLETED!")
        logger.info(f"Experiment: {os.path.basename(experiment_dir)}")
        logger.info(f"Results: {experiment_dir}/")
        for k, v in metrics.items():
            if 'mae' in k.lower() or 'mse' in k.lower() or 'rmse' in k.lower():
                logger.info(f"  {k}: {v:.4f}")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"❌ MODEL NOT FOUND: {e}")
        logger.error(f"Please train the model first using: python src/main.py")
        return 1
        
    except Exception as e:
        logger.error(f"❌ INFERENCE FAILED: {e}")
        return 1

if __name__ == '__main__':
    main()
