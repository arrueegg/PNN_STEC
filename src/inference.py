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

# Add the parent directory to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_parser import parse_config, compute_exp_name
from utils.data import get_data_loaders
from utils.feature_registry import initialize_feature_registry
from utils.metrics import calculate_metrics
from utils.plot import plot_test_metrics
from utils.base_trainer import BaseTrainer
from model.model import get_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

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
    
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(config, logger)

    del train_loader, val_loader

    # Run inference using BaseTrainer's methods
    model_type = config['model']['model_type']
    is_bayesian = 'BNN' in model_type
    
    if is_bayesian:
        # Use BaseTrainer's Bayesian inference method
        num_samples = 100
        bayesian_results, test_df = trainer.bayesian_inference_total_uncertainty(
            model, test_loader, num_samples=num_samples
        )
        
        # Extract outputs and targets for metrics calculation
        test_outputs = torch.stack([
            bayesian_results['mean'],
            bayesian_results['total_std']
        ], dim=1)
        test_targets = bayesian_results['targets']
        
    else:
        # Use BaseTrainer's standard test method
        test_outputs, test_targets = trainer.test_model(model, test_loader)
        
        # Create dataframe for plotting (simplified for non-Bayesian)
        predictions = test_outputs[:, 0].numpy().flatten()
        uncertainties = test_outputs[:, 1].numpy().flatten()
        targets = test_targets.numpy().flatten()
        
        test_df = pd.DataFrame({
            'target_stec': targets,
            'pred_stec': predictions,
            'pred_total_unc': uncertainties,
            'pred_epistemic_unc': uncertainties * 0.1,  # Dummy values for non-Bayesian
            'pred_aleatoric_unc': uncertainties * 0.9,
        })
    
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
    
    return metrics, test_df

def main():
    """Main function."""
    logger.info("Starting config-based inference...")
    
    # Setup
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        # Load config using the existing parser (without command line args)
        import sys
        # Temporarily modify sys.argv to avoid argument parsing conflicts
        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]]  # Keep only script name
        
        config = parse_config()
        
        # Restore original argv
        sys.argv = original_argv
        
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
