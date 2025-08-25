#!/usr/bin/env python3
"""
Config-based inference script.
Loads the model that matches the current config in config.yaml,
finds the corresponding experiment folder, and runs inference.
If the model doesn't exist, throws an error.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import glob
from tqdm import tqdm
import logging
import yaml

# Add the parent directory to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_parser import parse_config
from utils.data import get_data_loaders
from utils.feature_registry import initialize_feature_registry
from utils.metrics import calculate_metrics
from utils.plot import plot_test_metrics, plot_comprehensive_uncertainty_analysis
from model.model import get_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def generate_experiment_name(config):
    """Generate experiment name from config parameters."""
    mode = config.get('mode', 'pretrain').capitalize()
    target = config.get('target', 'stec').upper()
    
    # Model configuration
    model_type = config['model']['model_type']
    hidden_dim = config['model']['hidden_dim']
    num_layers = config['model']['num_layers']
    
    # Training configuration
    if mode.lower() == 'pretrain':
        epochs = config['pretrain']['epochs']
        batch_size = config['pretrain']['batchsize']
        learning_rate = config['pretrain']['learning_rate']
        scheduler = config['pretrain']['scheduler']
    else:  # finetune
        epochs = config['finetune']['epochs']
        batch_size = config['pretrain']['batchsize']  # Usually same as pretrain
        learning_rate = config['finetune']['learning_rate']
        scheduler = config['finetune']['scheduler']
    
    # Handle loss function name mapping
    loss_function = config['training']['loss_function']
    # Map full names to abbreviated versions used in experiment names
    loss_mapping = {
        'MSELoss': 'MSE',
        'MAELoss': 'MAE', 
        'GaussianNLLLoss': 'GNLL',
        'MSE': 'MSE',
        'MAE': 'MAE',
        'GNLL': 'GNLL'
    }
    loss_function = loss_mapping.get(loss_function, loss_function)
    
    optimizer = config['training']['optimizer']
    
    # Data configuration
    train_subset = config['data'].get('train_subset_size', 500000)
    sh_degree = config['data'].get('SH_degree', 0)
    loss_weight = config['training'].get('loss_weight', 1.0)
    use_swi = config['data'].get('use_SWI', True)
    
    # Format learning rate for filename
    lr_str = f"{learning_rate:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
    if 'e' not in lr_str:
        lr_str = f"{learning_rate:.3f}".rstrip('0').rstrip('.')
    
    # Format loss weight
    lw_str = f"{loss_weight:.0e}".replace('e-0', 'e-').replace('e+0', 'e+')
    if 'e' not in lw_str:
        lw_str = f"{loss_weight:.1f}".rstrip('0').rstrip('.')
    
    # Format subset size
    if train_subset >= 1000000:
        subset_str = f"{train_subset//1000000}M"
    elif train_subset >= 1000:
        subset_str = f"{train_subset//1000}K"
    else:
        subset_str = str(train_subset)
    
    # Handle scheduler name mapping
    scheduler_mapping = {
        'CosineAnnealingLR': 'CosineAnnealingLR',
        'StepLR': 'StepLR',
        'none': 'noSch',
        'None': 'noSch',
        None: 'noSch'
    }
    scheduler = scheduler_mapping.get(scheduler, scheduler)
    
    # Build experiment name
    experiment_name = (
        f"{mode}_{target}_{model_type}_"
        f"h{hidden_dim}_l{num_layers}_"
        f"lr{lr_str}_bs{batch_size}_"
        f"{loss_function}_{optimizer}_"
        f"{scheduler}_"
        f"sub{subset_str}_"
        f"SH{sh_degree}_"
        f"lw{lw_str}_"
        f"{'SWI' if use_swi else 'noSWI'}"
    )
    
    return experiment_name

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
    """Run the complete inference pipeline."""
    logger.info(f"Running inference...")
    
    # Setup device
    device = config['device']
    
    # Initialize feature registry
    feature_registry = initialize_feature_registry(config)
    config['feature_registry'] = feature_registry
    
    # Load model
    model = get_model(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(config, logger)
    
    # Run inference
    model_type = config['model']['model_type']
    is_bayesian = 'BNN' in model_type
    
    if is_bayesian:
        test_outputs, test_targets, test_df = run_bayesian_inference(
            model, test_loader, config, num_samples=100
        )
    else:
        test_outputs, test_targets = run_standard_inference(model, test_loader, config)
        test_df = create_results_dataframe(test_outputs, test_targets, config)
    
    # Calculate metrics
    metrics = calculate_metrics(test_outputs, test_targets, prefix="test")
    
    # Save results
    output_dir = os.path.join(experiment_dir, 'config_inference_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    results_path = os.path.join(output_dir, 'inference_results.csv')
    test_df.to_csv(results_path, index=False)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'inference_summary.txt')
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
    
    # Generate plots
    try:
        plot_test_metrics(test_df, output_dir=output_dir, 
                         feature_registry=config.get('feature_registry'))
        
        # Generate uncertainty analysis if available
        required_cols = ['pred_epistemic_unc', 'pred_aleatoric_unc', 'pred_total_unc']
        if all(col in test_df.columns for col in required_cols):
            plot_comprehensive_uncertainty_analysis(test_df, output_dir)
        
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")
    
    return metrics, test_df

def run_standard_inference(model, test_loader, config):
    """Run standard inference for non-Bayesian models."""
    model.eval()
    all_outputs = []
    all_targets = []
    
    device = config['device']
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Inference"):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                pred_mean, pred_var = outputs
                pred_std = torch.sqrt(torch.clamp(pred_var, min=1e-6))
            elif outputs.dim() > 1 and outputs.shape[-1] == 2:
                pred_mean = outputs[:, 0]
                pred_var = torch.nn.functional.softplus(outputs[:, 1]) + 1e-6
                pred_std = torch.sqrt(pred_var)
            else:
                pred_mean = outputs.squeeze()
                pred_std = torch.ones_like(pred_mean) * 0.1
            
            # Store results
            output_tensor = torch.stack([pred_mean, pred_std], dim=1)
            all_outputs.append(output_tensor.cpu())
            all_targets.append(targets.cpu())
    
    test_outputs = torch.cat(all_outputs)
    test_targets = torch.cat(all_targets)
    
    return test_outputs, test_targets

def run_bayesian_inference(model, test_loader, config, num_samples=100):
    """Run Bayesian inference for uncertainty quantification."""
    model.eval()
    all_outputs = []
    all_targets = []
    final_df = pd.DataFrame()
    
    device = config['device']
    feature_registry = config.get('feature_registry')
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Bayesian Inference"):
            bs = inputs.shape[0]
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Collect multiple samples
            sample_predictions = []
            sample_vars = []
            
            for _ in range(num_samples):
                outputs = model(inputs)
                
                if isinstance(outputs, tuple):
                    pred_mean, pred_var = outputs
                elif outputs.dim() > 1 and outputs.shape[-1] == 2:
                    pred_mean = outputs[:, 0]
                    pred_var = torch.nn.functional.softplus(outputs[:, 1]) + 1e-6
                else:
                    pred_mean = outputs.squeeze()
                    pred_var = torch.ones_like(pred_mean) * 1e-6
                
                sample_predictions.append(pred_mean.cpu())
                sample_vars.append(pred_var.cpu())
            
            # Calculate uncertainties
            sample_predictions = torch.stack(sample_predictions, dim=0)
            sample_vars = torch.stack(sample_vars, dim=0)
            
            pred_mean = sample_predictions.mean(dim=0)
            epistemic_var = sample_predictions.var(dim=0) if num_samples > 1 else torch.zeros_like(pred_mean)
            aleatoric_var = sample_vars.mean(dim=0)
            total_var = epistemic_var + aleatoric_var
            
            epistemic_std = torch.sqrt(epistemic_var)
            aleatoric_std = torch.sqrt(aleatoric_var)
            total_std = torch.sqrt(total_var)
            
            # Store results
            output_tensor = torch.stack([pred_mean, total_std], dim=1)
            all_outputs.append(output_tensor)
            all_targets.append(targets.cpu())
            
            # Create detailed dataframe
            inputs_original = inputs.cpu()  # Simplified - would need inverse transformation
            feature_order = [f"feature_{i}" for i in range(inputs.shape[1])]
            if feature_registry:
                try:
                    # Get feature names from registry
                    feature_order = []
                    for feature_type in feature_registry.feature_types:
                        features = feature_registry.get_features_by_type(feature_type)
                        feature_order.extend(features)
                except:
                    pass
            
            batch_df = pd.DataFrame(
                torch.cat([
                    inputs_original,
                    targets.cpu().view(bs, -1),
                    pred_mean.cpu().view(bs, -1),
                    epistemic_std.cpu().view(bs, -1),
                    aleatoric_std.cpu().view(bs, -1),
                    total_std.cpu().view(bs, -1)
                ], dim=1).numpy(),
                columns=[
                    *feature_order[:inputs.shape[1]],  # Ensure we don't exceed actual features
                    'target_stec', 'pred_stec',
                    'pred_epistemic_unc', 'pred_aleatoric_unc', 'pred_total_unc'
                ]
            )
            final_df = pd.concat([final_df, batch_df], ignore_index=True)
    
    test_outputs = torch.cat(all_outputs)
    test_targets = torch.cat(all_targets)
    
    return test_outputs, test_targets, final_df

def create_results_dataframe(test_outputs, test_targets, config):
    """Create results dataframe for non-Bayesian models."""
    predictions = test_outputs[:, 0].numpy().flatten()
    uncertainties = test_outputs[:, 1].numpy().flatten()
    targets = test_targets.numpy().flatten()
    
    df = pd.DataFrame({
        'target_stec': targets,
        'pred_stec': predictions,
        'pred_total_unc': uncertainties,
        'pred_epistemic_unc': uncertainties * 0.1,  # Dummy values
        'pred_aleatoric_unc': uncertainties * 0.9,  # Dummy values
    })
    
    return df

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
        # Generate expected experiment name from config
        experiment_name = generate_experiment_name(config)
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
        logger.info(f"Results: {experiment_dir}/config_inference_results/")
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
    import sys
    sys.exit(main())
