#!/usr/bin/env python3
"""
Streamlined Inference Script for PNN_STEC Project

This script performs inference on pre-trained neural network models by reusing
the existing refactored codebase infrastructure.

Key Features:
- Reuses BaseTrainer which includes InferenceManager internally
- Leverages existing model loading and uncertainty quantification methods
- Uses existing configuration and feature registry systems
- Automatically finds and loads trained models
- Generates comprehensive analysis using existing plotting systems

Usage:
    python src/inference_testset_refactored.py

Requirements:
- Model must be already trained and saved in experiments/ directory
- config/config.yaml must match an existing experiment

Output:
- All results saved to: experiments/<experiment_name>/
"""

import torch
import os
import sys
import logging
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_parser import parse_config, compute_exp_name
from utils.feature_registry import initialize_feature_registry
from training.base_trainer import BaseTrainer
from data_loader import get_test_data_loader
from model.model import get_model
from viz import plot_test_metrics
from utils.metrics import calculate_metrics


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def find_experiment_directory(experiment_name, base_dir="experiments"):
    """Find the experiment directory that matches the generated name exactly."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Experiments directory not found: {base_dir}")

    exact_path = os.path.join(base_dir, experiment_name)
    if os.path.exists(exact_path):
        return exact_path

    return None


def find_model_checkpoint(experiment_dir):
    """Find the model checkpoint in the experiment directory."""
    model_dir = os.path.join(experiment_dir, "model")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Look for any .pth file
    pth_files = list(Path(model_dir).glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No model checkpoint found in {model_dir}")

    return str(pth_files[0])


def run_inference_analysis(config, experiment_dir, model_path, logger):
    """Run complete inference analysis using existing infrastructure."""

    # Initialize feature registry
    feature_registry = initialize_feature_registry(config)

    # Add feature registry to config (required by BaseTrainer)
    config["feature_registry"] = feature_registry

    # Get test dataloader using existing function
    test_loader = get_test_data_loader(config, logger)

    # Create trainer which includes all managers
    trainer = BaseTrainer(config, logger)

    # Load the model (handles single or ensemble)
    from model.model import load_model_for_inference
    model = load_model_for_inference(config, experiment_dir, logger)

    # Run inference
    print("")
    logger.info("🚀 Running test evaluation...")
    trainer.test_model(model, test_loader)

    # Run Bayesian inference for uncertainty quantification
    logger.info("🧠 Running Bayesian inference for uncertainty analysis...")
    # Determine if model has Bayesian layers requiring MC sampling
    model_type = config["model"]["model_type"]
    is_bayesian = "BNN" in model_type or "Bayesian" in model_type or "FactorizedSTEC" in model_type
    num_mc_samples = 100 if is_bayesian else 1
    logger.info(f"Using {num_mc_samples} MC samples for model type: {model_type}")
    
    bayesian_results, test_df = trainer.bayesian_inference_total_uncertainty(
        model,
        test_loader,
        num_samples=num_mc_samples,
    )

    # Generate all plots and analysis using existing methods
    print("")
    logger.info("📊 Generating comprehensive analysis...")

    # Get scenario evaluation setting from config (default to False to save runtime)
    enable_scenarios = config.get("evaluation", {}).get("enable_scenarios", False)
    logger.info(f"Scenario-based evaluation: {'enabled' if enable_scenarios else 'disabled (saves runtime)'}")

    # Import the plot function from existing viz module
    # Generate main test plots
    plot_test_metrics(
        test_df, output_dir=experiment_dir, feature_registry=feature_registry,
        enable_scenarios=enable_scenarios
    )

    # Perform temporal split analysis
    logger.info("📅 Performing temporal split analysis...")
    interpolation_df, extrapolation_df, split_info = trainer.split_test_data_by_date(
        test_df
    )

    # SOLAR CYCLE ANALYSIS BLOCK (RESTORING)
    from evaluation.utils import get_solar_cycle_stats
    print("\n" + "="*80)
    print("      SOLAR CYCLE PERFORMANCE (TESTSET)")
    print("="*80)
    
    # Check if we have year information (requires essential_features in InferenceManager)
    if 'year' in test_df.columns:
        sc_metrics = get_solar_cycle_stats(test_df, logger)
        # Detailed logging
        if 'ACTIVE' in sc_metrics:
            logger.info(f"📊 ACTIVE (2014, 2023): RMSE={sc_metrics['ACTIVE']['rmse']:.3f}, MAE={sc_metrics['ACTIVE']['mae']:.3f}")
        if 'QUIET' in sc_metrics:
            logger.info(f"📊 QUIET (2019, 2020):  RMSE={sc_metrics['QUIET']['rmse']:.3f}, MAE={sc_metrics['QUIET']['mae']:.3f}")
    else:
        logger.warning("⚠️ Year column missing in test_df, skipping solar cycle stats.")
    print("="*80 + "\n")

    # Generate plots for interpolation and extrapolation if sufficient data
    # Disable scenario evaluation for these temporal splits
    if len(interpolation_df) > 1000:
        interpolation_dir = os.path.join(experiment_dir, "interpolation")
        plot_test_metrics(
            interpolation_df,
            output_dir=interpolation_dir,
            feature_registry=feature_registry,
            enable_scenarios=False,
        )
        logger.info("📈 Generated interpolation analysis plots")

    if len(extrapolation_df) > 1000:
        extrapolation_dir = os.path.join(experiment_dir, "extrapolation")
        plot_test_metrics(
            extrapolation_df,
            output_dir=extrapolation_dir,
            feature_registry=feature_registry,
            enable_scenarios=False,
        )
        logger.info("📈 Generated extrapolation analysis plots")

    # Calculate and log final metrics
    test_predictions = torch.stack(
        [
            torch.tensor(test_df["pred_stec"].values, dtype=torch.float32),
            torch.tensor(test_df["pred_total_unc"].values, dtype=torch.float32),
        ],
        dim=1,
    )
    test_targets = torch.tensor(test_df["target_stec"].values, dtype=torch.float32)

    metrics = calculate_metrics(test_predictions, test_targets, prefix="test")

    return metrics, test_df


def main():
    """Main inference pipeline."""
    logger = setup_logging()

    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    try:
        # Manual argument parsing to check for model_checkpoint
        import argparse
        parser = argparse.ArgumentParser(description="Inference Testset", add_help=False)
        parser.add_argument("--model_checkpoint", type=str, help="Path to specific model checkpoint")
        args, _ = parser.parse_known_args()

        if args.model_checkpoint:
            # Mode 1: Direct inference on specific model
            model_path = args.model_checkpoint
            if not os.path.exists(model_path):
                logger.error(f"❌ Model not found: {model_path}")
                return 1
            
            # Infer experiment directory (assuming structure experiments/EXP_NAME/model/model.pth)
            experiment_dir = os.path.dirname(os.path.dirname(model_path))
            
            logger.info(f"📍 Using experiment directory: {experiment_dir}")
            
            # Load config from experiment directory
            config_path = os.path.join(experiment_dir, "config.yaml")
            if os.path.exists(config_path):
                logger.info(f"📄 Loading config from: {config_path}")
                from utils.config_parser import load_config
                config = load_config(config_path)
            else:
                logger.warning(f"⚠️ Config not found in {experiment_dir}, loading from default/CLI")
                config = parse_config()

            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config["device"] = device
            config["output_dir"] = experiment_dir
            
            logger.info(
                f"🔧 Config: {config.get('mode', 'inference')} | {config['model']['model_type']} | Device: {device}"
            )
            
        else:
            # Mode 2: Standard workflow (find experiment by config name)
            # Load config
            config = parse_config()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            config["device"] = device

            logger.info(
                f"🔧 Config: {config['mode']} | {config['model']['model_type']} | Device: {device}"
            )

            # Generate experiment name
            experiment_name = compute_exp_name(config)
            logger.info(f"🔍 Looking for experiment: {experiment_name}")

            # Find experiment directory
            experiment_dir = find_experiment_directory(experiment_name)
            if experiment_dir is None:
                logger.error(f"❌ EXPERIMENT NOT FOUND: {experiment_name}")
                logger.error("Available experiments:")
                experiments_dir = "experiments"
                if os.path.exists(experiments_dir):
                    for exp in sorted(os.listdir(experiments_dir)):
                        if os.path.isdir(os.path.join(experiments_dir, exp)):
                            logger.error(f"  - {exp}")
                logger.error(
                    "Please train the model first or check your config.yaml settings."
                )
                return 1

            # Find model checkpoint
            model_path = find_model_checkpoint(experiment_dir)
        
        logger.info(f"✅ Found model: {model_path}")

        # Run complete inference analysis
        metrics, test_df = run_inference_analysis(
            config, experiment_dir, model_path, logger
        )

        # Summary
        print("")
        logger.info("✅ INFERENCE COMPLETED!")
        logger.info(f"📁 Experiment: {os.path.basename(experiment_dir)}")
        logger.info(f"📊 Samples processed: {len(test_df):,}")
        logger.info(f"📈 Results saved to: {experiment_dir}/")

        # Log key metrics
        for metric_name, value in metrics.items():
            if any(key in metric_name.lower() for key in ["mae", "mse", "rmse"]):
                logger.info(f"📊 {metric_name}: {value:.4f}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"❌ FILE NOT FOUND: {e}")
        logger.error("Please train the model first using: python src/main.py")
        return 1

    except Exception as e:
        logger.error(f"❌ INFERENCE FAILED: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
