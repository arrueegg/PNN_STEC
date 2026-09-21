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


def _store_test_predictions(test_df, config, logger):
    """Write the multi-year test frame to the prediction store, one file per day.

    The pretrained test set spans the whole 2014-2024 record, so it is split on
    its own year/doy columns rather than written as a single file. That keeps a
    later "just DOY 132-133" read cheap.
    """
    from evaluation import prediction_store

    if "year" not in test_df.columns or "doy" not in test_df.columns:
        logger.warning(
            "⚠️  Test frame has no year/doy columns - cannot partition prediction store, skipping"
        )
        return

    root = config.get("evaluation", {}).get(
        "prediction_store_root", prediction_store.DEFAULT_STORE_ROOT
    )
    # The partition used to be chosen by mode alone, so *any* pretrain-mode run wrote into
    # `pretrained_stec` - and the R2.2 fully-Bayesian evaluation therefore replaced the
    # paper model's 544-day partition with a different architecture's predictions. RMSE
    # over the same days read 21.99 TECU against the published 13.45 before this was
    # caught. Architecture is now part of the identity, so a non-canonical model gets its
    # own partition instead of overwriting the one the paper depends on.
    CANONICAL_ARCHITECTURES = {"BayesianResNetSTEC", "MLP_LaplacianNLL"}
    base = "pretrained_stec" if config.get("mode") == "pretrain" else "finetuned_stec"
    model_type = str(config.get("model", {}).get("model_type", ""))
    if model_type and model_type not in CANONICAL_ARCHITECTURES:
        base = f"{base}_{model_type.lower()}"
    variant = config.get("evaluation", {}).get("store_variant", base)

    day_keys = list(
        zip(test_df["year"].round().astype(int), test_df["doy"].round().astype(int))
    )
    frame = test_df.assign(
        _year=[k[0] for k in day_keys], _doy=[k[1] for k in day_keys]
    )

    written = 0
    for (year, doy), day_df in frame.groupby(["_year", "_doy"], sort=True):
        try:
            prediction_store.write_predictions(
                day_df.drop(columns=["_year", "_doy"]),
                model_variant=variant,
                dataset="own",
                year=int(year),
                doy=int(doy),
                root=root,
            )
            written += 1
        except Exception as exc:
            logger.error(f"❌ Prediction store failed for {year}-{doy:03d}: {exc}")

    logger.info(
        f"💾 Prediction store: wrote {written} day(s) under {root}/{variant}/own"
    )


def run_inference_analysis(config, experiment_dir, model_path, logger):
    """Run complete inference analysis using existing infrastructure."""

    # Station/satellite identity and the cycle-slip counter are not model
    # inputs, so they only reach the results frame through the metadata channel.
    # The prediction store needs them for per-station and per-arc analysis.
    if config.get("evaluation", {}).get("write_prediction_store", True):
        config["return_metadata"] = True
        config["metadata_fields"] = ["station", "sat", "slipc", "gfphase"]

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
    is_bayesian = (
        "BNN" in model_type
        or "Bayesian" in model_type
        or "FactorizedSTEC" in model_type
    )
    num_mc_samples = 100 if is_bayesian else 1
    logger.info(f"Using {num_mc_samples} MC samples for model type: {model_type}")

    bayesian_results, test_df = trainer.bayesian_inference_total_uncertainty(
        model,
        test_loader,
        num_samples=num_mc_samples,
    )

    # Persist the frame before plotting. This is the only source of Figures 4-9
    # and it used to live in memory only, so every re-binning or new
    # stratification meant repeating the MC pass over the whole test set.
    if config.get("evaluation", {}).get("write_prediction_store", True):
        _store_test_predictions(test_df, config, logger)

    # Generate all plots and analysis using existing methods
    print("")
    logger.info("📊 Generating comprehensive analysis...")

    # Get scenario evaluation setting from config (default to False to save runtime)
    enable_scenarios = config.get("evaluation", {}).get("enable_scenarios", False)
    logger.info(
        f"Scenario-based evaluation: {'enabled' if enable_scenarios else 'disabled (saves runtime)'}"
    )

    # Import the plot function from existing viz module
    # Generate main test plots
    plot_test_metrics(
        test_df,
        output_dir=experiment_dir,
        feature_registry=feature_registry,
        enable_scenarios=enable_scenarios,
    )

    # Perform temporal split analysis
    logger.info("📅 Performing temporal split analysis...")
    interpolation_df, extrapolation_df, split_info = trainer.split_test_data_by_date(
        test_df
    )

    # SOLAR CYCLE ANALYSIS BLOCK (RESTORING)
    from evaluation.utils import get_solar_cycle_stats

    print("\n" + "=" * 80)
    print("      TEMPORAL & SOLAR CYCLE PERFORMANCE (ALL YEARS)")
    print("=" * 80)

    # Check if we have year information (requires essential_features in InferenceManager)
    if "year" in test_df.columns:
        sc_metrics = get_solar_cycle_stats(test_df, logger)

        # Save metrics to a text file for yearly/temporal analysis
        metrics_file = os.path.join(experiment_dir, "yearly_temporal_analysis.txt")
        try:
            with open(metrics_file, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("      DETAILED TEMPORAL ANALYSIS (YEAR-BY-YEAR)\n")
                f.write("=" * 80 + "\n\n")

                # 1. Grouped summaries
                f.write("📊 GROUPED PERFORMANCE:\n")
                if "ACTIVE_GROUP" in sc_metrics:
                    m = sc_metrics["ACTIVE_GROUP"]
                    line = f"ACTIVE (2014, 2024): RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}, R²={m['r2']:.4f}, count={m['count']:,}"
                    f.write(line + "\n")
                    logger.info(f"📊 {line}")

                if "QUIET_GROUP" in sc_metrics:
                    m = sc_metrics["QUIET_GROUP"]
                    line = f"QUIET  (2019, 2020): RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}, R²={m['r2']:.4f}, count={m['count']:,}"
                    f.write(line + "\n")
                    logger.info(f"📊 {line}")

                # 2. Individual years
                f.write("\n📅 INDIVIDUAL YEARLY BREAKDOWN:\n")
                f.write(f"{'-' * 52}\n")
                f.write(
                    f"{'Year':<6} | {'RMSE':<8} | {'MAE':<8} | {'R²':<8} | {'Samples':<12}\n"
                )
                f.write(f"{'-' * 52}\n")

                # Extract year-like keys reliably (both string digits and actual years)
                # We handle potential mixed types from sc_metrics keys
                all_keys = list(sc_metrics.keys())
                year_keys = []
                for k in all_keys:
                    # Filter for keys that represent years (e.g., '2014' or 2014)
                    if str(k).isdigit() and len(str(k)) == 4:
                        year_keys.append(k)

                # Sort years numerically
                year_keys.sort(key=lambda x: int(x))

                for year in year_keys:
                    m = sc_metrics[year]
                    row = f"{str(year):<6} | {m['rmse']:<8.3f} | {m['mae']:<8.3f} | {m.get('r2', 0.0):<8.4f} | {m['count']:,}"
                    f.write(row + "\n")
                    # Log representative years to console
                    if str(year) in ["2014", "2019", "2023", "2024"]:
                        logger.info(
                            f"📅 Year {year}: RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}, R²={m.get('r2', 0):.4f}"
                        )

                if not year_keys:
                    f.write("No individual years found in metrics.\n")
                    logger.warning(
                        "⚠️ No individual years found in sc_metrics dictionary!"
                    )

                f.write("\n" + "=" * 80 + "\n")
            logger.info(f"✅ Full yearly temporal analysis saved to: {metrics_file}")
        except Exception as e:
            logger.error(f"❌ Failed to write yearly analysis file: {e}")
    else:
        logger.warning("⚠️ Year column missing in test_df, skipping solar cycle stats.")
    print("=" * 80 + "\n")

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

        parser = argparse.ArgumentParser(
            description="Inference Testset", add_help=False
        )
        parser.add_argument(
            "--model_checkpoint", type=str, help="Path to specific model checkpoint"
        )
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
                logger.warning(
                    f"⚠️ Config not found in {experiment_dir}, loading from default/CLI"
                )
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
