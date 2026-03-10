#!/usr/bin/env python3
"""
Add Pretrained STEC Baseline to Existing Daily Evaluations

Reads existing daily evaluation CSVs (STEC, VTEC+Mapping, GIM predictions),
runs pretrained model inference for each day, and saves extended results
to a separate output directory — preserving the originals untouched.

Usage:
    python src/add_pretrained_evaluation.py \
        --dates "2024-183:2024-189" \
        --source_dir multiday_results/mao_evaluation \
        --output_dir multiday_results/with_pretrained_baseline \
        --pretrained_baseline experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_... \
        --stec_config config/config.yaml
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
from typing import Dict, List, Tuple, Optional
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_parser import parse_config
from utils.feature_registry import initialize_feature_registry
from training.base_trainer import BaseTrainer
from data_loader import get_test_data_loader
from data_loader.collation import CollateWithSH
from compare_stec_vtec_gim import (
    load_experiment_config,
    find_best_checkpoint,
    load_model_from_checkpoint,
    compute_metrics,
)
from evaluation.publication_plots import generate_all_plots
from multiday_evaluation import (
    generate_date_list,
    collect_existing_results,
    generate_aggregate_report,
    extract_metrics_from_experiment,
    extract_elevation_metrics_from_experiment,
)

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def run_pretrained_inference(
    model: torch.nn.Module,
    test_loader,
    config: Dict,
    logger,
    num_samples: int = 100,
) -> pd.DataFrame:
    """Run inference with the pretrained model."""
    logger.info("Running Pretrained STEC Baseline inference...")

    trainer = BaseTrainer(config, logger)

    model_type = config["model"]["model_type"]
    is_bayesian = "BNN" in model_type or "Bayesian" in model_type or "FactorizedSTEC" in model_type
    samples = num_samples if is_bayesian else 1

    if not is_bayesian and num_samples > 1:
        logger.info(f"  {model_type} is deterministic — using 1 sample instead of {num_samples}")

    _, test_df = trainer.bayesian_inference_total_uncertainty(
        model, test_loader, num_samples=samples
    )

    logger.info(f"  Pretrained inference done: {len(test_df):,} predictions")
    return test_df


def compare_all_models_from_df(df: pd.DataFrame, logger) -> Dict[str, Dict[str, float]]:
    """Compute metrics for all model columns present in the dataframe."""
    ground_truth = df["true_stec"].values
    results = {}

    # Direct STEC model
    if "stec_pred" in df.columns:
        results["Direct STEC Model"] = compute_metrics(df["stec_pred"].values, ground_truth)

    # Pretrained STEC baseline
    if "pretrained_stec_pred" in df.columns:
        results["Pretrained STEC"] = compute_metrics(df["pretrained_stec_pred"].values, ground_truth)

    # VTEC + Mapping
    if "vtec_model_stec" in df.columns:
        results["VTEC + Mapping"] = compute_metrics(df["vtec_model_stec"].values, ground_truth)

    # IGS GIM
    if "gim_stec" in df.columns:
        gim_pred = df["gim_stec"].values
        gim_mask = ~np.isnan(gim_pred)
        results["IGS GIM"] = compute_metrics(gim_pred, ground_truth, mask=gim_mask)

    # Print summary
    logger.info("=" * 70)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 70)
    for model_name, m in results.items():
        logger.info(f"  {model_name}: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}  Bias={m['bias']:.4f}")
    logger.info("=" * 70)

    return results


def save_extended_results(
    metrics: Dict,
    df: pd.DataFrame,
    output_dir: Path,
    logger,
):
    """Save metrics and extended CSV to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics summary CSV
    rows = []
    for model_name, m in metrics.items():
        rows.append({
            "Model": model_name,
            "RMSE": m["rmse"],
            "MAE": m["mae"],
            "R²": m["r2"],
            "Bias": m["bias"],
            "Std": m["std"],
            "Count": m["count"],
        })
    pd.DataFrame(rows).to_csv(output_dir / "metrics_summary.csv", index=False)

    # Detailed predictions CSV
    csv_cols = ["true_stec", "stec_pred", "elevation"]
    if "pretrained_stec_pred" in df.columns:
        csv_cols.append("pretrained_stec_pred")
    if "vtec_model_stec" in df.columns:
        csv_cols.append("vtec_model_stec")
    if "gim_stec" in df.columns:
        csv_cols.append("gim_stec")

    df[csv_cols].to_csv(output_dir / "detailed_predictions.csv", index=False)

    # Text summary
    with open(output_dir / "comparison_summary.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("STEC Model Comparison Results (with Pretrained Baseline)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Samples: {len(df):,}\n\n")
        for model_name, m in metrics.items():
            f.write(f"{model_name}:\n")
            for k, v in m.items():
                f.write(f"  {k}: {v:,.6f}\n" if k != "count" else f"  {k}: {v:,}\n")
            f.write("\n")

    logger.info(f"  Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Add pretrained STEC baseline evaluation to existing daily results",
    )
    parser.add_argument("--dates", type=str, required=True,
                        help="Date(s) to process (e.g. '2024-183:2024-189')")
    parser.add_argument("--source_dir", type=str, default="multiday_results/mao_evaluation",
                        help="Directory with existing daily evaluations (default: multiday_results/mao_evaluation)")
    parser.add_argument("--output_dir", type=str, default="multiday_results/with_pretrained_baseline",
                        help="Output directory for extended results (default: multiday_results/with_pretrained_baseline)")
    parser.add_argument("--pretrained_baseline", type=str, required=True,
                        help="Path to pretrained STEC experiment folder")
    parser.add_argument("--stec_config", type=str, required=True,
                        help="Base STEC config file (for creating daily data loaders)")
    parser.add_argument("--num_inference_samples", type=int, default=100,
                        help="MC samples for Bayesian inference (default: 100)")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--dataset_type", type=str, default="own_vtec_gim",
                        help="Dataset subfolder to read from source (default: own_vtec_gim)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip days that already have pretrained results in output_dir")

    args = parser.parse_args()
    logger = setup_logging()

    source_dir = Path(args.source_dir)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    dates = generate_date_list(args.dates)
    logger.info(f"Processing {len(dates)} day(s): {dates}")

    # --- Load pretrained model ONCE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading pretrained model from: {args.pretrained_baseline}")

    pre_config, pre_dir = load_experiment_config(args.pretrained_baseline)
    if "data" in pre_config:
        pre_config["data"]["use_agg_h5"] = False
    pre_config["device"] = device

    # Increase batch size for speed
    for section in ("finetune", "pretrain"):
        if section in pre_config:
            pre_config[section]["batchsize"] = max(pre_config[section].get("batchsize", 2048), 4096)

    pre_checkpoint = find_best_checkpoint(pre_dir)
    pre_model, _ = load_model_from_checkpoint(pre_config, pre_checkpoint, logger)
    logger.info("Pretrained model loaded.\n")

    # --- Prepare base STEC config for daily data loaders ---
    with open(args.stec_config, "r") as f:
        base_stec_config = yaml.safe_load(f)

    base_stec_config["mode"] = "finetune"
    base_stec_config["device"] = device
    if "data" not in base_stec_config:
        base_stec_config["data"] = {}
    base_stec_config["data"]["use_agg_h5"] = False
    base_stec_config["data"]["test_size"] = "full"

    # Increase batch size
    for section in ("finetune", "pretrain"):
        if section in base_stec_config:
            base_stec_config[section]["batchsize"] = max(base_stec_config[section].get("batchsize", 2048), 4096)

    # Initialize feature registry once (only depends on feature set, not day)
    stec_registry = initialize_feature_registry(base_stec_config)
    base_stec_config["feature_registry"] = stec_registry
    CollateWithSH(base_stec_config)  # Sets output_indices in registry

    # --- Process each day ---
    batch_results = []

    for year, doy in dates:
        date_str = f"{year}-{doy:03d}"
        logger.info("=" * 70)
        logger.info(f"Processing {date_str}")
        logger.info("=" * 70)

        # Check source CSV
        source_csv = source_dir / f"{year}_DOY_{doy:03d}" / "evaluation" / args.dataset_type / "detailed_predictions.csv"
        if not source_csv.exists():
            logger.warning(f"  Source CSV not found: {source_csv}, skipping.")
            batch_results.append({"year": year, "doy": doy, "date": date_str, "success": False, "metrics": {}})
            continue

        # Check if already done
        out_eval_dir = output_base / f"{year}_DOY_{doy:03d}" / "evaluation" / args.dataset_type
        if args.skip_existing and (out_eval_dir / "detailed_predictions.csv").exists():
            existing_df = pd.read_csv(out_eval_dir / "detailed_predictions.csv")
            if "pretrained_stec_pred" in existing_df.columns:
                logger.info(f"  Already has pretrained results, skipping.")
                metrics = extract_metrics_from_experiment(output_base / f"{year}_DOY_{doy:03d}" / "evaluation")
                batch_results.append({
                    "year": year, "doy": doy, "date": date_str, "success": True,
                    "stec_experiment": f"recovered_{date_str}", "vtec_experiment": None,
                    "metrics": metrics,
                })
                continue

        # Read existing predictions
        existing_df = pd.read_csv(source_csv)
        logger.info(f"  Loaded {len(existing_df):,} samples from source CSV")

        # Set day in config and create test data loader
        day_config = {**base_stec_config}
        day_config["year"] = year
        day_config["doy"] = doy
        if "finetune" in day_config:
            day_config["finetune"] = {**base_stec_config["finetune"], "year": year, "doy": doy}
        if "pretrain" in day_config:
            day_config["pretrain"] = {**base_stec_config["pretrain"]}

        test_loader = get_test_data_loader(day_config, logger)

        # Run pretrained inference
        pre_df = run_pretrained_inference(
            pre_model, test_loader, pre_config, logger,
            num_samples=args.num_inference_samples,
        )

        if len(pre_df) != len(existing_df):
            logger.warning(
                f"  Size mismatch: pretrained={len(pre_df)}, existing={len(existing_df)}. "
                f"Skipping {date_str}."
            )
            batch_results.append({"year": year, "doy": doy, "date": date_str, "success": False, "metrics": {}})
            continue

        # Merge pretrained predictions into existing dataframe
        existing_df["pretrained_stec_pred"] = pre_df["pred_stec"].values

        # Rename elevation to match internal convention if needed
        if "elevation" in existing_df.columns and "satele" not in existing_df.columns:
            existing_df.rename(columns={"elevation": "satele"}, inplace=True)

        # Compute metrics
        metrics = compare_all_models_from_df(
            existing_df.rename(columns={"satele": "elevation"} if "satele" in existing_df.columns else {}),
            logger,
        )

        # Save extended results
        # Ensure columns use 'elevation' for output
        save_df = existing_df.copy()
        if "satele" in save_df.columns:
            save_df.rename(columns={"satele": "elevation"}, inplace=True)

        save_extended_results(metrics, save_df, out_eval_dir, logger)

        # Generate plots
        if not args.skip_plots:
            logger.info("  Generating plots...")
            # generate_all_plots expects 'satele' column
            plot_df = existing_df.copy()
            if "elevation" in plot_df.columns and "satele" not in plot_df.columns:
                plot_df.rename(columns={"elevation": "satele"}, inplace=True)

            generate_all_plots(
                test_df=plot_df,
                stec_col="stec_pred",
                vtec_col="vtec_model_stec" if "vtec_model_stec" in plot_df.columns else None,
                gim_col="gim_stec" if "gim_stec" in plot_df.columns else None,
                metrics=metrics,
                output_dir=out_eval_dir,
                logger=logger,
                pretrain_col="pretrained_stec_pred",
            )

        # Record result for aggregate (read back from saved CSV for consistent format)
        saved_metrics = extract_metrics_from_experiment(
            output_base / f"{year}_DOY_{doy:03d}" / "evaluation"
        )
        batch_results.append({
            "year": year,
            "doy": doy,
            "date": date_str,
            "success": True,
            "stec_experiment": f"recovered_{date_str}",
            "vtec_experiment": None,
            "metrics": saved_metrics,
        })

        # Free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"  Done: {date_str}\n")

    # --- Aggregate report ---
    success_count = sum(1 for r in batch_results if r.get("success"))
    logger.info("=" * 70)
    logger.info(f"COMPLETE: {success_count}/{len(batch_results)} days successful")
    logger.info("=" * 70)

    if success_count > 0:
        generate_aggregate_report(batch_results, output_base, skip_plots=args.skip_plots)

    logger.info(f"\nAll results saved to: {output_base}")


if __name__ == "__main__":
    main()
