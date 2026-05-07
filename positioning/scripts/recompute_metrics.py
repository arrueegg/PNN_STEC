#!/usr/bin/env python3
"""
Multi-Day Positioning Evaluation (Re-Aggregation Only)

Re-aggregates positioning metrics from existing .pos files using SNX ground-truth references.
Does NOT re-run positioning - only downloads SNX files and re-computes metrics.

Uses the same config-based and parallel structure as multiday_positioning.py.

Usage:
    python src/multiday_positioning_eval_only.py \\
        --stec_config config/config.yaml \\
        --vtec_config config/config_vtec_mlp_baseline.yaml \\
        --dates 2024-122:2024-366 \\
        --parallel 4
"""

import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.dates as mdates

_repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root / "positioning"))

from utils.config_parser import load_config, compute_exp_name

from positioning_eval.download_products import download_products
from positioning_eval.metrics import aggregate_daily_metrics


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def find_finetune_experiment_by_config(base_config_dict, year, doy):
    """
    Deterministically compute experiment name from base config.
    """
    config = base_config_dict.copy()
    config["mode"] = "finetune"
    config["year"] = year
    config["doy"] = doy

    # Ensure finetune section exists
    if "finetune" not in config:
        config["finetune"] = {}
    config["finetune"]["year"] = year
    config["finetune"]["doy"] = doy

    # Force use_agg_h5 False
    if "data" not in config:
        config["data"] = {}
    config["data"]["use_agg_h5"] = False

    exp_name = compute_exp_name(config)
    exp_path = Path("experiments") / exp_name

    if exp_path.exists():
        return str(exp_path)
    return None


def get_robust_limits(data, percentile=99.0):
    """Get robust axis limits excluding extreme outliers."""
    if len(data) == 0:
        return 0, 1
    return 0, np.percentile(data, percentile) * 1.2


def plot_trends(df, output_dir):
    """Generate paper-ready trend plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # UNIT CONVERSION (Standardizing to cm)
    # -------------------------------------------------------------------------
    if "3d_rms" not in df.columns:
        if "error_3d_rms" in df.columns:
            df["3d_rms"] = df["error_3d_rms"] * 100
        else:
            print("Could not find 3d_rms or error_3d_rms column")
            return

    if "2d_rms" not in df.columns and "error_2d_rms" in df.columns:
        df["2d_rms"] = df["error_2d_rms"] * 100

    if "up_rms" not in df.columns and "u_rms" in df.columns:
        df["up_rms"] = df["u_rms"] * 100

    # Common Style Settings
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "font.family": "sans-serif",
        }
    )

    # Define colors
    stec_color = "#1f77b4"  # Blue
    vtec_color = "#ff7f0e"  # Orange
    gim_color = "#2ca02c"  # Green

    # Helper to get style
    def get_style(method_name):
        m_lower = str(method_name).lower()
        if "stec" in m_lower or "direct" in m_lower:
            return stec_color, "Direct STEC", "o"
        elif "vtec" in m_lower:
            return vtec_color, "VTEC + Mapping", "s"
        elif "gim" in m_lower:
            return gim_color, "IGS GIM + Mapping", "^"
        return "gray", method_name, "x"

    # Pre-process Data
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    if "3d_rms" in df.columns and "method" in df.columns:
        # 1. High-Quality Time Series (Line Plot with Error Bands)
        # -------------------------------------------------------------------------
        plt.figure(figsize=(10, 6), dpi=300)

        daily_stats = (
            df.groupby(["date", "method"])["3d_rms"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        # Calculate standard error of the mean
        daily_stats["sem"] = daily_stats["std"] / (daily_stats["count"] ** 0.5)

        methods = daily_stats["method"].unique()

        for method in methods:
            subset = daily_stats[daily_stats["method"] == method]
            color, label, marker = get_style(method)

            plt.plot(
                subset["date"],
                subset["mean"],
                marker=marker,
                markersize=5,
                linewidth=2,
                label=label,
                color=color,
            )

            plt.fill_between(
                subset["date"],
                subset["mean"] - subset["sem"],
                subset["mean"] + subset["sem"],
                color=color,
                alpha=0.2,
            )

        plt.ylabel("3D RMS Error [cm]", fontweight="bold")
        plt.xlabel("Date", fontweight="bold")
        plt.title("Daily Positioning Performance Trend", fontweight="bold", pad=15)

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(frameon=True, framealpha=0.9, loc="best")

        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(rotation=45)

        # Robust Y-axis
        _, y_max = get_robust_limits(daily_stats["mean"], 99)
        plt.ylim(0, y_max)

        plt.tight_layout()
        plt.savefig(output_dir / "paper_trend_3d_rms_timeseries.png", dpi=300)
        plt.close()

        # 2. Daily Improvement vs GIM
        # -------------------------------------------------------------------------
        daily_pivot = daily_stats.pivot(index="date", columns="method", values="mean")

        # Locate GIM column
        gim_col = next(
            (c for c in daily_pivot.columns if "gim" in str(c).lower()), None
        )

        if gim_col:
            model_cols = [c for c in daily_pivot.columns if c != gim_col]

            if model_cols:
                plt.figure(figsize=(10, 6), dpi=300)

                for m_col in model_cols:
                    color, label, _ = get_style(m_col)

                    # Improvement: (GIM - Model) / GIM * 100
                    improvement = (
                        (daily_pivot[gim_col] - daily_pivot[m_col])
                        / daily_pivot[gim_col]
                        * 100
                    )

                    plt.plot(
                        improvement.index,
                        improvement.values,
                        marker="o",
                        markersize=4,
                        linewidth=2,
                        label=f"{label} vs GIM",
                        color=color,
                    )

                plt.axhline(0, color="black", linestyle="--", alpha=0.5)
                plt.ylabel("Improvement over IGS GIM [%]", fontweight="bold")
                plt.xlabel("Date", fontweight="bold")
                plt.title(
                    "Daily Relative Improvement in 3D Accuracy",
                    fontweight="bold",
                    pad=15,
                )
                plt.grid(True, linestyle="--", alpha=0.7)
                plt.legend()

                ax = plt.gca()
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(
                    output_dir / "paper_trend_improvement_timeseries.png", dpi=300
                )
                plt.close()

        # 3. Comparative Boxplot Distribution
        # -------------------------------------------------------------------------
        plt.figure(figsize=(8, 6), dpi=300)

        plot_df = df[df["method"].isin(methods)]

        method_colors = {method: get_style(method)[0] for method in methods}
        order = sorted(methods, key=lambda m: (str(m).lower().find("gim") >= 0, str(m)))

        sns.boxplot(
            data=plot_df,
            x="method",
            y="3d_rms",
            order=order,
            palette=method_colors,
            showfliers=True,
        )

        plt.ylabel("3D RMS Error [cm]", fontweight="bold")
        plt.xlabel("Method", fontweight="bold")
        plt.title("Error Distribution Across All Days", fontweight="bold", pad=15)
        plt.grid(True, linestyle="--", alpha=0.3, axis="y")

        # Rename x labels
        ax = plt.gca()
        labels = [get_style(t.get_text())[1] for t in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / "paper_boxplot_3d_rms_distribution.png", dpi=300)
        plt.close()


def process_day(current_date, stec_base_config, vtec_base_config, args):
    """
    Re-aggregate metrics for a single day from existing .pos files.
    Returns list of (date_obj, path, label) tuples for the consolidated report.
    """
    date_str = current_date.strftime("%Y-%m-%d")
    year = current_date.year
    doy = current_date.timetuple().tm_yday
    logger = logging.getLogger(f"Day-{doy}")

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s - %(levelname)s - [Day {doy}] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    logger.info(f"Processing {date_str}")

    # 1. Determine Experiments for this day
    stec_exp = find_finetune_experiment_by_config(stec_base_config, year, doy)
    vtec_exp = find_finetune_experiment_by_config(vtec_base_config, year, doy)

    experiments_to_run = []
    if stec_exp:
        experiments_to_run.append((stec_exp, "STEC"))
    else:
        logger.warning(f"Skipping STEC for {date_str}: Experiment not found")

    if vtec_exp:
        experiments_to_run.append((vtec_exp, "VTEC"))
    else:
        logger.warning(f"Skipping VTEC for {date_str}: Experiment not found")

    if not experiments_to_run:
        logger.error(f"No experiments found for {date_str}. Skipping.")
        return []

    day_results = []

    # Run process for each model type
    for exp_path, model_label in experiments_to_run:
        logger.info(f"--- Re-aggregating {model_label} Model ---")

        # Check if results exist
        res_path = (
            Path(exp_path)
            / "positioning"
            / "results"
            / f"{year}{doy:03d}"
            / "daily_summary.csv"
        )

        if not res_path.parent.exists():
            logger.warning(f"No results found for {model_label} (No {res_path.parent})")
            continue

        # 1. Download SNX file for this date
        products_dir = (
            Path(exp_path)
            / "positioning"
            / "evaluation"
            / f"{year}{doy:03d}"
            / "products"
        )
        products_dir.mkdir(parents=True, exist_ok=True)

        snx_file = products_dir / f"IGS0OPSSNX_{year}{doy:03d}0000_01D_01D_CRD.SNX"

        if not snx_file.exists():
            try:
                download_products(year, doy, str(products_dir), logger)
            except Exception as e:
                logger.warning(f"Could not download SNX: {e}")

        # 2. Re-aggregate metrics using SNX reference
        try:
            results_dir = (
                Path(exp_path) / "positioning" / "results" / f"{year}{doy:03d}"
            )

            # Get model label for method names
            if model_label == "STEC":
                model_method = "Direct STEC"
                gim_method = "IGS GIM + Mapping"
            else:
                model_method = "VTEC + Mapping"
                gim_method = "IGS GIM + Mapping"

            # Re-aggregate model metrics from model/ subdirectory
            logger.debug(f"Re-aggregating {model_method} metrics...")
            model_dir = results_dir / "model"
            metrics_model = (
                aggregate_daily_metrics(
                    model_dir,
                    year,
                    doy,
                    "model",
                    snx_file=snx_file if snx_file.exists() else None,
                )
                if model_dir.exists()
                else None
            )

            # Re-aggregate GIM metrics from gim/ subdirectory
            logger.debug(f"Re-aggregating {gim_method} metrics...")
            gim_dir = results_dir / "gim"
            metrics_gim = (
                aggregate_daily_metrics(
                    gim_dir,
                    year,
                    doy,
                    "gim",
                    snx_file=snx_file if snx_file.exists() else None,
                )
                if gim_dir.exists()
                else None
            )

            # Combine and save
            if metrics_model is not None or metrics_gim is not None:
                metrics_list = []
                if metrics_model is not None:
                    metrics_list.append(metrics_model)
                if metrics_gim is not None:
                    metrics_list.append(metrics_gim)

                combined = pd.concat(metrics_list, ignore_index=True)

                # Save updated summary
                combined.to_csv(res_path, index=False, float_format="%.4f")
                logger.info(f"✓ Re-aggregated {model_label} metrics for {date_str}")
                day_results.append((current_date, res_path, model_label))
            else:
                logger.warning(f"No metrics aggregated for {model_label}")

        except Exception as e:
            logger.error(f"Error re-aggregating {model_label}: {e}")
            import traceback

            traceback.print_exc()

    return day_results


def main():
    parser = argparse.ArgumentParser(
        description="Re-aggregate positioning metrics from existing .pos files with SNX ground-truth reference"
    )

    parser.add_argument(
        "--stec_config",
        required=True,
        help="Path to base STEC training config (e.g., config/config.yaml)",
    )
    parser.add_argument(
        "--vtec_config",
        required=True,
        help="Path to base VTEC training config (e.g., config/config_vtec_mlp_baseline.yaml)",
    )
    parser.add_argument(
        "--dates", required=True, help="Date range/list (e.g., 2024-122:2024-366)"
    )
    parser.add_argument(
        "--parallel", type=int, default=4, help="Number of DAYS to process in parallel"
    )

    args = parser.parse_args()
    logger = setup_logging()

    # Load Base Configs
    logger.info(f"Loading base STEC config: {args.stec_config}")
    try:
        stec_base_config = load_config(args.stec_config)
    except Exception as e:
        logger.error(f"Failed to load STEC config: {e}")
        return 1

    logger.info(f"Loading base VTEC config: {args.vtec_config}")
    try:
        vtec_base_config = load_config(args.vtec_config)
    except Exception as e:
        logger.error(f"Failed to load VTEC config: {e}")
        return 1

    # Parse Dates
    dates = []
    if ":" in args.dates:
        start_str, end_str = args.dates.split(":")

        def parse_d(d):
            if "-" in d and len(d.split("-")) == 2:
                # YYYY-DDD format
                parts = d.split("-")
                return datetime(int(parts[0]), 1, 1) + timedelta(days=int(parts[1]) - 1)
            else:
                raise ValueError(f"Expected YYYY-DDD format, got {d}")

        try:
            current = parse_d(start_str)
            end = parse_d(end_str)
            while current <= end:
                dates.append(current)
                current += timedelta(days=1)
        except ValueError as e:
            logger.error(
                f"Date format error: {e}. Use YYYY-DDD format (e.g., 2024-122:2024-366)"
            )
            return 1
    else:
        # Comma-separated list
        try:
            for d in args.dates.split(","):
                d = d.strip()
                parts = d.split("-")
                if len(parts) == 2:
                    year, doy = int(parts[0]), int(parts[1])
                    dates.append(datetime(year, 1, 1) + timedelta(days=doy - 1))
                else:
                    raise ValueError(f"Invalid format: {d}")
        except ValueError as e:
            logger.error(f"Date format error: {e}. Use YYYY-DDD format")
            return 1

    logger.info("=" * 80)
    logger.info("🚀 POSITIONING EVALUATION RE-AGGREGATION (SNX GROUND-TRUTH)")
    logger.info("=" * 80)
    logger.info(f"Processing {len(dates)} dates with {args.parallel} parallel workers")

    # Store results paths for aggregation
    daily_summary_paths = []

    # Process Days in Parallel
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(
                process_day, current_date, stec_base_config, vtec_base_config, args
            ): current_date
            for current_date in dates
        }

        for future in tqdm(
            as_completed(futures), total=len(dates), desc="Re-aggregation Progress"
        ):
            try:
                result = future.result()
                if result:
                    daily_summary_paths.extend(result)
            except Exception as e:
                date_failed = futures[future]
                logger.error(f"Failed to process {date_failed}: {e}")

    # 4. Aggregate Results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Combining Results...")
    logger.info("=" * 80)

    all_metrics = []

    for date_obj, csv_path, label in daily_summary_paths:
        try:
            df = pd.read_csv(csv_path)
            df["date"] = date_obj.strftime("%Y-%m-%d")
            df["doy"] = date_obj.timetuple().tm_yday

            # Normalize method names
            if "method" in df.columns:
                df["method"] = df["method"].replace(
                    {"model": "model", "Model": "model", "gim": "gim", "GIM": "gim"}
                )

                # Rename based on model type
                if label == "STEC":
                    df["method"] = df["method"].replace(
                        {"model": "Direct STEC", "gim": "IGS GIM + Mapping"}
                    )
                else:
                    df["method"] = df["method"].replace(
                        {"model": "VTEC + Mapping", "gim": "IGS GIM + Mapping"}
                    )

            all_metrics.append(df)
        except Exception as e:
            logger.warning(f"Error reading {csv_path}: {e}")

    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)

        # Deduplicate entries (especially for IGS GIM which appears in both STEC and VTEC runs)
        cols_to_check = ["date", "station", "method"]
        before_len = len(combined_df)
        combined_df.drop_duplicates(subset=cols_to_check, keep="first", inplace=True)
        dropped_count = before_len - len(combined_df)
        if dropped_count > 0:
            logger.info(
                f"Dropped {dropped_count} duplicate rows (mostly redundant IGS GIM entries)"
            )

        # Save to central folder
        base_output_dir = Path("multiday_results") / "positioning_snx"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        output_file = base_output_dir / "multiday_summary.csv"
        combined_df.to_csv(output_file, index=False, float_format="%.4f")
        logger.info(f"✅ Saved multi-day summary to: {output_file}")

        # Plot
        plot_trends(combined_df, base_output_dir)
        logger.info(f"Plots saved to: {base_output_dir}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ RE-AGGREGATION COMPLETED!")
        logger.info("=" * 80)

    else:
        logger.warning("No results to aggregate.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
