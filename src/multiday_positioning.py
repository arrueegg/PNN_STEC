#!/usr/bin/env python3
"""
Multi-Day Positioning Evaluation Script

Efficiently runs positioning evaluation over a range of dates.
1. Generates STEC corrections for ALL days in one go (loading model once).
2. Runs PPPx positioning evaluation day-by-day (can be parallelized internally).
3. Aggregates results into a single multi-day summary.

Usage:
    python src/multiday_positioning.py \
        --config config/config.yaml \
        --dates 2024-05-01:2024-05-30 \
        --parallel 4
"""

import os
import sys
import shutil
import argparse
import logging
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import utils for experiment name resolution
try:
    from utils.config_parser import load_config, compute_exp_name
except ImportError:
    # Use relative import if running as script from src
    sys.path.append(str(Path(__file__).parent.parent))
    from src.utils.config_parser import load_config, compute_exp_name


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def find_finetune_experiment_by_config(base_config_dict, year, doy):
    """
    Deterministically compute experiment name from base config.
    """
    config = base_config_dict.copy()
    config['mode'] = 'finetune'
    config['year'] = year
    config['doy'] = doy
    
    # Ensure finetune section exists (often needed by compute_exp_name indirectly, 
    # though usually just top-level params matter)
    if 'finetune' not in config:
        config['finetune'] = {}
    config['finetune']['year'] = year
    config['finetune']['doy'] = doy
    
    # Force use_agg_h5 False as in multiday_evaluation
    if 'data' not in config:
        config['data'] = {}
    config['data']['use_agg_h5'] = False
        
    exp_name = compute_exp_name(config)
    exp_path = Path("experiments") / exp_name
    
    if exp_path.exists():
        return str(exp_path)
    return None


def run_command(cmd, description, logger):
    """Run a shell command and log output."""
    logger.info(f"Running: {description}")
    # logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True
        )
        # Verify if it actually did anything relevant by checking output
        # (Optional: print output if it was too fast)
        logger.debug(f"Command stdout: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running command: {' '.join(cmd)}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False




import matplotlib.dates as mdates

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
    if '3d_rms' not in df.columns:
        if 'error_3d_rms' in df.columns:
             df['3d_rms'] = df['error_3d_rms'] * 100
        else:
            print("Could not find 3d_rms or error_3d_rms column")
            return

    if '2d_rms' not in df.columns and 'error_2d_rms' in df.columns:
        df['2d_rms'] = df['error_2d_rms'] * 100
        
    if 'up_rms' not in df.columns and 'u_rms' in df.columns:
        df['up_rms'] = df['u_rms'] * 100

    # Common Style Settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'font.family': 'sans-serif'
    })
    
    # Define colors
    stec_color = '#1f77b4'  # Blue
    vtec_color = '#ff7f0e'  # Orange
    gim_color = '#2ca02c'   # Green
    
    # Helper to get style
    def get_style(method_name):
        m_lower = str(method_name).lower()
        
        # Determine base label and color
        if 'stec' in m_lower:
            color = stec_color
            base_label = "Direct STEC"
            marker = 'o'
        elif 'vtec' in m_lower:
            color = vtec_color
            base_label = "VTEC + Mapping"
            marker = 's'
        elif 'gim' in m_lower:
            color = gim_color
            base_label = "IGS GIM + Mapping"
            marker = '^'
        else:
            return 'gray', method_name, 'x'
            
        # Append weight option to label if present
        if "_iono" in m_lower:
            label = f"{base_label} (Unc-Weighted)"
            # Slightly vary marker/color for distinction if needed, 
            # or just use color for "STEC" vs "VTEC"
        elif "_elev" in m_lower:
             label = f"{base_label} (Elev-Weighted)"
        elif "_snr" in m_lower:
             label = f"{base_label} (SNR-Weighted)"
        else:
            label = base_label
            
        return color, label, marker

    # Pre-process Data
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    if '3d_rms' in df.columns and 'method' in df.columns:
        
        # 1. High-Quality Time Series (Line Plot with Error Bands)
        # -------------------------------------------------------------------------
        plt.figure(figsize=(10, 6), dpi=300)
        
        daily_stats = df.groupby(['date', 'method'])['3d_rms'].agg(['mean', 'std', 'count']).reset_index()
        # Calculate standard error of the mean
        daily_stats['sem'] = daily_stats['std'] / (daily_stats['count'] ** 0.5)

        methods = daily_stats['method'].unique()
        
        for method in methods:
            subset = daily_stats[daily_stats['method'] == method]
            color, label, marker = get_style(method)
            
            plt.plot(subset['date'], subset['mean'], marker=marker, markersize=5, 
                     linewidth=2, label=label, color=color)
            
            plt.fill_between(subset['date'], 
                             subset['mean'] - subset['sem'], 
                             subset['mean'] + subset['sem'], 
                             color=color, alpha=0.2)
        
        plt.ylabel('3D RMS Error [cm]', fontweight='bold')
        plt.xlabel('Date', fontweight='bold')
        plt.title('Daily Positioning Performance Trend', fontweight='bold', pad=15)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(frameon=True, framealpha=0.9, loc='best')
        
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=45)
        
        # Robust Y-axis
        _, y_max = get_robust_limits(daily_stats['mean'], 99)
        plt.ylim(0, y_max)
        
        plt.tight_layout()
        plt.savefig(output_dir / "paper_trend_3d_rms_timeseries.png", dpi=300)
        plt.close()
        
        # 2. Daily Improvement vs GIM
        # -------------------------------------------------------------------------
        daily_pivot = daily_stats.pivot(index='date', columns='method', values='mean')
        
        # Locate GIM column
        gim_col = next((c for c in daily_pivot.columns if 'gim' in str(c).lower()), None)
        
        if gim_col:
            model_cols = [c for c in daily_pivot.columns if c != gim_col]
            
            if model_cols:
                plt.figure(figsize=(10, 6), dpi=300)
                
                for m_col in model_cols:
                    color, label, _ = get_style(m_col)
                    
                    # Improvement: (GIM - Model) / GIM * 100
                    improvement = (daily_pivot[gim_col] - daily_pivot[m_col]) / daily_pivot[gim_col] * 100
                    
                    plt.plot(improvement.index, improvement.values, marker='o', markersize=4,
                             linewidth=2, label=f"{label} vs GIM", color=color)
                
                plt.axhline(0, color='black', linestyle='--', alpha=0.5)
                plt.ylabel('Improvement over IGS GIM [%]', fontweight='bold')
                plt.xlabel('Date', fontweight='bold')
                plt.title('Daily Relative Improvement in 3D Accuracy', fontweight='bold', pad=15)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                ax = plt.gca()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / "paper_trend_improvement_timeseries.png", dpi=300)
                plt.close()

        # 3. Comparative Boxplot Distribution
        # -------------------------------------------------------------------------
        plt.figure(figsize=(8, 6), dpi=300)
        
        plot_df = df[df['method'].isin(methods)] 
        
        # Manually order palette and define order
        ordered_methods = sorted(methods, key=lambda x: (
            0 if 'stec' in str(x).lower() else 
            1 if 'vtec' in str(x).lower() else 
            2
        ))
        
        palette = {m: get_style(m)[0] for m in ordered_methods}

        sns.boxplot(x='method', y='3d_rms', hue='method', data=plot_df, 
                    order=ordered_methods, palette=palette, 
                    showfliers=False, legend=False) 
        
        plt.ylabel('3D RMS Error [cm]', fontweight='bold')
        plt.xlabel('Correction Method', fontweight='bold')
        plt.title('Overall Positioning Accuracy Distribution', fontweight='bold', pad=15)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # Rename x-ticks
        current_labels = [l.get_text() for l in plt.gca().get_xticklabels()]
        new_labels = [get_style(l)[1] for l in current_labels]
        ax = plt.gca()
        ax.set_xticklabels(new_labels)

        plt.tight_layout()
        plt.savefig(output_dir / "paper_overall_distribution_boxplot.png", dpi=300)
        plt.close()

        # 4. CDF Plot
        # -------------------------------------------------------------------------
        plt.figure(figsize=(10, 6), dpi=300)
        
        robust_max = 0
        for method in ordered_methods: 
            subset = df[df['method'] == method].sort_values('3d_rms')
            data = subset['3d_rms'].values
            y = np.arange(1, len(data) + 1) / len(data) * 100 
            
            _, local_max = get_robust_limits(data, 99)
            robust_max = max(robust_max, local_max)
            
            color, label, _ = get_style(method)
            
            plt.plot(data, y, linewidth=2.5, label=label, color=color)
            
            # P95 markers
            p95 = np.percentile(data, 95)
            plt.plot([0, p95], [95, 95], linestyle=':', color=color, alpha=0.5)
            plt.plot([p95, p95], [0, 95], linestyle=':', color=color, alpha=0.5)
            
        plt.xlabel('3D RMS Error [cm]', fontweight='bold')
        plt.ylabel('Cumulative Probability [%]', fontweight='bold')
        plt.title('Error Cumulative Distribution Function (CDF)', fontweight='bold', pad=15)
        
        plt.xlim(0, robust_max * 1.1)
        plt.ylim(0, 105)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_dir / "paper_cdf_3d_rms.png", dpi=300)
        plt.close()
        
        # 5. Scatter Plot (Vertical vs Horizontal)
        # -------------------------------------------------------------------------
        if '2d_rms' in df.columns and 'up_rms' in df.columns:
            plt.figure(figsize=(8, 8), dpi=300)
            
            _, x_max = get_robust_limits(df['2d_rms'], 99.5)
            _, y_max = get_robust_limits(df['up_rms'], 99.5)
            max_limit = max(x_max, y_max)
            
            for method in ordered_methods:
                subset = df[df['method'] == method]
                color, label, _ = get_style(method)
                
                plt.scatter(subset['2d_rms'], subset['up_rms'], 
                        alpha=0.4, label=label, color=color, s=20)
                
            plt.plot([0, max_limit], [0, max_limit], 'k--', alpha=0.3, label='1:1 Ratio')
            
            plt.xlabel('2D (Horizontal) RMS Error [cm]', fontweight='bold')
            plt.ylabel('Vertical (Up) RMS Error [cm]', fontweight='bold')
            plt.title('Vertical vs Horizontal Error', fontweight='bold', pad=15)
            plt.xlim(0, max_limit)
            plt.ylim(0, max_limit)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "analysis_vertical_vs_horizontal.png", dpi=300)
            plt.close()

    else:
        print(f"Required columns (3d_rms or error_3d_rms, method) not found for plotting. Columns: {df.columns.tolist()}")


from concurrent.futures import ProcessPoolExecutor, as_completed

def process_day(current_date, stec_base_config, vtec_base_config, args):
    """
    Process a single day: Inference + Evaluation.
    Returns a list of (date_obj, path, label) tuples for the consolidated report.
    """
    date_str = current_date.strftime("%Y-%m-%d")
    year = current_date.year
    doy = current_date.timetuple().tm_yday
    logger = logging.getLogger(f"Day-{doy}")
    
    # Deterministic logger for sub-processes
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(f'%(asctime)s - %(levelname)s - [Day {doy}] %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    logger.info(f"Starting processing for {date_str}")
    
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
        logger.info(f"--- Processing {model_label} Model ---")
        
        # 1. Run Inference (Single Day)
        run_inf = not args.skip_inference
        if args.only_vtec_inference and model_label == "STEC":
            run_inf = False
            logger.info(f"Skipping STEC inference as --only_vtec_inference is set")
        if args.only_stec_inference and model_label == "VTEC":
            run_inf = False
            logger.info(f"Skipping VTEC inference as --only_stec_inference is set")

        if run_inf:
            # Check if inference output already exists
            correction_dir = Path(exp_path) / "positioning" / "stec_corrections" / f"{year}{doy:03d}"
            # REDO logic: delete and regenerate if redo is set
            if args.redo and correction_dir.exists():
                logger.info(f"Redoing inference for {model_label} on {date_str}")
                import shutil
                shutil.rmtree(correction_dir)

            if correction_dir.exists() and any(correction_dir.iterdir()):
                 logger.info(f"Skipping inference for {model_label} (Output found at {correction_dir})")
            else:
                inf_cmd = [
                    sys.executable, "src/inference_positioning.py",
                    "--experiment", exp_path,
                    "--date", date_str
                ]
                if getattr(args, 'gnss_path', None):
                    inf_cmd.extend(["--gnss_path", args.gnss_path])
                    
                if not run_command(inf_cmd, f"Inference for {model_label} on {date_str}", logger):
                    continue

        # 2. Run Positioning Evaluation for each weighting option
        for w_opt in args.weight_opts:
            current_weight_opt = w_opt
            
            # Implementation of user requirement: 
            # Determine if model supports uncertainty weighting
            # "if the model only returns predictions (MLP), use elevation weighting even if 'iono' is specified"
            model_type = str(vtec_base_config.get('model', {}).get('model_type', '')).lower()
            loss_func = str(vtec_base_config.get('training', {}).get('loss_function', '')).lower()
            
            has_uncertainty = (
                "nll" in loss_func or 
                "nll" in model_type or 
                "bnn" in model_type or 
                "ensemble" in model_type or
                "mcdropout" in model_type
            )
            
            if current_weight_opt == "iono" and not has_uncertainty and model_label == "VTEC":
                logger.info(f"VTEC model is deterministic (no uncertainty). Mapping 'iono' weighting to 'elev' for {model_label}.")
                current_weight_opt = "elev"
                # If 'elev' weighting was already processed, we can skip this to avoid duplicates
                if "elev" in args.weight_opts and args.weight_opts.index("elev") < args.weight_opts.index("iono"):
                    logger.info(f"Skipping redundant 'iono' (mapped to 'elev') for {model_label}")
                    continue

            logger.info(f"Running evaluation with weight_opt: {current_weight_opt}")
            
            # Custom results path for different weightings
            summary_filename = "daily_summary.csv"
            if current_weight_opt != "elev":
                summary_filename = f"daily_summary_{current_weight_opt}.csv"
            
            res_path = Path(exp_path) / "positioning" / "results" / f"{year}{doy:03d}" / summary_filename
            if res_path.exists() and not args.redo:
                logger.info(f"Skipping evaluation for {model_label} ({current_weight_opt}) (Results found at {res_path})")
                day_results.append((current_date, res_path, f"{model_label}_{current_weight_opt}"))
                continue
            
            station_parallel = getattr(args, 'station_parallel', 1)
            eval_cmd = [
                sys.executable, "src/positioning_eval/run_positioning_evaluation.py",
                "--experiment", exp_path,
                "--date", date_str,
                "--all_test_stations",
                "--parallel", str(station_parallel),
                "--no_cleanup",
                "--weight_opt", current_weight_opt
            ]
            
            if args.skip_downloads:
                eval_cmd.append("--skip_downloads")
            if args.redo:
                eval_cmd.append("--redo")
            
            description = f"Evaluation for {model_label} ({current_weight_opt}) on {date_str}"
            if run_command(eval_cmd, description, logger):
                if res_path.exists():
                    day_results.append((current_date, res_path, f"{model_label}_{current_weight_opt}"))
    
    return day_results


def main():
    parser = argparse.ArgumentParser(description="Multi-Day Positioning Evaluation")
    
    # Updated arguments to support STEC and VTEC models
    parser.add_argument("--stec_config", required=True, 
                        help="Path to base STEC training config (e.g., config/config.yaml)")
    parser.add_argument("--vtec_config", required=True, 
                        help="Path to base VTEC training config (e.g., config/config_vtec_mlp_baseline.yaml)")
    parser.add_argument("--dates", required=True, 
                        help="Date range/list (e.g., 2024-122:2024-130)")
    
    parser.add_argument("--parallel", type=int, default=4, help="Number of DAYS to process in parallel")
    parser.add_argument("--station_parallel", type=int, default=4, help="Number of STATIONS per day to process in parallel")
    
    parser.add_argument("--skip_inference", action="store_true", help="Skip STEC/VTEC generation step")
    parser.add_argument("--only_vtec_inference", action="store_true", help="Only generate corrections for VTEC models (skip STEC inference)")
    parser.add_argument("--only_stec_inference", action="store_true", help="Only generate corrections for STEC models (skip VTEC inference)")
    parser.add_argument("--skip_downloads", action="store_true", help="Skip GNSS product/RINEX downloads")
    parser.add_argument("--no_cleanup", action="store_true", help="Do not delete downloaded RINEX/Product files after processing (default is to delete)")
    parser.add_argument("--redo", action="store_true", help="Force redo of positioning evaluation even if results exist")
    parser.add_argument("--weight_opt", type=str, default="elev", help="Weighting option: elev (elevation), snr (SNR), or iono (ionospheric uncertainty). Can be comma-separated list.")
    parser.add_argument("--gnss_path", type=str, help="Custom path to GNSS data (STEC_DB_CASDCB folder)")
    
    args = parser.parse_args()
    logger = setup_logging()

    # Parse Weighting Options
    weight_opts = [w.strip() for w in args.weight_opt.split(",")]
    args.weight_opts = weight_opts  # Store list in args
    
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
    if ':' in args.dates:
        start_str, end_str = args.dates.split(':')
        
        def parse_d(d):
            if '-' in d and len(d.split('-')) == 3: 
                return datetime.strptime(d, "%Y-%m-%d")
            # Handle YYYY-DDD
            parts = d.split('-')
            if len(parts) == 2 and len(parts[0]) == 4:
                 return datetime(int(parts[0]), 1, 1) + timedelta(days=int(parts[1]) - 1)
            # Handle just date string potentially?
            return datetime.strptime(d, "%Y-%m-%d")

        try:
            current = parse_d(start_str)
            end = parse_d(end_str)
            while current <= end:
                dates.append(current)
                current += timedelta(days=1)
        except ValueError as e:
            logger.error(f"Date format error: {e}. Use YYYY-MM-DD or YYYY-DDD")
            return 1
    else:
        # Assuming comma separated YYYY-MM-DD or YYYY-DDD
        try:
            for d in args.dates.split(','):
                d = d.strip()
                if '-' in d and len(d.split('-')) == 3:
                    dates.append(datetime.strptime(d, "%Y-%m-%d"))
                elif '-' in d and len(d.split('-')) == 2:
                     parts = d.split('-')
                     dates.append(datetime(int(parts[0]), 1, 1) + timedelta(days=int(parts[1]) - 1))
        except ValueError:
            logger.error("Invalid date format in list.")
            return 1
            
    # Store results paths for aggregation
    # Tuple format: (date_obj, path, label)
    daily_summary_paths = []

    # Process Days in Parallel
    logger.info(f"\n🚀 Starting Multiday Processing with {args.parallel} concurrent days...")
    
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(
                process_day, 
                current_date, stec_base_config, vtec_base_config, args
            ): current_date for current_date in dates
        }
        
        for future in tqdm(as_completed(futures), total=len(dates), desc="Multiday Progress"):
            try:
                result = future.result()
                if result:
                    daily_summary_paths.extend(result)
            except Exception as e:
                date_failed = futures[future]
                logger.error(f"Failed to process {date_failed}: {e}")
        
    # 4. Aggregate Results
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Aggregating Results...")
    logger.info("="*80)
    
    all_metrics = []
    
    # We need to handle GIM comparison. GIM results are generated in BOTH runs (STEC/VTEC vs GIM).
    # We should deduplicate GIM results or just take them from one source.
    # The 'method' column in daily_summary.csv likely has "Model" and "IGS GIM".
    # We will rename "Model" to "STEC Model" or "VTEC Model" depending on source.
    
    # Keep track of unique (date, station, method) combinations to avoid GIM duplication if needed,
    # or simpler: just trust the aggregation plotting logic to handle duplicates or separate them.
    # Actually, simpler approach: Rename "Model" -> "STEC/VTEC Model", and keep "IGS GIM".
    # If we concat everything, we'll have duplicate "IGS GIM" entries. Seaborn handles this fine usually (aggregating),
    # but for trends we might want to be cleaner.
    
    for date_obj, csv_path, label in daily_summary_paths:
        try:
            df = pd.read_csv(csv_path)
            df['date'] = date_obj.strftime("%Y-%m-%d")
            df['doy'] = date_obj.timetuple().tm_yday
            
            # Normalize method names using our label (which contains model + weight)
            if 'method' in df.columns:
                # Normalize typical names (handle case variations)
                df['method'] = df['method'].str.lower()
                
                # Replace 'model' with the descriptive label we stored in day_results
                # label is e.g. "STEC_iono", "VTEC_elev", etc.
                df.loc[df['method'].str.startswith('model'), 'method'] = label
                
                # Handle GIM - preserve weighting suffix from the current run
                weight_suffix = label.split('_')[-1]
                gim_mask = df['method'].str.contains('gim')
                if gim_mask.any():
                    df.loc[gim_mask, 'method'] = f"gim_{weight_suffix}"
            
            all_metrics.append(df)
        except Exception as e:
            logger.warning(f"Error reading {csv_path}: {e}")
            
    if all_metrics:
        combined_df = pd.concat(all_metrics, ignore_index=True)
        
        # Deduplicate entries to correct statistics (especially for IGS GIM which appears in both runs)
        cols_to_check = ['date', 'method']
        if 'station' in combined_df.columns:
            cols_to_check.append('station')
        if 'time' in combined_df.columns:
            cols_to_check.append('time')
            
        before_len = len(combined_df)
        combined_df.drop_duplicates(subset=cols_to_check, inplace=True)
        dropped_count = before_len - len(combined_df)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} duplicate rows (mostly redundant IGS GIM entries)")
        
        # Save to a central "multiday_results" folder
        # Use a unique name based on the current run configuration
        run_identifier = datetime.now().strftime("%Y%m%d_%H%M")
        base_output_dir = Path("multiday_results") / f"positioning_{run_identifier}"
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = base_output_dir / "multiday_summary.csv"
        combined_df.to_csv(output_file, index=False, float_format='%.4f')
        logger.info(f"✅ Saved multi-day summary to: {output_file}")
        
        # Plot
        plot_trends(combined_df, base_output_dir)
        logger.info(f"Plots saved to: {base_output_dir}")
        
    else:
        logger.warning("No results to aggregate.")

    logger.info("DONE.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
