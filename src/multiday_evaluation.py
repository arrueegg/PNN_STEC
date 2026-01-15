"""
Multi-Day Evaluation Pipeline for Robust Paper Results

Automates training and evaluation across multiple test days for statistically robust results.

For each specified day:
1. Finetune STEC model from pretrained weights
2. Finetune VTEC model from scratch
3. Run comprehensive comparison (STEC vs VTEC+Mapping vs GIM)
4. Store results in organized structure

Finally generates aggregate statistics and plots across all days.
"""

import os
import sys
import yaml
import logging
import math
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import torch
from types import SimpleNamespace
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_parser import parse_config, compute_exp_name
from main import main as run_main_training
from compare_stec_vtec_gim import main as run_main_comparison
# from inference_positioning import main as run_main_positioning # TODO: refactor inference_positioning.py first

class LogFilter:
    """Filter out lines starting with specific prefixes from stream."""
    def __init__(self, stream, ignore_prefixes):
        self.stream = stream
        self.ignore_prefixes = ignore_prefixes

    def write(self, data):
        # Handle empty writes
        if not data:
            return

        # Quick check for exact match start (optimization)
        if any(data.lstrip().startswith(p) for p in self.ignore_prefixes):
            return
            
        # Line-by-line check for multiline writes
        # Use splitlines(keepends=True) to preserve formatting
        # Note: if data is just a partial line (no newline), this logic still works for simple cases,
        # but sophisticated stream filtering would require buffering. 
        # For standard print/tqdm output, this often suffices.
        if '\n' in data or '\r' in data:
            lines = data.splitlines(keepends=True)
            for line in lines:
                if not any(line.lstrip().startswith(p) for p in self.ignore_prefixes):
                    self.stream.write(line)
        else:
            # Single chunk without newline
            if not any(data.lstrip().startswith(p) for p in self.ignore_prefixes):
                self.stream.write(data)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        """Allow checks for TTY (needed by tools like wandb)."""
        if hasattr(self.stream, 'isatty'):
            return self.stream.isatty()
        return False

@contextmanager
def capture_execution(log_file_path):
    """
    Context manager to redirect stdout, stderr AND logging to a file.
    Critically important for in-process execution to mimic subprocess isolation.
    """
    # 1. Setup logging to file
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    # Save original handlers and remove them (to silence console)
    original_handlers = root_logger.handlers[:]
    for h in original_handlers:
        root_logger.removeHandler(h)
    
    root_logger.addHandler(file_handler)
    
    # 2. Redirect stdout/stderr with filtering
    with open(log_file_path, 'a') as f:
        # Filter out tqdm updates for Bayesian Inference specifically
        filtered_stream = LogFilter(f, ignore_prefixes=["Bayesian Inference :", "Bayesian Inference:"])
        
        with redirect_stdout(filtered_stream), redirect_stderr(filtered_stream):
            try:
                yield
            finally:
                # Restore logging
                root_logger.removeHandler(file_handler)
                for h in original_handlers:
                    root_logger.addHandler(h)
                file_handler.close()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_date_string(date_str: str) -> Tuple[int, int]:
    """Parse date string to (year, doy).
    
    Supports formats:
    - YYYY-DOY (e.g., "2024-183")
    - YYYY-MM-DD (e.g., "2024-07-01")
    """
    if '-' in date_str and len(date_str.split('-')) == 3:
        # YYYY-MM-DD format
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        year = dt.year
        doy = dt.timetuple().tm_yday
    else:
        # YYYY-DOY format
        parts = date_str.split('-')
        year = int(parts[0])
        doy = int(parts[1])
    
    return year, doy


def generate_date_list(dates_arg: str) -> List[Tuple[int, int]]:
    """Generate list of (year, doy) tuples from date argument.
    
    Supports:
    - Single date: "2024-183"
    - Multiple dates: "2024-183,2024-184,2024-185"
    - Date range: "2024-183:2024-192" (inclusive)
    """
    dates = []
    
    if ':' in dates_arg:
        # Date range
        start_str, end_str = dates_arg.split(':')
        start_year, start_doy = parse_date_string(start_str)
        end_year, end_doy = parse_date_string(end_str)
        
        # Convert to datetime for easy iteration
        start_dt = datetime(start_year, 1, 1) + timedelta(days=start_doy - 1)
        end_dt = datetime(end_year, 1, 1) + timedelta(days=end_doy - 1)
        
        current_dt = start_dt
        while current_dt <= end_dt:
            year = current_dt.year
            doy = current_dt.timetuple().tm_yday
            dates.append((year, doy))
            current_dt += timedelta(days=1)
    
    else:
        # Single date or comma-separated list
        for date_str in dates_arg.split(','):
            date_str = date_str.strip()
            year, doy = parse_date_string(date_str)
            dates.append((year, doy))
    
    return dates


def create_modified_config(base_config_path: str, year: int, doy: int, 
                          output_dir: Path, is_vtec: bool = False,
                          pretrain_folder: str = None) -> str:
    """Create modified config file for specific date.
    
    Args:
        base_config_path: Path to base config
        year: Year for this training day
        doy: Day of year for this training day
        output_dir: Directory to save temp config
        is_vtec: Whether this is VTEC model (finetune from scratch)
        pretrain_folder: Explicit pretrain folder to use (optional)
    
    Returns path to temporary config file.
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for this date
    config['mode'] = 'finetune'
    config['year'] = year  # Top-level year for experiment name generation
    config['doy'] = doy    # Top-level doy for experiment name generation
    config['finetune']['year'] = year
    config['finetune']['doy'] = doy
    
    # CRITICAL: Disable aggregated H5 to validly force day-specific loading
    # This prevents loading the 100GB train.h5 or 6GB test.h5
    if 'data' not in config:
        config['data'] = {}
    config['data']['use_agg_h5'] = False
    
    # Set finetune_from_scratch for VTEC
    if is_vtec:
        config['finetune']['finetune_from_scratch'] = True
    else:
        config['finetune']['finetune_from_scratch'] = False
        # Set pretrain_folder if provided
        if pretrain_folder:
            config['pretrain_folder'] = pretrain_folder
            logger.info(f"Using specified pretrain_folder: {pretrain_folder}")
    
    # Create temporary config file
    model_type = "vtec" if is_vtec else "stec"
    temp_config_path = output_dir / f"temp_config_{model_type}_{year}_{doy:03d}.yaml"
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(temp_config_path)


def check_experiment_exists(config_path: str) -> Tuple[bool, str]:
    """Check if an experiment described by the config already exists."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure year/doy strings match what main.py expects/formats
        if 'year' in config:
            config['year'] = str(config['year'])
        if 'doy' in config:
            config['doy'] = str(config['doy']).zfill(3)

        exp_name = compute_exp_name(config)
        exp_path = Path("experiments") / exp_name
        
        # Check if experiment exists with a trained model
        # 1. Check for root .pt files (common in some setups)
        # 2. Check for model/*.pth files (standard in this repo)
        model_dir = exp_path / "model"
        
        has_root_model = (exp_path / "best_model.pt").exists() or (exp_path / "model.pt").exists()
        has_subdir_model = model_dir.exists() and any(model_dir.glob("*.pth"))
        
        if exp_path.exists() and (has_root_model or has_subdir_model):
            return True, exp_name
            
        return False, exp_name
    except Exception as e:
        logger.warning(f"Failed to check if experiment exists: {e}")
        return False, None


def ensure_pretrain_exists(base_config_path: str, output_dir: Path) -> Tuple[bool, str]:
    """Check if pretrain experiment exists, run if not.
    
    Returns (success, pretrain_folder_path).
    """
    # Load config to determine what pretrain experiment should exist
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Temporarily set mode to pretrain to compute pretrain experiment name
    original_mode = config['mode']
    config['mode'] = 'pretrain'
    
    # Compute what the pretrain experiment name should be
    try:
        pretrain_exp_name = compute_exp_name(config)
    except Exception as e:
        logger.error(f"Failed to compute pretrain experiment name: {e}")
        return False, None
    
    pretrain_folder = Path("experiments") / pretrain_exp_name
    model_folder = pretrain_folder / "model"
    
    # Check if pretrain experiment exists with trained model
    if pretrain_folder.exists() and model_folder.exists() and list(model_folder.glob("*.pth")):
        logger.info(f"✓ Found existing pretrain experiment: {pretrain_exp_name}")
        return True, str(pretrain_folder)
    
    # Pretrain doesn't exist - need to run it
    logger.info(f"⚠️  Pretrain experiment not found: {pretrain_exp_name}")
    logger.info("Running pretraining first...")
    
    # Create pretrain config
    config['mode'] = 'pretrain'
    temp_pretrain_config = output_dir / "temp_config_pretrain.yaml"
    
    with open(temp_pretrain_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run pretrain
    success, exp_name = run_training(str(temp_pretrain_config))
    
    if not success:
        logger.error("✗ Pretraining failed")
        return False, None
    
    # Verify model was trained
    pretrain_folder = Path("experiments") / exp_name
    model_folder = pretrain_folder / "model"
    
    if not (model_folder.exists() and list(model_folder.glob("*.pth"))):
        logger.error(f"✗ Pretrain completed but no model found in {model_folder}")
        return False, None
    
    logger.info(f"✓ Pretraining completed: {exp_name}")
    return True, str(pretrain_folder)


def run_training(config_path: str) -> Tuple[bool, str]:
    """Run training for a given config.
    
    Returns (success, experiment_name).
    """
    logger.info(f"Training with config: {config_path}")
    
    # Determine log file path (in same directory as config)
    config_path_obj = Path(config_path)
    log_file = config_path_obj.parent / f"{config_path_obj.stem}_training.log"
    
    logger.info(f"Training log: {log_file}")
    
    # Pre-calculate experiment name to ensure we get the correct one
    # This avoids issues where we guess the wrong directory (e.g. STEC instead of VTEC due to timestamps)
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Ensure year/doy strings match what main.py expects
        if 'year' in config_dict:
            config_dict['year'] = str(config_dict['year'])
        if 'doy' in config_dict:
            config_dict['doy'] = str(config_dict['doy']).zfill(3)
            
        expected_experiment_name = compute_exp_name(config_dict)
    except Exception as e:
        logger.warning(f"Could not pre-calculate experiment name: {e}")
        expected_experiment_name = None

    try:
        # Disable tqdm progress bars for cleaner logs and redirect output
        os.environ['TQDM_DISABLE'] = '1'
        
        # Clear garbage and cache before training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        with capture_execution(log_file):
            # Run training in-process
            run_main_training(config_path=config_path)
            
        # Use pre-calculated name if available, otherwise fallback to parsing/guessing
        if expected_experiment_name:
            experiment_name = expected_experiment_name
            logger.info(f"✓ Training completed (Exp: {experiment_name})")
            return True, experiment_name

        # For in-process execution, we assume success if no exception was raised.
        # However, finding the experiment name without parsing stdout is harder if we didn't compute it.
        # But we computed expected_experiment_name above, so we should be good.
        
        if not expected_experiment_name:
             # Fallback: find most recent experiment directory (less reliable now that we don't have stdout lines easily)
            experiments_dir = Path(__file__).parent.parent / "experiments"
            exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
            if exp_dirs:
                experiment_name = max(exp_dirs, key=lambda x: x.stat().st_mtime).name
            else:
                 logger.warning("Could not determine experiment name after training.")
                 return True, None # Return True but no name?

            return True, experiment_name
            
        return True, expected_experiment_name
    
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        # traceback
        import traceback
        traceback.print_exc()
        with open(log_file, 'a') as f:
            f.write(f"\n\n=== ERROR ===\n{e}\n")
            traceback.print_exc(file=f)
        return False, None
    finally:
         # Clean up env var
        if 'TQDM_DISABLE' in os.environ:
            del os.environ['TQDM_DISABLE']


def run_positioning_pipeline(experiment_name: str, year: int, doy: int) -> bool:
    """Run complete positioning pipeline for a given experiment and date.
    
    1. Generate STEC corrections (inference_positioning.py)
    2. Run PPPx and evaluate (run_positioning_evaluation.py)
    
    Returns success status.
    """
    date_str = f"{year}-{doy:03d}"
    logger.info(f"Running positioning pipeline for {experiment_name} on {date_str}")
    
    # 1. Generate STEC corrections
    logger.info("Step 1: Generating STEC corrections...")
    cmd_inference = [
        sys.executable,
        "src/inference_positioning.py",
        "--experiment", experiment_name,
        "--year", str(year),
        "--doy", str(doy)
    ]
    
    try:
        subprocess.run(
            cmd_inference,
            cwd=Path(__file__).parent.parent,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✓ STEC corrections generated")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ STEC inference failed: {e}")
        logger.error(f"stderr: {e.stderr}")
        return False
        
    # 2. Run Positioning Evaluation
    # Convert to YYYY-MM-DD for this script
    dt = datetime(year, 1, 1) + timedelta(days=doy - 1)
    date_formatted = dt.strftime("%Y-%m-%d")
    
    logger.info("Step 2: Running PPPx evaluation...")
    cmd_eval = [
        sys.executable,
        "src/positioning_eval/run_positioning_evaluation.py",
        "--experiment", experiment_name,
        "--date", date_formatted,
        "--all_test_stations",
        "--cleanup" # Clean up downloaded files to save space
    ]
    
    try:
        # stream output to show progress bars
        subprocess.run(
            cmd_eval,
            cwd=Path(__file__).parent.parent,
            check=True
        )
        logger.info("✓ Positioning evaluation completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Positioning evaluation failed: {e}")
        return False


def run_comparison(stec_exp: str, vtec_exp: str, output_dir: Path, 
                  num_samples: int = 100) -> bool:
    """Run comprehensive comparison evaluation.
    
    Returns success status.
    """
    logger.info(f"Running comparison: {stec_exp} vs {vtec_exp}")

    # Ensure output directory exists for logging
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "comparison.log"
    logger.info(f"Comparison log: {log_file}")
    
    try:
        # Disable tqdm to keep log clean
        os.environ['TQDM_DISABLE'] = '1'

        # Clear garbage and cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with capture_execution(log_file):
            # Construct args object explicitly mimicking argparse.Namespace
            # or pass direct arguments if we refactored run_comparison signature (which we did partially, but using Namespace is safer for now as I kept 'args' support)
            
            # Create a namespace object with all arguments expected by run_main_comparison
            # Note: I modified run_main_comparison (aka main) to accept 'args'.
            # I can just pass a Namespace object.
            
            args = SimpleNamespace(
                stec_experiment=stec_exp,
                vtec_experiment=vtec_exp,
                num_inference_samples=num_samples,
                test_size=None, # Default as per CLI
                madrigal_path="/home/space/data/iono/Madrigal_STEC", # Default
                no_gim=False, # Default
                gim_path="/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex", # Default
                mapping_function="MSLM", # Default
                output_dir=str(output_dir)
            )
            
            run_main_comparison(args=args)
        
        logger.info(f"✓ Comparison completed")
        return True
    
    except Exception as e:
        logger.error(f"✗ Comparison failed. See log at {log_file}")
        import traceback
        with open(log_file, 'a') as f:
            f.write(f"\n\n=== ERROR ===\n{e}\n")
            traceback.print_exc(file=f)
        return False
    finally:
         # Clean up env var
        if 'TQDM_DISABLE' in os.environ:
            del os.environ['TQDM_DISABLE']


def extract_metrics_from_experiment(evaluation_dir: Path) -> Dict[str, Dict]:
    """Extract metrics from experiment evaluation results.
    
    Returns dict with metrics for each dataset type.
    """
    metrics = {}
    
    # Check for both dataset types
    for dataset_type in ["own_vtec_gim", "madrigal_vtec_gim"]:
        dataset_dir = evaluation_dir / dataset_type
        metrics_file = dataset_dir / "metrics_summary.csv"
        
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            metrics[dataset_type] = df.to_dict('records')
    
    return metrics


def extract_elevation_metrics_from_experiment(evaluation_dir: Path) -> Dict[str, pd.DataFrame]:
    """Extract elevation-binned metrics from detailed predictions.
    
    Returns dict with elevation-binned metrics for the dataset.
    """
    elevation_metrics = {}
    
    # Check for both dataset types
    for dataset_type in ["own_vtec_gim", "madrigal_vtec_gim"]:
        dataset_dir = evaluation_dir / dataset_type
        predictions_file = dataset_dir / "detailed_predictions.csv"
        
        if predictions_file.exists():
            # Read the predictions file
            df = pd.read_csv(predictions_file)
            
            # Bin elevations (typical GNSS elevations are 5-90 degrees)
            elevation_bins = np.arange(0, 91, 5)  # 0-5, 5-10, ..., 85-90
            df['elevation_bin'] = pd.cut(df['elevation'], bins=elevation_bins, labels=elevation_bins[:-1])
            
            # Calculate metrics per elevation bin
            binned_metrics = []
            for bin_start, group in df.groupby('elevation_bin'):
                if len(group) > 100:  # Only include bins with sufficient data
                    # Direct STEC metrics
                    stec_rmse = np.sqrt(np.mean((group['true_stec'] - group['stec_pred'])**2))
                    stec_mae = np.mean(np.abs(group['true_stec'] - group['stec_pred']))
                    stec_bias = np.mean(group['stec_pred'] - group['true_stec'])
                    
                    # VTEC+Mapping metrics
                    vtec_rmse = np.sqrt(np.mean((group['true_stec'] - group['vtec_model_stec'])**2))
                    vtec_mae = np.mean(np.abs(group['true_stec'] - group['vtec_model_stec']))
                    vtec_bias = np.mean(group['vtec_model_stec'] - group['true_stec'])
                    
                    # GIM metrics
                    gim_rmse = np.sqrt(np.mean((group['true_stec'] - group['gim_stec'])**2))
                    gim_mae = np.mean(np.abs(group['true_stec'] - group['gim_stec']))
                    gim_bias = np.mean(group['gim_stec'] - group['true_stec'])
                    
                    binned_metrics.append({
                        'elevation_bin': bin_start,
                        'count': len(group),
                        'Direct STEC RMSE': stec_rmse,
                        'Direct STEC MAE': stec_mae,
                        'Direct STEC Bias': stec_bias,
                        'VTEC + Mapping RMSE': vtec_rmse,
                        'VTEC + Mapping MAE': vtec_mae,
                        'VTEC + Mapping Bias': vtec_bias,
                        'IGS GIM RMSE': gim_rmse,
                        'IGS GIM MAE': gim_mae,
                        'IGS GIM Bias': gim_bias
                    })
            
            elevation_metrics[dataset_type] = pd.DataFrame(binned_metrics)
    
    return elevation_metrics


def generate_aggregate_report(batch_results: List[Dict], output_dir: Path):
    """Generate aggregate statistics and plots across all days."""
    
    logger.info("="*70)
    logger.info("Generating Aggregate Report")
    logger.info("="*70)
    
    # Create summary directory
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all metrics
    all_results = []
    
    for result in batch_results:
        if not result['success']:
            continue
        
        year = result['year']
        doy = result['doy']
        date_str = f"{year}-{doy:03d}"
        
        # Extract metrics from both datasets
        for dataset_type, dataset_metrics in result['metrics'].items():
            for metric_row in dataset_metrics:
                row = {
                    'date': date_str,
                    'year': year,
                    'doy': doy,
                    'dataset': dataset_type,
                    **metric_row
                }
                all_results.append(row)
        
        # Extract elevation metrics
        # Note: output_dir is typically multiday_results, so day folders are inside it
        evaluation_dir = output_dir / f"{result['year']}_DOY_{result['doy']}" / "evaluation"
        elevation_metrics = extract_elevation_metrics_from_experiment(evaluation_dir)
        result['elevation_metrics'] = elevation_metrics
    
    if not all_results:
        logger.warning("No successful results to aggregate")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save complete results
    df.to_csv(summary_dir / "all_results.csv", index=False)
    logger.info(f"Saved complete results: {summary_dir / 'all_results.csv'}")
    
    # Generate summary statistics
    summary_stats = []
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        for model_type in dataset_df['Model'].unique():
            model_df = dataset_df[dataset_df['Model'] == model_type]
            
            stats = {
                'Dataset': dataset,
                'Model': model_type,
                'RMSE_mean': model_df['RMSE'].mean(),
                'RMSE_std': model_df['RMSE'].std(),
                'MAE_mean': model_df['MAE'].mean(),
                'MAE_std': model_df['MAE'].std(),
                'R2_mean': model_df['R²'].mean(),
                'R2_std': model_df['R²'].std(),
                'Num_days': len(model_df)
            }
            summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(summary_dir / "summary_statistics.csv", index=False)
    logger.info(f"Saved summary statistics: {summary_dir / 'summary_statistics.csv'}")
    
    # Print summary table
    logger.info("\n" + "="*70)
    logger.info("SUMMARY STATISTICS (across all days)")
    logger.info("="*70)
    logger.info(summary_df.to_string(index=False))
    
    # Generate plots
    generate_aggregate_plots(df, batch_results, summary_dir)
    
    logger.info(f"\n✅ Aggregate report saved to: {summary_dir}")


def generate_aggregate_plots(df: pd.DataFrame, batch_results: List[Dict], output_dir: Path):
    """Generate publication-ready aggregate plots."""
    
    logger.info("Generating aggregate plots...")
    
    # Dataset name mapping used for filenames
    name_map = {
        'own_vtec_gim': 'ownDS',
        'madrigal_vtec_gim': 'Madrigal'
    }
    
    # Set aesthetics for publication-quality plots
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 8
    
    # Define consistent colors for models using a colorblind-friendly palette
    # Direct STEC: Blue, VTEC+Mapping: Orange, IGS GIM: Green
    colors = sns.color_palette("colorblind")
    model_colors = {
        'Direct STEC': colors[0],
        'VTEC + Mapping': colors[1],
        'IGS GIM + Mapping': colors[2]
    }
    
    # Define baseline colors (subset of model_colors for consistency in improvement plots)
    baseline_colors = {
        'VTEC + Mapping': model_colors['VTEC + Mapping'],
        'IGS GIM + Mapping': model_colors['IGS GIM + Mapping']
    }
    
    # Ensure date column is datetime for proper plotting
    if 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'], format='%Y-%j')
        df = df.sort_values('datetime')
    else: 
        logger.warning("No 'date' column found in dataframe, skipping datetime conversion")
    
    unique_datasets = df['dataset'].unique()
    
    # Normalize model names in df to ensure consistency
    df['Model'] = df['Model'].replace({
        'Direct STEC Model': 'Direct STEC',
        'IGS GIM': 'IGS GIM + Mapping'
    })
    
    # -------------------------------------------------------------------------
    # 1. RMSE comparison across days (Separate plot per dataset)
    # -------------------------------------------------------------------------
    for dataset in unique_datasets:
        mapped_name = name_map.get(dataset, dataset)
        dataset_df = df[df['dataset'] == dataset]
        
        plt.figure(figsize=(14, 7))
        
        # Determine x-axis values (datetime if available, else string)
        use_date_obj = 'datetime' in dataset_df.columns
        x_col = 'datetime' if use_date_obj else 'date'
        
        pivot_df = dataset_df.pivot(index=x_col, columns='Model', values='RMSE')
        
        # Plot each model with consistent colors
        for model in pivot_df.columns:
            color = model_colors.get(model, 'gray')
            plt.plot(pivot_df.index, pivot_df[model], marker='o', label=model, color=color)
        
        plt.ylabel('RMSE (TECU)')
        plt.title(f'RMSE by Date ({mapped_name})')
        # Move legend further down
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=len(model_colors), frameon=True)
        
        # Improved date formatting
        if use_date_obj:
            plt.xlabel('Date')
            ax = plt.gca()
            locator = mdates.AutoDateLocator()
            formatter = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        else:
            plt.xlabel('Date (YYYY-DOY)')
            plt.xticks(rotation=45)
            
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        filename = output_dir / f'rmse_by_date_{mapped_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved RMSE vs Date plot: {filename}")
    
    # -------------------------------------------------------------------------
    # 2. Box plots comparing models (Separate plot per dataset AND metric)
    # -------------------------------------------------------------------------
    metrics_list = ['RMSE', 'MAE', 'R²', 'Bias']
    
    for dataset in unique_datasets:
        mapped_name = name_map.get(dataset, dataset)
        dataset_df = df[df['dataset'] == dataset]
        
        for metric in metrics_list:
            if metric not in dataset_df.columns:
                continue
                
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=dataset_df, x='Model', y=metric, palette=model_colors, hue='Model', legend=False)
            
            plt.title(f'{metric} Distribution ({mapped_name})')
            plt.xlabel('')
            plt.ylabel(metric)
            plt.tick_params(axis='x', rotation=15)
            plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            # Clean filename
            metric_clean = metric.replace('²', '2')
            filename = output_dir / f'{metric_clean.lower()}_boxplot_{mapped_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved {metric} boxplot: {filename}")
    
    # -------------------------------------------------------------------------
    # 3. Improvement statistics (Separate plot per dataset)
    # -------------------------------------------------------------------------
    for dataset in unique_datasets:
        mapped_name = name_map.get(dataset, dataset)
        dataset_df = df[df['dataset'] == dataset]
        
        improvements = []
        
        # Group by date/datetime
        use_date_obj = 'datetime' in dataset_df.columns
        x_col = 'datetime' if use_date_obj else 'date'
        
        for date_val in dataset_df[x_col].unique():
            subset = dataset_df[dataset_df[x_col] == date_val]
            
            # Get values safely
            direct_stec = subset[subset['Model'] == 'Direct STEC']
            vtec_map = subset[subset['Model'] == 'VTEC + Mapping']
            igs_gim = subset[subset['Model'] == 'IGS GIM + Mapping']
            
            if direct_stec.empty: 
                continue
                
            stec_rmse = direct_stec['RMSE'].values[0]
            
            if not vtec_map.empty:
                vtec_rmse = vtec_map['RMSE'].values[0]
                imp_vtec = (1 - stec_rmse / vtec_rmse) * 100
                improvements.append({'date_val': date_val, 'baseline': 'VTEC + Mapping', 'improvement': imp_vtec})
            
            if not igs_gim.empty:
                gim_rmse = igs_gim['RMSE'].values[0]
                imp_gim = (1 - stec_rmse / gim_rmse) * 100
                improvements.append({'date_val': date_val, 'baseline': 'IGS GIM + Mapping', 'improvement': imp_gim})
        
        if improvements:
            plt.figure(figsize=(14, 7))
            imp_df = pd.DataFrame(improvements)
            
            # Use scatter + line for time series with consistent baseline colors
            sns.scatterplot(data=imp_df, x='date_val', y='improvement', hue='baseline', 
                            palette=baseline_colors, s=80, alpha=0.9, legend=False)
            sns.lineplot(data=imp_df, x='date_val', y='improvement', hue='baseline', 
                         palette=baseline_colors, alpha=0.9)
            
            plt.ylabel('RMSE Improvement (%)')
            plt.title(f'Direct STEC Improvement Over Baselines ({mapped_name})')
            plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
            # Move legend further down
            plt.legend(title='Baseline', loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=len(baseline_colors), frameon=True)
            
             # Improved date formatting
            if use_date_obj:
                plt.xlabel('Date')
                ax = plt.gca()
                locator = mdates.AutoDateLocator()
                formatter = mdates.DateFormatter('%Y-%m-%d')
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            else:
                plt.xlabel('Date (YYYY-DOY)')
                plt.xticks(rotation=45)

            plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            filename = output_dir / f'improvement_by_date_{mapped_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved improvement plot: {filename}")

    # -------------------------------------------------------------------------
    # 4. Elevation-dependent plots
    # -------------------------------------------------------------------------
    logger.info("Generating elevation-dependent plots...")
    
    # Prepare elevation data
    all_elevation_data = []
    
    # First try to gather from batch_results
    for result in batch_results:
        # Check if elevation metrics exist in memory
        if result.get('success', False) and 'elevation_metrics' in result:
             for dataset_type, elev_df in result['elevation_metrics'].items():
                if not elev_df.empty:
                    df_copy = elev_df.copy()
                    df_copy['dataset'] = dataset_type
                    all_elevation_data.append(df_copy)
    
    elevation_agg = None
    
    if all_elevation_data:
        elevation_df = pd.concat(all_elevation_data, ignore_index=True)
        # Aggregate across all days
        elevation_agg = elevation_df.groupby(['elevation_bin', 'dataset']).agg({
            'Direct STEC RMSE': ['mean', 'std'],
            'Direct STEC MAE': ['mean', 'std'],
            'VTEC + Mapping RMSE': ['mean', 'std'],
            'VTEC + Mapping MAE': ['mean', 'std'],
            'IGS GIM RMSE': ['mean', 'std'],
            'IGS GIM MAE': ['mean', 'std'],
            'count': 'sum'
        }).round(3)
        
        # Flatten columns
        elevation_agg.columns = ['_'.join(col).strip() for col in elevation_agg.columns.values]
        elevation_agg = elevation_agg.reset_index()
        
        # Save metrics
        csv_path = output_dir / 'elevation_metrics.csv'
        elevation_agg.to_csv(csv_path, index=False)
        logger.info(f"Saved elevation metrics: {csv_path}")

    elif (output_dir / 'elevation_metrics.csv').exists():
        # Fallback to loading existing CSV if no batch results provided (summary_only mode)
        elevation_agg = pd.read_csv(output_dir / 'elevation_metrics.csv')
        logger.info(f"Loaded existing elevation metrics: {output_dir / 'elevation_metrics.csv'}")

    # Plotting Elevation Results
    if elevation_agg is not None:
        for dataset in elevation_agg['dataset'].unique():
            mapped_name = name_map.get(dataset, dataset)
            dataset_elev = elevation_agg[elevation_agg['dataset'] == dataset]
            
            # Helper for elevation plots
            def plot_elevation_metric(metric_name, ylabel, filename_prefix):
                plt.figure(figsize=(10, 6))
                
                # Add jitter to x-axis to prevent overlap
                x = dataset_elev['elevation_bin'].values.astype(float)
                offset = 0.8
                
                # Direct STEC (Shift Left)
                plt.errorbar(x - offset, 
                           dataset_elev[f'Direct STEC {metric_name}_mean'],
                           yerr=dataset_elev[f'Direct STEC {metric_name}_std'],
                           label='Direct STEC', marker='o', capsize=4, 
                           color=model_colors['Direct STEC'],
                           markersize=6, alpha=0.9)
                
                # VTEC + Mapping (Center)
                plt.errorbar(x, 
                           dataset_elev[f'VTEC + Mapping {metric_name}_mean'],
                           yerr=dataset_elev[f'VTEC + Mapping {metric_name}_std'],
                           label='VTEC + Mapping', marker='s', capsize=4, 
                           color=model_colors['VTEC + Mapping'],
                           markersize=6, alpha=0.9)
                
                # IGS GIM using simplified columns but new label (Shift Right)
                plt.errorbar(x + offset, 
                           dataset_elev[f'IGS GIM {metric_name}_mean'],
                           yerr=dataset_elev[f'IGS GIM {metric_name}_std'],
                           label='IGS GIM + Mapping', marker='^', capsize=4, 
                           color=model_colors['IGS GIM + Mapping'],
                           markersize=6, alpha=0.9)
                
                plt.xlabel('Elevation Angle (degrees)')
                plt.ylabel(ylabel)
                plt.title(f'{metric_name} vs Elevation ({mapped_name})')
                # Move legend further down
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=len(model_colors), frameon=True)
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.xlim(0, 90)
                plt.tight_layout()
                
                fname = output_dir / f'{filename_prefix}_{mapped_name}.png'
                plt.savefig(fname, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved {metric_name} plot: {fname}")

            # Create RMSE Plot
            plot_elevation_metric('RMSE', 'RMSE (TECU)', 'rmse_vs_elevation')
            
            # Create MAE Plot
            plot_elevation_metric('MAE', 'MAE (TECU)', 'mae_vs_elevation')
    
    logger.info(f"✓ Plots saved to {output_dir}")


def collect_existing_results(output_base: Path) -> List[Dict]:
    """Collect results from existing experiment directories."""
    batch_results = []
    
    # Find all date directories
    date_dirs = [d for d in output_base.iterdir() if d.is_dir() and '_' in d.name and 'DOY' in d.name]
    
    for date_dir in sorted(date_dirs):
        try:
            # Parse year and doy from directory name
            parts = date_dir.name.split('_')
            if len(parts) >= 3 and parts[1] == 'DOY':
                year = int(parts[0])
                doy = int(parts[2])
                date_str = f"{year}-{doy:03d}"
                
                # Check if evaluation exists
                eval_dir = date_dir / "evaluation"
                if not eval_dir.exists():
                    continue
                
                # Extract metrics directly from evaluation directory
                metrics = {}
                for dataset_type in ["own_vtec_gim", "madrigal_vtec_gim"]:
                    dataset_dir = eval_dir / dataset_type
                    metrics_file = dataset_dir / "metrics_summary.csv"
                    
                    if metrics_file.exists():
                        df = pd.read_csv(metrics_file)
                        metrics[dataset_type] = df.to_dict('records')
                
                if metrics:  # Only add if we found metrics
                    result = {
                        'year': year,
                        'doy': doy,
                        'date': date_str,
                        'success': True,
                        'stec_experiment': f"multiday_{date_str}",  # Dummy name
                        'vtec_experiment': None,
                        'metrics': metrics
                    }
                    
                    # Extract elevation metrics
                    evaluation_dir = date_dir / "evaluation"
                    elevation_metrics = extract_elevation_metrics_from_experiment(evaluation_dir)
                    result['elevation_metrics'] = elevation_metrics
                    
                    batch_results.append(result)
                    logger.info(f"✓ Collected results for {date_str}")
                
        except (ValueError, IndexError) as e:
            logger.warning(f"Skipping invalid directory {date_dir.name}: {e}")
            continue
    
    return batch_results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Day Evaluation Pipeline for Robust Paper Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single day
  python src/batch_evaluation.py \\
      --dates "2024-183" \\
      --stec_config config/config.yaml \\
      --vtec_config config/config_vtec_mlp_baseline.yaml
  
  # Multiple specific days
  python src/batch_evaluation.py \\
      --dates "2024-183,2024-184,2024-185" \\
      --stec_config config/config.yaml \\
      --vtec_config config/config_vtec_mlp_baseline.yaml
  
  # Date range (inclusive)
  python src/batch_evaluation.py \\
      --dates "2024-183:2024-192" \\
      --stec_config config/config.yaml \\
      --vtec_config config/config_vtec_mlp_baseline.yaml \\
      --output_dir multiday_results/july_2024
  
  # Quick test with fewer samples
  python src/batch_evaluation.py \\
      --dates "2024-183,2024-184" \\
      --stec_config config/config.yaml \\
      --vtec_config config/config_vtec_mlp_baseline.yaml \\
      --num_inference_samples 10 \\
      --test_size 1000

Date formats supported:
  - YYYY-DOY: "2024-183"
  - YYYY-MM-DD: "2024-07-01"
  - Range: "2024-183:2024-192" or "2024-07-01:2024-07-10"
  - List: "2024-183,2024-184,2024-185"
        """
    )
    
    parser.add_argument("--dates", type=str,
                       help="Date(s) to evaluate (single, comma-separated, or range)")
    parser.add_argument("--stec_config", type=str,
                       help="Base config file for STEC training")
    parser.add_argument("--vtec_config", type=str,
                       help="Base config file for VTEC training")
    parser.add_argument("--output_dir", type=str, default="multiday_results",
                       help="Output directory for all results (default: multiday_results)")
    parser.add_argument("--num_inference_samples", type=int, default=100,
                       help="MC samples for Bayesian inference (default: 100)")
    parser.add_argument("--test_size", type=int, default=None,
                       help="Test set size (default: full)")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training, only run evaluation (experiments must exist)")
    parser.add_argument("--skip_comparison", action="store_true",
                       help="Skip comparison evaluation (Step 3)")
    parser.add_argument("--skip_existing", action="store_true",
                       help="Skip training if experiment already exists")
    parser.add_argument("--pretrain_folder", type=str, default=None,
                       help="Pretrain experiment folder to use for STEC model (optional, auto-runs pretrain if needed)")
    parser.add_argument("--no_aggregate", action="store_true",
                       help="Skip aggregate report generation (for parallel execution)")
    parser.add_argument("--summary_only", action="store_true",
                       help="Skip processing, only generate aggregate report from existing results")
    parser.add_argument("--positioning", action="store_true",
                       help="Run positioning evaluation for each day")
    
    args = parser.parse_args()
    
    if not args.summary_only and (not args.dates or not args.stec_config or not args.vtec_config):
        parser.error("--dates, --stec_config, and --vtec_config are required unless --summary_only is used")
    
    logger.info("="*70)
    logger.info("MULTI-DAY EVALUATION PIPELINE")
    logger.info("="*70)
    
    # Create output directory
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    if args.summary_only:
        # Skip processing, collect existing results from summary CSV
        summary_csv = output_base / "summary" / "all_results.csv"
        if summary_csv.exists():
            logger.info(f"Reading existing results from {summary_csv}")
            df = pd.read_csv(summary_csv)
            batch_results = []
            # Group by date and reconstruct batch_results
            for (date, year, doy), group in df.groupby(['date', 'year', 'doy']):
                metrics = {}
                for dataset in group['dataset'].unique():
                    dataset_df = group[group['dataset'] == dataset]
                    metrics[dataset] = dataset_df.drop(columns=['date', 'year', 'doy', 'dataset']).to_dict('records')
                
                batch_results.append({
                    'year': int(year),
                    'doy': int(doy),
                    'date': date,
                    'success': True,
                    'stec_experiment': f"multiday_{date}",
                    'vtec_experiment': None,
                    'metrics': metrics
                })
        else:
            logger.info("No existing summary found, collecting from individual evaluations...")
            batch_results = collect_existing_results(output_base)
        
        success_count = len([r for r in batch_results if r['success']])
        logger.info(f"Found {success_count} successful experiments")
        
        if success_count > 0:
            generate_aggregate_report(batch_results, output_base)
        
        logger.info(f"\n✅ Summary generated from existing results in: {output_base}")
        return
    
    # Parse dates
    dates = generate_date_list(args.dates)
    logger.info(f"Evaluating {len(dates)} day(s): {dates}")
    
    # Store results
    batch_results = []
    
    # Process each date
    for year, doy in dates:
        date_str = f"{year}-{doy:03d}"
        logger.info("\n" + "="*70)
        logger.info(f"Processing Date: {date_str} ({year} DOY {doy})")
        logger.info("="*70)
        
        # Create date-specific directory
        date_dir = output_base / f"{year}_DOY_{doy:03d}"
        date_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            'year': year,
            'doy': doy,
            'date': date_str,
            'success': False,
            'stec_experiment': None,
            'vtec_experiment': None,
            'metrics': {}
        }
        
        if not args.skip_training:
            # Step 0: Ensure pretrain exists for STEC (only check once, first day)
            pretrain_folder = args.pretrain_folder
            if pretrain_folder is None and (year, doy) == dates[0]:
                logger.info("\n[0/3] Checking pretrain experiment...")
                pretrain_success, pretrain_folder = ensure_pretrain_exists(args.stec_config, date_dir)
                if not pretrain_success:
                    logger.error("✗ Failed to ensure pretrain experiment exists, aborting")
                    break
                # Use this pretrain folder for all subsequent days
                args.pretrain_folder = pretrain_folder
                pretrain_folder = args.pretrain_folder
            
            # Check if all output processing is already done for this day
            try:
                # Create temporary configs to check for existence
                temp_stec_config = create_modified_config(args.stec_config, year, doy, 
                                                    date_dir, is_vtec=False,
                                                    pretrain_folder=pretrain_folder)
                temp_vtec_config = create_modified_config(args.vtec_config, year, doy,
                                                    date_dir, is_vtec=True)
                
                stec_exists, stec_name = check_experiment_exists(temp_stec_config)
                vtec_exists, vtec_name = check_experiment_exists(temp_vtec_config)
                
                # Check for evaluation results
                eval_dir = date_dir / "evaluation"
                comparison_log = eval_dir / "comparison.log"
                # Check primarily for the result file of the first dataset (own_vtec_gim)
                # We check this specific path because extract_metrics_from_experiment looks here
                metrics_file = eval_dir / "own_vtec_gim" / "metrics_summary.csv"
                
                eval_exists = comparison_log.exists() and metrics_file.exists()
                
                if stec_exists and vtec_exists and eval_exists:
                    # Attempt to load metrics to ensure the run was actually successful/readable where we expect it
                    metrics = extract_metrics_from_experiment(eval_dir)
                    if not metrics:
                         # If metrics dict is empty, the run was incomplete
                         raise ValueError("Empty metrics extracted")

                    logger.info(f"✓ All steps completed for {date_str}, skipping day.")
                    result['success'] = True
                    result['stec_experiment'] = stec_name
                    result['vtec_experiment'] = vtec_name
                    result['metrics'] = metrics
                    batch_results.append(result)
                    continue
            except Exception as e:
                # Fallback to re-running if check fails
                # logger.debug(f"Day {date_str} not skipped due to: {e}")
                pass

            # Step 1: Train STEC model
            logger.info(f"\n[1/3] Training STEC model for {date_str}")
            stec_config = create_modified_config(args.stec_config, year, doy, 
                                                date_dir, is_vtec=False,
                                                pretrain_folder=args.pretrain_folder)
            
            # Check if STEC experiment already exists
            stec_exists, stec_exp_name = check_experiment_exists(stec_config)
            
            if args.skip_existing and stec_exists:
                logger.info(f"✓ Found existing STEC experiment: {stec_exp_name}, skipping training")
                stec_success = True
                stec_exp = stec_exp_name
            else:
                stec_success, stec_exp = run_training(stec_config)
            
            if not stec_success:
                logger.error(f"✗ STEC training failed for {date_str}, skipping...")
                batch_results.append(result)
                continue
            
            result['stec_experiment'] = stec_exp
            
            # Step 2: Train VTEC model
            logger.info(f"\n[2/3] Training VTEC model for {date_str}")
            vtec_config = create_modified_config(args.vtec_config, year, doy,
                                                date_dir, is_vtec=True)
            
            # Check if VTEC experiment already exists to skip redundant training
            vtec_exists, vtec_exp_name = check_experiment_exists(vtec_config)
            
            if (args.skip_existing or True) and vtec_exists: # existing logic always skipped VTEC if existed, keeping that but making it explicit or ensuring skip_existing covers it?
                # The original code unconditionally skipped VTEC if it existed.
                # "if vtec_exists:" was the original check.
                # So I should probably keep it as is, or maybe apply skip_existing to it too?
                # The user only asked for STEC. But usually skip_existing implies both.
                # The original code for VTEC was:
                # if vtec_exists: ...
                # So it was ALWAYS skipping existing VTEC.
                # I will touch STEC logic only as requested, but maybe it's better to use the flag for STEC.
                pass

            if vtec_exists: # Original behavior: always skip existing VTEC
                logger.info(f"✓ Found existing VTEC experiment: {vtec_exp_name}, skipping training")
                vtec_success = True
                vtec_exp = vtec_exp_name
            else:
                vtec_success, vtec_exp = run_training(vtec_config)
            
            if not vtec_success:
                logger.error(f"✗ VTEC training failed for {date_str}, skipping...")
                batch_results.append(result)
                continue
            
            result['vtec_experiment'] = vtec_exp
        
        else:
            # Find existing experiments for this date
            logger.info(f"Looking for existing experiments for {date_str}...")
            
            # Helper to check existence
            def find_exp(config_path, is_vtec):
                try:
                    with open(config_path, 'r') as f:
                        cfg = yaml.safe_load(f)
                    
                    cfg['mode'] = 'finetune'
                    cfg['year'] = str(year)
                    cfg['doy'] = str(doy).zfill(3)
                    
                    if is_vtec:
                        cfg['finetune']['finetune_from_scratch'] = True
                        if 'target' not in cfg: cfg['target'] = 'vtec'
                    else:
                        cfg['finetune']['finetune_from_scratch'] = False
                        if 'target' not in cfg: cfg['target'] = 'stec'

                    exp_name = compute_exp_name(cfg)
                    exp_path = Path("experiments") / exp_name
                    
                    if exp_path.exists():
                         return True, exp_name
                    else:
                         return False, exp_name
                except Exception as e:
                    logger.error(f"Error computing experiment name: {e}")
                    return False, None

            # STEC
            stec_found, stec_exp = find_exp(args.stec_config, is_vtec=False)
            if stec_found:
                logger.info(f"✓ Found existing STEC experiment: {stec_exp}")
                result['stec_experiment'] = stec_exp
            else:
                logger.error(f"✗ STEC experiment not found (expected: {stec_exp}), skipping {date_str}")
                batch_results.append(result)
                continue

            # VTEC
            vtec_found, vtec_exp = find_exp(args.vtec_config, is_vtec=True)
            if vtec_found:
                 logger.info(f"✓ Found existing VTEC experiment: {vtec_exp}")
                 result['vtec_experiment'] = vtec_exp
            else:
                 logger.error(f"✗ VTEC experiment not found (expected: {vtec_exp}), skipping {date_str}")
                 batch_results.append(result)
                 continue
        
        # Step 3: Run comparison evaluation
        comp_success = False
        if not args.skip_comparison:
            logger.info(f"\n[3/3] Running comparison evaluation for {date_str}")
            comp_success = run_comparison(
                result['stec_experiment'],
                result['vtec_experiment'],
                date_dir / "evaluation",
                args.num_inference_samples
            )
        else:
            logger.info(f"\n[3/3] Skipping comparison evaluation for {date_str}")
            comp_success = True

        if comp_success:
            result['success'] = True
            if not args.skip_comparison:
                result['metrics'] = extract_metrics_from_experiment(date_dir / "evaluation")
                logger.info(f"✓ All steps completed for {date_str}")
            else:
                logger.info(f"✓ Training steps completed for {date_str}")
            
            # Step 4: Run positioning evaluation (Optional)
            if args.positioning:
                logger.info(f"\n[4/4] Running positioning evaluation for {date_str}")
                pos_success = run_positioning_pipeline(result['stec_experiment'], year, doy)
                if pos_success:
                    logger.info(f"✓ Positioning evaluation completed for {date_str}")
                else:
                    logger.warning(f"⚠️ Positioning evaluation failed for {date_str}")
                    # We don't mark the whole day as failed just because positioning failed, 
                    # as the main STEC/VTEC comparison might be the primary goal.
                    # Unless strict mode is desired. For now, just warn.
        else:
            logger.error(f"✗ Comparison failed for {date_str}")
        
        batch_results.append(result)
    
    # Generate aggregate report
    logger.info("\n" + "="*70)
    logger.info("MULTI-DAY PROCESSING COMPLETE")
    logger.info("="*70)
    
    success_count = sum(1 for r in batch_results if r['success'])
    logger.info(f"Successful: {success_count}/{len(batch_results)} days")
    
    if success_count > 0 and not args.no_aggregate:
        generate_aggregate_report(batch_results, output_base)
    
    logger.info(f"\n✅ All results saved to: {output_base}")


if __name__ == "__main__":
    main()
