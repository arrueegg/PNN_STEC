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
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_parser import parse_config

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
                          output_dir: Path, is_vtec: bool = False) -> str:
    """Create modified config file for specific date.
    
    Returns path to temporary config file.
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for this date
    config['mode'] = 'finetune'
    config['finetune']['year'] = year
    config['finetune']['doy'] = doy
    
    # Set finetune_from_scratch for VTEC
    if is_vtec:
        config['finetune']['finetune_from_scratch'] = True
    else:
        config['finetune']['finetune_from_scratch'] = False
    
    # Create temporary config file
    model_type = "vtec" if is_vtec else "stec"
    temp_config_path = output_dir / f"temp_config_{model_type}_{year}_{doy:03d}.yaml"
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return str(temp_config_path)


def run_training(config_path: str) -> Tuple[bool, str]:
    """Run training for a given config.
    
    Returns (success, experiment_name).
    """
    logger.info(f"Training with config: {config_path}")
    
    # Build command - use sys.executable to get current Python interpreter
    cmd = [sys.executable, "src/main.py", "--config", config_path]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract experiment name from output
        # Look for "Experiment directory: experiments/..."
        experiment_name = None
        for line in result.stdout.split('\n') + result.stderr.split('\n'):
            if "Experiment directory:" in line or "experiment_dir" in line.lower():
                # Extract experiment folder name
                if "experiments/" in line:
                    parts = line.split("experiments/")
                    if len(parts) > 1:
                        experiment_name = parts[1].strip().split()[0]
                        break
        
        if not experiment_name:
            # Fallback: find most recent experiment directory
            experiments_dir = Path(__file__).parent.parent / "experiments"
            exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
            if exp_dirs:
                experiment_name = max(exp_dirs, key=lambda x: x.stat().st_mtime).name
        
        logger.info(f"✓ Training completed: {experiment_name}")
        return True, experiment_name
    
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Training failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False, None


def run_comparison(stec_exp: str, vtec_exp: str, output_dir: Path, 
                  num_samples: int = 100) -> bool:
    """Run comprehensive comparison evaluation.
    
    Returns success status.
    """
    logger.info(f"Running comparison: {stec_exp} vs {vtec_exp}")
    
    # Build command - use sys.executable to get current Python interpreter
    cmd = [
        sys.executable,
        "src/compare_stec_vtec_gim.py",
        "--stec_experiment", stec_exp,
        "--vtec_experiment", vtec_exp,
        "--num_inference_samples", str(num_samples),
        "--output_dir", str(output_dir)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"✓ Comparison completed")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Comparison failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False


def extract_metrics_from_experiment(stec_exp_name: str) -> Dict[str, Dict]:
    """Extract metrics from experiment evaluation results.
    
    Returns dict with metrics for each dataset type.
    """
    exp_dir = Path("experiments") / stec_exp_name / "evaluation"
    
    metrics = {}
    
    # Check for both dataset types
    for dataset_type in ["own_vtec_gim", "madrigal_vtec_gim"]:
        dataset_dir = exp_dir / dataset_type
        metrics_file = dataset_dir / "metrics_summary.csv"
        
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            metrics[dataset_type] = df.to_dict('records')
    
    return metrics


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
    generate_aggregate_plots(df, summary_dir)
    
    logger.info(f"\n✅ Aggregate report saved to: {summary_dir}")


def generate_aggregate_plots(df: pd.DataFrame, output_dir: Path):
    """Generate publication-ready aggregate plots."""
    
    logger.info("Generating aggregate plots...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 10
    
    # 1. RMSE comparison across days
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, dataset in enumerate(df['dataset'].unique()):
        ax = axes[idx]
        dataset_df = df[df['dataset'] == dataset]
        
        # Pivot for plotting
        pivot_df = dataset_df.pivot(index='date', columns='Model', values='RMSE')
        
        pivot_df.plot(ax=ax, marker='o', linewidth=2, markersize=6)
        ax.set_xlabel('Date (YYYY-DOY)', fontsize=11)
        ax.set_ylabel('RMSE (TECU)', fontsize=11)
        ax.set_title(f'RMSE by Date - {dataset}', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_by_date.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plots comparing models
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['RMSE', 'MAE', 'R²', 'Bias']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Combine all datasets
        sns.boxplot(data=df, x='Model', y=metric, hue='dataset', ax=ax)
        ax.set_title(f'{metric} Distribution Across All Days', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.legend(title='Dataset', loc='best')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Improvement statistics
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate improvements for Direct STEC over baselines
    improvements = []
    
    for date in df['date'].unique():
        for dataset in df['dataset'].unique():
            subset = df[(df['date'] == date) & (df['dataset'] == dataset)]
            
            stec_rmse = subset[subset['Model'] == 'Direct STEC']['RMSE'].values
            vtec_rmse = subset[subset['Model'] == 'VTEC + Mapping']['RMSE'].values
            gim_rmse = subset[subset['Model'] == 'IGS GIM']['RMSE'].values
            
            if len(stec_rmse) > 0 and len(vtec_rmse) > 0:
                imp_vtec = (1 - stec_rmse[0] / vtec_rmse[0]) * 100
                improvements.append({'date': date, 'dataset': dataset, 
                                   'baseline': 'VTEC+Mapping', 'improvement': imp_vtec})
            
            if len(stec_rmse) > 0 and len(gim_rmse) > 0:
                imp_gim = (1 - stec_rmse[0] / gim_rmse[0]) * 100
                improvements.append({'date': date, 'dataset': dataset,
                                   'baseline': 'GIM', 'improvement': imp_gim})
    
    if improvements:
        imp_df = pd.DataFrame(improvements)
        sns.barplot(data=imp_df, x='date', y='improvement', hue='baseline', ax=ax)
        ax.set_xlabel('Date (YYYY-DOY)', fontsize=11)
        ax.set_ylabel('RMSE Improvement (%)', fontsize=11)
        ax.set_title('Direct STEC Improvement Over Baselines', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.legend(title='Baseline', loc='best')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'improvement_by_date.png', dpi=300, bbox_inches='tight')
    
    plt.close()
    
    logger.info(f"✓ Plots saved to {output_dir}")


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
    
    parser.add_argument("--dates", type=str, required=True,
                       help="Date(s) to evaluate (single, comma-separated, or range)")
    parser.add_argument("--stec_config", type=str, required=True,
                       help="Base config file for STEC training")
    parser.add_argument("--vtec_config", type=str, required=True,
                       help="Base config file for VTEC training")
    parser.add_argument("--output_dir", type=str, default="multiday_results",
                       help="Output directory for all results (default: multiday_results)")
    parser.add_argument("--num_inference_samples", type=int, default=100,
                       help="MC samples for Bayesian inference (default: 100)")
    parser.add_argument("--test_size", type=int, default=None,
                       help="Test set size (default: full)")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training, only run evaluation (experiments must exist)")
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("MULTI-DAY EVALUATION PIPELINE")
    logger.info("="*70)
    
    # Parse dates
    dates = generate_date_list(args.dates)
    logger.info(f"Evaluating {len(dates)} day(s): {dates}")
    
    # Create output directory
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
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
            # Step 1: Train STEC model
            logger.info(f"\n[1/3] Training STEC model for {date_str}")
            stec_config = create_modified_config(args.stec_config, year, doy, 
                                                date_dir, is_vtec=False)
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
            vtec_success, vtec_exp = run_training(vtec_config)
            
            if not vtec_success:
                logger.error(f"✗ VTEC training failed for {date_str}, skipping...")
                batch_results.append(result)
                continue
            
            result['vtec_experiment'] = vtec_exp
        
        else:
            # Find existing experiments for this date
            logger.info(f"Looking for existing experiments for {date_str}...")
            # User needs to provide experiment names or we find them
            # For now, skip this complexity
            logger.error("--skip_training not yet fully implemented")
            continue
        
        # Step 3: Run comparison evaluation
        logger.info(f"\n[3/3] Running comparison evaluation for {date_str}")
        comp_success = run_comparison(
            result['stec_experiment'],
            result['vtec_experiment'],
            date_dir / "evaluation",
            args.num_inference_samples
        )
        
        if comp_success:
            result['success'] = True
            result['metrics'] = extract_metrics_from_experiment(result['stec_experiment'])
            logger.info(f"✓ All steps completed for {date_str}")
        else:
            logger.error(f"✗ Comparison failed for {date_str}")
        
        batch_results.append(result)
    
    # Generate aggregate report
    logger.info("\n" + "="*70)
    logger.info("MULTI-DAY PROCESSING COMPLETE")
    logger.info("="*70)
    
    success_count = sum(1 for r in batch_results if r['success'])
    logger.info(f"Successful: {success_count}/{len(batch_results)} days")
    
    if success_count > 0:
        generate_aggregate_report(batch_results, output_base)
    
    logger.info(f"\n✅ All results saved to: {output_base}")


if __name__ == "__main__":
    main()
