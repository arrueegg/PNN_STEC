#!/bin/bash
"""
Parallel Multi-Day Evaluation on Cluster

This script splits a date range into chunks and submits parallel SLURM jobs,
each processing a subset of days for faster cluster execution.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def parse_date_string(date_str: str) -> Tuple[int, int]:
    """Parse date string to (year, doy)."""
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
    """Generate list of (year, doy) tuples from date argument."""
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

def split_dates_into_chunks(dates: List[Tuple[int, int]], chunk_size: int) -> List[List[Tuple[int, int]]]:
    """Split date list into chunks of specified size."""
    return [dates[i:i + chunk_size] for i in range(0, len(dates), chunk_size)]

def ensure_pretrain_exists(stec_config_path: str) -> str:
    """Check if pretrain experiment exists, run if not. Returns pretrain folder path."""
    import yaml
    from pathlib import Path
    
    # Load config to determine what pretrain experiment should exist
    with open(stec_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Temporarily set mode to pretrain to compute pretrain experiment name
    original_mode = config.get('mode', 'pretrain')
    config['mode'] = 'pretrain'
    
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from utils.config_parser import compute_exp_name
    
    # Compute what the pretrain experiment name should be
    try:
        pretrain_exp_name = compute_exp_name(config)
    except Exception as e:
        print(f"Failed to compute pretrain experiment name: {e}")
        return None
    
    pretrain_folder = Path("experiments") / pretrain_exp_name
    model_folder = pretrain_folder / "model"
    
    # Check if pretrain experiment exists with trained model
    if pretrain_folder.exists() and model_folder.exists() and list(model_folder.glob("*.pth")):
        print(f"✓ Found existing pretrain experiment: {pretrain_exp_name}")
        return str(pretrain_folder)
    
    # Pretrain doesn't exist - need to run it
    print(f"⚠️  Pretrain experiment not found: {pretrain_exp_name}")
    print("Running pretraining first...")
    
    # Run pretrain using CLI
    import subprocess
    result = subprocess.run([
        "python", "cli.py", "train", "--config", stec_config_path
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"✗ Pretraining failed: {result.stderr}")
        return None
    
    # Verify model was trained
    if not (model_folder.exists() and list(model_folder.glob("*.pth"))):
        print(f"✗ Pretrain completed but no model found in {model_folder}")
        return None
    
    print(f"✓ Pretraining completed: {pretrain_exp_name}")
    return str(pretrain_folder)

def create_slurm_script(chunk_dates: List[Tuple[int, int]], chunk_id: int,
                       stec_config: str, vtec_config: str, output_dir: str,
                       num_inference_samples: int, base_dir: str, pretrain_folder: str) -> str:
    """Create SLURM script for a chunk of dates."""

    # Convert dates to string format
    date_strings = [f"{year}-{doy:03d}" for year, doy in chunk_dates]
    dates_arg = ",".join(date_strings)

    script_content = f'''#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=multiday_parallel/logs/chunk_{chunk_id:02d}-%j.out
#SBATCH --job-name=multiday_{chunk_id:02d}

set -euo pipefail

############################
# 1) Modules & environment
############################
module load stack/2024-06 python_cuda/3.11.6
module load eth_proxy

main_dir="{base_dir}"
cd "$main_dir"
source "${{main_dir}}/env/bin/activate"

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}

############################
# 2) Run multiday evaluation for this chunk
############################
python cli.py multiday \\
    --dates "{dates_arg}" \\
    --stec_config "{stec_config}" \\
    --vtec_config "{vtec_config}" \\
    --output_dir "{output_dir}" \\
    --num_inference_samples {num_inference_samples} \\
    --pretrain_folder "{pretrain_folder}" \\
    --no_aggregate

echo "✅ Chunk {chunk_id} completed successfully"
'''

    return script_content

def main():
    parser = argparse.ArgumentParser(
        description="Submit parallel multi-day evaluation jobs to SLURM cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit 7-day evaluation in chunks of 2 days each (4 parallel jobs)
  python scripts/submit_parallel_multiday.py \\
      --dates "2024-183:2024-189" \\
      --chunk_size 2 \\
      --stec_config config/config.yaml \\
      --vtec_config config/config_vtec_mlp_baseline.yaml

  # Submit 10-day evaluation in chunks of 3 days each (4 jobs, last has 1 day)
  python scripts/submit_parallel_multiday.py \\
      --dates "2024-183:2024-192" \\
      --chunk_size 3 \\
      --stec_config config/config.yaml \\
      --vtec_config config/config_vtec_mlp_baseline.yaml \\
      --output_dir multiday_results/parallel_july_2024
        """
    )

    parser.add_argument("--dates", type=str, required=True,
                       help="Date range to evaluate (e.g., '2024-183:2024-192')")
    parser.add_argument("--chunk_size", type=int, default=3,
                       help="Number of days per SLURM job (default: 3)")
    parser.add_argument("--stec_config", type=str, required=True,
                       help="Base config file for STEC training")
    parser.add_argument("--vtec_config", type=str, required=True,
                       help="Base config file for VTEC training")
    parser.add_argument("--output_dir", type=str, default="multiday_results_parallel",
                       help="Base output directory (default: multiday_results_parallel)")
    parser.add_argument("--num_inference_samples", type=int, default=100,
                       help="MC samples for Bayesian inference (default: 100)")
    parser.add_argument("--base_dir", type=str,
                       default="/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC",
                       help="Base directory on cluster")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be submitted without actually submitting")

    args = parser.parse_args()

    print("="*70)
    print("PARALLEL MULTI-DAY EVALUATION SUBMISSION")
    print("="*70)

    # Step 1: Ensure pretrain exists
    print("\n[1/2] Checking pretrain experiment...")
    pretrain_folder = ensure_pretrain_exists(args.stec_config)
    if pretrain_folder is None:
        print("❌ Failed to ensure pretrain experiment exists")
        return
    
    print(f"Pretrain folder: {pretrain_folder}")

    # Step 2: Generate all dates
    print("\n[2/2] Setting up parallel jobs...")
    all_dates = generate_date_list(args.dates)
    print(f"Total dates to process: {len(all_dates)}")
    print(f"Dates: {[f'{y}-{d:03d}' for y,d in all_dates]}")

    # Split into chunks
    date_chunks = split_dates_into_chunks(all_dates, args.chunk_size)
    print(f"Number of chunks: {len(date_chunks)} (chunk size: {args.chunk_size})")

    # Create output directories
    scripts_dir = Path("multiday_parallel/scripts")
    logs_dir = Path("multiday_parallel/logs")
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Submit jobs
    job_ids = []

    for i, chunk in enumerate(date_chunks):
        chunk_id = i + 1
        date_strings = [f"{year}-{doy:03d}" for year, doy in chunk]
        print(f"\nChunk {chunk_id}: {len(chunk)} days - {date_strings}")

        # Create SLURM script
        script_content = create_slurm_script(
            chunk, chunk_id, args.stec_config, args.vtec_config,
            args.output_dir, args.num_inference_samples, args.base_dir, pretrain_folder
        )

        script_path = scripts_dir / f"chunk_{chunk_id:02d}.sh"

        # Write script
        with open(script_path, 'w') as f:
            f.write(script_content)

        print(f"  Created script: {script_path}")

        if not args.dry_run:
            # Submit job
            try:
                result = subprocess.run(
                    ["sbatch", "--parsable", str(script_path)],
                    capture_output=True, text=True, check=True
                )
                job_id = result.stdout.strip()
                job_ids.append(job_id)
                print(f"  Submitted job: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to submit job: {e}")
                continue
        else:
            print("  (Dry run - not submitted)")

    print("\n" + "="*70)
    if not args.dry_run:
        print(f"✅ Submitted {len(job_ids)} jobs to cluster")
        if job_ids:
            print(f"Job IDs: {', '.join(job_ids)}")
            print(f"Monitor with: squeue -u $USER")
            print(f"Cancel all with: scancel {' '.join(job_ids)}")
    else:
        print("✅ Dry run complete - scripts created but not submitted")

    print(f"Results will be in: {args.output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
