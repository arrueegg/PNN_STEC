#!/usr/bin/env python3
"""
Simple Hyperparameter Grid Search

Clean and practical hyperparameter tuning with sensible defaults.
"""

import os
import yaml
import itertools
import argparse
from pathlib import Path
from datetime import datetime


def define_grids():
    """Define parameter grids - edit these as needed"""

    grids = {
        # mini test grid (2 combinations)
        "mini": {
            "model.hidden_dim": [128],
            "model.num_layers": [3],
            "model.model_type": ["BNN_NLL"],
            "pretrain.learning_rate": [0.01, 0.05],
            "data.train_subset_size": [10_000],
        },
        # Standard grid
        "standard": {
            "target": ["stec"],
            "model.hidden_dim": [256, 512, 1024],
            "model.num_layers": [4, 8],
            "model.model_type": ["BNN_NLL", "MLP_NLL"],
            "pretrain.learning_rate": [0.01, 0.001],
            "training.loss_weight": [1.0],
            "training.loss_function": ["GaussianNLLLoss", "MSELoss"],
            "training.target_weighting.enabled": [True, False],
        },
        # Custom grid (many combinations)
        "custom": {
            "target": ["stec", "vtec"],
            "model.hidden_dim": [128, 256, 512],
            "model.num_layers": [4],
            "model.model_type": ["BNN_NLL", "MLP_NLL"],
            "training.loss_function": ["GaussianNLLLoss", "MSELoss"],
            "training.loss_weight": [0.1, 1.0],
            "pretrain.learning_rate": [0.001, 0.01, 0.1],
            "pretrain.batchsize": [256, 512, 4096],
        },
    }

    return grids


def load_base_config():
    """Load the base configuration"""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def apply_params_to_config(config, params, cluster_mode=False):
    """Apply hyperparameters to config"""
    import copy

    new_config = copy.deepcopy(config)

    for param_name, value in params.items():
        keys = param_name.split(".")
        current = new_config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value

    # Set cluster flag and adjust paths for cluster execution
    new_config["cluster"] = cluster_mode
    if cluster_mode:
        new_config["data"][
            "scratch_dir"
        ] = "/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data/"
        new_config["data"][
            "GNSS_data_path"
        ] = "/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data/STEC_DB_CASDCB"
        new_config["data"][
            "SWI_data_path"
        ] = "/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data/SWI/"

    return new_config


def generate_slurm_script(trial_id, config_file, output_path):
    """Generate SLURM script for a single trial"""
    # Import cluster configuration
    try:
        import sys
        import os

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
        )
        sys.path.insert(0, config_path)
        # Add parent directory of 'scripts' to sys.path to allow import from 'config'
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, parent_dir)
        from config.cluster_config import (
            CLUSTER_PATHS,
            MODULE_COMMANDS,
            DEFAULT_SLURM_SETTINGS,
        )
    except ImportError as e:
        print(f"Error importing cluster configuration: {e}")
        # Fallback to default values if cluster_config not available
        CLUSTER_PATHS = {"main_dir": "/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC"}
        MODULE_COMMANDS = [
            "module load stack/2024-06 python_cuda/3.11.6",
            "module load eth_proxy",
        ]
        DEFAULT_SLURM_SETTINGS = {
            "ntasks": 1,
            "cpus_per_task": 12,
            "time": "2:00:00",
            "mem_per_cpu": "10G",
            "gpus": 1,
        }

    slurm_script_path = output_path / "slurm_scripts" / f"trial_{trial_id:03d}.sh"
    log_path = output_path / "logs" / f"trial_{trial_id:03d}-%j.out"

    with open(slurm_script_path, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"#SBATCH --ntasks={DEFAULT_SLURM_SETTINGS['ntasks']}\n")
        f.write(f"#SBATCH --cpus-per-task={DEFAULT_SLURM_SETTINGS['cpus_per_task']}\n")
        f.write(f"#SBATCH --time={DEFAULT_SLURM_SETTINGS['time']}\n")
        f.write(f"#SBATCH --mem-per-cpu={DEFAULT_SLURM_SETTINGS['mem_per_cpu']}\n")
        if DEFAULT_SLURM_SETTINGS.get("gpus"):
            f.write(f"#SBATCH --gpus={DEFAULT_SLURM_SETTINGS['gpus']}\n")
        f.write(f"#SBATCH --output={log_path}\n")
        f.write(f"#SBATCH --job-name=hp_trial_{trial_id:03d}\n\n")

        f.write("# Load modules\n")
        for module_cmd in MODULE_COMMANDS:
            f.write(f"{module_cmd}\n")
        f.write("\n")

        f.write("# Setup environment\n")
        f.write(f"main_dir=\"{CLUSTER_PATHS['main_dir']}\"\n")
        f.write("cd $main_dir\n")
        f.write("source ${main_dir}/env/bin/activate\n\n")

        f.write("# Run trial\n")
        f.write(f'echo "🚀 Starting hyperparameter trial {trial_id}"\n')
        f.write(f"python src/main.py --config_path {config_file}\n")
        f.write(f'echo "✅ Completed hyperparameter trial {trial_id}"\n')

    os.chmod(slurm_script_path, 0o755)
    return slurm_script_path


def generate_search(grid_name="quick", output_dir="hp_search", cluster_mode=False):
    """Generate hyperparameter search"""

    # Setup
    grids = define_grids()
    if grid_name not in grids:
        raise ValueError(f"Unknown grid: {grid_name}. Available: {list(grids.keys())}")

    param_grid = grids[grid_name]
    base_config = load_base_config()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    if cluster_mode:
        (output_path / "slurm_scripts").mkdir(exist_ok=True)
        (output_path / "logs").mkdir(exist_ok=True)

    # Generate parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"🎯 Grid: {grid_name}")
    print(f"📊 Combinations: {len(combinations)}")
    print(f"📁 Output: {output_path}")
    if cluster_mode:
        print("🖥️ Cluster mode: ENABLED")
        print(
            "📂 Data paths adjusted for cluster: /cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data/"
        )

    # Generate configs and run scripts
    run_commands = []
    slurm_jobs = []

    for i, combination in enumerate(combinations, 1):
        params = dict(zip(keys, combination))

        # Create trial config
        trial_config = apply_params_to_config(base_config, params, cluster_mode)

        # Set trial output directory
        trial_config["output_dir"] = f"{output_dir}/results/trial_{i:03d}/"

        # Save config
        os.makedirs(output_path / "configs", exist_ok=True)
        config_file = output_path / "configs" / f"config_{i:03d}.yaml"
        with open(config_file, "w") as f:
            yaml.dump(trial_config, f, default_flow_style=False, indent=2)

        # Add run command
        run_commands.append(f"python src/main.py --config_path {config_file}")

        # Generate SLURM script if in cluster mode
        if cluster_mode:
            slurm_script = generate_slurm_script(i, config_file, output_path)
            slurm_jobs.append(slurm_script)

        # Print progress
        param_str = ", ".join([f"{k.split('.')[-1]}={v}" for k, v in params.items()])
        print(f"  Trial {i:2d}: {param_str}")

    # Create run scripts
    if cluster_mode:
        # Create SLURM submission script
        submit_script = output_path / "submit_all.sh"
        with open(submit_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Submit all hyperparameter trials to SLURM\n")
            f.write(f"# Grid: {grid_name}, Trials: {len(combinations)}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

            f.write('echo "🚀 Submitting hyperparameter trials to cluster..."\n')
            f.write("job_ids=()\n\n")

            for i, slurm_script in enumerate(slurm_jobs, 1):
                f.write(f'echo "Submitting trial {i}/{len(slurm_jobs)}"\n')
                f.write(f"job_id=$(sbatch --parsable {slurm_script})\n")
                f.write("job_ids+=($job_id)\n")
                f.write('echo "  Job ID: $job_id"\n\n')

            f.write('echo "📊 Submitted ${#job_ids[@]} jobs to cluster"\n')
            f.write('echo "Job IDs: ${job_ids[*]}"\n')
            f.write("echo\n")
            f.write('echo "📋 Monitor with: squeue -u $USER"\n')
            f.write('echo "📋 Cancel all with: scancel ${job_ids[*]}"\n')

        os.chmod(submit_script, 0o755)

        print(f"\n✅ Generated {len(combinations)} configs and SLURM scripts")
        print(f"🖥️ Submit to cluster: ./{output_dir}/submit_all.sh")
        print(f"🔧 Single trial: sbatch {output_dir}/slurm_scripts/trial_001.sh")
        print("📋 Monitor: squeue -u $USER")

    else:
        # Create local run script
        run_script = output_path / "run_search.sh"
        with open(run_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Hyperparameter search: {grid_name}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("source env/bin/activate\n\n")

            for i, cmd in enumerate(run_commands, 1):
                f.write(f'echo "🚀 Trial {i}/{len(run_commands)}"\n')
                f.write(f"{cmd}\n")
                f.write("echo\n")

            f.write('echo "✅ Search complete!"\n')

        os.chmod(run_script, 0o755)

        print(f"\n✅ Generated {len(combinations)} configs")
        print(f"🚀 Run search: ./{output_dir}/run_search.sh")
        print(
            f"🔧 Single trial: python src/main.py --config_path {output_dir}/configs/config_001.yaml"
        )


def main():
    parser = argparse.ArgumentParser(description="Simple hyperparameter search")
    parser.add_argument(
        "--grid",
        choices=["mini", "standard", "custom"],
        default="standard",
        help="Parameter grid to use",
    )
    parser.add_argument("--output", default="hp_search", help="Output directory")
    parser.add_argument(
        "--cluster",
        action="store_true",
        default=True,
        help="Generate SLURM scripts for cluster execution",
    )

    args = parser.parse_args()

    generate_search(args.grid, args.output, args.cluster)


if __name__ == "__main__":
    main()
