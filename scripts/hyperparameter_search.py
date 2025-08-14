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
        'mini': {
            'model.hidden_dim': [128],
            'model.num_layers': [3],
            'model.model_type': ['BNN_NLL'],
            'pretrain.learning_rate': [0.01, 0.05],
            'data.train_subset_size': [10_000],
        },
        
        # Standard grid (72 combinations) 
        'standard': {
            'model.hidden_dim': [128, 256, 512],
            'model.num_layers': [3, 4],
            'model.model_type': ['BNN_NLL'],
            'pretrain.learning_rate': [0.01, 0.05],
            'pretrain.batchsize': [512],
        },

        # Custom grid (360 combinations)
        'custom': {
            'model.hidden_dim': [128, 256],
            'model.num_layers': [3, 4],
            'model.model_type': ['BNN_NLL'],
            'training.loss_function': ['GaussianNLLLoss', 'MSELoss'],
            'training.loss_weight': [0.1, 0.5, 1.0],
            'pretrain.learning_rate': [0.001, 0.01, 0.05],
            'pretrain.batchsize': [128, 256, 512, 1024, 4096],
        }
    }
    
    return grids

def load_base_config():
    """Load the base configuration"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def apply_params_to_config(config, params):
    """Apply hyperparameters to config"""
    import copy
    new_config = copy.deepcopy(config)
    
    for param_name, value in params.items():
        keys = param_name.split('.')
        current = new_config
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        current[keys[-1]] = value
    
    return new_config

def generate_slurm_script(trial_id, config_file, output_path):
    """Generate SLURM script for a single trial"""
    slurm_script_path = output_path / 'slurm_scripts' / f'trial_{trial_id:02d}.sh'
    log_path = output_path / 'logs' / f'trial_{trial_id:02d}-%j.out'
    
    with open(slurm_script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("#SBATCH --ntasks=1\n")
        f.write("#SBATCH --cpus-per-task=8\n")
        f.write("#SBATCH --time=24:00:00\n")
        f.write("#SBATCH --mem-per-cpu=4G\n")
        f.write("#SBATCH --gres=gpu:1\n")  # Request 1 GPU
        f.write(f"#SBATCH --output={log_path}\n")
        f.write(f"#SBATCH --job-name=hp_trial_{trial_id:02d}\n\n")
        
        f.write("# Load modules\n")
        f.write("module load stack/2024-06 python_cuda/3.11.6\n")
        f.write("module load eth_proxy\n\n")
        
        f.write("# Setup environment\n")
        f.write("main_dir=\"/scratch2/arrueegg/WP4/PNN_STEC\"\n")
        f.write("cd $main_dir\n")
        f.write("source ${main_dir}/env/bin/activate\n\n")
        
        f.write("# Run trial\n")
        f.write(f"echo \"🚀 Starting hyperparameter trial {trial_id}\"\n")
        f.write(f"python src/main.py --config_path {config_file}\n")
        f.write(f"echo \"✅ Completed hyperparameter trial {trial_id}\"\n")
    
    os.chmod(slurm_script_path, 0o755)
    return slurm_script_path

def generate_search(grid_name='quick', output_dir='hp_search', cluster_mode=False):
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
        (output_path / 'slurm_scripts').mkdir(exist_ok=True)
        (output_path / 'logs').mkdir(exist_ok=True)
    
    # Generate parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"🎯 Grid: {grid_name}")
    print(f"📊 Combinations: {len(combinations)}")
    print(f"📁 Output: {output_path}")
    if cluster_mode:
        print(f"🖥️ Cluster mode: ENABLED")
    
    # Generate configs and run scripts
    run_commands = []
    slurm_jobs = []
    
    for i, combination in enumerate(combinations, 1):
        params = dict(zip(keys, combination))
        
        # Create trial config
        trial_config = apply_params_to_config(base_config, params)
        
        # Set trial output directory
        trial_config['output_dir'] = f"{output_dir}/results/trial_{i:02d}/"
        
        # Reduce epochs for hyperparameter search
        trial_config['pretrain']['epochs'] = min(trial_config['pretrain']['epochs'], 20)
        
        # Save config
        config_file = output_path / f'config_{i:02d}.yaml'
        with open(config_file, 'w') as f:
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
        submit_script = output_path / 'submit_all.sh'
        with open(submit_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Submit all hyperparameter trials to SLURM\n")
            f.write(f"# Grid: {grid_name}, Trials: {len(combinations)}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("echo \"🚀 Submitting hyperparameter trials to cluster...\"\n")
            f.write("job_ids=()\n\n")
            
            for i, slurm_script in enumerate(slurm_jobs, 1):
                f.write(f"echo \"Submitting trial {i}/{len(slurm_jobs)}\"\n")
                f.write(f"job_id=$(sbatch --parsable {slurm_script})\n")
                f.write("job_ids+=($job_id)\n")
                f.write(f"echo \"  Job ID: $job_id\"\n\n")
            
            f.write("echo \"📊 Submitted ${#job_ids[@]} jobs to cluster\"\n")
            f.write("echo \"Job IDs: ${job_ids[*]}\"\n")
            f.write("echo\n")
            f.write("echo \"📋 Monitor with: squeue -u $USER\"\n")
            f.write("echo \"📋 Cancel all with: scancel ${job_ids[*]}\"\n")
        
        os.chmod(submit_script, 0o755)
        
        print(f"\n✅ Generated {len(combinations)} configs and SLURM scripts")
        print(f"🖥️ Submit to cluster: ./{output_dir}/submit_all.sh")
        print(f"🔧 Single trial: sbatch {output_dir}/slurm_scripts/trial_01.sh")
        print(f"📋 Monitor: squeue -u $USER")
    
    else:
        # Create local run script
        run_script = output_path / 'run_search.sh'
        with open(run_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Hyperparameter search: {grid_name}\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write("source env/bin/activate\n\n")
            
            for i, cmd in enumerate(run_commands, 1):
                f.write(f"echo \"🚀 Trial {i}/{len(run_commands)}\"\n")
                f.write(f"{cmd}\n")
                f.write("echo\n")
            
            f.write("echo \"✅ Search complete!\"\n")
        
        os.chmod(run_script, 0o755)
        
        print(f"\n✅ Generated {len(combinations)} configs")
        print(f"🚀 Run search: ./{output_dir}/run_search.sh")
        print(f"🔧 Single trial: python src/main.py --config_path {output_dir}/config_01.yaml")

def main():
    parser = argparse.ArgumentParser(description='Simple hyperparameter search')
    parser.add_argument('--grid', choices=['mini', 'standard', 'custom'], 
                       default='mini', help='Parameter grid to use')
    parser.add_argument('--output', default='hp_search',
                       help='Output directory')
    parser.add_argument('--cluster', action='store_true',
                       help='Generate SLURM scripts for cluster execution')
    
    args = parser.parse_args()
    
    generate_search(args.grid, args.output, args.cluster)

if __name__ == "__main__":
    main()
