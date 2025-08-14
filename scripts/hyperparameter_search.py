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

def generate_search(grid_name='quick', output_dir='hp_search'):
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
    
    # Generate parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"🎯 Grid: {grid_name}")
    print(f"📊 Combinations: {len(combinations)}")
    print(f"📁 Output: {output_path}")
    
    # Generate configs and run script
    run_commands = []
    
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
        
        # Print progress
        param_str = ", ".join([f"{k.split('.')[-1]}={v}" for k, v in params.items()])
        print(f"  Trial {i:2d}: {param_str}")
    
    # Create run script
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
    
    args = parser.parse_args()
    
    generate_search(args.grid, args.output)

if __name__ == "__main__":
    main()
