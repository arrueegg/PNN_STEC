#!/usr/bin/env python3
"""
WandB Sweep Manager for Euler Cluster
Creates wandb sweeps and submits parallel agents to SLURM.
"""

import os
import sys
import argparse
import subprocess
import yaml
from pathlib import Path
import wandb

def create_sweep(sweep_config_path, project_name="PNN_STEC"):
    """Create a wandb sweep and return the sweep ID"""
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    print(f"📋 Creating wandb sweep for project: {project_name}")
    print(f"📄 Using config: {sweep_config_path}")
    
    # Get the entity name
    entity = wandb.api.default_entity
    if not entity:
        entity = "arno-rueegg"  # fallback to known entity
    
    sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)
    
    print(f"🆔 Sweep created with ID: {sweep_id}")
    print(f"🌐 View sweep at: https://wandb.ai/{entity}/{project_name}/sweeps/{sweep_id}")
    
    return sweep_id

def submit_agents(sweep_id, num_agents, project_name="PNN_STEC"):
    """Submit multiple wandb agents to SLURM"""
    print(f"🚀 Submitting {num_agents} agents to cluster...")
    
    job_ids = []
    script_dir = Path("hp_search/wandb_slurm_scripts")
    script_dir.mkdir(exist_ok=True)
    
    for agent_id in range(1, num_agents + 1):
        slurm_script_path = script_dir / f"wandb_agent_{agent_id:03d}.sh"
        slurm_content = generate_slurm_script(sweep_id, project_name, agent_id)
        
        with open(slurm_script_path, 'w') as f:
            f.write(slurm_content)
        
        os.chmod(slurm_script_path, 0o755)
        
        try:
            result = subprocess.run(
                ['sbatch', '--parsable', str(slurm_script_path)],
                capture_output=True,
                text=True,
                check=True
            )
            job_id = result.stdout.strip()
            job_ids.append(job_id)
            print(f"  ✅ Agent {agent_id:2d}//{num_agents}: Job ID {job_id}")
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to submit agent {agent_id}: {e}")
            print(f"     stderr: {e.stderr}")
    
    print(f"\n📊 Successfully submitted {len(job_ids)}/{num_agents} agents")
    print(f"Job IDs: {' '.join(job_ids)}")
    print("\n📋 Monitor jobs with:")
    print(f"   squeue -u $USER")
    print(f"   scancel {' '.join(job_ids)}  # to cancel all")
    print(f"\n🌐 Monitor sweep progress at:")
    print(f"   https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")
    
    return job_ids

def generate_slurm_script(sweep_id, project_name, agent_id):
    """Generate SLURM script content for a wandb agent"""
    
    CLUSTER_PATHS = {'main_dir': '/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC'}
    MODULE_COMMANDS = [
        'module load stack/2024-06 python_cuda/3.11.6',
        'module load eth_proxy'
    ]
    DEFAULT_SLURM_SETTINGS = {
        'ntasks': 1,
        'cpus_per_task': 12,
        'time': '4:00:00',
        'mem_per_cpu': '10G',
        'gpus': 1
    }
    
    log_path = f"hp_search/logs/wandb_agent_{agent_id:03d}-%j.out"

    slurm_script = f"""#!/bin/bash

#SBATCH --ntasks={DEFAULT_SLURM_SETTINGS['ntasks']}
#SBATCH --cpus-per-task={DEFAULT_SLURM_SETTINGS['cpus_per_task']}
#SBATCH --time={DEFAULT_SLURM_SETTINGS['time']}
#SBATCH --mem-per-cpu={DEFAULT_SLURM_SETTINGS['mem_per_cpu']}
#SBATCH --gpus={DEFAULT_SLURM_SETTINGS['gpus']}
#SBATCH --output={log_path}
#SBATCH --job-name=wandb_agent_{agent_id:03d}

# Load modules
"""
    
    for module_cmd in MODULE_COMMANDS:
        slurm_script += f"{module_cmd}\n"
    
    slurm_script += f"""

# Setup environment
main_dir="{CLUSTER_PATHS['main_dir']}"
cd $main_dir
source ${{main_dir}}/env/bin/activate

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}}
export CLUSTER_MODE=true

# W&B configuration
export WANDB_DIR="$SCRATCH/wandb_runs/$SLURM_JOB_ID"
export TMPDIR="$SCRATCH/tmp/$SLURM_JOB_ID"
mkdir -p "$WANDB_DIR" "$TMPDIR"

unset WANDB_DISABLE_SERVICE
unset WANDB_DISABLED

export WANDB_START_METHOD=thread
export WANDB_CONSOLE=off
export WANDB_SILENT=true
export WANDB_PROJECT="{project_name}"
export WANDB_SWEEP_ID="{sweep_id}"
export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0}}

echo "🚀 Starting WandB agent {agent_id}"
wandb agent arno-rueegg/{project_name}/{sweep_id}
echo "✅ Completed WandB agent {agent_id}"
"""
    
    return slurm_script

def main():
    parser = argparse.ArgumentParser(description="Manage WandB sweeps on Euler cluster")
    parser.add_argument("--config", default="config/wandb_sweep_config.yaml",
                       help="Path to sweep configuration YAML file")
    parser.add_argument("--project", default="PNN_STEC",
                       help="WandB project name")
    parser.add_argument("--agents", type=int, default=8,
                       help="Number of parallel agents to submit")
    parser.add_argument("--sweep-id", type=str,
                       help="Existing sweep ID (skip creation)")
    parser.add_argument("--create-only", action="store_true",
                       help="Only create sweep, don't submit agents")
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    Path("hp_search/logs").mkdir(parents=True, exist_ok=True)
    
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"🔄 Using existing sweep: {sweep_id}")
    else:
        sweep_id = create_sweep(args.config, args.project)
    
    if not args.create_only:
        submit_agents(sweep_id, args.agents, args.project)
    else:
        print(f"✅ Sweep created. Use --sweep-id {sweep_id} to submit agents later.")

if __name__ == "__main__":
    main()