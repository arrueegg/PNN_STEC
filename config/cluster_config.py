#!/bin/bash
"""
Cluster Configuration Template

This file contains cluster-specific settings that you can modify
according to your cluster's requirements and policies.
"""

# Default SLURM settings - modify as needed
DEFAULT_SLURM_SETTINGS = {
    'ntasks': 1,
    'cpus_per_task': 12,
    'time': '4:00:00',
    'mem_per_cpu': '10G',
    'gpus': 1,
}

# Cluster paths - IMPORTANT: Update these paths for your cluster
CLUSTER_PATHS = {
    'main_dir': '/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC',  # Updated for cluster
    'log_dir': '/cluster/work/igp_psr/arrueegg/WP4/logs',  # Update this path
    'work_dir': '/cluster/work/igp_psr/arrueegg/WP4',  # Update this path
}

# Module loading commands - modify for your cluster
MODULE_COMMANDS = [
    'module load stack/2024-06 python_cuda/3.11.6',
    'module load eth_proxy',
]

# Environment setup commands
ENV_COMMANDS = [
    'cd ${main_dir}',
    'source ${main_dir}/env/bin/activate',
]
