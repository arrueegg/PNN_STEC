#!/bin/bash
: '
Simple WandB Sweep Launcher

Usage:
    ./launch_wandb_sweep.sh [num_agents] [sweep_config]

Examples:
    ./launch_wandb_sweep.sh 8                             # Use 8 agents with default config
    ./launch_wandb_sweep.sh 16 config/custom_sweep.yaml   # Use 16 agents with custom config
'

# Default values
NUM_AGENTS=${1:-8}
#SWEEP_CONFIG=${2:-"config/wandb_sweep_config_all.yaml"}
SWEEP_CONFIG=${2:-"config/wandb_sweep_config_BNN.yaml"}
#SWEEP_CONFIG=${2:-"config/wandb_sweep_config_DE.yaml"}
#SWEEP_CONFIG=${2:-"config/wandb_sweep_config_MLP.yaml"}

echo "🚀 Launching WandB Sweep"
echo "📊 Agents: $NUM_AGENTS"
echo "📄 Config: $SWEEP_CONFIG"
echo

# Activate environment
source env/bin/activate

# Run the sweep manager
python scripts/wandb_sweep_manager.py \
    --config "$SWEEP_CONFIG" \
    --agents "$NUM_AGENTS" \
    --project "PNN_STEC"

echo
echo "✅ Sweep launched! Check the output above for monitoring instructions."