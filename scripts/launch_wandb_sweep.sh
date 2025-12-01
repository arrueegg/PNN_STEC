#!/bin/bash
: '
Simple WandB Sweep Launcher

Usage:
    ./launch_wandb_sweep.sh [num_agents] [model_type|sweep_config]

Arguments:
    num_agents    : Number of agents (default: 8)
    model_type    : BNN, DE, MC, MLP, baseline, attention, or all (default: BNN)
    sweep_config  : Custom config file path (if not using model_type)

Examples:
    ./launch_wandb_sweep.sh 8 BNN                         # Use 8 agents with BNN config
    ./launch_wandb_sweep.sh 16 DE                         # Use 16 agents with DE config
    ./launch_wandb_sweep.sh 4 MC                          # Use 4 agents with MCDropout config
    ./launch_wandb_sweep.sh 12 MLP                        # Use 12 agents with MLP config
    ./launch_wandb_sweep.sh 8 baseline                    # Use 8 agents with baseline config
    ./launch_wandb_sweep.sh 8 attention                   # Use 8 agents with AttentionMLP_BNN config
    ./launch_wandb_sweep.sh 8 all                         # Use 8 agents with all models config
    ./launch_wandb_sweep.sh 16 config/custom_sweep.yaml   # Use 16 agents with custom config
'

# Default values
NUM_AGENTS=${1:-8}
MODEL_OR_CONFIG=${2:-"BNN"}

# Determine config file based on input
case "$MODEL_OR_CONFIG" in
    "BNN"|"bnn")
        SWEEP_CONFIG="config/wandb_sweep_config_BNN.yaml"
        ;;
    "DE"|"de")
        SWEEP_CONFIG="config/wandb_sweep_config_DE.yaml"
        ;;
    "MC"|"mc"|"MCDropout"|"mcdropout")
        SWEEP_CONFIG="config/wandb_sweep_config_MCDropout.yaml"
        ;;
    "MLP"|"mlp")
        SWEEP_CONFIG="config/wandb_sweep_config_MLP.yaml"
        ;;
    "baseline"|"BASELINE")
        SWEEP_CONFIG="config/wandb_sweep_config_baseline.yaml"
        ;;
    "attention"|"ATTENTION"|"attn")
        SWEEP_CONFIG="config/wandb_sweep_config_AttentionMLP_NLL.yaml"
        ;;
    "all"|"ALL")
        SWEEP_CONFIG="config/wandb_sweep_config_all.yaml"
        ;;
    *)
        # If it doesn't match any model type, assume it's a custom config path
        SWEEP_CONFIG="$MODEL_OR_CONFIG"
        ;;
esac

echo "🚀 Launching WandB Sweep"
echo "📊 Agents: $NUM_AGENTS"
echo "📄 Config: $SWEEP_CONFIG"
echo "🤖 Model Type: $MODEL_OR_CONFIG"
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