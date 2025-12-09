#!/bin/bash
#SBATCH --job-name=sweep_attention
#SBATCH --output=hp_search/logs/sweep_attention_%j.out
#SBATCH --error=hp_search/logs/sweep_attention_%j.err
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1

# Hyperparameter sweep for AttentionMLP_BNN_NLL model only
# Usage: sbatch hp_search/sweep_attention_cluster.sh [num_agents]

# Load modules
module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy

# Activate virtual environment
source /cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/env/bin/activate

# Set working directory
cd /cluster/work/igp_psr/arrueegg/WP4/PNN_STEC

# Create logs directory
mkdir -p hp_search/logs

# Configuration
MODEL_NAME="AttentionMLP_BNN_NLL"
SWEEP_CONFIG="config/wandb_sweep_config_AttentionMLP_BNN_NLL_cluster.yaml"
BASE_CONFIG="config/config_cluster_AttentionMLP_BNN_NLL.yaml"
NUM_AGENTS=${1:-4}  # Default to 4 agents if not specified

echo "=========================================="
echo "AttentionMLP_BNN_NLL Hyperparameter Sweep"
echo "=========================================="
echo "Sweep config: $SWEEP_CONFIG"
echo "Base config: $BASE_CONFIG"
echo "Number of agents: $NUM_AGENTS"
echo "=========================================="

# Initialize W&B sweep
echo "Initializing W&B sweep..."
SWEEP_ID=$(wandb sweep --project PNN_STEC_Cluster $SWEEP_CONFIG 2>&1 | grep "Run sweep agent with:" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to create sweep"
    exit 1
fi

echo "Sweep ID: $SWEEP_ID"
echo "$SWEEP_ID" > hp_search/logs/${MODEL_NAME}_sweep_id.txt

# Launch sweep agents
echo "Launching $NUM_AGENTS sweep agents..."
for i in $(seq 1 $NUM_AGENTS); do
    echo "Starting agent $i/$NUM_AGENTS"
    WANDB_CONFIG_PATH=$BASE_CONFIG wandb agent $SWEEP_ID &
    sleep 2  # Stagger agent starts
done

# Wait for all agents to complete
wait

echo "=========================================="
echo "Sweep completed!"
echo "=========================================="
