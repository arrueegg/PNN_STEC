#!/bin/bash
#SBATCH --job-name=sweep_3models
#SBATCH --output=hp_search/logs/sweep_3models_%A_%a.out
#SBATCH --error=hp_search/logs/sweep_3models_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --array=1-3

# Hyperparameter sweep launcher for three key models:
# 1. BayesianResNetSTEC
# 2. AttentionMLP_BNN_NLL  
# 3. FactorizedSTEC

# Load modules
module load gcc/8.2.0 python_gpu/3.11.2 eth_proxy

# Activate virtual environment
source /cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/env/bin/activate

# Set working directory
cd /cluster/work/igp_psr/arrueegg/WP4/PNN_STEC

# Create logs directory
mkdir -p hp_search/logs

# Determine which model to run based on array task ID
case $SLURM_ARRAY_TASK_ID in
    1)
        MODEL_NAME="BayesianResNetSTEC"
        SWEEP_CONFIG="config/wandb_sweep_config_BayesianResNetSTEC_cluster.yaml"
        BASE_CONFIG="config/config_cluster_BayesianResNetSTEC.yaml"
        NUM_AGENTS=4  # Run 4 parallel agents for this model
        ;;
    2)
        MODEL_NAME="AttentionMLP_BNN_NLL"
        SWEEP_CONFIG="config/wandb_sweep_config_AttentionMLP_BNN_NLL_cluster.yaml"
        BASE_CONFIG="config/config_cluster_AttentionMLP_BNN_NLL.yaml"
        NUM_AGENTS=3  # Run 3 parallel agents for this model
        ;;
    3)
        MODEL_NAME="FactorizedSTEC"
        SWEEP_CONFIG="config/wandb_sweep_config_FactorizedSTEC_cluster.yaml"
        BASE_CONFIG="config/config_cluster_FactorizedSTEC.yaml"
        NUM_AGENTS=3  # Run 3 parallel agents for this model
        ;;
    *)
        echo "Invalid array task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "=========================================="
echo "Starting sweep for: $MODEL_NAME"
echo "Sweep config: $SWEEP_CONFIG"
echo "Base config: $BASE_CONFIG"
echo "Number of agents: $NUM_AGENTS"
echo "=========================================="

# Initialize W&B sweep and get sweep ID
echo "Initializing W&B sweep..."
SWEEP_ID=$(wandb sweep --project PNN_STEC_Cluster $SWEEP_CONFIG 2>&1 | grep "Run sweep agent with:" | awk '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to create sweep"
    exit 1
fi

echo "Sweep ID: $SWEEP_ID"

# Save sweep ID to file for reference
echo "$SWEEP_ID" > hp_search/logs/${MODEL_NAME}_sweep_id.txt

# Launch multiple sweep agents in parallel
echo "Launching $NUM_AGENTS sweep agents..."
for i in $(seq 1 $NUM_AGENTS); do
    echo "Starting agent $i/$NUM_AGENTS for $MODEL_NAME"
    WANDB_CONFIG_PATH=$BASE_CONFIG wandb agent $SWEEP_ID &
done

# Wait for all agents to complete
wait

echo "=========================================="
echo "Sweep completed for: $MODEL_NAME"
echo "=========================================="
