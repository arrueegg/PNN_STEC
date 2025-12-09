#!/bin/bash
# Launch hyperparameter sweeps for the three key models
# Usage: bash scripts/launch_full_sweep.sh [num_agents_per_model]
#
# This script launches sweeps for:
#   1. BayesianResNetSTEC - Hybrid ResNet + Bayesian head
#   2. AttentionMLP_BNN_NLL - Lightweight Attention + Bayesian head
#   3. FactorizedSTEC - Physics-based VTEC × MF factorization

set -e

NUM_AGENTS=${1:-4}

echo "================================================================"
echo "Three-Model Hyperparameter Sweeps (Cluster)"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  • Number of agents per model: $NUM_AGENTS"
echo "  • Number of models: 3"
echo "  • Total agents: $((3 * NUM_AGENTS))"
echo ""
echo "Models (in launch order):"
echo "  1. BayesianResNetSTEC    - Hybrid ResNet + Bayesian head"
echo "  2. AttentionMLP_BNN_NLL  - Lightweight Attention + Bayesian head"
echo "  3. FactorizedSTEC        - Physics-based VTEC × MF"
echo ""

# Array to track sweep IDs
declare -a SWEEP_IDS

echo "Launching sweeps..."
echo ""

# 1. BayesianResNetSTEC
echo "📤 BayesianResNetSTEC..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_BayesianResNetSTEC_cluster.yaml 2>&1 | grep -oP 'Sweep ID: \\K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("BayesianResNetSTEC: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# 2. AttentionMLP_BNN_NLL
echo "📤 AttentionMLP_BNN_NLL..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_AttentionMLP_BNN_NLL_cluster.yaml 2>&1 | grep -oP 'Sweep ID: \\K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("AttentionMLP_BNN_NLL: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# 3. FactorizedSTEC
echo "📤 FactorizedSTEC..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_FactorizedSTEC_cluster.yaml 2>&1 | grep -oP 'Sweep ID: \\K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("FactorizedSTEC: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

echo "================================================================"
echo "✅ All Sweeps Launched!"
echo "================================================================"
echo ""
echo "Summary:"
printf '%s\n' "${SWEEP_IDS[@]}"
echo ""
echo "Monitor on WandB:"
echo "  https://wandb.ai/your-entity/PNN_STEC_Cluster"
echo ""
echo "Check sweep status:"
echo "  wandb sweep --project PNN_STEC_Cluster [sweep_id]"
echo ""
