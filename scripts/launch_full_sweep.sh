#!/bin/bash
# Launch full hyperparameter sweep for all 6 models
# Usage: bash scripts/launch_full_sweep.sh [num_agents_per_model]
#
# Each model gets num_agents_per_model agents for its sweep

set -e

NUM_AGENTS=${1:-2}

echo "================================================================"
echo "Full Hyperparameter Sweep: All 6 Models"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  • Number of agents per model: $NUM_AGENTS"
echo "  • Total models: 6"
echo "  • Total agents: $((6 * NUM_AGENTS))"
echo ""
echo "Models:"
echo "  1. MLP_NLL"
echo "  2. Branch_BNN_NLL"
echo "  3. ResNet_MSE"
echo "  4. ResNet_NLL"
echo "  5. AttentionMLP_MSE"
echo "  6. AttentionMLP_NLL"
echo ""

# Array to track sweep IDs
declare -a SWEEP_IDS

echo "Launching sweeps..."
echo ""

# MLP_NLL
echo "📤 MLP_NLL..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_MLP_NLL.yaml 2>&1 | grep -oP 'Sweep ID: \K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("MLP_NLL: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# Branch_BNN_NLL
echo "📤 Branch_BNN_NLL..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_Branch_BNN_NLL.yaml 2>&1 | grep -oP 'Sweep ID: \K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("Branch_BNN_NLL: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# ResNet_MSE
echo "📤 ResNet_MSE..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_ResNet_MSE.yaml 2>&1 | grep -oP 'Sweep ID: \K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("ResNet_MSE: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# ResNet_NLL
echo "📤 ResNet_NLL..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_ResNet_NLL.yaml 2>&1 | grep -oP 'Sweep ID: \K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("ResNet_NLL: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# AttentionMLP_MSE
echo "📤 AttentionMLP_MSE..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_AttentionMLP_MSE.yaml 2>&1 | grep -oP 'Sweep ID: \K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("AttentionMLP_MSE: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# AttentionMLP_NLL
echo "📤 AttentionMLP_NLL..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_AttentionMLP_NLL.yaml 2>&1 | grep -oP 'Sweep ID: \K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("AttentionMLP_NLL: $RESULT")
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
echo "View on WandB:"
echo "  https://wandb.ai/arrueegg/pnn_stec"
echo ""
