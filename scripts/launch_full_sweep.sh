#!/bin/bash
# Launch selected hyperparameter sweeps
# Usage: bash scripts/launch_full_sweep.sh [num_agents_per_model]
#
# This script launches a reduced set of sweeps (selected runs only).

set -e

NUM_AGENTS=${1:-2}

echo "================================================================"
echo "Selected Hyperparameter Sweeps"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  • Number of agents per model: $NUM_AGENTS"
echo "  • Number of models: 8"
echo "  • Total agents: $((8 * NUM_AGENTS))"
echo ""
echo "Models (in launch order):"
echo "  1. MLP_MSE"
echo "  2. Branch3Way_MLP_MSE"
echo "  3. Branch3Way_BNN_NLL"
echo "  4. ResNet_MSE"
echo "  5. ResNet_BNN_NLL"
echo "  6. AttentionMLP_MSE"
echo "  7. AttentionMLP_NLL"
echo "  8. BayesianResNet_NLL"
echo ""

# Array to track sweep IDs
declare -a SWEEP_IDS

echo "Launching selected sweeps..."
echo ""

# 1. MLP_MSE
echo "📤 MLP_MSE..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_MLP_MSE.yaml 2>&1 | grep -oP 'Sweep ID: \\K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("MLP_MSE: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# 2. Branch3Way_MLP_MSE
echo "📤 Branch3Way_MLP_MSE..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_Branch3Way_MLP_MSE.yaml 2>&1 | grep -oP 'Sweep ID: \\K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("Branch3Way_MLP_MSE: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# 3. Branch3Way_BNN_NLL
echo "📤 Branch3Way_BNN_NLL..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_Branch3Way_BNN_NLL.yaml 2>&1 | grep -oP 'Sweep ID: \\K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("Branch3Way_BNN_NLL: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# 4. ResNet_MSE
echo "📤 ResNet_MSE..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_ResNet_MSE.yaml 2>&1 | grep -oP 'Sweep ID: \\K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("ResNet_MSE: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# 5. ResNet_BNN_NLL
echo "📤 ResNet_BNN_NLL..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_ResNet_BNN_NLL.yaml 2>&1 | grep -oP 'Sweep ID: \\K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("ResNet_BNN_NLL: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# 6. AttentionMLP_MSE
echo "📤 AttentionMLP_MSE..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_AttentionMLP_MSE.yaml 2>&1 | grep -oP 'Sweep ID: \\K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("AttentionMLP_MSE: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# 7. AttentionMLP_NLL
echo "📤 AttentionMLP_NLL..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_AttentionMLP_NLL.yaml 2>&1 | grep -oP 'Sweep ID: \\K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("AttentionMLP_NLL: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# 8. BayesianResNet_NLL (new)
echo "📤 BayesianResNet_NLL..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_BayesianResNetSTEC.yaml 2>&1 | grep -oP 'Sweep ID: \\K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("BayesianResNet_NLL: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

echo "================================================================"
echo "✅ Selected Sweeps Launched!"
echo "================================================================"
echo ""
echo "Summary:"
printf '%s\n' "${SWEEP_IDS[@]}"
echo ""
echo "View on WandB:"
echo "  https://wandb.ai/arrueegg/pnn_stec"
echo ""
