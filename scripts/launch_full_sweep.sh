#!/bin/bash
# Launch full hyperparameter sweep for all models
# Usage: bash scripts/launch_full_sweep.sh [num_agents_per_model]
#
# Each model gets num_agents_per_model agents for its sweep

set -e

NUM_AGENTS=${1:-2}

echo "================================================================"
echo "Full Hyperparameter Sweep: All Models"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  • Number of agents per model: $NUM_AGENTS"
echo "  • Number of models: 13"
echo "  • Total agents: $((13 * NUM_AGENTS))"
echo ""
echo "Models:"
echo "  BASELINES:"
echo "    1. MLP_MSE"
echo "    2. MLP_NLL"
echo "  BRANCH MODELS:"
echo "    3. Branch_MLP_MSE"
echo "    4. Branch_MLP_NLL"
echo "    5. Branch_BNN_NLL"
echo "    6. Branch3Way_MLP_MSE (optimized)"
echo "    7. Branch3Way_MLP_NLL (optimized)"
echo "    8. Branch3Way_BNN_NLL (optimized)"
echo "  NEW ARCHITECTURES:"
echo "    9. ResNet_MSE"
echo "    10. ResNet_NLL"
echo "    11. AttentionMLP_MSE"
echo "    12. AttentionMLP_NLL"
echo ""

# Array to track sweep IDs
declare -a SWEEP_IDS

echo "Launching sweeps..."
echo ""

# MLP_MSE
echo "📤 MLP_MSE..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_MLP_MSE.yaml 2>&1 | grep -oP 'Sweep ID: \K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("MLP_MSE: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# MLP_NLL
echo "📤 MLP_NLL..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_MLP_NLL.yaml 2>&1 | grep -oP 'Sweep ID: \K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("MLP_NLL: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# Branch_MLP_MSE
echo "📤 Branch_MLP_MSE..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_Branch_MLP_MSE.yaml 2>&1 | grep -oP 'Sweep ID: \K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("Branch_MLP_MSE: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# Branch_MLP_NLL
echo "📤 Branch_MLP_NLL..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_Branch_MLP_NLL.yaml 2>&1 | grep -oP 'Sweep ID: \K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("Branch_MLP_NLL: $RESULT")
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

# Branch3Way_MLP_MSE
echo "📤 Branch3Way_MLP_MSE (optimized)..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_Branch3Way_MLP_MSE.yaml 2>&1 | grep -oP 'Sweep ID: \K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("Branch3Way_MLP_MSE: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# Branch3Way_MLP_NLL
echo "📤 Branch3Way_MLP_NLL (optimized)..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_Branch3Way_MLP_NLL.yaml 2>&1 | grep -oP 'Sweep ID: \K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("Branch3Way_MLP_NLL: $RESULT")
  echo "   ✓ Sweep ID: $RESULT"
fi
echo ""

# Branch3Way_BNN_NLL
echo "📤 Branch3Way_BNN_NLL (optimized)..."
RESULT=$(./scripts/launch_wandb_sweep.sh $NUM_AGENTS config/wandb_sweep_config_Branch3Way_BNN_NLL.yaml 2>&1 | grep -oP 'Sweep ID: \K[^ ]+' || true)
if [ ! -z "$RESULT" ]; then
  SWEEP_IDS+=("Branch3Way_BNN_NLL: $RESULT")
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
