#!/bin/bash
# Quick Test: Compare All New Model Architectures on Euler Cluster
# Usage: bash hp_search/test_new_models_cluster.sh

set -e

PROJECT_DIR="/scratch2/arrueegg/WP4/PNN_STEC"
cd "$PROJECT_DIR"

source env/bin/activate

echo "================================================================"
echo "Testing All 4 New Model Architectures"
echo "================================================================"
echo ""
echo "Models to test:"
echo "  1. ResNet_MSE"
echo "  2. ResNet_NLL"
echo "  3. AttentionMLP_MSE"
echo "  4. AttentionMLP_NLL"
echo ""

# Create output directory
mkdir -p experiments/test_new_models

# Test 1: ResNet_MSE
echo "================================================================"
echo "Test 1/4: ResNet_MSE (Deterministic)"
echo "================================================================"
python src/main.py \
  --config_path config/config.yaml \
  --override "model.model_type=ResNet_MSE" \
  --override "model.hidden_dim=512" \
  --override "model.num_layers=4" \
  --override "training.loss_function=MSELoss" \
  --override "pretrain.epochs=10" \
  --override "pretrain.batchsize=512" \
  --override "data.train_subset_size=100000" \
  --override "debug_single_batch=False"
echo "✓ Test 1 complete"
echo ""

# Test 2: ResNet_NLL
echo "================================================================"
echo "Test 2/4: ResNet_NLL (Uncertainty)"
echo "================================================================"
python src/main.py \
  --config_path config/config.yaml \
  --override "model.model_type=ResNet_NLL" \
  --override "model.hidden_dim=512" \
  --override "model.num_layers=4" \
  --override "training.loss_function=GaussianNLLLoss" \
  --override "pretrain.epochs=10" \
  --override "pretrain.batchsize=512" \
  --override "data.train_subset_size=100000" \
  --override "debug_single_batch=False"
echo "✓ Test 2 complete"
echo ""

# Test 3: AttentionMLP_MSE
echo "================================================================"
echo "Test 3/4: AttentionMLP_MSE (Attention + Deterministic)"
echo "================================================================"
python src/main.py \
  --config_path config/config.yaml \
  --override "model.model_type=AttentionMLP_MSE" \
  --override "model.hidden_dim=512" \
  --override "model.num_layers=2" \
  --override "model.num_heads=4" \
  --override "training.loss_function=MSELoss" \
  --override "pretrain.epochs=10" \
  --override "pretrain.batchsize=512" \
  --override "data.train_subset_size=100000" \
  --override "debug_single_batch=False"
echo "✓ Test 3 complete"
echo ""

# Test 4: AttentionMLP_NLL
echo "================================================================"
echo "Test 4/4: AttentionMLP_NLL (Attention + Uncertainty)"
echo "================================================================"
python src/main.py \
  --config_path config/config.yaml \
  --override "model.model_type=AttentionMLP_NLL" \
  --override "model.hidden_dim=512" \
  --override "model.num_layers=2" \
  --override "model.num_heads=4" \
  --override "training.loss_function=GaussianNLLLoss" \
  --override "pretrain.epochs=10" \
  --override "pretrain.batchsize=512" \
  --override "data.train_subset_size=100000" \
  --override "debug_single_batch=False"
echo "✓ Test 4 complete"
echo ""

echo "================================================================"
echo "✅ All 4 models tested successfully!"
echo "================================================================"
echo ""
echo "Results saved in: experiments/"
echo ""
echo "To view results:"
echo "  ls -lh experiments/ | grep -E 'ResNet|Attention'"
echo ""
