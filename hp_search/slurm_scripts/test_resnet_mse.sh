#!/bin/bash
#SBATCH --job-name=test_resnet_mse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --output=hp_search/logs/test_resnet_mse_%j.log
#SBATCH --error=hp_search/logs/test_resnet_mse_%j.err

# Test: ResNet_MSE on Euler Cluster

PROJECT_DIR="/scratch2/arrueegg/WP4/PNN_STEC"
cd "$PROJECT_DIR"

source env/bin/activate

echo "Job started: $(date)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

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

echo ""
echo "Job completed: $(date)"
