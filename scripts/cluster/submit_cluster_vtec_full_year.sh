#!/bin/bash
# =====================================================================
# DEPRECATED - Use independent job submission instead
# =====================================================================
#
# This file is DEPRECATED. Job arrays were causing all jobs to run on 
# the same node. Use the new independent job approach instead:
#
# STEPS:
# 1. Generate independent scripts:
#    bash generate_independent_jobs.sh
#
# 2. Submit all jobs:
#    bash submit_all_jobs.sh
#
# Each DOY will now run as a completely independent job on different nodes.
#
# =====================================================================

echo "❌ DEPRECATED: Use the new independent job approach"
echo ""
echo "Steps:"
echo "1. bash generate_independent_jobs.sh"
echo "2. bash submit_all_jobs.sh"
echo ""
exit 1

# OLD CODE BELOW (DO NOT USE)
# =====================================================================

PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)

# Alternatively, hardcode cluster path:
# PROJECT_DIR="/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC"

# DOY from job array index
DOY=$SLURM_ARRAY_TASK_ID

# Year to train
YEAR=2024

# Configuration file
CONFIG_FILE="config/config_cluster_mao_laplacian.yaml"

# =====================================================================
# Setup Environment
# =====================================================================

echo "=========================================="
echo "VTEC Training: DOY $DOY / 245 total"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start Time: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/slurm_logs"

# Change to project directory
cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }

# Check if Python environment exists, activate it
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
else
    echo "❌ No Python environment found (venv or env)"
    exit 1
fi

# =====================================================================
# Verify Configuration
# =====================================================================

echo ""
echo "Verifying setup..."

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "src/main.py" ]; then
    echo "❌ src/main.py not found"
    exit 1
fi

# Check data files exist
if [ ! -f "data/train.h5" ]; then
    echo "⚠️  Warning: data/train.h5 not found - data may be on scratch"
fi

echo "✅ Configuration verified"

# =====================================================================
# GPU Setup
# =====================================================================

# Set GPU device (if multiple GPUs, use first one)
export CUDA_VISIBLE_DEVICES=0

# CUDA settings for stability
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:16:8

# PyTorch optimizations
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# =====================================================================
# Training
# =====================================================================

echo ""
echo "Starting training for DOY $DOY with 10 ensemble members..."
echo "Using config: $CONFIG_FILE"
echo "Python version: $(python --version)"
echo ""

# Run training with verbose output
python -u src/main.py \
  --config "$CONFIG_FILE" \
  --year "$YEAR" \
  --doy "$DOY" \
  --output_dir "experiments/" \
  --cluster true 2>&1 | tee -a slurm_logs/vtec_doy_${DOY}_verbose.log

TRAIN_EXIT_CODE=$?

# =====================================================================
# Results Summary
# =====================================================================

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ DOY $DOY Training COMPLETED SUCCESSFULLY"
else
    echo "❌ DOY $DOY Training FAILED (exit code: $TRAIN_EXIT_CODE)"
fi
echo "End Time: $(date)"
echo "=========================================="

# Clean up GPU memory
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

exit $TRAIN_EXIT_CODE
