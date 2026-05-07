#!/bin/bash
# =====================================================================
# DEPRECATED - Updated approach uses independent jobs
# =====================================================================
#
# After running the new independent job training approach:
# 1. bash generate_independent_jobs.sh
# 2. bash submit_all_jobs.sh
#
# Once those complete, you can run this positioning script:
#SBATCH --job-name=VTEC_Positioning_2024
#SBATCH --time=24:00:00               # Long timeout (positioning is intensive)
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1             # One GPU per node
#SBATCH --mem-per-cpu=16G             # Memory per CPU
#SBATCH --output=slurm_logs/positioning_%j.log

PROJECT_DIR=$(cd "$(dirname "$0")/../../.." && pwd)

echo "=========================================="
echo "VTEC Positioning Evaluation"
echo "=========================================="
echo "Start Time: $(date)"
echo "=========================================="

cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }

# Activate environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "env" ]; then
    source env/bin/activate
fi

# =====================================================================
# Positioning Evaluation
# =====================================================================

python src/multiday_evaluation.py \
  --dates "2024-001:2024-365" \
  --vtec_config config/config_mao_laplacian.yaml \
  --stec_config config/config_BNN.yaml \
  --output_dir multiday_results/Mao_FullYear_2024_positioning \
  --positioning \
  --skip_training

POSITION_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $POSITION_EXIT_CODE -eq 0 ]; then
    echo "✅ Positioning evaluation COMPLETED"
else
    echo "❌ Positioning evaluation FAILED"
fi
echo "End Time: $(date)"
echo "=========================================="

exit $POSITION_EXIT_CODE
