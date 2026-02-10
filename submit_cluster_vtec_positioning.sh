#!/bin/bash
# =====================================================================
# SLURM Job: Run Positioning Evaluation after VTEC training
# Run this AFTER main training completes (365 jobs done)
# =====================================================================
#SBATCH --job-name=VTEC_Positioning_2024
#SBATCH --time=24:00:00               # Long timeout (positioning is intensive)
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1             # One GPU per node
#SBATCH --mem-per-cpu=8G              # Memory per CPU
#SBATCH --output=slurm_logs/positioning_%j.log
#SBATCH --error=slurm_logs/positioning_%j.err

PROJECT_DIR="/scratch2/arrueegg/WP4/PNN_STEC"

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
