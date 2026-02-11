#!/bin/bash
# =====================================================================
# Generate and submit individual SBATCH scripts for each DOY
# One completely independent job per DOY (122-366)
# =====================================================================

PROJECT_DIR=$(pwd)
SCRIPTS_DIR="$PROJECT_DIR/sbatch_scripts"

# Create directory for individual scripts
mkdir -p "$SCRIPTS_DIR"

echo "Generating 245 independent SBATCH scripts..."
echo "Output directory: $SCRIPTS_DIR"
echo ""

# Loop through DOYs 122-366
for DOY in $(seq 122 366); do
    SCRIPT_FILE="$SCRIPTS_DIR/vtec_doy_${DOY}.sh"
    
    # Create individual sbatch script
    cat > "$SCRIPT_FILE" << 'EOF'
#!/bin/bash
# =====================================================================
# Single VTEC Training Job - DOY specific
# This is a completely independent job (no job arrays)
# =====================================================================
#SBATCH --job-name=VTEC_DOY_INDIVIDUAL
#SBATCH --time=08:00:00                # Max 8 hours per DOY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # Allocate 8 CPU cores
#SBATCH --gpus-per-node=1              # One GPU
#SBATCH --mem-per-cpu=1G              # Memory per CPU core
#SBATCH --output=slurm_logs/vtec_doy_${DOY}_%j.log
#SBATCH --mail-type=FAIL               # Email on job failure

# =====================================================================
# Configuration
# =====================================================================

PROJECT_DIR=$(pwd)
DOY=PLACEHOLDER_DOY
YEAR=2024
CONFIG_FILE="config/config_cluster_mao_laplacian.yaml"

# =====================================================================
# Setup Environment
# =====================================================================

echo "=========================================="
echo "VTEC Training: DOY $DOY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
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

echo "✅ Configuration verified"

# =====================================================================
# GPU Setup
# =====================================================================

export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:16:8
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export CLUSTER_MODE=true                # Enable cluster mode detection

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
EOF

    # Replace placeholder with actual DOY
    sed -i "s/PLACEHOLDER_DOY/$DOY/g" "$SCRIPT_FILE"
    
    # Make executable
    chmod +x "$SCRIPT_FILE"
    
    echo "✅ Generated: $SCRIPT_FILE"
done

echo ""
echo "=========================================="
echo "✅ Generated 245 independent SBATCH scripts"
echo "=========================================="
echo ""
echo "To submit all jobs:"
echo "  bash submit_all_jobs.sh"
echo ""
echo "To submit specific DOYs:"
echo "  sbatch sbatch_scripts/vtec_doy_122.sh"
echo "  sbatch sbatch_scripts/vtec_doy_123.sh"
echo "  ..."
echo ""
