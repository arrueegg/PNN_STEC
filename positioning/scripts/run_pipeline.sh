#!/bin/bash
# Quick wrapper script for running the complete positioning evaluation pipeline

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT" || { echo "Failed to cd to $REPO_ROOT"; exit 1; }

# Activate virtual environment if it exists and not already activated
if [ -d "env/bin" ] && [ -z "$VIRTUAL_ENV" ]; then
    source env/bin/activate
fi

EXPERIMENT="$1"
DATE="$2"
SKIP_DOWNLOADS="${3:-false}"  # Optional third argument, default to false

# Strip leading "experiments/" or "experiment/" if present
EXPERIMENT="${EXPERIMENT#experiments/}"

# Remove trailing slash if present
EXPERIMENT="${EXPERIMENT%/}"

if [ -z "$EXPERIMENT" ] || [ -z "$DATE" ]; then
    echo "Usage: bash run_positioning_pipeline.sh <experiment_name> <date> [skip_downloads]"
    echo ""
    echo "Arguments:"
    echo "  experiment_name: Name of the experiment folder"
    echo "  date: Date in YYYY-MM-DD format"
    echo "  skip_downloads: Optional flag 'skip' to skip downloading products/RINEX (default: download)"
    echo ""
    echo "Examples:"
    echo "  bash run_positioning_pipeline.sh Finetune_STEC_2024_183 2024-07-01"
    echo "  bash run_positioning_pipeline.sh BayesianResNetSTEC 2024-07-01 skip"
    echo ""
    echo "This will:"
    echo "  1. Download GNSS products (unless skip_downloads=skip)"
    echo "  2. Download RINEX files for test stations (unless skip_downloads=skip)"
    echo "  3. Run positioning with your model's STEC corrections"
    echo "  4. Run positioning with IGS GIM for comparison"
    echo "  5. Generate performance metrics"
    exit 1
fi

# Default settings - adjust as needed
PPPX_PATH="$REPO_ROOT/positioning/positioning_eval/pppx"
GIM_PATH="/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex"
PARALLEL_JOBS=4

# Build skip_downloads flag
SKIP_FLAG=""
if [ "$SKIP_DOWNLOADS" = "skip" ]; then
    SKIP_FLAG="--skip_downloads"
    echo "Note: Will skip downloading GNSS products and RINEX files"
fi

echo "========================================================================"
echo "  Positioning Evaluation Pipeline"
echo "========================================================================"
echo "  Experiment: $EXPERIMENT"
echo "  Date: $DATE"
echo "  Parallel jobs: $PARALLEL_JOBS"
echo "  Skip downloads: $SKIP_DOWNLOADS"
echo "========================================================================"
echo ""

# Step 1: Generate STEC corrections (if not already done)
echo "Step 1: Generating STEC corrections..."
python positioning/scripts/generate_stec_corrections.py \
    --experiment "$EXPERIMENT" \
    --date "$DATE"

if [ $? -ne 0 ]; then
    echo "ERROR: STEC correction generation failed"
    exit 1
fi

echo ""
echo "Step 2: Running positioning evaluation..."

# Step 2: Run positioning evaluation
python positioning/positioning_eval/run_positioning_evaluation.py \
    --experiment "$EXPERIMENT" \
    --date "$DATE" \
    --all_test_stations \
    --parallel $PARALLEL_JOBS \
    --pppx_path "$PPPX_PATH" \
    --gim_base_path "$GIM_PATH" \
    $SKIP_FLAG

if [ $? -ne 0 ]; then
    echo "ERROR: Positioning evaluation failed"
    exit 1
fi

echo ""
echo "========================================================================"
echo "  ✅ Pipeline completed successfully!"
echo "========================================================================"
echo ""
echo "Results are saved in:"
echo "  experiments/$EXPERIMENT/positioning/results/"
echo ""
