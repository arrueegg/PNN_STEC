#!/bin/bash
# Quick wrapper script for running the complete positioning evaluation pipeline

# Activate virtual environment if it exists and not already activated
if [ -d "env/bin" ] && [ -z "$VIRTUAL_ENV" ]; then
    source env/bin/activate
fi

EXPERIMENT="$1"
DATE="$2"

# Strip leading "experiments/" or "experiment/" if present
EXPERIMENT="${EXPERIMENT#experiments/}"

# Remove trailing slash if present
EXPERIMENT="${EXPERIMENT%/}"

if [ -z "$EXPERIMENT" ] || [ -z "$DATE" ]; then
    echo "Usage: bash run_positioning_pipeline.sh <experiment_name> <date>"
    echo ""
    echo "Examples:"
    echo "  bash run_positioning_pipeline.sh Finetune_STEC_2024_183 2024-07-01"
    echo "  bash run_positioning_pipeline.sh BayesianResNetSTEC 2024-07-01"
    echo ""
    echo "This will:"
    echo "  1. Download GNSS products"
    echo "  2. Download RINEX files for test stations"
    echo "  3. Run positioning with your model's STEC corrections"
    echo "  4. Run positioning with IGS GIM for comparison"
    echo "  5. Generate performance metrics"
    exit 1
fi

# Default settings - adjust as needed
PPPX_PATH="./src/positioning_eval/pppx"
GIM_PATH="/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex"
PARALLEL_JOBS=4

echo "========================================================================"
echo "  Positioning Evaluation Pipeline"
echo "========================================================================"
echo "  Experiment: $EXPERIMENT"
echo "  Date: $DATE"
echo "  Parallel jobs: $PARALLEL_JOBS"
echo "========================================================================"
echo ""

# Step 1: Generate STEC corrections (if not already done)
echo "Step 1: Generating STEC corrections..."
python src/inference_positioning.py \
    --experiment "$EXPERIMENT" \
    --date "$DATE"

if [ $? -ne 0 ]; then
    echo "ERROR: STEC correction generation failed"
    exit 1
fi

echo ""
echo "Step 2: Running positioning evaluation..."

# Step 2: Run positioning evaluation
python src/positioning_eval/run_positioning_evaluation.py \
    --experiment "$EXPERIMENT" \
    --date "$DATE" \
    --all_test_stations \
    --parallel $PARALLEL_JOBS \
    --pppx_path "$PPPX_PATH" \
    --gim_base_path "$GIM_PATH"

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
