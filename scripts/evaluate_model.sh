#!/bin/bash
# =============================================================================
# STEC Model Evaluation Script
# =============================================================================
# This script evaluates a STEC model against baselines (VTEC+mapping, IGS GIM)
# and optionally using independent Madrigal STEC observations as ground truth.
#
# Usage:
#   ./scripts/evaluate_model.sh <experiment_folder> [options]
#
# Examples:
#   # Evaluate with normal test set + GIM comparison
#   ./scripts/evaluate_model.sh experiments/Finetune_STEC_2024_183_... --include_gim
#
#   # Evaluate with independent Madrigal data (100K samples)
#   ./scripts/evaluate_model.sh experiments/Finetune_STEC_2024_183_... --use_madrigal --test_size 100000
#
#   # Full comparison: Madrigal + GIM (recommended for publication)
#   ./scripts/evaluate_model.sh experiments/Finetune_STEC_2024_183_... --use_madrigal --include_gim --test_size 100000
# =============================================================================

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Python interpreter
PYTHON="${PROJECT_ROOT}/env/bin/python"

# Default paths
GIM_PATH="/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex"
MADRIGAL_PATH="/home/space/data/iono/Madrigal_STEC"

# Check if experiment folder is provided
if [ $# -lt 1 ]; then
    echo "Error: Experiment folder required"
    echo "Usage: $0 <experiment_folder> [options]"
    echo ""
    echo "Options:"
    echo "  --include_gim            Compare with IGS GIM"
    echo "  --use_madrigal           Use independent Madrigal STEC data"
    echo "  --test_size <N>          Limit to N test samples"
    echo "  --vtec_experiment <path> Include VTEC+mapping comparison"
    echo ""
    exit 1
fi

# First argument is the experiment folder
STEC_EXPERIMENT="$1"
shift

# Check if experiment exists
if [ ! -d "$STEC_EXPERIMENT" ]; then
    echo "Error: Experiment folder not found: $STEC_EXPERIMENT"
    exit 1
fi

echo "========================================================================"
echo "STEC Model Evaluation Pipeline"
echo "========================================================================"
echo "Experiment: $(basename "$STEC_EXPERIMENT")"
echo "Python: $PYTHON"
echo ""

# Run comparison
$PYTHON src/compare_stec_vtec_gim.py \
    --stec_experiment "$STEC_EXPERIMENT" \
    --gim_path "$GIM_PATH" \
    --madrigal_path "$MADRIGAL_PATH" \
    "$@"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ Evaluation completed successfully!"
    echo "========================================================================"
    echo ""
    
    # Find the evaluation directory
    EVAL_DIR=""
    if [[ "$*" == *"--use_madrigal"* ]]; then
        EVAL_DIR="${STEC_EXPERIMENT}/evaluation/madrigal_comparison"
    else
        EVAL_DIR="${STEC_EXPERIMENT}/evaluation/normal_comparison"
    fi
    
    if [ -d "$EVAL_DIR" ]; then
        echo "📁 Results location:"
        echo "   $EVAL_DIR"
        echo ""
        echo "📊 Generated files:"
        ls -lh "$EVAL_DIR"/*.png "$EVAL_DIR"/*.pdf 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
        echo ""
        echo "📄 Summary:"
        echo "   ${EVAL_DIR}/comparison_summary.txt"
        echo ""
        echo "💡 Tip: Open PNG files for quick viewing, use PDF for publications"
    fi
else
    echo ""
    echo "❌ Evaluation failed. Check the error messages above."
    exit 1
fi
