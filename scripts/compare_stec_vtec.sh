#!/bin/bash
# Quick-start script for STEC vs VTEC comparison
# This script helps you run the fair comparison with sensible defaults

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}STEC vs VTEC Fair Comparison${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check if arguments provided
if [ "$#" -lt 2 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}\n"
    echo "Usage: $0 <STEC_EXPERIMENT> <VTEC_EXPERIMENT> [OPTIONS]"
    echo ""
    echo "Example:"
    echo "  $0 Pretrain_STEC_BNN_NLL_h256_l4_... Pretrain_VTEC_BNN_NLL_h256_l4_..."
    echo ""
    echo "Optional arguments:"
    echo "  --mapping MSLM|SLM        Mapping function (default: MSLM)"
    echo "  --samples N               MC samples for BNN (default: 100)"
    echo "  --test-size N             Test samples (default: 1000000)"
    echo "  --output DIR              Output directory (default: auto-generated)"
    exit 1
fi

STEC_EXP="$1"
VTEC_EXP="$2"
shift 2

# Default values
MAPPING="MSLM"
MC_SAMPLES=100
TEST_SIZE=1000000
OUTPUT_DIR=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mapping)
            MAPPING="$2"
            shift 2
            ;;
        --samples)
            MC_SAMPLES="$2"
            shift 2
            ;;
        --test-size)
            TEST_SIZE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Auto-generate output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="comparisons/stec_vs_vtec_${TIMESTAMP}"
fi

# Verify experiments exist
if [ ! -d "experiments/${STEC_EXP}" ]; then
    echo -e "${RED}Error: STEC experiment not found: experiments/${STEC_EXP}${NC}"
    exit 1
fi

if [ ! -d "experiments/${VTEC_EXP}" ]; then
    echo -e "${RED}Error: VTEC experiment not found: experiments/${VTEC_EXP}${NC}"
    exit 1
fi

# Verify STEC experiment has target=stec
STEC_TARGET=$(grep "^target:" "experiments/${STEC_EXP}/config.yaml" | awk '{print $2}' | tr -d '"' | tr -d "'")
if [ "$STEC_TARGET" != "stec" ]; then
    echo -e "${YELLOW}Warning: STEC experiment has target=${STEC_TARGET}, expected 'stec'${NC}"
fi

# Verify VTEC experiment has target=vtec
VTEC_TARGET=$(grep "^target:" "experiments/${VTEC_EXP}/config.yaml" | awk '{print $2}' | tr -d '"' | tr -d "'")
if [ "$VTEC_TARGET" != "vtec" ]; then
    echo -e "${YELLOW}Warning: VTEC experiment has target=${VTEC_TARGET}, expected 'vtec'${NC}"
fi

# Display configuration
echo -e "${GREEN}Configuration:${NC}"
echo "  STEC Experiment: ${STEC_EXP}"
echo "  VTEC Experiment: ${VTEC_EXP}"
echo "  Mapping Function: ${MAPPING}"
echo "  MC Samples: ${MC_SAMPLES}"
echo "  Test Size: ${TEST_SIZE}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo ""

# Ask for confirmation
read -p "Proceed with comparison? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Run comparison
echo -e "\n${GREEN}Running comparison...${NC}\n"

python src/compare_stec_vtec.py \
    --stec_experiment "${STEC_EXP}" \
    --vtec_experiment "${VTEC_EXP}" \
    --mapping_function "${MAPPING}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_mc_samples ${MC_SAMPLES} \
    --test_size ${TEST_SIZE}

# Check if successful
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Comparison completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    echo -e "Results saved to: ${GREEN}${OUTPUT_DIR}${NC}\n"
    echo "Generated files:"
    echo "  📊 scatter_comparison.png    - Prediction scatter plots"
    echo "  📈 error_analysis.png        - Error distribution analysis"
    echo "  📉 elevation_analysis.png    - Performance vs elevation"
    echo "  📄 comparison_summary.txt    - Detailed metrics report"
    echo "  📁 detailed_predictions.csv  - All predictions for analysis"
    echo ""
    echo "Quick view of results:"
    echo "----------------------------------------"
    cat "${OUTPUT_DIR}/comparison_summary.txt"
    echo "----------------------------------------"
else
    echo -e "\n${RED}Comparison failed. Check error messages above.${NC}\n"
    exit 1
fi
