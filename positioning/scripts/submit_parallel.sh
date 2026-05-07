#!/bin/bash
# Convenience script to submit parallel multiday evaluation jobs.
#
# Usage:
#   ./positioning/scripts/submit_parallel.sh --dates "2024-183:2024-189" [options]

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT" || { echo "Failed to cd to $REPO_ROOT"; exit 1; }

# Default values
DATES=""
CHUNK_SIZE=3
STEC_CONFIG="config/config.yaml"
VTEC_CONFIG="config/config_vtec_mlp_baseline.yaml"
OUTPUT_DIR="multiday_results_parallel"
NUM_SAMPLES=100
DRY_RUN=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dates)
            DATES="$2"
            shift 2
            ;;
        --chunk_size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --stec_config)
            STEC_CONFIG="$2"
            shift 2
            ;;
        --vtec_config)
            VTEC_CONFIG="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_inference_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN="--dry_run"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --dates DATES [options]"
            echo "Options:"
            echo "  --dates DATES              Date range (required)"
            echo "  --chunk_size SIZE          Days per job (default: 3)"
            echo "  --stec_config FILE         STEC config file"
            echo "  --vtec_config FILE         VTEC config file"
            echo "  --output_dir DIR           Output directory"
            echo "  --num_inference_samples N  Inference samples (default: 100)"
            echo "  --dry_run                  Show what would be submitted"
            exit 1
            ;;
    esac
done

if [ -z "$DATES" ]; then
    echo "Error: --dates is required"
    echo "Usage: $0 --dates DATES [options]"
    exit 1
fi

echo "🚀 Submitting parallel multiday evaluation..."
echo "Dates: $DATES"
echo "Chunk size: $CHUNK_SIZE days per job"
echo "STEC config: $STEC_CONFIG"
echo "VTEC config: $VTEC_CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo

python positioning/scripts/submit_parallel.py \
    --dates "$DATES" \
    --chunk_size "$CHUNK_SIZE" \
    --stec_config "$STEC_CONFIG" \
    --vtec_config "$VTEC_CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    --num_inference_samples "$NUM_SAMPLES" \
    $DRY_RUN