#!/bin/bash
# =====================================================================
# Submit all 245 independent SBATCH jobs
# =====================================================================

PROJECT_DIR=$(pwd)
SCRIPTS_DIR="$PROJECT_DIR/sbatch_scripts"

if [ ! -d "$SCRIPTS_DIR" ]; then
    echo "❌ sbatch_scripts directory not found!"
    echo "First run: bash generate_independent_jobs.sh"
    exit 1
fi

echo "========================================"
echo "Submitting 245 independent SBATCH jobs"
echo "========================================"
echo ""

# Array to store job IDs
JOB_IDS=()
SUBMITTED=0
SKIPPED=0
FAILED=0

# Submit each DOY script
for SCRIPT in "$SCRIPTS_DIR"/vtec_doy_*.sh; do
    DOY=$(basename "$SCRIPT" | sed 's/vtec_doy_//g' | sed 's/.sh//g')
    
    # Prepare padded DOY for folder matching (ensure 3 digits as in experiment names)
    DOY_PADDED=$(printf "%03d" "$DOY")

    # Check if experiment already has all 10 ensemble members
    ALREADY_DONE=false
    if [ -d "experiments" ]; then
        # Look for folders matching the current DOY (supports any target and any year)
        for EXP_DIR in experiments/Finetune_VTEC_*"${DOY_PADDED}"*LaplacianNLL*; do
            if [ -d "$EXP_DIR" ]; then
                # Check both 'model' and 'models' as per user comment
                MEMBER_COUNT=0
                if [ -d "$EXP_DIR/model" ]; then
                    MEMBER_COUNT=$(ls -1 "$EXP_DIR/model"/*.pth 2>/dev/null | wc -l)
                elif [ -d "$EXP_DIR/models" ]; then
                    MEMBER_COUNT=$(ls -1 "$EXP_DIR/models"/*.pth 2>/dev/null | wc -l)
                fi

                if [ "$MEMBER_COUNT" -ge 10 ]; then
                    #echo "⏭️  Skipping DOY $DOY: $MEMBER_COUNT ensemble members already exist in $(basename "$EXP_DIR")"
                    ALREADY_DONE=true
                    break
                fi
            fi
        done
    fi

    if [ "$ALREADY_DONE" = true ]; then
        #SKIPPED=$((SKIPPED + 1))
        continue
    fi
    
    # Submit job
    #JOB_ID=$(sbatch "$SCRIPT" 2>&1 | grep -oP 'Submitted batch job \K[0-9]+')
    
    if [ -z "$JOB_ID" ]; then
        echo "❌ Failed to submit DOY $DOY"
        FAILED=$((FAILED + 1))
    else
        echo "✅ Submitted DOY $DOY: Job ID $JOB_ID"
        JOB_IDS+=($JOB_ID)
        SUBMITTED=$((SUBMITTED + 1))
    fi
    
    # Small delay to avoid overwhelming scheduler
    sleep 0.5
done

echo ""
echo "========================================"
echo "Submission Summary"
echo "========================================"
echo "Submitted: $SUBMITTED jobs"
echo "Skipped:   $SKIPPED jobs (already completed)"
echo "Failed:    $FAILED jobs"
echo ""

if [ $SUBMITTED -gt 0 ]; then
    echo "Monitor all jobs:"
    echo "  squeue -u \$USER"
    echo ""
    echo "Monitor specific job:"
    echo "  squeue -j ${JOB_IDS[0]}"
    echo ""
    echo "View logs:"
    echo "  tail -f slurm_logs/vtec_doy_*.log"
    echo ""
    echo "Cancel all jobs:"
    echo "  scancel -u \$USER"
fi
