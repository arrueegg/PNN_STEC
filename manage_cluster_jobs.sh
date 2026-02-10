#!/bin/bash
# =====================================================================
# DEPRECATED - Use independent job submission scripts instead
# =====================================================================
#
# This management script was for the old job array approach.
# Use the new independent job approach:
#
# 1. Generate independent scripts:
#    bash generate_independent_jobs.sh
#
# 2. Submit all jobs:
#    bash submit_all_jobs.sh
#
# 3. Monitor:
#    squeue -u $USER
#
# =====================================================================

echo "❌ DEPRECATED: Use the new independent job approach"
echo ""
echo "Steps:"
echo "1. bash generate_independent_jobs.sh"
echo "2. bash submit_all_jobs.sh"
echo "3. squeue -u \$USER (to monitor)"
echo ""
exit 1

# OLD CODE BELOW (DO NOT USE)
# =====================================================================

PROJECT_DIR=$(pwd)

# Alternatively, hardcode cluster path:
# PROJECT_DIR="/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =====================================================================
# Submit Jobs
# =====================================================================

submit_jobs() {
    echo -e "${GREEN}Submitting VTEC cluster jobs (365 DOYs)...${NC}"
    
    cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }
    
    # Make script executable
    chmod +x submit_cluster_vtec_full_year.sh
    
    # Create logs directory
    mkdir -p slurm_logs
    
    # Submit job array
    JOBID=$(sbatch submit_cluster_vtec_full_year.sh | awk '{print $4}')
    
    if [ -z "$JOBID" ]; then
        echo -e "${RED}❌ Failed to submit jobs${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Jobs submitted with ID: $JOBID${NC}"
    echo ""
    echo "Monitor progress with:"
    echo "  squeue -j $JOBID"
    echo "  squeue -j $JOBID --state=RUNNING"
    echo ""
    echo "View logs:"
    echo "  tail -f slurm_logs/vtec_doy_*.log"
    echo ""
    echo "Failed jobs:"
    echo "  squeue -j $JOBID --state=FAILED"
}

# =====================================================================
# Monitor Jobs
# =====================================================================

monitor_jobs() {
    JOBID=${1:-""}
    
    if [ -z "$JOBID" ]; then
        echo "Usage: $0 monitor <job_id>"
        exit 1
    fi
    
    echo -e "${GREEN}Monitoring job array $JOBID${NC}"
    echo ""
    
    while true; do
        clear
        echo -e "${GREEN}=== VTEC Training Progress ===${NC}"
        echo "Time: $(date)"
        echo ""
        
        # Overall stats
        TOTAL=$(squeue -j $JOBID --noheader | wc -l)
        RUNNING=$(squeue -j $JOBID --state=RUNNING --noheader | wc -l)
        PENDING=$(squeue -j $JOBID --state=PENDING --noheader | wc -l)
        COMPLETED=$(squeue -j $JOBID --state=COMPLETED --noheader | wc -l)
        FAILED=$(squeue -j $JOBID --state=FAILED --noheader | wc -l)
        
        echo "Total jobs: $TOTAL"
        echo "Running: $RUNNING"
        echo "Pending: $PENDING"
        echo "Completed: $COMPLETED"
        echo "Failed: $FAILED"
        echo ""
        
        # Check if all jobs are done
        if [ $TOTAL -eq 0 ]; then
            echo -e "${GREEN}✅ All jobs completed!${NC}"
            break
        fi
        
        # Show currently running DOYs
        if [ $RUNNING -gt 0 ]; then
            echo "Running DOYs:"
            squeue -j $JOBID --state=RUNNING --noheader | awk '{print $3}' | sort -u
        fi
        
        echo ""
        echo "Press Ctrl+C to exit, will refresh in 30 seconds..."
        sleep 30
    done
}

# =====================================================================
# Check Results
# =====================================================================

check_results() {
    echo -e "${GREEN}Checking training results...${NC}"
    echo ""
    
    cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }
    
    # Count generated experiments
    VTEC_DIRS=$(find experiments -maxdepth 1 -name "*VTEC*" -type d | wc -l)
    STEC_DIRS=$(find experiments -maxdepth 1 -name "*STEC*" -type d | wc -l)
    
    echo "VTEC experiments created: $VTEC_DIRS"
    echo "STEC experiments created: $STEC_DIRS"
    echo ""
    
    # Count ensemble members
    ENSEMBLE_TOTAL=0
    for dir in experiments/Finetune_VTEC_*/; do
        if [ -d "$dir" ]; then
            MEMBERS=$(find "$dir" -name "*.pth" -o -name "*.pt" | wc -l)
            ENSEMBLE_TOTAL=$((ENSEMBLE_TOTAL + MEMBERS))
        fi
    done
    
    echo "Total VTEC ensemble members: $ENSEMBLE_TOTAL"
    echo "Expected (365 DOY × 10 members): 3650"
    echo ""
    
    # Show failed DOYs if any
    if [ -f "slurm_logs/failed_doys.txt" ]; then
        echo -e "${YELLOW}Failed DOYs:${NC}"
        cat slurm_logs/failed_doys.txt
    fi
}

# =====================================================================
# Aggregate Results
# =====================================================================

aggregate_results() {
    echo -e "${GREEN}Aggregating results...${NC}"
    echo ""
    
    cd "$PROJECT_DIR" || { echo "Failed to cd to $PROJECT_DIR"; exit 1; }
    
    # Run multiday evaluation with skip_training to aggregate only
    python src/multiday_evaluation.py \
        --dates "2024-001:2024-365" \
        --vtec_config config/config_mao_laplacian.yaml \
        --stec_config config/config_BNN.yaml \
        --output_dir multiday_results/Mao_FullYear_2024 \
        --skip_training \
        --summary_only
    
    echo -e "${GREEN}✅ Aggregation complete!${NC}"
    echo "Results in: multiday_results/Mao_FullYear_2024/"
}

# =====================================================================
# Main
# =====================================================================

COMMAND=${1:-"submit"}

case "$COMMAND" in
    submit)
        submit_jobs
        ;;
    monitor)
        monitor_jobs "$2"
        ;;
    check)
        check_results
        ;;
    aggregate)
        aggregate_results
        ;;
    *)
        echo "Usage: $0 {submit|monitor|check|aggregate} [args]"
        echo ""
        echo "Commands:"
        echo "  submit           - Submit 365-day job array"
        echo "  monitor <jobid>  - Monitor job progress"
        echo "  check            - Check training results"
        echo "  aggregate        - Aggregate all results into summary"
        echo ""
        echo "Example workflow:"
        echo "  $0 submit                    # Launch jobs"
        echo "  $0 monitor 12345678          # Watch progress (replace with actual job ID)"
        echo "  $0 check                     # After complete, verify results"
        echo "  $0 aggregate                 # Generate aggregate metrics"
        exit 1
        ;;
esac
