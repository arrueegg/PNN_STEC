#!/bin/bash
# Submit all hyperparameter trials to SLURM
# Grid: mini, Trials: 2
# Generated: 2025-08-16 08:04

echo "🚀 Submitting hyperparameter trials to cluster..."
job_ids=()

echo "Submitting trial 1/2"
job_id=$(sbatch --parsable hp_search/slurm_scripts/trial_001.sh)
job_ids+=($job_id)
echo "  Job ID: $job_id"

echo "Submitting trial 2/2"
job_id=$(sbatch --parsable hp_search/slurm_scripts/trial_002.sh)
job_ids+=($job_id)
echo "  Job ID: $job_id"

echo "📊 Submitted ${#job_ids[@]} jobs to cluster"
echo "Job IDs: ${job_ids[*]}"
echo
echo "📋 Monitor with: squeue -u $USER"
echo "📋 Cancel all with: scancel ${job_ids[*]}"
