# Cluster Training Guide: VTEC Full Year (365 DOYs)

## Quick Start

### Step 1: Make scripts executable
```bash
chmod +x submit_cluster_vtec_full_year.sh manage_cluster_jobs.sh
```

### Step 2: Submit all 365 jobs
```bash
bash manage_cluster_jobs.sh submit
```

This will:
- Launch a SLURM job array with 365 jobs (one per DOY)
- Each job trains 10 ensemble members (~20-30 min per DOY)
- Return a job ID (e.g., `12345678`)

**Output:** Job ID to use in next step

---

## Monitor Progress

### Watch real-time (auto-refreshes every 30 sec)
```bash
bash manage_cluster_jobs.sh monitor 12345678
```

Replace `12345678` with your actual job ID from submit step.

**Shows:**
- Total jobs, running, pending, completed, failed
- Which DOYs are currently training
- Overall progress

**Exit:** Press `Ctrl+C`

### Manual monitoring
```bash
# See all jobs
squeue -j 12345678

# Only running jobs
squeue -j 12345678 --state=RUNNING

# Failed jobs
squeue -j 12345678 --state=FAILED

# View log for specific DOY (e.g., DOY 122)
tail -f slurm_logs/vtec_doy_122.log

# View all logs in real-time
tail -f slurm_logs/vtec_doy_*.log
```

---

## After Training Completes

### Check Results
```bash
bash manage_cluster_jobs.sh check
```

Shows:
- Number of VTEC experiments created
- Total ensemble members trained
- Failed DOYs (if any)

### Aggregate Results

Generate summary statistics across all 365 DOYs:

```bash
bash manage_cluster_jobs.sh aggregate
```

This creates:
- `multiday_results/Mao_FullYear_2024/aggregate_metrics.csv` - Overall statistics
- `multiday_results/Mao_FullYear_2024/plots/` - Comparison plots
- Summary reports

---

## Computational Resources

### Per Job (per DOY)
- **Time:** 20-30 min (10 ensemble members)
- **GPU:** 1 GPU required
- **Memory:** 64 GB RAM
- **CPUs:** 1 task

### Total for Full Year
- **Wall-clock time:** ~30 min (all 365 jobs in parallel)
- **If sequential queue:** 365 × 25 min ≈ 150 hours compute

---

## Troubleshooting

### Jobs fail immediately
Check your job log:
```bash
cat slurm_logs/vtec_doy_001.err
```

Common issues:
- Python environment not found → Update `PROJECT_DIR` in script
- Data not found → Transfer H5 files to cluster
- CUDA issues → Check GPU availability with `nvidia-smi`

### Out of memory
Increase `--mem` in `submit_cluster_vtec_full_year.sh` (currently 64G)

### Missing Python packages
On cluster login node:
```bash
cd /scratch2/arrueegg/WP4/PNN_STEC
source venv/bin/activate
pip install -r requirements.txt
```

### Resubmit failed DOYs
```bash
# Check which DOYs failed
bash manage_cluster_jobs.sh check | grep "Failed"

# Resubmit just those DOYs manually:
# Edit submit script and use --array=122,145,200 etc.
```

---

## Configuration Details

### File: submit_cluster_vtec_full_year.sh
- **#SBATCH --array=1-365** - Range of DOYs to train
- **#SBATCH --time=08:00:00** - Timeout per job
- **#SBATCH --gpus-per-task=1** - GPUs per job
- **#SBATCH --mem=64G** - RAM per job

### File: manage_cluster_jobs.sh
- **submit** - Launch jobs
- **monitor** - Watch progress
- **check** - Verify results
- **aggregate** - Create summary

---

## Expected Output Structure

After all jobs complete:
```
experiments/
├── Finetune_VTEC_2024_001_MLP_LaplacianNLL_*.pth  (10 members)
├── Finetune_VTEC_2024_002_MLP_LaplacianNLL_*.pth  (10 members)
├── ...
└── Finetune_VTEC_2024_365_MLP_LaplacianNLL_*.pth  (10 members)

multiday_results/
└── Mao_FullYear_2024/
    ├── 2024_DOY_001/
    │   ├── metrics_vtec.csv
    │   └── plots/
    ├── 2024_DOY_002/
    ...
    ├── aggregate_metrics.csv
    └── plots/

positioning_results/  (if --positioning used)
├── 2024_DOY_001/
├── 2024_DOY_002/
...
└── aggregate_positioning.csv
```

---

## Advanced Options

### Limit parallelism (e.g., max 20 jobs at once)
Edit `submit_cluster_vtec_full_year.sh`:
```bash
#SBATCH --array=1-365%20   # Max 20 running simultaneously
```

### Train specific DOY range (e.g., DOY 100-110)
```bash
#SBATCH --array=100-110
```

### Run with positioning evaluation (slower, ~2x time)
Add to `src/main.py` call:
```bash
--positioning true
```

### Different random seeds for reproducibility
Edit line in script:
```bash
python src/main.py ... --random_seed $((42 + DOY))
```

---

## Example Session

```bash
# 1. Submit
$ bash manage_cluster_jobs.sh submit
✅ Jobs submitted with ID: 12345678

# 2. Monitor (wait ~30 minutes for parallelization)
$ bash manage_cluster_jobs.sh monitor 12345678

# 3. Check after completion
$ bash manage_cluster_jobs.sh check
VTEC experiments created: 365
Total VTEC ensemble members: 3650

# 4. Aggregate results
$ bash manage_cluster_jobs.sh aggregate
✅ Aggregation complete!
Results in: multiday_results/Mao_FullYear_2024/
```

---

## Next Steps

After full year training:

1. **Review metrics:** `cat multiday_results/Mao_FullYear_2024/aggregate_metrics.csv`
2. **Run positioning evaluation:** `sbatch submit_cluster_vtec_positioning.sh` (if needed)
3. **Transfer results back:** 
   ```bash
   rsync -avz experiments/ local_machine:/path/to/results/
   rsync -avz multiday_results/ local_machine:/path/to/results/
   ```

---
