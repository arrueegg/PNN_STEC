# Multi-Day Evaluation Pipeline Guide

## Overview

The multi-day evaluation pipeline automates training and comprehensive evaluation across multiple test days. This is essential for producing robust, statistically significant results for your paper, rather than relying on a single day's evaluation.

## What It Does

For each specified date, the pipeline:

1. **Finetunes STEC model** on that day's data (from pretrained weights)
2. **Finetunes VTEC model** on that day's data (from scratch)
3. **Runs comprehensive comparison** against all baselines:
   - Direct STEC predictions
   - VTEC + MSLM mapping
   - IGS GIM
   - Evaluates on both own test set AND Madrigal independent test set (if available)
4. **Stores organized results** for each day

After processing all dates, it generates:
- **Aggregate statistics** across all days (mean, std, etc.)
- **Publication-ready plots** showing trends over time
- **Summary tables** for your paper
- **Individual day results** for detailed analysis

## Quick Start

### Basic Usage

```bash
# Single day
python cli.py multiday \
    --dates "2024-183" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml

# One week of evaluation
python cli.py multiday \
    --dates "2024-183:2024-189" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml
```

### Recommended for Papers

Evaluate at least 7-10 days for statistical significance:

```bash
python cli.py multiday \
    --dates "2024-183:2024-192" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml \
    --output_dir multiday_results/paper_evaluation
```

## Date Formats

The pipeline supports multiple date input formats:

### 1. Single Date
```bash
--dates "2024-183"           # Year-DOY format
--dates "2024-07-01"         # YYYY-MM-DD format
```

### 2. Comma-Separated List
```bash
--dates "2024-183,2024-184,2024-185"
--dates "2024-07-01,2024-07-02,2024-07-03"
```

### 3. Date Range (Inclusive)
```bash
--dates "2024-183:2024-189"           # One week
--dates "2024-07-01:2024-07-10"       # 10 days
--dates "2024-183:2024-213"           # One month
```

You can mix formats in lists:
```bash
--dates "2024-183,2024-07-05,2024-190"
```

## Output Structure

Results are organized in a clear, paper-friendly structure:

```
multiday_results/
├── 2024_DOY_183/
│   ├── stec_experiment/          # STEC model for this day
│   ├── vtec_experiment/          # VTEC model for this day
│   └── evaluation/               # Comparison results
│       ├── own_vtec_gim/         # Own test set results
│       └── madrigal_vtec_gim/    # Independent test results
├── 2024_DOY_184/
│   └── ...
├── 2024_DOY_185/
│   └── ...
└── summary/
    ├── all_results.csv           # Complete results table
    ├── summary_statistics.csv    # Mean/std across days
    ├── rmse_by_date.png          # Time series plots
    ├── metrics_boxplots.png      # Distribution plots
    └── improvement_by_date.png   # Improvement over baselines
```

## Complete Options

```bash
python cli.py multiday \
    --dates <dates>                          # Required: dates to process
    --stec_config <path>                     # Required: base STEC config
    --vtec_config <path>                     # Required: base VTEC config
    --output_dir <path>                      # Optional: output directory (default: multiday_results)
    --num_inference_samples <int>            # Optional: MC samples (default: 100)
    --test_size <int>                        # Optional: test set size (default: full)
    --python_exec <path>                     # Optional: Python executable (default: env/bin/python)
```

## Examples

### Example 1: Quick Test (2 days, fast)
```bash
python cli.py multiday \
    --dates "2024-183,2024-184" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml \
    --num_inference_samples 10 \
    --output_dir multiday_results/quick_test
```
**Runtime:** ~30-60 minutes (depending on model size and hardware)

### Example 2: One Week Evaluation
```bash
python cli.py multiday \
    --dates "2024-183:2024-189" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml \
    --output_dir multiday_results/week1_july
```
**Runtime:** ~4-8 hours (7 days × training + evaluation)

### Example 3: Monthly Evaluation (Robust Paper Results)
```bash
python cli.py multiday \
    --dates "2024-183:2024-213" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml \
    --output_dir multiday_results/july_2024_monthly
```
**Runtime:** ~2-3 days (30 days × training + evaluation)

### Example 4: Specific Days (Different Space Weather Conditions)
```bash
# Evaluate quiet, moderate, and disturbed days
python cli.py multiday \
    --dates "2024-183,2024-195,2024-210,2024-225" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml \
    --output_dir multiday_results/varied_conditions
```

## Understanding the Results

### 1. Individual Day Results

Each day's evaluation produces:
- **Scatter plots**: True vs predicted STEC for all models
- **Residual analysis**: Error patterns by elevation
- **Error distributions**: Histograms and Q-Q plots
- **Metrics summary**: RMSE, MAE, R², Bias for each model
- **Detailed predictions CSV**: All predictions for further analysis

Access via: `batch_results/2024_DOY_183/evaluation/`

### 2. Aggregate Summary

The `summary/` directory contains:

**all_results.csv** - Complete table with all metrics from all days:
```csv
date,year,doy,dataset,Model,RMSE,MAE,Bias,R²,Count
2024-183,2024,183,own_vtec_gim,Direct STEC,10.34,6.12,-0.45,0.927,150000
2024-183,2024,183,own_vtec_gim,VTEC + Mapping,12.56,7.89,-1.23,0.891,150000
...
```

**summary_statistics.csv** - Mean and std across all days:
```csv
Dataset,Model,RMSE_mean,RMSE_std,MAE_mean,MAE_std,R2_mean,R2_std,Num_days
own_vtec_gim,Direct STEC,10.45,1.23,6.34,0.89,0.925,0.015,7
own_vtec_gim,VTEC + Mapping,12.89,1.45,8.12,1.02,0.885,0.022,7
...
```

**Plots:**
- `rmse_by_date.png`: Time series showing how each model performs across days
- `metrics_boxplots.png`: Box plots comparing model distributions
- `improvement_by_date.png`: Direct STEC improvement over baselines by day

### 3. Using Results in Your Paper

**Statistical Significance:**
```
Direct STEC: 10.45 ± 1.23 TECU (mean ± std across 7 days)
VTEC+Mapping: 12.89 ± 1.45 TECU
IGS GIM: 14.23 ± 1.78 TECU

Improvement: 18.9% over VTEC+Mapping, 26.6% over GIM
```

**Include in paper:**
1. Summary statistics table (from `summary_statistics.csv`)
2. Time series plot showing performance stability
3. Box plots showing distribution differences
4. Individual day examples for detailed analysis

## Tips & Best Practices

### 1. Choose Representative Days

For robust evaluation, select days that represent:
- **Different seasons** (if evaluating across months)
- **Varying space weather conditions** (quiet, moderate, disturbed)
- **Different geographic coverage** (if applicable)

### 2. Monitor Progress

The pipeline prints detailed progress for each day:
```
======================================================================
Processing Date: 2024-183 (2024 DOY 183)
======================================================================

[1/3] Training STEC model for 2024-183
✓ Training completed: Finetune_STEC_2024_183_...

[2/3] Training VTEC model for 2024-183
✓ Training completed: Finetune_VTEC_2024_183_...

[3/3] Running comparison evaluation for 2024-183
✓ All steps completed for 2024-183
```

If a day fails, the pipeline continues with remaining days.

### 3. Computational Resources

**Disk space:** Each day generates ~500MB-2GB of results
- 7 days: ~5-15 GB
- 30 days: ~20-60 GB

**Runtime estimates (per day):**
- Small model + quick test: ~10-20 minutes
- Full model + full evaluation: ~2-4 hours

**Parallelization:**
Currently sequential. For multiple days, consider:
- Running batch job on cluster
- Using SLURM array jobs (one day per job)

### 4. Resuming Failed Runs

If the pipeline fails mid-execution, you can:
1. Check which days completed successfully in `output_dir/`
2. Remove failed days from the date range
3. Re-run with remaining dates
4. Manually combine results later

Future enhancement: `--skip_existing` flag to resume automatically.

## Configuration Files

### Base Configs Should Include

**STEC Config** (`config/config.yaml`):
```yaml
mode: finetune  # Will be set automatically
model:
  model_type: FactorizedSTEC  # Or BNN_NLL, etc.
  
finetune:
  pretrained_path: "experiments/Pretrain_STEC_..."  # Required
  # year and doy will be set automatically
  
# Other training parameters...
```

**VTEC Config** (`config/config_vtec_mlp_baseline.yaml`):
```yaml
mode: finetune
target: vtec  # Must be vtec for VTEC model

model:
  model_type: MLP
  
finetune:
  finetune_from_scratch: true  # Will be set automatically
  # year and doy will be set automatically

# Other training parameters...
```

The pipeline automatically:
- Sets `mode: finetune`
- Sets `finetune.year` and `finetune.doy`
- Sets `finetune_from_scratch` for VTEC models

## Common Issues

### Issue 1: CUDA Out of Memory
**Solution:** Reduce batch size in config or add `--test_size 50000`

### Issue 2: Training Takes Too Long
**Solution:** 
- Use `--num_inference_samples 10` for quick testing
- Reduce `--test_size` for faster evaluation
- Full runs can be scheduled overnight/weekend

### Issue 3: Pretrained Model Not Found
**Solution:** Ensure `finetune.pretrained_path` in STEC config points to valid pretrained experiment

### Issue 4: GIM Data Not Available
The comparison script will automatically skip GIM evaluation if data is not found. No error.

## Advanced Usage

### Custom Python Environment
```bash
python cli.py batch \
    --dates "2024-183:2024-189" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml \
    --python_exec /path/to/custom/python
```

### Custom Output Structure
```bash
python cli.py multiday \
    --dates "2024-183:2024-189" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml \
    --output_dir experiments/paper_results/multi_day_evaluation
```

## Next Steps

After running batch evaluation:

1. **Review aggregate statistics** in `summary/summary_statistics.csv`
2. **Check plots** for any anomalies or trends
3. **Select representative days** for detailed analysis in paper
4. **Include summary table** with mean ± std in paper
5. **Show time series plots** to demonstrate stability
6. **Discuss outlier days** if any (space weather events?)

## Paper Checklist

- [ ] Evaluate at least 7-10 days for statistical significance
- [ ] Include days with varied space weather conditions
- [ ] Generate summary statistics (mean ± std)
- [ ] Create time series plots showing performance stability
- [ ] Select 2-3 representative days for detailed case studies
- [ ] Document any failed days and reasons
- [ ] Include both own test set AND Madrigal independent test results
- [ ] Report improvements over baselines with confidence intervals
