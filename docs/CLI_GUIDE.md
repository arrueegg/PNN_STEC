# PNN_STEC Command Line Interface Guide

## Overview

The new `cli.py` provides a unified entry point for all PNN_STEC workflows with clean subcommands.

## Quick Start

```bash
# Get help
python cli.py --help
python cli.py <command> --help

# Or use the virtual environment
env/bin/python cli.py --help
```

## Available Commands

### 1. Train Models

Train STEC or VTEC models (pretrain or finetune).

```bash
# Standard training
python cli.py train --config config/config.yaml

# VTEC baseline training
python cli.py train --config config/config_vtec_mlp_baseline.yaml

# Finetune on single day
python cli.py train --config config/config_finetune.yaml
```

**Equivalent to:** `python src/main.py --config <config>`

---

### 2. Compare Against Baselines

Comprehensive comparison: Direct STEC vs VTEC+Mapping vs IGS GIM.

```bash
# Full comparison (recommended)
python cli.py compare \
    --stec_experiment "Finetune_STEC_2024_183_FactorizedSTEC_..." \
    --vtec_experiment "Finetune_VTEC_2024_183_MLP_..."

# STEC only (no VTEC baseline)
python cli.py compare \
    --stec_experiment "Finetune_STEC_..."

# Quick test with subset
python cli.py compare \
    --stec_experiment "Finetune_STEC_..." \
    --vtec_experiment "Finetune_VTEC_..." \
    --test_size 1000 \
    --num_inference_samples 10

# Skip GIM comparison
python cli.py compare \
    --stec_experiment "Finetune_STEC_..." \
    --vtec_experiment "Finetune_VTEC_..." \
    --no_gim
```

**Automatic evaluation:**
- Own test set (always)
- Madrigal independent test set (if available and model is finetuned)
- VTEC+Mapping baseline (if `--vtec_experiment` provided)
- IGS GIM baseline (enabled by default)

**Equivalent to:** `python src/compare_stec_vtec_gim.py ...`

---

### 3. Evaluate Model — removed

`cli.py evaluate` did `from evaluation import main`, which has always resolved to the
*package* `src/evaluation/` (no `main` attribute) rather than the flat module
`src/evaluation.py` it was written against - an `ImportError` on every invocation, since
before the `stec/` rebuild and unrelated to it. It has been removed rather than fixed:
section 4 below (`cli.py inference`) already does "compute metrics and generate plots"
via the same underlying driver (`src/inference_testset.py`) and has always been the
command real evaluation runs used (CLAUDE.md: "not the one used for paper numbers").

---

### 4. Run Inference

Generate predictions on test dataset.

```bash
# Run inference and save predictions
python cli.py inference --experiment "Finetune_STEC_..."

# Inference on subset
python cli.py inference \
    --experiment "Finetune_STEC_..." \
    --test_size 10000 \
    --output_file predictions.csv
```

**Equivalent to:** `python src/inference_testset.py --experiment ...`

---

### 5. Positioning Evaluation — removed

`cli.py positioning` targeted `inference_positioning.py`, a module that does not exist
anywhere in this repository, under `src/` or otherwise, and never has - not a casualty
of the `stec/` rebuild. Use the real positioning driver directly:

```bash
# Full pipeline (products, RINEX, PPPx, metrics) for one experiment/day range
positioning/scripts/run_pipeline.py --experiment "Finetune_STEC_..." \
    --start_date 2024-07-01 --end_date 2024-07-07

# One experiment/day directly
positioning/positioning_eval/run_positioning_evaluation.py \
    --experiment "Finetune_STEC_..." --date 2024-07-01 --stations ZIMM BRUS WTZR
```

See CLAUDE.md's positioning workflow and `docs/revision/retirement_inventory.md`.

---

### 6. Generate Spatial Maps

Generate spatial STEC maps for visualization.

```bash
# Generate map for specific time
python cli.py map \
    --experiment "Finetune_STEC_..." \
    --date 2024-07-01 \
    --time 12:00

# Generate time series of maps
python cli.py map \
    --experiment "Finetune_STEC_..." \
    --date 2024-07-01 \
    --time_series 00:00-23:59 \
    --interval 1h
```

**Equivalent to:** `python src/inference_map.py --experiment ...`

---

### 7. Multi-Day Evaluation

Automated pipeline for training and evaluating models across multiple test days. Essential for statistically robust paper results.

```bash
# One week evaluation
python cli.py multiday \
    --dates "2024-183:2024-189" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml

# Specific days with independent positioning analysis
python cli.py multiday \
    --dates "2024-183,2024-184" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml \
    --positioning

# Monthly evaluation for paper
python cli.py multiday \
    --dates "2024-183:2024-213" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml \
    --output_dir multiday_results/july_2024

# Quick test with fewer samples (for debugging)
python cli.py multiday \
    --dates "2024-183,2024-184" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml \
    --num_inference_samples 10 \
    --test_size 1000
```

**Key Arguments:**
- `--dates`: Date range (e.g., "2024-183:2024-189") or list ("2024-183,2024-184")
- `--stec_config`: Path to STEC model config
- `--vtec_config`: Path to VTEC baseline config
- `--positioning`: **(NEW)** Run positioning domain analysis for each day
- `--pretrain_folder`: Use a specific pretrained model instead of training from scratch
- `--skip_training`: Only run evaluation on existing models
- `--no_aggregate`: Skip the final aggregation step (useful for parallel runs)
- `--summary_only`: Skip all processing and only generate reports from existing CSVs

**What it does:**
For each date:
1. Finetune STEC model on that day (unless `--skip_training`)
2. Finetune VTEC model on that day (unless `--skip_training`)
3. Run comprehensive comparison (STEC vs VTEC+Mapping vs GIM)
4. Evaluate on both own test set and Madrigal independent test
5. (Optional) Run positioning analysis if `--positioning` is set

Finally generates aggregate statistics, comparison plots, and summary tables.

**Equivalent to:** `python src/multiday_evaluation.py --dates ...`

See [MULTIDAY_EVALUATION_GUIDE.md](docs/MULTIDAY_EVALUATION_GUIDE.md) for complete documentation.

---

### 8. Standalone Utility Scripts

Some advanced analysis tasks have specialized scripts not yet integrated into the main CLI.

#### Positioning Analysis Plotting
Regenerate high-quality positioning plots from multi-day results with advanced filtering.

```bash
python src/plot_positioning_manual.py \
    --input multiday_results/positioning_snx/multiday_summary.csv \
    --output_dir plots/paper_ready \
    --exclude_threshold 5
```

#### Differential STEC (dSTEC) Evaluation
Evaluate using the dSTEC metric to remove geometry-dependent errors.

```bash
python src/dstec_evaluation.py "experiments/Finetune_STEC_..."
```


---

## Common Options

### Experiment Names

You can provide either:
- Short name: `"Finetune_STEC_2024_183_..."`
- Full path: `"experiments/Finetune_STEC_2024_183_..."`

The CLI will automatically resolve to the correct path.

### Inference Samples

For Bayesian models, control uncertainty estimation:
- `--num_inference_samples 100` (default, for production)
- `--num_inference_samples 10` (for quick testing)

### Test Set Size

Control dataset size:
- `--test_size 100000` (specific number)
- No flag = use full test set (recommended for final results)

---

## Workflow Examples

### Complete Training & Evaluation Pipeline

```bash
# 1. Train STEC model
python cli.py train --config config/config.yaml

# 2. Train VTEC baseline
python cli.py train --config config/config_vtec_mlp_baseline.yaml

# 3. Comprehensive comparison
python cli.py compare \
    --stec_experiment "Finetune_STEC_..." \
    --vtec_experiment "Finetune_VTEC_..."

# 4. Generate visualization maps
python cli.py map \
    --experiment "Finetune_STEC_..." \
    --date 2024-07-01 \
    --time 12:00
```

### Quick Testing Workflow

```bash
# Fast evaluation on small subset
python cli.py compare \
    --stec_experiment "Finetune_STEC_..." \
    --vtec_experiment "Finetune_VTEC_..." \
    --test_size 1000 \
    --num_inference_samples 10 \
    --no_gim
```

---

## Backward Compatibility

All original scripts remain functional:
- `python src/main.py --config ...` still works
- `python src/compare_stec_vtec_gim.py ...` still works
- etc.

The CLI is a convenient wrapper - use whichever interface you prefer.

---

## Tips

1. **Get help for any command:**
   ```bash
   python cli.py train --help
   python cli.py compare --help
   ```

2. **Tab completion** (if your shell supports it):
   ```bash
   python cli.py <TAB>
   ```

3. **Use virtual environment:**
   ```bash
   env/bin/python cli.py <command> [options]
   ```

4. **Check available experiments:**
   ```bash
   ls experiments/
   ```
