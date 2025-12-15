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

### 3. Evaluate Model

Basic model evaluation on test set.

```bash
# Full evaluation
python cli.py evaluate --experiment "Finetune_STEC_..."

# Quick evaluation on subset
python cli.py evaluate \
    --experiment "Finetune_STEC_..." \
    --test_size 10000
```

**Equivalent to:** `python src/evaluation.py --experiment ...`

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

### 5. Positioning Evaluation

Evaluate STEC model impact on GNSS positioning accuracy.

```bash
# Run positioning evaluation
python cli.py positioning --experiment "Finetune_STEC_..."

# Specify date range
python cli.py positioning \
    --experiment "Finetune_STEC_..." \
    --start_date 2024-07-01 \
    --end_date 2024-07-07
```

**Equivalent to:** `python src/inference_positioning.py --experiment ...`

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

# Specific days
python cli.py multiday \
    --dates "2024-183,2024-184,2024-185" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml

# Monthly evaluation for paper
python cli.py multiday \
    --dates "2024-183:2024-213" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml \
    --output_dir multiday_results/july_2024

# Quick test
python cli.py multiday \
    --dates "2024-183,2024-184" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml \
    --num_inference_samples 10
```

**What it does:**
For each date:
1. Finetune STEC model on that day
2. Finetune VTEC model on that day  
3. Run comprehensive comparison (STEC vs VTEC+Mapping vs GIM)
4. Evaluate on both own test set and Madrigal independent test

Finally generates aggregate statistics, comparison plots, and summary tables.

**Equivalent to:** `python src/multiday_evaluation.py --dates ...`

See [MULTIDAY_EVALUATION_GUIDE.md](docs/MULTIDAY_EVALUATION_GUIDE.md) for complete documentation.

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
