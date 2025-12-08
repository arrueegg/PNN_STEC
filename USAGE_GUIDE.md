# PNN_STEC Usage Guide

Quick reference for running different parts of the codebase.

## 🚀 Main Training Pipeline

### `src/main.py`
**Primary entry point for model training (pretrain + finetune)**

```bash
python src/main.py
```

- Automatically handles both **pretrain** and **finetune** modes based on `config/config.yaml`
- Set `mode: "pretrain"` or `mode: "finetune"` in config
- If finetuning and no pretrained model exists, automatically runs pretraining first
- All experiments saved to `experiments/<auto_generated_name>/`

---

## 🔍 Inference & Evaluation

### `src/inference_testset.py`
**Run inference on test set and generate metrics/plots**

```bash
python src/inference_testset.py
```

- Loads trained model from experiment folder (auto-detected from config)
- Generates comprehensive test set metrics
- Creates plots and saves to `experiments/<experiment_name>/`

### `src/inference_positioning.py`
**Generate STEC corrections for positioning applications**

```bash
# Single date
python src/inference_positioning.py --experiment <exp_folder> --date 2024-07-01

# Date range
python src/inference_positioning.py --experiment <exp_folder> --start_date 2024-07-01 --end_date 2024-07-05
```

- Filters observations by test stations (from `test_station.list`)
- Exports CSV files per station/day: `experiments/<exp>/positioning/stec_corrections/YYYYDDD/<station>.csv`

### `src/inference_map.py`
**Generate global STEC maps**

```bash
python src/inference_map.py --date 2024-05-15 --elevation 30.0 --azimuth 180.0
```

- Creates hourly global STEC maps with configurable resolution
- Saves numpy files (`.npz`) to `experiments/<experiment_name>/global_maps/`

### `src/evaluation.py`
**Compare model predictions vs IGS GIM products**

```bash
python src/evaluation.py
```

- Groups test results by day
- Compares model STEC vs GIM STEC vs ground truth
- Saves comparison plots and CSV files

### `src/dstec_evaluation.py`
**Differential STEC (dSTEC) evaluation**

```bash
python src/dstec_evaluation.py
```

- Evaluates using differential STEC metric (removes common-mode errors)
- Configuration set directly in file (not in `config.yaml`)
- Compares model vs GIM using elevation-based differences

---

## 📊 Positioning Evaluation

### `src/positioning_eval/run_positioning_evaluation.py`
**Complete positioning evaluation pipeline**

```bash
# Single station
python src/positioning_eval/run_positioning_evaluation.py --experiment <exp_folder> --date 2024-07-01 --stations ZIMM BRUS

# All test stations
python src/positioning_eval/run_positioning_evaluation.py --experiment <exp_folder> --date 2024-07-01 --all_test_stations
```

- Downloads GNSS products (orbits, clocks, GIM, RINEX)
- Runs PPPx positioning with model corrections vs IGS GIM
- Computes positioning accuracy metrics
- Saves daily summary reports

---

## 🔧 Hyperparameter Search

### `scripts/hyperparameter_search.py`
**Grid search for hyperparameter tuning**

```bash
python scripts/hyperparameter_search.py --grid standard --output_dir hp_search/
```

- Generates multiple config files with parameter combinations
- Edit parameter grids directly in the script
- Use `hp_search/run_search.sh` to launch all configs

### WandB Sweeps
**Distributed hyperparameter optimization**

```bash
# Initialize sweep (returns sweep_id)
wandb sweep config/wandb_sweep_config_<model>.yaml

# Launch agents
wandb agent <sweep_id>
```

- Sweep configs in `config/wandb_sweep_config_*.yaml`
- Supports parallel agents for faster optimization

---

## 📝 Configuration

All experiments controlled by **`config/config.yaml`**:

- `mode`: "pretrain" or "finetune"
- `model.model_type`: "BNN_NLL", "MLP_MSE", "Branch_BNN_NLL", etc.
- `target`: "stec" or "vtec"
- `use_SWI`: Enable Space Weather Indices
- `target_weighting.enabled`: Address high STEC underprediction
- `kl_annealing`: Gradually increase KL weight for BNN training

---

## 📂 Output Locations

- **Training**: `experiments/<auto_generated_name>/`
  - Models: `model/best_model.pth`
  - Logs: `logs/`
  - Plots: `plots/`
  
- **Inference**: `experiments/<experiment_name>/`
  - Test metrics: `test_results/`
  - Positioning CSVs: `positioning/stec_corrections/`
  - Global maps: `global_maps/`

- **Hyperparameter search**: `hp_search/config_*.yaml`
