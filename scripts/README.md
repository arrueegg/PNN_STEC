# Scripts Directory

This directory contains various utility scripts for the PNN_STEC project.

## WandB Sweep Scripts

### Core Files
- **`wandb_sweep_manager.py`**: Python script that creates WandB sweeps and submits parallel SLURM jobs
- **`launch_wandb_sweep.sh`**: Simple bash wrapper for easy sweep launching

### Configuration Files (located in `config/`)
- **`config/wandb_sweep_config.yaml`**: Default sweep configuration (balanced parameters)
- **`config/wandb_sweep_config_quick.yaml`**: Quick test configuration (minimal parameters, fast execution)
- **`config/wandb_sweep_config_comprehensive.yaml`**: Extensive search space (many parameters, thorough exploration)

### Quick Usage
```bash
# Launch 8 parallel agents with default config
./scripts/launch_wandb_sweep.sh 8

# Use quick test config
./scripts/launch_wandb_sweep.sh 4 config/wandb_sweep_config_quick.yaml

# Advanced usage
python scripts/wandb_sweep_manager.py --config config/wandb_sweep_config.yaml --agents 12
```

For detailed documentation, see [`docs/wandb_sweep_guide.md`](../docs/wandb_sweep_guide.md).

## Other Scripts
- **`hyperparameter_search.py`**: Original grid search hyperparameter tuning
- **`plot_dataset_counts.py`**: Generate dataset visualization plots
- **`plot_dataset_statistics.py`**: Create dataset statistics plots

## Note on Organization
- **Scripts**: Located in `scripts/` (user-created, version controlled)
- **Configurations**: Located in `config/` (alongside main config.yaml)
- **Generated Files**: Auto-created in `hp_search/logs/` and `hp_search/wandb_slurm_scripts/`
- **Original HP Search**: Grid search files remain in `hp_search/` (auto-generated configs)