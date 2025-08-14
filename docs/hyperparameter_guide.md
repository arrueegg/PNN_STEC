# Hyperparameter Tuning Guide

## 🎯 Simple and Clean Approach

This project has a streamlined hyperparameter tuning system that is:
- **Simple**: One script, multiple grid options
- **Clean**: No complex directory structures or metadata  
- **Practical**: Reasonable search spaces that actually work
- **Scalable**: Supports both local execution and cluster submission

## 📝 Quick Start

### Local Execution
```bash
# Mini test (2 combinations) - for quick testing
python scripts/hyperparameter_search.py --grid mini

# Standard search (72 combinations) - for thorough optimization  
python scripts/hyperparameter_search.py --grid standard

# Custom output directory
python scripts/hyperparameter_search.py --grid standard --output my_hp_test
```

### �️ Cluster Execution (NEW!)
```bash
# Generate SLURM scripts for cluster submission
python scripts/hyperparameter_search.py --grid standard --cluster

# Submit all jobs to cluster
./hp_search/submit_all.sh

# Monitor progress
squeue -u $USER
```

👉 **For detailed cluster usage, see [Cluster Guide](cluster_hyperparameter_guide.md)**

## �🔧 What Gets Generated

- **Config files**: `config_01.yaml`, `config_02.yaml`, etc.
- **Run script**: `run_search.sh` for local sequential execution
- **SLURM scripts**: `slurm_scripts/trial_*.sh` for cluster execution (with `--cluster`)
- **Submit script**: `submit_all.sh` for easy cluster submission (with `--cluster`)
- **Results**: Each trial saves to its own subdirectory

## 🚀 Running the Search

```bash
# Generate configs
python scripts/hyperparameter_search.py --grid standard

# Run the search
cd hp_search
./run_search.sh

# Or run single trial for testing
python src/main.py --config_path hp_search/config_01.yaml
```
