# 🖥️ Cluster Hyperparameter Tuning Guide

## 🎯 Overview

The hyperparameter search system now supports **automatic SLURM cluster submission**! This allows you to run hundreds of hyperparameter combinations in parallel across multiple cluster nodes.

## 🚀 Quick Start

### Local Execution (Original)
```bash
# Generate configs for local execution
python scripts/hyperparameter_search.py --grid standard

# Run locally
./hp_search/run_search.sh
```

### Cluster Execution (New!)
```bash
# Generate configs + SLURM scripts for cluster
python scripts/hyperparameter_search.py --grid standard --cluster

# Submit all jobs to cluster
./hp_search/submit_all.sh

# Monitor progress
squeue -u $USER
```

## 📋 Parameter Grids

| Grid | Combinations | Best For |
|------|-------------|----------|
| `mini` | 2 | Quick testing |
| `standard` | 72 | Thorough optimization |
| `custom` | 360 | Extensive search |

## 🔧 SLURM Configuration

### Default Settings (modify in script)
```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
```

### Cluster Paths (update these!)
Edit the `generate_slurm_script()` function in `scripts/hyperparameter_search.py`:
```python
f.write("main_dir=\"/YOUR/CLUSTER/PATH/PNN_STEC\"\n")
```

## 📁 Generated Structure

```
hp_search/
├── config_01.yaml, config_02.yaml, ...    # Trial configurations
├── submit_all.sh                           # Submit all jobs
├── slurm_scripts/
│   ├── trial_01.sh, trial_02.sh, ...     # Individual SLURM scripts
├── logs/
│   └── trial_01-%j.out, trial_02-%j.out, ...  # Job outputs
└── results/
    ├── trial_01/, trial_02/, ...          # Trial results
```

## 🎛️ Usage Examples

### 1. Quick Test on Cluster
```bash
# Test with mini grid (2 jobs)
python scripts/hyperparameter_search.py --grid mini --cluster --output test_hp
./test_hp/submit_all.sh
```

### 2. Standard Search
```bash
# Full search (72 jobs)
python scripts/hyperparameter_search.py --grid standard --cluster --output hp_standard
./hp_standard/submit_all.sh
```

### 3. Custom Grid
```bash
# Large search (360 jobs) - use with caution!
python scripts/hyperparameter_search.py --grid custom --cluster --output hp_large
./hp_large/submit_all.sh
```

### 4. Single Trial Testing
```bash
# Submit just one trial to test setup
sbatch hp_search/slurm_scripts/trial_01.sh
```

## 📊 Monitoring and Management

### Check Job Status
```bash
# View your jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# View job output (while running)
tail -f hp_search/logs/trial_01-<job_id>.out
```

### Cancel Jobs
```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel specific job pattern
scancel --name="hp_trial_*"
```

## ⚙️ Customization

### 1. Modify SLURM Settings
Edit the `generate_slurm_script()` function in `scripts/hyperparameter_search.py`:
```python
f.write("#SBATCH --time=12:00:00\n")      # Reduce time
f.write("#SBATCH --mem-per-cpu=8G\n")     # More memory
f.write("#SBATCH --partition=gpu\n")      # Specific partition
```

### 2. Add Your Parameter Grid
```python
def define_grids():
    grids = {
        'my_grid': {
            'model.hidden_dim': [64, 128, 256],
            'model.num_layers': [2, 3, 4],
            'pretrain.learning_rate': [0.001, 0.01],
            # ... your parameters
        }
    }
```

### 3. Cluster-Specific Modules
Update module loading in `generate_slurm_script()`:
```python
f.write("module load your_python_module\n")
f.write("module load your_cuda_module\n")
```

## 💡 Best Practices

1. **Start Small**: Test with `--grid mini` first
2. **Check Paths**: Update cluster paths in the script
3. **Monitor Resources**: Check memory/time usage of first few jobs
4. **Use Arrays**: For very large searches, consider SLURM job arrays
5. **Save Results**: Results are saved to individual trial directories

## 🛠️ Troubleshooting

### Common Issues

**Jobs fail immediately**
- Check module loading commands
- Verify cluster paths
- Test single trial first

**Out of memory**
- Increase `--mem-per-cpu`
- Reduce batch size in configs
- Check GPU memory usage

**Jobs pending**
- Check cluster queue policies
- Verify requested resources are available
- Consider using different partition

### Debug Single Trial
```bash
# Run single trial interactively
srun --pty bash
cd /your/path/PNN_STEC
source env/bin/activate
python src/main.py --config_path hp_search/config_01.yaml
```

## 🎉 Ready to Scale!

Your hyperparameter tuning system is now cluster-ready! You can efficiently search through hundreds of parameter combinations using the full power of your cluster.
