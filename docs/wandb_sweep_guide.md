# WandB Sweep Hyperparameter Tuning on Euler Cluster

This guide explains how to use the new WandB (Weights & Biases) sweep functionality for distributed hyperparameter optimization on the Euler cluster.

## Overview

The WandB sweep system provides several advantages over the existing grid search:

- **Intelligent Search**: Uses Bayesian optimization to efficiently explore hyperparameter space
- **Parallel Execution**: Launches multiple agents on different cluster nodes simultaneously  
- **Real-time Monitoring**: Track progress and results through WandB's web interface
- **Early Termination**: Automatically stops poorly performing runs to save compute time
- **Experiment Tracking**: All runs are centrally logged with full reproducibility

## Quick Start

### **Quick Start (Recommended)**
```bash
cd /cluster/work/igp_psr/arrueegg/WP4/PNN_STEC

# Launch 8 parallel agents with default config
./scripts/launch_wandb_sweep.sh 8
```

### **Test Configuration First**
```bash
# Start with quick test (fast, minimal resources)
./scripts/launch_wandb_sweep.sh 4 config/wandb_sweep_config_quick.yaml
```

### **Advanced Usage**
```bash
# Create sweep without submitting jobs
python scripts/wandb_sweep_manager.py --create-only --config config/wandb_sweep_config.yaml

# Submit agents to existing sweep
python scripts/wandb_sweep_manager.py --sweep-id YOUR_SWEEP_ID --agents 16
```

### **Available Configurations**
- **`config/wandb_sweep_config.yaml`**: Balanced default configuration
- **`config/wandb_sweep_config_quick.yaml`**: Fast testing (minimal parameters)
- **`config/wandb_sweep_config_comprehensive.yaml`**: Extensive exploration (your customized version)

### 2. Custom Configuration

Create your own sweep config and launch with 16 agents:

```bash
# Copy and modify the default config
cp config/wandb_sweep_config.yaml config/my_custom_sweep.yaml
# Edit config/my_custom_sweep.yaml as needed

# Launch with custom config
./scripts/launch_wandb_sweep.sh 16 config/my_custom_sweep.yaml
```

### 3. Advanced Usage

Use the Python manager directly for more control:

```bash
# Create sweep only (don't submit agents yet)
python scripts/wandb_sweep_manager.py --create-only

# Submit agents to existing sweep
python scripts/wandb_sweep_manager.py --sweep-id YOUR_SWEEP_ID --agents 12
```

## Sweep Configuration

### Default Search Space

The default configuration (`hp_search/wandb_sweep_config.yaml`) searches over:

**Model Architecture:**
- Model types: BNN_NLL, MLP_NLL, Branch_BNN_NLL, MLP_MCDropout_NLL
- Hidden dimensions: 128, 256, 512, 1024
- Number of layers: 3, 4, 6, 8

**Training Hyperparameters:**
- Learning rate: log-uniform between 0.0001 and 0.1
- Batch size: 256, 512, 1024, 2048
- Loss weight: log-uniform between 0.01 and 10.0
- Optimizers: Adam, AdamW
- Weight decay: log-uniform between 0.0 and 0.01

**BNN-Specific Parameters:**
- KL annealing warmup epochs: 10, 20, 30, 50
- KL end weight: log-uniform between 0.001 and 1.0

**Target Weighting:**
- Enabled/disabled
- Weight functions: linear, quadratic, log, quantile
- High value weight: uniform between 1.0 and 5.0

### Customizing Search Space

Edit `config/wandb_sweep_config.yaml` to modify the search space:

```yaml
parameters:
  # Example: Add new parameter
  pretrain.patience:
    values: [10, 20, 30, 50]
  
  # Example: Change distribution
  pretrain.learning_rate:
    distribution: uniform  # instead of log_uniform_values
    min: 0.001
    max: 0.01
  
  # Example: Single value (fixed parameter)
  model.num_layers:
    value: 4
```

## Monitoring and Management

### 1. Web Interface

After launching a sweep, you'll see a URL like:
```
https://wandb.ai/your-username/PNN_STEC/sweeps/sweep-id
```

This interface shows:
- Real-time progress of all agents
- Best performing configurations
- Parameter importance analysis
- Performance visualizations

### 2. Command Line Monitoring

```bash
# Check SLURM job status
squeue -u $USER

# View job logs
ls hp_search/logs/wandb_agent_*

# Cancel all sweep jobs
scancel $(squeue -u $USER -h -o %i)
```

### 3. Early Stopping

The default configuration includes Hyperband early termination:
- Stops poorly performing runs after 10 epochs
- Saves compute time for promising configurations
- Can be disabled by removing the `early_terminate` section

## Understanding Results

### 1. Best Configurations

WandB automatically tracks the best performing configurations based on your metric (default: `val_loss`).

### 2. Parameter Importance

The sweep interface shows which parameters have the most impact on performance, helping guide future experiments.

### 3. Parallel Coordinates Plot

Visualize relationships between hyperparameters and performance metrics.

## Resource Management

### Default Resource Allocation (per agent)
- **CPUs**: 12 cores (matches git-synced hyperparameter search configuration)
- **GPU**: 1 GPU
- **Memory**: 10GB per CPU (120GB total)
- **Time**: 2 hours (standard training duration)
- **Scratch**: Standard cluster allocation

**Enhanced Features vs. Standard Grid Search:**
- **Intelligent parameter exploration**: Bayesian optimization vs. exhaustive grid search
- **Consistent resource allocation**: Same cluster configuration as standard hyperparameter search
- **Environment variable management**: Proper CPU thread limiting and cluster mode detection

### Scaling Considerations

**Small Scale (2-4 agents):**
- Good for initial exploration
- Total: 24-48 CPUs, 240-480GB RAM
- Faster feedback

**Medium Scale (8-12 agents):**
- Recommended for most use cases
- Total: 96-144 CPUs, 960-1440GB RAM
- Good balance of speed and resource usage
- Default configuration

**Large Scale (16+ agents):**
- For extensive hyperparameter spaces
- Total: 192+ CPUs, 1920+ GB RAM
- Requires significant cluster resources
- Consider cluster load and fair usage policies

**Resource Planning:**
- Each agent uses 12 CPUs × 10GB = 120GB RAM
- 2-hour jobs suitable for most hyperparameter trials
- More conservative resource usage allows for larger sweeps

## Troubleshooting

### Common Issues

**1. Authentication Error**
```bash
# Login to WandB (one-time setup)
wandb login your-api-key
```

**2. Module Loading Errors**
```bash
# Check if running on correct cluster environment
module list
```

**3. Out of Memory**
```bash
# Reduce batch size or model size in sweep config
# Check GPU memory usage in job logs
```

**4. Sweep Not Starting**
```bash
# Check if WandB project exists
# Verify network connectivity from cluster nodes
```

### Debug Mode

For testing, add debug mode to your sweep:

```yaml
parameters:
  debug:
    value: true
  pretrain.epochs:
    value: 1  # Short runs for testing
```

## Best Practices

### 1. Start Small
- Begin with a small number of agents (2-4)
- Use a focused parameter space
- Validate the setup works before scaling up

### 2. Use Appropriate Search Methods
- **Bayesian (bayes)**: Best for continuous parameters, budget-conscious
- **Random**: Good baseline, works well for discrete parameters  
- **Grid**: Exhaustive but expensive, use for small spaces only

### 3. Monitor Resource Usage
- Check cluster load before launching large sweeps
- Use `squeue` to monitor your jobs
- Be mindful of fair usage policies

### 4. Organize Experiments
- Use descriptive sweep names
- Tag related sweeps
- Document significant findings

## Advanced Features

### 1. Multi-Objective Optimization

Optimize for both accuracy and model size:

```yaml
metric:
  name: custom_metric
  goal: maximize
# Then compute custom_metric = accuracy / model_parameters in your code
```

### 2. Conditional Parameters

Some parameters only apply to certain model types:

```yaml
parameters:
  model.model_type:
    values: ['BNN_NLL', 'MLP_NLL']
  
  # Only for BNN models
  training.kl_annealing.enabled:
    values: [true, false]
```

### 3. Sweep Continuation

Resume a sweep with additional agents:

```bash
python scripts/wandb_sweep_manager.py \
    --sweep-id your-existing-sweep-id \
    --agents 8
```

## Integration with Existing Workflow

The WandB sweep system integrates seamlessly with your existing codebase:

- **Feature Registry**: Fully compatible, no changes needed
- **Config System**: Sweep parameters override base config values
- **Model Types**: All existing model architectures supported
- **Data Pipeline**: No modifications required
- **Logging**: Enhanced with WandB's experiment tracking

## Support and Further Reading

- **WandB Documentation**: https://docs.wandb.ai/guides/sweeps
- **Euler Cluster Guide**: Check ETH cluster documentation
- **Project Issues**: Use GitHub issues for bug reports
- **Parameter Tuning**: See `docs/hyperparameter_guide.md` for domain-specific advice

## File Organization

- **Core Scripts**: `scripts/wandb_sweep_manager.py`, `scripts/launch_wandb_sweep.sh`
- **Configurations**: `config/wandb_sweep_config*.yaml` 
- **Generated Files**: `hp_search/logs/`, `hp_search/wandb_slurm_scripts/` (auto-created)
- **Documentation**: `docs/wandb_sweep_guide.md`