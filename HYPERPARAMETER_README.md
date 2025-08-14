# 🎯 Hyperparameter Tuning System

A clean and practical hyperparameter tuning system with cluster support for efficient parallel execution.

## ⚡ Quick Start

### Local Execution
```bash
# Quick test
python scripts/hyperparameter_search.py --grid mini

# Run locally
./hp_search/run_search.sh
```

### Cluster Execution
```bash
# Generate SLURM scripts
python scripts/hyperparameter_search.py --grid standard --cluster

# Submit to cluster
./hp_search/submit_all.sh

# Monitor
squeue -u $USER
```

## 📊 Available Grids

| Grid | Combinations | Use Case |
|------|-------------|----------|
| `mini` | 2 | Quick testing |
| `standard` | 72 | Thorough search |
| `custom` | 360 | Extensive optimization |

## 📚 Documentation

- [**Main Guide**](docs/hyperparameter_guide.md) - Complete usage guide
- [**Cluster Guide**](docs/cluster_hyperparameter_guide.md) - Detailed cluster setup and usage

## 🛠️ Features

✅ **Simple** - One script, multiple grids  
✅ **Clean** - No complex directory structures  
✅ **Scalable** - Local and cluster execution  
✅ **Configurable** - Easy to customize parameters  
✅ **Practical** - Sensible parameter ranges  

## 🎛️ Customization

Edit `scripts/hyperparameter_search.py` to add your own parameter grids:

```python
def define_grids():
    grids = {
        'my_grid': {
            'model.hidden_dim': [64, 128, 256],
            'model.num_layers': [2, 3, 4],
            'pretrain.learning_rate': [0.001, 0.01],
            # Add your parameters here
        }
    }
```

## 🔧 System Requirements

- **Local**: Python environment with PyTorch
- **Cluster**: SLURM scheduler with GPU support

## 💡 Tips

1. Start with `--grid mini` to test your setup
2. Use `--cluster` for parallel execution on cluster
3. Monitor jobs with `squeue -u $USER`
4. Results saved to individual trial directories

Ready to optimize your neural networks efficiently! 🚀
