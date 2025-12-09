# Hyperparameter Tuning for Three Key Models

This guide explains how to run hyperparameter sweeps on the Euler cluster for the three key models:
1. **BayesianResNetSTEC**: Hybrid architecture with deterministic ResNet backbone + Bayesian output head
2. **AttentionMLP_BNN_NLL**: Lightweight attention-based model with Bayesian head
3. **FactorizedSTEC**: Physics-based STEC = MF × VTEC factorization model

## Model Overview

### BayesianResNetSTEC
- **Architecture**: Deterministic ResNet backbone (4 residual blocks) + Bayesian output layer
- **Parameters**: ~400-500K (depending on hidden_dim)
- **Bayesian Component**: ~5% of parameters (output head only)
- **Strengths**: Strong performance with efficient epistemic uncertainty quantification
- **Hyperparameters**: hidden_dim, num_layers, prior_sigma, KL weight

### AttentionMLP_BNN_NLL  
- **Architecture**: Deterministic multi-head attention backbone + Bayesian output layer
- **Parameters**: ~150-500K (lightweight design)
- **Bayesian Component**: ~0.1% of parameters (output head only)
- **Strengths**: Feature interactions via attention, computationally efficient
- **Hyperparameters**: hidden_dim, num_layers, num_heads, prior_sigma, KL weight

### FactorizedSTEC
- **Architecture**: VTEC network (with Bayesian variance head) × Geometry network (MF prediction)
- **Parameters**: ~50-200K (very compact)
- **Bayesian Component**: ~1% of parameters (VTEC variance head only)
- **Strengths**: Physically interpretable, separates ionospheric field from geometry
- **Hyperparameters**: vtec_hidden, geom_hidden, vtec_layers, geom_layers, prior_sigma, KL weight

## Quick Start

### Option 1: Run All Three Models Simultaneously
```bash
cd /cluster/work/igp_psr/arrueegg/WP4/PNN_STEC
sbatch hp_search/run_3models_sweep_cluster.sh
```
This launches 3 separate SLURM array jobs (one per model) with multiple agents each.

### Option 2: Run Individual Model Sweeps

#### BayesianResNetSTEC
```bash
sbatch hp_search/sweep_bayesresnet_cluster.sh 5  # Launch with 5 parallel agents
```

#### AttentionMLP_BNN_NLL
```bash
sbatch hp_search/sweep_attention_cluster.sh 4   # Launch with 4 parallel agents
```

#### FactorizedSTEC
```bash
sbatch hp_search/sweep_factorized_cluster.sh 4  # Launch with 4 parallel agents
```

## Configuration Files

### Base Configurations (Cluster-Compatible)
- `config/config_cluster_BayesianResNetSTEC.yaml`
- `config/config_cluster_AttentionMLP_BNN_NLL.yaml`
- `config/config_cluster_FactorizedSTEC.yaml`

These configs include:
- Cluster-specific data paths (`/cluster/work/igp_psr/arrueegg/...`)
- Optimized batch sizes and worker counts
- Reasonable default hyperparameters
- W&B integration enabled

### Sweep Configurations (W&B)
- `config/wandb_sweep_config_BayesianResNetSTEC_cluster.yaml`
- `config/wandb_sweep_config_AttentionMLP_BNN_NLL_cluster.yaml`
- `config/wandb_sweep_config_FactorizedSTEC_cluster.yaml`

These define:
- Search space for each hyperparameter
- Optimization method (Bayesian optimization)
- Metric to optimize (val_MAE)
- Run caps (60-80 runs per model)

## Hyperparameter Search Spaces

### BayesianResNetSTEC
| Parameter | Values | Notes |
|-----------|--------|-------|
| hidden_dim | [256, 512, 1024] | Wider networks for capacity |
| num_layers | [3, 4, 5] | Number of residual blocks |
| prior_sigma | [0.01, 0.05, 0.1] | Bayesian prior uncertainty |
| learning_rate | [1e-4, 5e-4, 1e-3] | Adam optimizer |
| loss_weight (KL) | [0.01, 0.05, 0.1] | Balance data fit vs prior |
| SH_degree | [0, 5] | Spherical harmonic embeddings |

### AttentionMLP_BNN_NLL
| Parameter | Values | Notes |
|-----------|--------|-------|
| hidden_dim | [128, 256, 512] | Smaller for efficiency |
| num_layers | [2, 3, 4] | Fewer layers than MLP |
| num_heads | [4, 8] | Multi-head attention |
| dropout_rate | [0.0, 0.1, 0.2] | Regularization |
| prior_sigma | [0.01, 0.05, 0.1] | Bayesian prior |
| loss_weight (KL) | [0.01, 0.05] | Lower - small Bayesian component |

### FactorizedSTEC
| Parameter | Values | Notes |
|-----------|--------|-------|
| vtec_hidden | [128, 256] | VTEC network width |
| vtec_layers | [3, 4] | VTEC network depth |
| geom_hidden | [64, 128] | Geometry network width |
| geom_layers | [2, 3] | Geometry network depth |
| prior_sigma | [0.05, 0.1] | For VTEC variance head |
| loss_weight (KL) | [0.005, 0.01, 0.02] | Very low - minimal Bayesian |
| warmup_epochs | [10, 15] | Longer warmup for physics model |

## Monitoring Sweeps

### Check Sweep Status
```bash
# View logs
tail -f hp_search/logs/sweep_*.out

# Check running jobs
squeue -u $USER

# View specific model sweep ID
cat hp_search/logs/BayesianResNetSTEC_sweep_id.txt
```

### W&B Dashboard
All sweeps report to the W&B project: **PNN_STEC_Cluster**

View progress at: https://wandb.ai/your-entity/PNN_STEC_Cluster

### Resume/Add Agents to Existing Sweep
```bash
# Get sweep ID from logs
SWEEP_ID=$(cat hp_search/logs/FactorizedSTEC_sweep_id.txt)

# Launch additional agents
wandb agent $SWEEP_ID
```

## Expected Results

### Training Time Per Run
- **BayesianResNetSTEC**: ~4-6 hours (larger model)
- **AttentionMLP_BNN_NLL**: ~2-4 hours (medium)
- **FactorizedSTEC**: ~2-3 hours (compact)

Times assume:
- 500K training samples per epoch
- 1M validation samples
- 150 epochs max with early stopping (patience=20)
- 1 GPU (RTX 3090 or similar)

### Target Performance (Validation MAE)
Based on preliminary tests:
- **BayesianResNetSTEC**: 2.5-3.0 TECU (best performer)
- **AttentionMLP_BNN_NLL**: 2.8-3.2 TECU (good efficiency/performance tradeoff)
- **FactorizedSTEC**: 3.0-3.5 TECU (interpretable, physics-based)

## Best Practices

1. **Start Small**: Test with 1-2 agents first to verify configuration
2. **Monitor Resources**: Check GPU utilization with `nvidia-smi`
3. **Check Logs Regularly**: Ensure no errors in sweep execution
4. **Budget Time**: Each sweep may take 1-3 days depending on agents
5. **Use Bayesian Optimization**: More efficient than grid/random search

## Troubleshooting

### Sweep Fails to Initialize
```bash
# Check W&B login
wandb login

# Verify config files exist
ls -l config/wandb_sweep_config_*_cluster.yaml
```

### Out of Memory Errors
Reduce batch size in base config:
```yaml
pretrain:
  batchsize: 512  # Reduce from 1024
```

### Slow Training
- Verify data moved to scratch: `data.move_to_scratch: True`
- Check GPU utilization: `nvidia-smi` should show >80%
- Reduce `val_size` for faster validation: `1_000_000 → 500_000`

## Next Steps After Sweep

1. **Analyze Results**: Use W&B parallel coordinates plot to identify best hyperparameters
2. **Best Model Selection**: Pick top 3 configurations per model based on val_MAE
3. **Full Training**: Train best configs for full 150 epochs without early stopping
4. **Ensemble**: Consider ensembling top-3 models for each architecture
5. **Fine-tuning**: Use best pretrained models for day-specific fine-tuning

## Contact

For issues or questions:
- Check existing logs in `hp_search/logs/`
- Review model verification: `tests/verify_three_models.py`
- See main documentation: `README.md`, `USAGE_GUIDE.md`
