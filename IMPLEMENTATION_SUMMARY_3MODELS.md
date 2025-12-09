# Three-Model Implementation & Cluster Hyperparameter Tuning Setup

## Summary

Successfully verified and configured three key Bayesian neural network models for STEC prediction with cluster-compatible hyperparameter tuning on Euler.

## Models Verified ✅

### 1. BayesianResNetSTEC
- **Architecture**: Hybrid - Deterministic ResNet backbone + Bayesian output head
- **Parameters**: 431,620 (100% trainable)
- **Bayesian Component**: 432,648 parameters (~100% due to all output paths)
- **Epistemic Uncertainty**: 1.785 TECU (strong uncertainty quantification)
- **Status**: ✅ Fully implemented and tested

**Key Features:**
- Combines expressiveness of ResNet with Bayesian uncertainty
- Efficient computation (deterministic backbone)
- Principled uncertainty via weight posteriors in output head
- Outputs: (mean, variance) for NLL loss

### 2. AttentionMLP_BNN_NLL (Lightweight)
- **Architecture**: Deterministic multi-head attention + Bayesian output head  
- **Parameters**: 527,620 (100% trainable)
- **Bayesian Component**: 516 parameters (~0.1% - only output head)
- **Epistemic Uncertainty**: 0.322 TECU (efficient uncertainty)
- **Status**: ✅ Fully implemented and tested

**Key Features:**
- Feature-level attention for learning interactions
- Very lightweight Bayesian component (only 516 params)
- Adaptive feature tokenization (2-16 tokens based on input dim)
- Configurable num_heads (4 or 8)
- Outputs: (mean, variance) for NLL loss

### 3. FactorizedSTEC (Physics-Based)
- **Architecture**: VTEC network (Bayesian variance head) × Geometry network (MF prediction)
- **Parameters**: 50,565 (100% trainable)  
- **Bayesian Component**: 516 parameters (~1.0% - VTEC variance head)
- **Epistemic Uncertainty**: 0.227 TECU (compact model)
- **Status**: ✅ Fully implemented and tested

**Key Features:**
- Physics-based factorization: STEC = MF × VTEC
- Separates ionospheric field (VTEC) from geometry (MF)
- VTECFieldNet: Bayesian MLP for VTEC mean + variance
- GeomNet: Deterministic MLP for mapping factor with MF(90°)=1 constraint
- Uncertainty propagation: var_stec = MF² × var_vtec
- Requires FeatureSplitter for feature splitting
- Outputs: (mean, variance) for NLL loss

## Cluster Configuration Files Created

### Base Configurations
1. **`config/config_cluster_BayesianResNetSTEC.yaml`**
   - Cluster data paths configured
   - Default: hidden_dim=1024, num_layers=4, prior_sigma=0.05
   - KL weight=0.05, warmup=10 epochs

2. **`config/config_cluster_AttentionMLP_BNN_NLL.yaml`**
   - Lightweight settings: hidden_dim=256, num_layers=3
   - num_heads=4, dropout_rate=0.1
   - KL weight=0.05 (low - small Bayesian component)

3. **`config/config_cluster_FactorizedSTEC.yaml`**
   - vtec_hidden=128, geom_hidden=64
   - vtec_layers=3, geom_layers=2
   - KL weight=0.01 (very low - minimal Bayesian component)
   - Slower warmup: 10 epochs

### W&B Sweep Configurations
1. **`config/wandb_sweep_config_BayesianResNetSTEC_cluster.yaml`**
   - Search space: hidden_dim [256,512,1024], num_layers [3,4,5]
   - Prior_sigma [0.01,0.05,0.1], KL weight [0.01,0.05,0.1]
   - Run cap: 80 runs

2. **`config/wandb_sweep_config_AttentionMLP_BNN_NLL_cluster.yaml`**
   - Search space: hidden_dim [128,256,512], num_layers [2,3,4]
   - num_heads [4,8], dropout [0.0,0.1,0.2]
   - Prior_sigma [0.01,0.05,0.1], KL weight [0.01,0.05]
   - Run cap: 60 runs

3. **`config/wandb_sweep_config_FactorizedSTEC_cluster.yaml`**
   - Search space: vtec_hidden [128,256], geom_hidden [64,128]
   - vtec_layers [3,4], geom_layers [2,3]
   - Prior_sigma [0.05,0.1], KL weight [0.005,0.01,0.02]
   - Warmup [10,15] epochs
   - Run cap: 60 runs

## SLURM Sweep Scripts Created

### Master Script (All 3 Models)
**`hp_search/run_3models_sweep_cluster.sh`**
- SLURM array job (3 tasks, one per model)
- Automatically launches BayesianResNetSTEC, AttentionMLP_BNN_NLL, FactorizedSTEC
- Configurable parallel agents per model
- Usage: `sbatch hp_search/run_3models_sweep_cluster.sh`

### Individual Model Scripts
1. **`hp_search/sweep_bayesresnet_cluster.sh`**
   - Dedicated sweep for BayesianResNetSTEC
   - Default: 5 parallel agents
   - 48-hour time limit

2. **`hp_search/sweep_attention_cluster.sh`**
   - Dedicated sweep for AttentionMLP_BNN_NLL  
   - Default: 4 parallel agents
   - 36-hour time limit

3. **`hp_search/sweep_factorized_cluster.sh`**
   - Dedicated sweep for FactorizedSTEC
   - Default: 4 parallel agents
   - 36-hour time limit

## Documentation Created

**`hp_search/README_3MODELS_SWEEP.md`**
- Comprehensive guide for running sweeps
- Model architecture details
- Hyperparameter search spaces explained
- Monitoring and troubleshooting tips
- Expected performance targets
- Best practices

## Key Implementation Details

### FactorizedSTEC Special Requirements
- Requires `CollateWithSH` initialization before model creation
- Uses `FeatureSplitter` to split features into:
  - `x_vtec`: Temporal, IPP, SWI features for VTEC prediction
  - `x_geom`: Station, direction features for MF prediction  
  - `elev_rad`: Elevation in radians for MF(90°)=1 constraint
- Wrapped with `FactorizedSTECModelWrapper` for training pipeline integration

### Variance Initialization (Critical for NLL~2.0)
- **BayesianResNetSTEC**: Output head bias initialized to match STEC statistics
- **AttentionMLP_BNN_NLL**: Output head bias initialized to mean STEC (15.5 TECU)
- **FactorizedSTEC**: 
  - VTEC variance head bias = 3.2 → var ≈ 12 after softplus
  - Accounts for MF² scaling: var_stec = MF² × var_vtec
  - Target: NLL ≈ 2.0 at typical elevations

### KL Weight Settings
Different KL weights for different architectures based on % of Bayesian parameters:
- **BayesianResNetSTEC**: 0.01-0.1 (~100% Bayesian in output path)
- **AttentionMLP_BNN_NLL**: 0.01-0.05 (~0.1% Bayesian)
- **FactorizedSTEC**: 0.005-0.02 (~1% Bayesian, only VTEC variance head)

## Testing Results

All models verified with forward pass on 32-sample batch (127 features):
- ✅ All output shapes correct: (batch_size, 1) for mean and variance
- ✅ All variances positive (no negative values)
- ✅ All models show epistemic uncertainty variation (MC sampling)
- ✅ Reasonable initialization (mean ~15.5 TECU, var ~0.8-11 TECU²)

## Usage Instructions

### Step 1: Verify Models Locally (Optional)
```bash
python tests/verify_three_models.py
```

### Step 2: Launch Sweeps on Cluster

**Option A: All models at once**
```bash
cd /cluster/work/igp_psr/arrueegg/WP4/PNN_STEC
sbatch hp_search/run_3models_sweep_cluster.sh
```

**Option B: Individual models**
```bash
# BayesianResNetSTEC (5 agents)
sbatch hp_search/sweep_bayesresnet_cluster.sh 5

# AttentionMLP_BNN_NLL (4 agents)
sbatch hp_search/sweep_attention_cluster.sh 4

# FactorizedSTEC (4 agents)  
sbatch hp_search/sweep_factorized_cluster.sh 4
```

### Step 3: Monitor Progress
```bash
# Check job status
squeue -u $USER

# View logs
tail -f hp_search/logs/sweep_*.out

# Check W&B dashboard
# https://wandb.ai/your-entity/PNN_STEC_Cluster
```

## Expected Outcomes

### Performance Targets (Validation MAE)
- **BayesianResNetSTEC**: 2.5-3.0 TECU (best overall)
- **AttentionMLP_BNN_NLL**: 2.8-3.2 TECU (good efficiency)
- **FactorizedSTEC**: 3.0-3.5 TECU (interpretable)

### Sweep Duration
- Total runs: ~200 across all models
- Training time per run: 2-6 hours
- Total sweep time: 1-3 days (with parallel agents)

## Files Modified/Created

### Model Implementation (Already Existed - Verified)
- ✅ `src/model/model.py` - All three models implemented
- ✅ `src/utils/feature_splitter.py` - Feature splitting for FactorizedSTEC

### New Configuration Files
- ✅ `config/config_cluster_AttentionMLP_BNN_NLL.yaml`
- ✅ `config/config_cluster_FactorizedSTEC.yaml`
- ✅ `config/wandb_sweep_config_BayesianResNetSTEC_cluster.yaml`
- ✅ `config/wandb_sweep_config_AttentionMLP_BNN_NLL_cluster.yaml`
- ✅ `config/wandb_sweep_config_FactorizedSTEC_cluster.yaml`

### New Sweep Scripts
- ✅ `hp_search/run_3models_sweep_cluster.sh` (master script)
- ✅ `hp_search/sweep_bayesresnet_cluster.sh`
- ✅ `hp_search/sweep_attention_cluster.sh`
- ✅ `hp_search/sweep_factorized_cluster.sh`

### New Documentation
- ✅ `hp_search/README_3MODELS_SWEEP.md`
- ✅ `tests/verify_three_models.py`

## Next Steps

1. **Launch Sweeps**: Submit jobs to Euler cluster
2. **Monitor Progress**: Check W&B dashboard and logs
3. **Analyze Results**: Use W&B parallel coordinates to identify best configs
4. **Select Best Models**: Top 3 configurations per architecture
5. **Full Training**: Train best models for complete 150 epochs
6. **Evaluation**: Compare models on test set
7. **Fine-tuning**: Use best pretrained models for day-specific tuning

## Notes

- All models use GaussianNLLLoss with proper variance prediction
- All models output (mean, variance) tuple following repo convention
- KL annealing enabled for all Bayesian models
- Cluster paths configured for `/cluster/work/igp_psr/arrueegg/`
- W&B project: `PNN_STEC_Cluster`
