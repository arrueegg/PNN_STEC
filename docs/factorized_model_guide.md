# Factorized VTEC × MF Model Guide

## Overview

The `FactorizedSTEC` model separates STEC prediction into two physically meaningful components:

**STEC = MF × VTEC**

Where:
- **VTEC** (Vertical TEC): Ionospheric electron content in the vertical direction (field property)
- **MF** (Mapping Factor): Geometric scaling factor converting VTEC to STEC along the slant path

This factorization provides:
1. **Physical interpretability**: Separate ionospheric field from geometric effects
2. **Targeted fine-tuning**: Adapt only VTEC or MF during daily fine-tuning
3. **Uncertainty propagation**: Uncertainty from VTEC propagates to STEC via: σ_stec = |MF| × σ_vtec

## Architecture

### VTECFieldNet
Predicts VTEC and its uncertainty at the Ionospheric Pierce Point (IPP).

**Inputs** (VTEC field features):
- IPP geographic coordinates (lat, lon)
- IPP solar-magnetic coordinates (sm_lat, sm_lon)  
- Temporal features (year, day-of-year, time-of-day, local solar time)
- Space Weather Indices (F10.7, Kp, Dst, AE, ap, sunspot number)
- Spherical harmonic embeddings of IPP location

**Outputs**:
- `vtec_mean`: Mean VTEC prediction at IPP [TECU]
- `vtec_log_sigma`: Log of VTEC standard deviation (aleatoric uncertainty)

**Architecture**:
- Multi-layer MLP backbone (default: 3 layers × 128 hidden units)
- Dual output heads: one for mean, one for log(σ)
- ReLU or Tanh activation (Tanh may produce smoother spatial fields)

### GeomNet
Predicts the mapping factor from geometry features.

**Inputs** (geometry features):
- Station geographic coordinates (lat, lon)
- Station solar-magnetic coordinates (sm_lat, sm_lon)
- Direction features (elevation, azimuth as Cartesian unit vector)
- Spherical harmonic embeddings of station location

**Outputs**:
- `mf`: Mapping factor satisfying physical constraints

**Physical Constraints**:
The MF is computed to guarantee:
1. **MF(90°) = 1**: Vertical rays (zenith) have no path elongation
2. **MF ≥ 1**: Slant path is always ≥ vertical path
3. **MF increases as elevation decreases**: Lower elevation → longer slant path

Implementation:
```python
g(elev) = 1 - sin(elev)  # 0 at 90°, ~1 at 0°
MF = 1 + g(elev) * softplus(mf_raw)
```

**Architecture**:
- Multi-layer MLP backbone (default: 2 layers × 64 hidden units)
- Single output head for MF
- ReLU or Tanh activation

### STEC Computation and Uncertainty Propagation

Given VTEC and MF predictions:

```python
σ_v = exp(vtec_log_sigma)
μ_stec = MF × vtec_mean
σ_stec = |MF| × σ_v
```

The model returns `(μ_stec, σ_stec²)` as a tuple following the repo convention.

## Configuration

### Basic Configuration

```yaml
model:
  model_type: FactorizedSTEC
  
  # Activation function for both networks
  activation: "relu"  # or "tanh" for smoother fields
  
  # VTECFieldNet configuration
  vtec_hidden: 128    # Hidden dimension
  vtec_layers: 3      # Number of layers
  
  # GeomNet configuration  
  geom_hidden: 64     # Hidden dimension
  geom_layers: 2      # Number of layers
```

### Training Configuration

```yaml
training:
  loss_function: "GaussianNLLLoss"  # Required for uncertainty
  
  # No KL annealing needed (no Bayesian layers by default)
  kl_annealing:
    enabled: false
    start_weight: 0.0
    end_weight: 0.0
  
  # Recommended: weight high STEC values more
  target_weighting:
    enabled: true
    weight_function: "linear"
```

### Feature Control

**Required features** for `FactorizedSTEC`:
- Direction features: `satazi: true`, `satele: true` (needed for MF)
- Station features: All station lat/lon features (needed for MF)
- IPP features: All IPP lat/lon features (needed for VTEC)

**Recommended**:
- `use_SWI: True`: Space weather helps VTEC prediction
- `SH_degree: 5`: Spatial encoding improves both VTEC and MF

## Training Workflow

### 1. Pretraining

Train both VTEC and MF networks on ~15 years of data:

```yaml
mode: pretrain
year: 2024
doy: 122  # Can train on single day or multiple days

training:
  pretrain:
    num_epochs: 50
    batch_size: 1024
    lr: 1e-3
    lr_scheduler: "ReduceLROnPlateau"
```

Run:
```bash
python src/main.py
```

This creates a pretrained model capturing ionospheric climatology and geometric mapping.

### 2. Daily Fine-tuning

Fine-tune the pretrained model on a specific day with selective freezing.

**Option A: Freeze MF, adapt VTEC** (RECOMMENDED for daily fine-tuning)
```yaml
mode: finetune
pretrain_folder: "experiments/Pretrain_STEC_FactorizedSTEC_..."

finetune:
  freeze_vtec_net: false  # Adapt to daily ionospheric conditions
  freeze_geom_net: true   # Keep geometric model fixed (deterministic physics)
  
  num_epochs: 20
  batch_size: 512
  lr: 1e-4
```

**Rationale:** 
- MF depends on elevation (geometry) → deterministic, doesn't change day-to-day
- VTEC depends on ionospheric conditions → varies with space weather, local time, storms
- Therefore: freeze geometry, adapt ionosphere

**Option B: Freeze VTEC, adapt MF** (for station-specific calibration)
```yaml
finetune:
  freeze_vtec_net: true   # Keep ionospheric climatology
  freeze_geom_net: false  # Adapt MF for new stations/receivers
```

**Use case:** When deploying to new receiver stations not seen during pretraining,
you can keep the learned VTEC field and only calibrate the geometric model.

**Option C: Fine-tune everything**
```yaml
finetune:
  freeze_vtec_net: false
  freeze_geom_net: false
```

## Feature Splitting

The `FeatureSplitter` automatically divides collated features into VTEC and geometry components:

### VTEC Features
- Temporal: year, doy, sod, local_time_hours (transformed to sin/cos/norm)
- IPP location: lat_ipp, lon_ipp, sm_lat_ipp, sm_lon_ipp
- IPP SH embeddings: sh_ipp_geo, sh_ipp_sm
- Space Weather: Kp, F10.7, Dst, AE, ap, sunspot number

### Geometry Features  
- Station location: lat_sta, lon_sta, sm_lat_sta, sm_lon_sta
- Direction: elevation and azimuth (as Cartesian unit vector e_up, e_east, e_north)
- Station SH embeddings: sh_sta_geo, sh_sta_sm

### Elevation Extraction
Elevation in radians is extracted from the direction Cartesian vector:
```python
e_up = sin(elevation)  # From collation
elevation_rad = arcsin(e_up)  # Recovered in splitter
```

## Accessing Detailed Outputs

For analysis and visualization, use `forward_detailed`:

```python
from utils.feature_splitter import FeatureSplitter

# Get splitter from model wrapper
splitter = model.splitter

# Split features
x_vtec, x_geom, elev_rad = splitter.split_features(features)

# Get detailed outputs
outputs = model.forward_detailed(features)

# Access components
vtec_mean = outputs["vtec_mean"]        # VTEC prediction
vtec_uncertainty = outputs["sigma_v"]   # VTEC uncertainty
mf = outputs["mf"]                      # Mapping factor
stec_mean = outputs["mu_stec"]          # STEC prediction
stec_uncertainty = outputs["sigma_stec"] # STEC uncertainty
```

## Inference and Evaluation

The factorized model integrates seamlessly with existing inference scripts:

```bash
# Test set evaluation
python src/inference_testset.py

# Positioning evaluation
bash positioning/scripts/run_pipeline.sh "experiments/Pretrain_STEC_FactorizedSTEC_..." 2024-06-01

# Map visualization
python src/inference_map.py
```

All scripts work transparently - the model wrapper handles feature splitting internally.

## Hyperparameter Tuning

Key hyperparameters for WandB sweeps:

```yaml
parameters:
  # VTEC network
  vtec_hidden:
    values: [64, 128, 256]
  vtec_layers:
    values: [2, 3, 4]
  
  # Geometry network
  geom_hidden:
    values: [32, 64, 128]
  geom_layers:
    values: [1, 2, 3]
  
  # Activation
  activation:
    values: ["relu", "tanh"]
  
  # Training
  learning_rate:
    min: 1e-4
    max: 1e-2
  target_weighting.enabled:
    values: [true, false]
```

## Advanced: Bayesian Extensions

To add epistemic uncertainty to VTEC predictions, you can:

1. Replace VTECFieldNet backbone with MC Dropout layers
2. Use Bayesian linear layers (torchbnn.BayesLinear) in VTECFieldNet
3. Run multiple stochastic forward passes to sample VTEC distributions

Keep GeomNet deterministic - uncertainty should come from VTEC field, not geometry.

## Troubleshooting

### Error: "Elevation component (e_up) not found"
**Cause**: Direction features (satazi, satele) are disabled.  
**Solution**: Set `satazi: true` and `satele: true` in feature_control.

### Error: "Feature registry not found"
**Cause**: Feature registry not initialized before model creation.  
**Solution**: Ensure `initialize_feature_registry(config)` is called in main.py.

### Warning: "Both VTECNet and GeomNet are frozen"
**Cause**: Both freeze flags are True during fine-tuning.  
**Solution**: Set at least one of freeze_vtec_net or freeze_geom_net to false.

### Poor high STEC predictions
**Solution**: Enable target weighting:
```yaml
target_weighting:
  enabled: true
  weight_function: "linear"  # or "quadratic" for stronger weighting
```

### MF values far from 1 at high elevation
**Check**: Verify elevation is in radians and MF constraint is applied correctly.  
**Debug**: Use forward_detailed to inspect MF values across elevations.

## Example Complete Workflow

```bash
# 1. Pretrain on climatology
python src/main.py  # with config_FactorizedSTEC.yaml, mode: pretrain

# 2. Fine-tune for specific day (freeze VTEC, adapt MF)
# Edit config: mode: finetune, freeze_vtec_net: true
python src/main.py

# 3. Evaluate on test set
python src/inference_testset.py

# 4. Positioning evaluation
bash positioning/scripts/run_pipeline.sh "experiments/Finetune_STEC_FactorizedSTEC_..." 2024-06-01

# 5. Analyze detailed outputs (in Python)
from model.model import get_model
model = get_model(config)
outputs = model.forward_detailed(features)
# Plot VTEC field, MF vs elevation, uncertainties, etc.
```

## References

For implementation details, see:
- `src/model/model.py`: VTECFieldNet, GeomNet, FactorizedSTECModel classes
- `src/utils/feature_splitter.py`: Feature splitting logic
- `src/utils/model_utils.py`: Selective freezing for fine-tuning
- `config/config_FactorizedSTEC.yaml`: Example configuration
