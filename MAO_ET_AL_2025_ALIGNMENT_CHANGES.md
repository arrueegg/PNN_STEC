# Alignment Changes: Current VTEC Models → Mao et al., 2025 Specification

## Executive Summary

Your current daily VTEC models are close in spirit but differ significantly in technical details. The table below shows the critical misalignments:

| Aspect | Your Current Implementation | Mao et al., 2025 Spec | Status |
|--------|----------------------------|----------------------|--------|
| **Loss Function** | MSELoss | Laplacian NLL | ❌ MAJOR |
| **Network Output** | Single VTEC value | 2 scalars: μ (VTEC) + d (scale) | ❌ MAJOR |
| **Hidden Activation** | ReLU | tanh | ❌ Partial |
| **Output Activation** | Linear/no clip | Softplus on uncertainty term | ❌ Partial |
| **Spatial Encoding** | Generic lat/lon | Spherical harmonics deg/order 15 | ❌ MAJOR |
| **Spatial Frame** | Geographic lat/lon | Magnetic +sun-fixed lon | ❌ MAJOR |
| **Spherical Harmonics Degree** | 5 | 15 | ❌ Critical |
| **Total Features** | ~40–50 (incl. SWI) | 259 (256 SH + 3 temporal) | ❌ MAJOR |
| **Space Weather Indices (SWI)** | Included (F10.7, Dst, Kp, etc.) | **NOT included** | ❌ CRITICAL |
| **Time Features** | Multiple (sod, doy) | Only 3: SODnorm, sin(SOD), cos(SOD) | ⚠️ Partial |
| **Architecture** | MLP h512 L4 | MLP h90 L3 | ⚠️ Size mismatch |
| **Ensemble Strategy** | Single model per day | 10 independent models per day | ❌ MAJOR |
| **Uncertainty Quantification** | Variance = 0 (MSE only) | Per-model variance + ensemble spread | ❌ MAJOR |
| **Optimizer** | Adam with weight decay | Adam (basic; no details in paper) | ⚠️ Weight decay unclear |
| **Early Stopping** | ReduceLROnPlateau | Not specified (choose reasonable default) | ⚠️ OK for now |

---

## Detailed Change Breakdown

### 1. **Loss Function: MSELoss → Laplacian NLL**

**Current:**
```yaml
training:
  loss_function: MSELoss
  optimizer: Adam
  weight_decay: 0.0001
```

**Required Change:**
The loss must switch from MSELoss (assumes Gaussian errors) to **Laplacian Negative Log-Likelihood**—more robust to outliers and aligns with the paper's uncertainty model.

**Laplacian NLL Loss:**
```
ℓ_NLL = log(d(X_i)) + |Y_i - μ(X_i)| / d(X_i)
```

Where:
- `μ(X_i)`: predicted VTEC (network output #1, unsaturated)
- `d(X_i)`: predicted Laplacian scale parameter, related to uncertainty
- `Y_i`: observed VTEC target

**Variance from Laplacian NLL:**
```
σ²(X_i) = 2 * d²(X_i)   (relates Laplacian scale to variance)
```

**TODO:**
- [ ] Implement `LaplacianNLLLoss` in `src/training/losses.py` (or similar)
- [ ] Ensure `d(X_i)` is positive after softplus: `d = softplus(d_raw) + eps` where `eps ≈ 1e-3`
- [ ] Update config YAML to use new loss function
- [ ] Remove MSELoss references; validate gradient flow

---

### 2. **Model Output Architecture: 1-scalar → 2-scalar**

**Current:**
```python
# MLP class returns:
return x, torch.zeros_like(x)  # Single prediction + dummy zero variance
```

**Required Change:**
Create a new model variant `MLP_LaplacianNLL` that outputs **two scalars per sample:**
1. `μ_raw`: raw VTEC mean (linear output, unbounded)
2. `d_raw`: log-scale parameter (to be softplus'd)

The network's final layer should have `output_size=2`:
```python
class MLP_LaplacianNLL(torch.nn.Module):
    def __init__(self, n_in=259, n_out=2, hidden_dim=90, num_layers=3, activation='tanh'):
        super().__init__()
        # Build network...
        self.output_layer = Linear(hidden_dim, n_out)  # outputs [μ_raw, d_raw]
        
    def forward(self, x):
        for layer in self.layers:
            x = activation_fn(layer(x))  # tanh for hidden
        outputs = self.output_layer(x)  # shape: (batch_size, 2)
        
        mu = outputs[:, 0:1]  # VTEC prediction
        d_raw = outputs[:, 1:2]  # log-scale (to be softplus'd)
        
        d = F.softplus(d_raw) + 1e-3  # Ensure positive scale
        
        # Convert Laplacian scale to variance for tracking
        variance = 2.0 * d ** 2
        
        return mu, d, variance
```

**TODO:**
- [ ] Create `MLP_LaplacianNLL` class in `src/model/model.py`
- [ ] Ensure output layer bias initialization (μ output bias ≈ 15.5, d output bias ≈ 0)
- [ ] Modify loss function call to accept both `mu` and `d` predictions
- [ ] Update config to specify `model_type: MLP_LaplacianNLL` or similar

---

### 3. **Network Architecture: h512 L4 → h90 L3 + tanh activation**

**Current:**
```yaml
model:
  hidden_dim: 512
  num_layers: 4
  # (implicit activation: ReLU)
```

**Mao et al. Selection (from tuning):**
```yaml
model:
  hidden_dim: 90
  num_layers: 3
  activation: tanh  # NOT relu
```

**Notes:**
- The paper tuned over `L ∈ {1,2,3}` and `N ∈ {10,20,...,120}` and selected L=3, N=90.
- Your current (h512, L4) is likely over-parameterized for a daily interpolator; may overfit.
- Use **tanh** in hidden layers, **softplus** on the scale output `d`.

**TODO:**
- [ ] Update config: `hidden_dim: 90`, `num_layers: 3`
- [ ] Change activation in `MLP_LaplacianNLL` from ReLU to tanh
- [ ] Consider re-tuning on a subset if drift is large (but paper's choice is principled)

---

### 4. **Spatial Encoding: Generic lat/lon → Spherical Harmonics (deg/order 15)**

**Current State:**
Your config has:
```yaml
data:
  SH_degree: 5  # Spherical harmonic degree
```

And your features include raw `lat_ipp`, `lon_ipp`, etc.

**Mao et al. Requirement:**
- **Not** raw geographic lat/lon
- **Spherical harmonics expansion** up to degree/order **15** (not 5)
- **Coordinate transformation before SH expansion:**
  1. Use **magnetic latitude** instead of geographic latitude
  2. Use **sun-fixed longitude** instead of geographic longitude
- **Output:** 256 spatial features (from SH basis)

**Why this matters:**
- SH basis (degree 15) gives $(L+1)^2 = 256$ features (order 0 to 14)
- Magnetic coordinates align with ionospheric symmetry (EIA, auroral zones)
- Sun-fixed frame reduces day/night oscillations due to Earth rotation

### 4.1 **Magnetic Latitude Transformation**

Use the IGRF (International Geomagnetic Reference Field) or similar dipole model:

```python
def geographic_to_magnetic_coords(lat_geo, lon_geo, year, doy):
    """
    Convert geographic coordinates to magnetic latitude.
    
    Recommended library: pyIGRF (or equivalent)
    from pyigrf import get_dipole_tilt, coord_transform
    
    Args:
        lat_geo: Geographic latitude [degrees]
        lon_geo: Geographic longitude [degrees]
        year, doy: Date for IGRF model
        
    Returns:
        mag_lat: Magnetic latitude [degrees]
        mag_lon: Magnetic longitude [degrees]
    """
    # Use IGRF or simple dipole approximation
    # For daily models, a single fixed dipole tilt per day may suffice
    # (or approximate dipole tilt from Dst/other solar index—but paper doesn't use SWI!)
    pass
```

### 4.2 **Sun-Fixed Longitude Transformation**

The paper uses "sun-fixed" (or "solar-fixed") longitude to reduce side-effects of Earth rotation:

```python
def geographic_to_sunfixed_coords(lon_geo, year, doy, sod):
    """
    Convert geographic longitude to sun-fixed longitude.
    
    In sun-fixed frame, 0° is always at subsolar point.
    Common approach: subtract solar hour angle from geographic longitude.
    
    Args:
        lon_geo: Geographic longitude [degrees, -180 to 180]
        year, doy: Date
        sod: Seconds of day [0, 86400)
        
    Returns:
        lon_sf: Sun-fixed longitude [degrees]
    """
    # Compute solar hour angle (Greenwich solar time offset)
    # lon_sf = lon_geo - solar_hour_angle
    # where solar_hour_angle depends on year/doy/sod
    # Common formula uses equation of time and Greenwich hour angle
    
    # Simplified:
    # 1. Compute Greenwich hour angle (GHA) for the given time
    # 2. Subtract from lon_geo
    # Reference: Vallado et al., Fundamentals of Astrodynamics and Applications
    pass
```

### 4.3 **Spherical Harmonics Expansion**

```python
def spherical_harmonics_embedding(mag_lat, lon_sf, degree=15):
    """
    Compute spherical harmonics basis up to degree/order `degree`.
    
    Output: (degree+1)² features = (15+1)² = 256 features
    
    Args:
        mag_lat: Magnetic latitude in radians
        lon_sf: Sun-fixed longitude in radians
        degree: Max degree/order (15 for Mao et al.)
        
    Returns:
        features: Array of shape (n_samples, 256) or (256,)
    """
    # Use scipy.special.spherical_jn, legendre, or similar
    # Or pre-computed library like pyshtools
    
    # SH basis: Y_{l,m}(lat, lon) for l=0..15, m=0..l
    # Real-valued spherical harmonics (not complex)
    
    # For efficiency on large batch, use vectorized computation
    pass
```

**TODO:**
- [ ] Implement `geographic_to_magnetic_coords()` (use pyIGRF or dipole model)
- [ ] Implement `geographic_to_sunfixed_coords()` (solar hour angle correction)
- [ ] Implement `spherical_harmonics_embedding()` up to degree 15 (256 features)
- [ ] Integrate into data loading / feature engineering pipeline
- [ ] Update config: `SH_degree: 15` (from 5)
- [ ] Verify time-consistency: magnetic/solar transforms must use same sod, year, doy as features

---

### 5. **Temporal Features: Complex → Simple (3 features only)**

**Current:**
Your config includes many temporal features: sod, doy, local_time_hours, year, etc.

**Mao et al. Usage:**
**Only 3 temporal features:**
1. Normalized seconds of day (SOD_norm)
2. sin(2π · SOD / 86400)
3. cos(2π · SOD / 86400)

**Definition:**
```
SOD_norm = 2 · SOD / 86400 - 1     (maps [0, 86400] → [-1, 1])
SOD_sin = sin(2π · SOD / 86400)    (periodic, period = 1 day)
SOD_cos = cos(2π · SOD / 86400)    (periodic, period = 1 day)
```

**Why NOT DOY, year, local_time, etc.:**
- Daily training: model is fitted per calendar day, so all samples share the same DOY and year.
- Local time: subsumed by longitude (sun-fixed) + SOD (universal time).
- Storm indices (Kp, Dst, Ap, F10.7): paper explicitly does **not** include them for daily models (storms observed in that day's data).

**TODO:**
- [ ] Remove from feature engineering: DOY, year, local_time_hours, all SWI indices
- [ ] Keep only: SOD_norm, SOD_sin, SOD_cos
- [ ] Update feature registry / config to disable SWI: `use_SWI: false`
- [ ] Verify resulting feature count: 256 (SH) + 3 (temporal) = **259 features**

---

### 6. **Space Weather Indices (SWI): REMOVE**

**Current:**
```yaml
feature_control:
  AE-index,_nT: true
  Dst-index,_nT: true
  Kp_index: true
  R_Sunspot_No: true
  ap_index,_nT: true
  f107_index: true
  ...

data:
  use_SWI: true
```

**Mao et al.:**
```
EXPLICITLY NOT INCLUDED
```

**Rationale from Paper:**
> "They did not include space-weather indices like F10.7 and Dst as inputs, because training is daily and storm-time effects are already 'in' the observations of that day."

**Why This Matters:**
- Including SWI artificially decouples model inputs from observations → potential data leakage or inconsistency.
- Daily model: all inputs (GNSS VTEC, coordinates, time) are from one calendar day; adding external SWI adds exogenous information not directly observed in GNSS data.
- Paper chose to rely solely on the day's observational snapshot + spatial/temporal structure.

**TODO:**
- [ ] Set `use_SWI: false` in config
- [ ] Remove all `feature_control` entries for SWI: Dst, Kp, F10.7, Ap, AE, etc.
- [ ] Update data loading to skip SWI and only ingest: coordinates, time, target
- [ ] Recalculate feature count: should drop from ~40–50 to exactly 259

---

### 7. **Elevation Cutoff: Verify 15°**

**Current Config:**
```yaml
data:
  min_elevation: 5.0
  max_elevation: 90.0
```

**Mao et al.:**
```
min_elevation: 15.0°
```

**Reason:**
Reduces low-elevation mapping-function issues and measurement errors.

**TODO:**
- [ ] Update config: `min_elevation: 15.0`

---

### 8. **Deep Ensemble: Single Model → 10 Independent Models per Day**

**Current:**
```
Trains one MLP per day.
```

**Mao et al.**
```
Trains M = 10 independent NNs per day, differing by:
  1. Random weight initialization
  2. Shuffling of training data order (different batch permutations)
```

**Final Prediction:**
```
Mean prediction:
  μ_e(X_i) = (1/M) Σ μ_m(X_i)    for m=1..10

Ensemble variance (combines per-model uncertainty + disagreement):
  σ_e²(X_i) = (1/M) Σ 2·d_m²(X_i) + (1/M) Σ μ_m² - μ_e²
```

**Implementation:**
```python
def train_ensemble(config, daily_data, M=10):
    """
    Train M independent models for one calendar day.
    
    Args:
        config: Configuration dict
        daily_data: Data for one day (train/val/test splits already done)
        M: Number of ensemble members (default 10)
        
    Yields:
        models: List of M trained model checkpoints
    """
    ensemble = []
    for i in range(M):
        # Unique seed per ensemble member
        seed = config['random_seed'] + i
        setup_seed(seed)
        
        # Create new model with random initialization
        model = MLP_LaplacianNLL(...)
        
        # Optional: shuffle data differently each iteration
        # (PyTorch DataLoader with shuffle=True, different worker seeds)
        
        # Train model
        model = train_one_model(config, daily_data, model, seed=seed)
        
        ensemble.append(model)
    
    return ensemble

def ensemble_forward(models, X):
    """
    Forward pass through all ensemble members.
    
    Args:
        models: List of M trained models
        X: Input batch (n_samples, 259)
        
    Returns:
        mu_e: Ensemble mean prediction (n_samples, 1)
        sig2_e: Ensemble variance (n_samples, 1)
    """
    mu_list = []
    d_list = []
    
    for model in models:
        mu, d, _ = model(X)  # Each returns (μ, d, σ²)
        mu_list.append(mu)
        d_list.append(d)
    
    # Stack: (M, n_samples, 1)
    mu_stack = torch.cat(mu_list, dim=1)  # (n_samples, M)
    d_stack = torch.cat(d_list, dim=1)    # (n_samples, M)
    
    # Ensemble mean
    mu_e = mu_stack.mean(dim=1, keepdim=True)
    
    # Ensemble variance
    var_per_model = 2.0 * (d_stack ** 2)  # (n_samples, M)
    var_aleatoric = var_per_model.mean(dim=1, keepdim=True)  # Avg per-model var
    var_epistemic = (mu_stack ** 2).mean(dim=1, keepdim=True) - mu_e ** 2  # Ensemble spread
    
    sig2_e = var_aleatoric + var_epistemic
    
    return mu_e, sig2_e
```

**TODO:**
- [ ] Modify training loop to train M=10 separate models per day
- [ ] Each model: unique seed, possibly shuffled data
- [ ] Implement `ensemble_forward()` for prediction
- [ ] Store all 10 checkpoints (or save ensemble metadata)
- [ ] Update inference to compute ensemble mean + variance
- [ ] Update config (or script) to specify `num_ensemble_members: 10`

---

### 9. **Optimizer & Scheduler: Clarify Defaults**

**Current:**
```yaml
training:
  optimizer: Adam
  weight_decay: 0.0001
  learning_rate: 0.0001

finetune:
  scheduler: ReduceLROnPlateau
  scheduler_step_size: 100
```

**Mao et al.:**
```
Not fully specified beyond "Adam optimizer"
```

**Recommendation:**
The paper likely used:
- **Optimizer:** Adam / AdamW (your choice)
- **Learning rate:** Research standard, e.g., 1e-3 or 1e-4 (tuned with CV)
- **Early stopping:** On validation metric (MAE, RMSE, or NLL)
- **Weight decay:** Probably no weight decay (or very light), not specified
- **Scheduler:** Possibly ReduceLROnPlateau or StepLR

**For alignment:**
- Keep Adam optimizer
- Disable weight decay (set to 0) unless you justify it
- Use early stopping on validation MAE or NLL
- Document your choice (not specified in paper, so your defaults are acceptable)

**TODO:**
- [ ] Verify optimizer is Adam
- [ ] Set `weight_decay: 0.0` (or very small, e.g., 1e-6)
- [ ] Confirm early stopping is based on validation metric (not just epoch count)
- [ ] Document in comments that paper doesn't specify these details

---

### 10. **Train/Test Split: Verify Alignment with Paper**

**Current (from your experiments folder naming):**
```
Finetune_VTEC_2024_291_...
```
Appears to be for DOY 291 (Oct 18), year 2024.

**Mao et al. Paper:**
```
Train stations: 348
Test stations: 52 (each with ≥ 80% data availability)
(For replication: use this split or document your own choice)
```

**TODO:**
- [ ] If you have a custom split, document it
- [ ] If replicating, use paper's split: 348 train / 52 test
- [ ] Ensure stations meet ≥ 80% data availability for test set

---

### 11. **VTEC Targets: Verify Generation Matches Specification**

**Current:**
Your data pipeline loads GNSS-derived VTEC from HDF5 files.

**Mao et al. Specification:**
```
1. Use GPS L1/L2 and Galileo E1/E5a
2. Form geometry-free (GF) combinations
3. Resolve GF ambiguities using pseudorange (for CCL extraction)
4. Extract STEC from dual-frequency code/carrier
5. Map to VTEC at IPP using SLM (single-layer model, H_ion = 450 km)
6. Use CAS satellite DCBs
7. Estimate receiver DCBs (not always available from catalogs)
8. Apply 15° elevation cutoff
9. Sampling: 30 s
```

**Your Data Source:**
```yaml
data:
  GNSS_data_path: /home/space/data/iono/STEC_DB_CASDCB
```

**Question to Verify:**
- [ ] Are your VTEC targets computed as above (CCL+SLM, 450 km shell)?
- [ ] Are satellite DCBs from CAS products?
- [ ] Are receiver DCBs estimated or from a catalog?
- [ ] Is 15° elevation cutoff applied in your h5 file generation?
- [ ] Sampling rate 30 s or different?

If targets are already processed into your h5 files, ensure the processing matched the spec above. If not, you may need to regenerate.

**TODO:**
- [ ] Document VTEC target generation pipeline
- [ ] Confirm 450 km SLM mapping
- [ ] Confirm elevation cutoff: 15°
- [ ] Confirm CAS satellite DCBs
- [ ] Confirm 30 s sampling (or document what you use)

---

### 12. **Gridding & Output Format (Post-Training)**

**Current:**
VTEC models trained; specific output format not clear from your config.

**Mao et al.**
```
Converts continuous daily NN to gridded "NN-GIM" products:
  - Spatial resolution: same as conventional IAAC products (~5° × 2.5°)
  - Temporal resolution: ~1 hour
  - Output format: IONEX-style grids (or similar)
  - Also stores uncertainty (RMS) maps aligned with grids
```

**TODO (Post-Training):**
- [ ] Implement gridder that samples daily trained NN on spatial/temporal grid
- [ ] Grid spacing: ~5° lon × 2.5° lat; ~1 hour temporal
- [ ] Output: GIM-formatted files (IONEX, custom HDF5, or NetCDF)
- [ ] Also output uncertainty maps (from ensemble variance)
- [ ] This is for evaluation/comparison; not needed for model training itself

---

## Summary: Minimum Changes Checklist

**CRITICAL (Must-Have) Changes:**

- [ ] **Loss function**: MSELoss → Laplacian NLL
- [ ] **Model output**: 1 scalar → 2 scalars (μ, d)
- [ ] **Spatial features**: Implement spherical harmonics (degree 15, 256 features)
- [ ] **Coordinate transforms**: Add magnetic latitude + sun-fixed longitude
- [ ] **Remove SWI**: Disable all space-weather indices (F10.7, Dst, Kp, etc.)
- [ ] **Temporal features**: Simplify to 3 (SODnorm, SODsin, SODcos)
- [ ] **Ensemble**: Train 10 independent models per day
- [ ] **Ensemble fusion**: Combine predictions + variances correctly
- [ ] **Activation functions**: Use tanh (hidden), softplus (scale output)
- [ ] **Architecture**: h90, 3 layers (from h512, 4 layers)
- [ ] **Elevation cutoff**: Verify 15° minimum

**RECOMMENDED (Best Practice) Changes:**

- [ ] Remove weight decay (set to 0) or justify keeping it
- [ ] Use early stopping on validation metric
- [ ] Document optimizer / scheduler choices (paper doesn't specify)
- [ ] Implement output gridding (GIM format) for evaluation

**INFORMATIONAL (Already OK or Out-of-Scope) Changes:**

- [ ] Train/test split: 348 stations train / 52 test (document if different)
- [ ] VTEC target generation: verify matches SLM 450 km, CAS DCBs, 15° cutoff, 30 s sampling
- [ ] Data preprocessing: ensure correct GNSS processing (CCL, GF combinations)

---

## Implementation Order (Recommended)

1. **Feature engineering (1–2 days):**
   - Magnetic latitude transformation (IGRF or dipole)
   - Sun-fixed longitude transformation
   - Spherical harmonics up to degree 15 (256 features)
   - Remove SWI; collapse temporal to 3 features

2. **Model architecture (1 day):**
   - Create `MLP_LaplacianNLL` with 2 outputs (μ, d)
   - Set activations: tanh hidden, softplus output scale
   - Resize to h90, 3 layers

3. **Loss function (1 day):**
   - Implement Laplacian NLL loss
   - Test gradient flow and loss values

4. **Ensemble training (2–3 days):**
   - Modify training loop to train M=10 models per day
   - Implement ensemble fusion (forward + variance computation)
   - Test on small subset first

5. **Validation & gridding (2–3 days):**
   - Retrain one daily model end-to-end
   - Verify ensemble predictions + uncertainty
   - Implement GIM gridding for output
   - Compare to official GIM products if possible

6. **Full retraining (variable):**
   - Retrain all daily models with new pipeline
   - Log metrics (MAE, RMSE, ensemble spread, etc.)
   - Document results

---

## Expected Outcomes Post-Alignment

- **Feature space:** 259 features (256 SH + 3 temporal)
- **Model complexity:** Smaller (h90×3 vs. h512×4) → faster training, less overfit risk
- **Uncertainty quantification:** 10-member ensembles with per-model variance + epistemic spread
- **Robustness:** Laplacian NLL more tolerant of outliers than MSE
- **Replicability:** Close alignment with Mao et al., 2025 allows direct comparison

---

## References & Useful Links

- **Mao et al., 2025** — "NN-based global ionospheric mapping..." (Space Weather, 10.1029/2025SW004446)
- **Rußwurm et al., 2024** — Spherical harmonics positional embeddings for geospatial data
- **pyIGRF** — https://github.com/cmac1000/pyigrf (magnetic coordinate conversion)
- **pyshtools** — https://shtools.github.io/SHTOOLS/ (spherical harmonics toolkit)
- **IONEX format** — Widely used for GIM products (NOAA, CDDIS documentation available)

---

## Notes for Future Experiments

Once aligned:
- Use this setup as **baseline for all subsequent ionospheric modeling work**
- Consider next steps: multi-day forecasting, 3D reconstruction, alternative uncertainty (e.g., heteroscedastic NN)
- Publish/archive the replica implementation for reproducibility

