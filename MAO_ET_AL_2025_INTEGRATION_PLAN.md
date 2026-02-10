# Mao et al., 2025 Integration Plan
## Practical Step-by-Step Implementation for Daily VTEC Models

---

## Part 0: Overview & Key Insights

Your codebase already has **most of the required infrastructure**:
- ✅ Feature registry system with feature ordering
- ✅ Spherical harmonics implementation (`locationencoder/pe/spherical_harmonics.py`)
- ✅ Coordinate transforms module (has spacepy integration)
- ✅ LaplaceLoss already implemented
- ✅ Modular training pipeline with managers

**Strategy:** Rather than rebuild, we **extend and integrate existing modules** step-by-step. This minimizes disruption and leverages what's already proven.

---

## Part 1: Feature Engineering Integration
### Goal: 259 features = 256 SH-based spatial + 3 temporal

### 1.1 Add New Features to Registry (Days 1–2)

**File:** `src/utils/feature_registry.py`

**Current state:** Features are manually listed in `DEFAULT_FEATURE_CONTROL`

**Changes needed:**
1. Remove old raw spatial features (lat_ipp, lon_ipp, sm_lat_ipp, sm_lon_ipp, etc.)
2. Add new computed features:
   - `sh_embedding_256`: 256 spherical harmonics features (pre-computed)
   - `sod_normalized`: Normalized seconds of day [-1, 1]
   - `sod_sin`: sin(2π·SOD/86400)
   - `sod_cos`: cos(2π·SOD/86400)
3. Remove ALL space weather indices (Kp, Dst, F10.7, etc.)
4. Keep only the target feature (vtec)

**Implementation sketch:**

```python
# In feature_registry.py - update DEFAULT_FEATURE_CONTROL
DEFAULT_FEATURE_CONTROL = {
    # Spatial features (will be replaced with SH at data loading time)
    # DEPRECATED: "lat_ipp", "lon_ipp", "sm_lat_ipp", "sm_lon_ipp" - REMOVED
    
    # Temporal features (3 total)
    "sod_normalized": True,      # (2*SOD/86400 - 1) normalized to [-1, 1]
    "sod_sin": True,             # sin(2π·SOD/86400)
    "sod_cos": True,             # cos(2π·SOD/86400)
    
    # Spatial encoding (computed on-the-fly at data load)
    "sh_embedding_256": True,    # Spherical harmonics degree 15
    
    # REMOVED: All SWI features (Kp, Dst, F10.7, etc.)
    # REMOVED: All station features (for daily VTEC, not needed)
    # REMOVED: Direction features (satazi, satele - only for STEC)
    
    # Target (always enabled)
    "vtec": True,
}

# Update FeatureType enum to include new types
class FeatureType(Enum):
    TEMPORAL = "temporal"     # sod_normalized, sod_sin, sod_cos
    SPATIAL_SH = "spatial_sh" # sh_embedding_256 (spherical harmonics)
    TARGET = "target"         # vtec
```

**Register new features during initialization:**

```python
def initialize_feature_registry(config):
    """
    Initialize feature registry for Mao et al. daily VTEC models.
    """
    registry = FeatureRegistry()
    
    # Register in order (deterministic!)
    # Position matters: Temporal features first (3), then SH features (256)
    
    # Temporal features (3 features, indices 0-2)
    registry.register_feature(
        name="sod_normalized",
        feature_type=FeatureType.TEMPORAL,
        position=0,
        description="Normalized seconds of day: (2*SOD/86400 - 1)"
    )
    registry.register_feature(
        name="sod_sin",
        feature_type=FeatureType.TEMPORAL,
        position=1,
        description="sin(2π·SOD/86400)"
    )
    registry.register_feature(
        name="sod_cos",
        feature_type=FeatureType.TEMPORAL,
        position=2,
        description="cos(2π·SOD/86400)"
    )
    
    # Spatial SH features (256 features, indices 3-258)
    registry.register_feature(
        name="sh_embedding_256",
        feature_type=FeatureType.SPATIAL_SH,
        position=3,
        description="Spherical harmonics embedding degree 15: (L+1)² = 256 features"
    )
    
    # Target (index 259)
    registry.register_feature(
        name="vtec",
        feature_type=FeatureType.TARGET,
        description="Vertical Total Electron Content (TECU)"
    )
    
    return registry

# Verify total features in tests:
# assert registry.get_total_features() == 259  # 3 temporal + 256 SH
```

**TODO:**
- [ ] Update `DEFAULT_FEATURE_CONTROL` to only include new features
- [ ] Add `SPATIAL_SH` to `FeatureType` enum
- [ ] Update `initialize_feature_registry()` function
- [ ] Update feature registry unit tests to verify 259 total features
- [ ] Update config YAML templates to reflect new features
- [ ] Add migration guide for removing SWI from existing configs

---

### 1.2 Implement Coordinate Transforms (Days 1–2)

**Files:**
- `src/utils/coordinate_transforms.py` (extend)
- `src/utils/preprocessing.py` (integrate into data loading)

**Tasks:**

#### A) Magnetic Latitude Transformation

```python
# In coordinate_transforms.py - ADD THIS FUNCTION

def geographic_to_magnetic_latitude(lat_geo: np.ndarray, 
                                  lon_geo: np.ndarray,
                                  year: int, 
                                  doy: int) -> np.ndarray:
    """
    Convert geographic latitude to magnetic latitude using IGRF dipole model.
    
    For a daily model, we can either:
    1. Use an annual average dipole tilt (simpler, sufficient for daily)
    2. Use spacepy's coord_transform for exact IGRF (more complex)
    
    Args:
        lat_geo: Geographic latitude in degrees
        lon_geo: Geographic longitude in degrees
        year: Year (for IGRF model)
        doy: Day of year
        
    Returns:
        mag_lat: Magnetic latitude in degrees
    """
    
    # Option 1: Use spacepy (if already available)
    try:
        from spacepy import coordinates as coord
        from spacepy.time import Ticktock
        
        # Convert DOY to datetime
        dt = datetime(year, 1, 1) + timedelta(days=int(doy) - 1)
        
        # Create Ticktock object
        ticks = Ticktock([dt], 'UTC')
        
        # Input coordinates (GEO = geographic)
        cvals = coord.Coords(
            [lon_geo, lat_geo, np.zeros_like(lat_geo)],  # lon, lat, radius
            'GEO',
            'sph'
        )
        cvals.ticks = ticks
        
        # Transform to SM (Solar Magnetic) - alternative: MAG
        cvals_sm = cvals.convert('SM', 'sph')
        
        # Extract magnetic latitude from SM coords
        mag_lat = cvals_sm.lats
        
        return mag_lat
        
    except ImportError:
        logger.warning("spacepy not available, using geographic latitude")
        return lat_geo
        
    except Exception as e:
        logger.warning(f"Magnetic latitude conversion failed: {e}, using geographic")
        return lat_geo
```

#### B) Sun-Fixed Longitude Transformation

```python
# In coordinate_transforms.py - ADD THIS FUNCTION

def geographic_to_sunfixed_longitude(lon_geo: np.ndarray,
                                   year: int,
                                   doy: int,
                                   sod: np.ndarray) -> np.ndarray:
    """
    Convert geographic longitude to sun-fixed (solar-fixed) longitude.
    
    In sun-fixed frame: 0° = subsolar point (always illuminated side towards sun)
    This reduces oscillations due to Earth's rotation.
    
    Method: lon_sf = lon_geo - Greenwich_Hour_Angle(time)
    
    Args:
        lon_geo: Geographic longitude in degrees [-180, 180]
        year: Year
        doy: Day of year
        sod: Seconds of day [0, 86400)
        
    Returns:
        lon_sf: Sun-fixed longitude in degrees
        
    References:
        - Vallado et al. "Fundamentals of Astrodynamics and Applications"
        - Standard GIM practice (from IGS documentation)
    """
    
    # Compute Day Number from year/doy
    # JD = Julian Day Number at 0h UTC
    a = (14 - 1) // 12  # January = month 1, so (14-1)//12 = 1
    y = year + 4800 - a
    m = 1 + 12 * a - 3
    
    jd = doy + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    
    # Julian centuries from J2000.0 (Jan 1, 2000, 12:00 UT)
    jd_j2000 = 2451545.0
    t_ut1 = (jd - jd_j2000) / 36525.0
    
    # GMST (Greenwich Mean Sidereal Time) in seconds
    # Simplified polynomial (accurate enough for daily models)
    gmst_seconds = (67310.54841 +
                   (876600.0 * 3600.0 + 8640184.812866) * t_ut1 +
                   0.093104 * t_ut1**2 -
                   6.2e-6 * t_ut1**3)
    
    gmst_seconds = gmst_seconds % 86400.0  # Modulo 1 day
    gmst_hours = gmst_seconds / 3600.0
    
    # Greenwich Hour Angle of Sun (simplified)
    # For exact computation, would need SOFA library or Astropy
    # Simplified: GHA_sun ≈ GMST + equation of time + 0° (sun on meridian at noon)
    
    # Equation of Time (minutes, simplified)
    b = (doy - 1) * 2 * np.pi / 365.25
    eot_minutes = 9.87 * np.sin(2*b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    eot_hours = eot_minutes / 60.0
    
    # Approximate solar hour angle
    # At noon UTC: GHA_sun ≈ 0° (by definition)
    # Hour offset from noon: sod/3600 - 12
    hour_from_noon = sod / 3600.0 - 12.0
    
    gha_sun_hours = (gmst_hours + eot_hours + hour_from_noon) % 24.0
    gha_sun_degrees = gha_sun_hours * 15.0  # 15°/hour
    
    # Sun-fixed longitude
    lon_sf = lon_geo - gha_sun_degrees
    
    # Normalize to [-180, 180]
    lon_sf = ((lon_sf + 180) % 360) - 180
    
    return lon_sf
```

#### C) Integrate Into Data Loading

**File:** `src/data_loader/datasets.py`

Current code loads raw lat_ipp, lon_ipp. We need to compute spherical harmonics on-the-fly.

```python
# In H5Dataset.__init__() after loading data

# Add coordinate transform initialization
self.use_mag_coords = config.get("use_magnetic_coordinates", True)
self.use_sunfixed_lon = config.get("use_sunfixed_longitude", True)
self.sh_degree = config["data"].get("SH_degree", 15)

# Lazy-load SH embedding module
self.sh_encoder = None
if config["data"].get("compute_sh_on_load", True):
    from utils.locationencoder.pe.spherical_harmonics import SphericalHarmonics
    # For degree 15: (15+1)² = 256 features
    self.sh_encoder = SphericalHarmonics(
        legendre_polys=16,  # degree 15 = (0..15) = 16 values
        harmonics_calculation="analytic"
    )

# In H5Dataset.__getitem__()
def __getitem__(self, idx):
    # ... existing code to load raw data ...
    
    # Get raw lat/lon from IPP
    lat_ipp_geo = data['lat_ipp']
    lon_ipp_geo = data['lon_ipp']
    year = data['year']
    doy = data['doy']
    sod = data['sod']
    
    # Transform coordinates
    if self.use_mag_coords:
        lat_ipp = geographic_to_magnetic_latitude(lat_ipp_geo, lon_ipp_geo, year, doy)
    else:
        lat_ipp = lat_ipp_geo
    
    if self.use_sunfixed_lon:
        lon_ipp = geographic_to_sunfixed_longitude(lon_ipp_geo, year, doy, sod)
    else:
        lon_ipp = lon_ipp_geo
    
    # Compute spherical harmonics embedding (256 features)
    if self.sh_encoder is not None:
        # SH takes (lon, lat) as input
        sh_features = self.sh_encoder(torch.tensor([[lon_ipp, lat_ipp]], dtype=torch.float32))
        sh_features = sh_features.squeeze(0)  # Remove batch dim -> (256,)
    else:
        sh_features = torch.zeros(256, dtype=torch.float32)
    
    # Compute temporal features (3)
    sod_norm = 2.0 * sod / 86400.0 - 1.0
    sod_sin = np.sin(2 * np.pi * sod / 86400.0)
    sod_cos = np.cos(2 * np.pi * sod / 86400.0)
    
    # Assemble final feature vector (259 = 3 temporal + 256 spatial)
    temporal_features = torch.tensor([sod_norm, sod_sin, sod_cos], dtype=torch.float32)
    features = torch.cat([temporal_features, sh_features])  # (259,)
    
    # Get target (vtec)
    target = torch.tensor(data['vtec'], dtype=torch.float32)
    
    return features, target
```

**TODO:**
- [ ] Implement `geographic_to_magnetic_latitude()` function
- [ ] Implement `geographic_to_sunfixed_longitude()` function
- [ ] Integrate into `H5Dataset.__getitem__()` to compute features on-load
- [ ] Add config flags: `use_magnetic_coordinates`, `use_sunfixed_longitude`, `compute_sh_on_load`
- [ ] Update tests to verify feature shapes (should be torch.Size([259]))
- [ ] Benchmark: measure on-load computation time (should be <100μs per sample)
- [ ] Consider caching transforms if needed (but for daily data, usually fast)

---

### 1.3 Remove Space Weather Indices (Day 1)

**File:** `src/utils/feature_registry.py` + `config/config.yaml`

**Changes:**
```yaml
# BEFORE (current config)
feature_control:
  AE-index,_nT: true
  Dst-index,_nT: true
  Kp_index: true
  f107_index: true
  ap_index,_nT: true
  ...

data:
  use_SWI: true

# AFTER (Mao et al. aligned)
feature_control:
  # REMOVED: All SWI and station features
  # Only include temporal + spatial (handled in feature registry)

data:
  use_SWI: false
```

**Code changes:**
```python
# In feature_registry.initialize_feature_registry()
# Simply don't register SWI features

# In data_loader/datasets.py
# skip SWI loading entirely

def __init__(self, config, h5_path, split):
    # ...
    self.use_SWI = config["data"].get("use_SWI", False)
    if self.use_SWI:
        raise ValueError(
            "Mao et al. models do not use SWI. "
            "Set use_SWI: false in config."
        )
```

**TODO:**
- [ ] Remove SWI feature registration from `initialize_feature_registry()`
- [ ] Add guard in data loader to reject configs with `use_SWI: true`
- [ ] Update config templates to remove SWI entries
- [ ] Document why Mao et al. excludes SWI (daily model captures storm effects in observations)

---

## Part 2: Model Architecture
### Goal: MLP (3×90 tanh, softplus uncertainty)

### 2.1 Create MLP_LaplacianNLL Model (Days 2–3)

**File:** `src/model/model.py`

Current status:
- You have `MLP` class (ReLU, single output)
- You have `MLP_NLL` class (Gaussian NLL, 2 outputs)
- Need: `MLP_LaplacianNLL` (tanh, Laplacian NLL, 2 outputs)

```python
# In src/model/model.py - ADD NEW CLASS

class MLP_LaplacianNLL(torch.nn.Module):
    """
    MLP for daily VTEC modeling per Mao et al., 2025.
    
    Architecture:
    - Input: 259 features (3 temporal + 256 SH spatial)
    - Hidden: 3 layers × 90 neurons, tanh activation
    - Output: 2 scalars (μ for VTEC, d for Laplacian scale)
    - Loss: Laplacian NLL (robust to outliers)
    
    Outputs:
        mu: VTEC prediction [batch, 1]
        d: Laplacian scale (positive, softplus'd) [batch, 1]
        variance: Predicted variance = 2*d² [batch, 1] (for logging)
    """
    
    def __init__(self, n_in=259, hidden_dim=90, num_layers=3):
        super().__init__()
        
        assert n_in == 259, f"Expected 259 input features, got {n_in}"
        assert hidden_dim == 90, f"Mao et al. uses 90 neurons, got {hidden_dim}"
        assert num_layers == 3, f"Mao et al. uses 3 layers, got {num_layers}"
        
        # Input layer
        self.input_layer = torch.nn.Linear(n_in, hidden_dim)
        
        # Hidden layers (with tanh activation)
        self.hidden_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers - 1)
        ])
        
        # Output layer (2 outputs: μ, d)
        self.output_layer = torch.nn.Linear(hidden_dim, 2)
        
        # Initialize output bias
        with torch.no_grad():
            # μ output bias initialized to typical VTEC value (~15.5 TECU)
            self.output_layer.bias[0].fill_(15.5)
            # d output bias initialized to small value (~0.1, will be softplus'd)
            self.output_layer.bias[1].fill_(0.1)
            # Small weights for stable initialization
            self.output_layer.weight.normal_(0, 0.01)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, 259]
            
        Returns:
            mu: VTEC predictions [batch_size, 1]
            d: Laplacian scale [batch_size, 1] (positive, softplus'd)
            variance: Predicted variance [batch_size, 1] (for logging/metrics)
        """
        # Input layer + tanh
        x = torch.tanh(self.input_layer(x))
        
        # Hidden layers + tanh
        for hidden_layer in self.hidden_layers:
            x = torch.tanh(hidden_layer(x))
        
        # Output layer
        outputs = self.output_layer(x)  # [batch, 2]
        
        mu = outputs[:, 0:1]      # VTEC prediction [batch, 1]
        d_raw = outputs[:, 1:2]   # Raw scale (unbounded)
        
        # Ensure d is positive: apply softplus + small epsilon
        d = torch.nn.functional.softplus(d_raw) + 1e-3
        
        # Convert to variance for logging/uncertainty metrics
        # Laplacian property: σ² = 2*d²
        variance = 2.0 * (d ** 2)
        
        return mu, d, variance
```

**Integration into get_model():**

```python
# In src/model/model.py - UPDATE get_model() function

def get_model(config):
    """Get model instance based on config."""
    model_type = config["model"]["model_type"]
    device = config.get("device", torch.device("cpu"))
    n_in = config.get("n_input_features", 259)  # NEW: for Mao et al. alignment
    
    if model_type == "MLP":
        return MLP(
            n_in=n_in,
            n_out=config["model"].get("output_size", 1),
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
        )
    
    elif model_type == "MLP_LaplacianNLL":  # NEW
        assert config["model"]["hidden_dim"] == 90, \
            "Mao et al. uses hidden_dim=90"
        assert config["model"]["num_layers"] == 3, \
            "Mao et al. uses num_layers=3"
        return MLP_LaplacianNLL(
            n_in=n_in,
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
        )
    
    elif model_type == "MLP_NLL":
        # ... existing code ...
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

**Config template:**

```yaml
# config_mao_et_al_2025.yaml

model:
  model_type: MLP_LaplacianNLL  # NEW model class
  hidden_dim: 90               # Per Mao et al. tuning
  num_layers: 3                # Per Mao et al. tuning
  output_size: 2               # μ (VTEC) + d (scale)

training:
  loss_function: LaplacianNLLLoss  # NEW loss type (see next section)
  optimizer: Adam
  weight_decay: 0.0            # Paper doesn't specify - removed
  learning_rate: 0.0001        # Standard, can be tuned
  
finetune:
  epochs: 150
  early_stopping: true
  patience: 10
  scheduler: ReduceLROnPlateau
```

**TODO:**
- [ ] Implement `MLP_LaplacianNLL` class in `model.py`
- [ ] Update `get_model()` to support new model type
- [ ] Add unit test: verify output shape (batch, 1) for each of μ, d, variance
- [ ] Add unit test: verify d is always positive (softplus works)
- [ ] Create config template `config_mao_et_al_2025.yaml`
- [ ] Verify initialization: μ bias ≈ 15.5, d bias ≈ 0.1
- [ ] Test gradient flow through softplus and tanh

---

### 2.2 Implement LaplacianNLLLoss (Days 2–3)

**File:** `src/utils/loss_function.py`

Current state: You have a `LaplaceLoss` class, but need proper `LaplacianNLLLoss`.

```python
# In src/utils/loss_function.py - ADD/REPLACE

class LaplacianNLLLoss(torch.nn.Module):
    """
    Laplacian Negative Log-Likelihood Loss (Mao et al., 2025).
    
    Assumes target Y follows a Laplacian distribution with:
    - Mean (location): μ(X)
    - Scale (diversity): d(X)
    
    Loss per sample:
        ℓ_NLL = log(d) + |Y - μ| / d
    
    Rationale:
    - More robust to outliers than Gaussian NLL (uses L1, not L2 distance)
    - Allows network to predict both position (μ) and uncertainty (d)
    - Paper shows better performance than Gaussian for VTEC targets
    
    Network outputs:
    - μ: VTEC prediction (unbounded)
    - d: Scale parameter (positive, ensured via softplus)
    
    Args:
        reduction: 'mean' or 'none'
        eps: Small epsilon to prevent log(0), default 1e-6
    """
    
    def __init__(self, reduction='mean', eps=1e-6):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, mu, d, y):
        """
        Args:
            mu: Predicted mean (VTEC) [batch, 1]
            d: Predicted scale (positive) [batch, 1]
            y: Observed VTEC target [batch, 1]
            
        Returns:
            loss: Scalar (if reduction='mean') or vector [batch] (if 'none')
        """
        # Ensure positive d (should already be via softplus in model)
        d = torch.clamp(d, min=self.eps)
        
        # Laplacian NLL: log(d) + |y - μ| / d
        abs_error = torch.abs(y - mu)
        loss = torch.log(d) + abs_error / d
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


# Update get_criterion() to include new loss

def get_criterion(config, loss_fn=None):
    loss_type = config["training"]["loss_function"]
    if loss_fn is not None:
        loss_type = loss_fn
    
    # ... existing code ...
    
    elif loss_type == "LaplacianNLLLoss":      # NEW
        return LaplacianNLLLoss(reduction='mean', eps=1e-6)
    
    else:
        raise Exception(f"unknown loss {loss_type}")
```

**Integration into training loop:**

```python
# In src/training/train_manager.py - UPDATE train_epoch()

def train_epoch(self, model, dataloader, criterion_mse, criterion_nll, criterion_kld, 
                optimizer, epoch=0):
    """
    Training epoch (existing structure, just update loss computation).
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(self.device), y.to(self.device)
        
        # Forward pass
        if isinstance(model, MLP_LaplacianNLL):
            # NEW: Laplacian NLL model returns (μ, d, variance)
            mu, d, variance = model(X)
            
            # Compute loss using Laplacian NLL
            loss = criterion_nll(mu, d, y)  # Pass μ, d, y
        
        else:
            # Existing logic for other models
            outputs = model(X)
            if isinstance(outputs, tuple):
                pred, pred_var = outputs
                loss = criterion_nll(pred, y, pred_var)
            else:
                loss = criterion_mse(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

**Unit tests:**

```python
# In tests/test_losses.py - NEW

def test_laplacian_nll_loss():
    import torch
    from src.utils.loss_function import LaplacianNLLLoss
    
    criterion = LaplacianNLLLoss()
    
    # Test case 1: Perfect prediction
    mu = torch.tensor([[10.0]])
    d = torch.tensor([[1.0]])
    y = torch.tensor([[10.0]])
    
    loss = criterion(mu, d, y)
    expected = np.log(1.0) + 0.0 / 1.0  # = 0.0
    assert np.isclose(loss.item(), expected, atol=1e-5)
    
    # Test case 2: 1 TECU error, scale=1
    mu = torch.tensor([[10.0]])
    d = torch.tensor([[1.0]])
    y = torch.tensor([[11.0]])
    
    loss = criterion(mu, d, y)
    expected = np.log(1.0) + 1.0 / 1.0  # = 1.0
    assert np.isclose(loss.item(), expected, atol=1e-5)
    
    # Test case 3: Batch averaging
    mu = torch.tensor([[10.0], [20.0]])
    d = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[11.0], [22.0]])
    
    loss = criterion(mu, d, y)
    loss1 = np.log(1.0) + 1.0 / 1.0       # = 1.0
    loss2 = np.log(2.0) + 2.0 / 2.0       # ≈ 1.693
    expected = (loss1 + loss2) / 2
    assert np.isclose(loss.item(), expected, atol=1e-5)
```

**TODO:**
- [ ] Implement `LaplacianNLLLoss` class in `loss_function.py`
- [ ] Update `get_criterion()` to recognize "LaplacianNLLLoss"
- [ ] Update training loop to pass (μ, d, y) to Laplacian loss
- [ ] Add unit tests for loss computation (verify gradients)
- [ ] Benchmark: compare loss values to Gaussian NLL on real data
- [ ] Verify gradient flow: d loss should reflect both prediction error AND uncertainty

---

## Part 3: Ensemble Training
### Goal: 10 independent models per day with proper variance fusion

### 3.1 Create Ensemble Trainer (Days 4–5)

**File:** `src/training/ensemble_trainer.py` (NEW)

```python
# In src/training/ensemble_trainer.py - NEW FILE

import torch
import logging
import os
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """
    Trains M independent models per day for deep ensembles.
    
    Each ensemble member:
    - Uses unique random seed (config_seed + member_id)
    - Sees full training data (may be shuffled differently)
    - Trained independently to convergence
    - Saves checkpoint for later ensemble inference
    
    Ensemble fusion at inference:
    - Combine means: μ_e = (1/M) Σ μ_m(X)
    - Combine variances: σ_e² = (1/M) Σ 2*d_m² + (1/M) Σ μ_m² - μ_e²
      (First term: aleatoric/per-model uncertainty)
      (Second term: epistemic/ensemble disagreement)
    """
    
    def __init__(self, config, logger, base_trainer_class):
        """
        Args:
            config: Configuration dict
            logger: Logger instance
            base_trainer_class: Class to use for individual training
                (e.g., Finetuner from src/finetune.py)
        """
        self.config = config
        self.logger = logger
        self.base_trainer_class = base_trainer_class
        self.num_ensemble_members = config.get("ensemble", {}).get("num_members", 10)
        self.device = config.get("device", torch.device("cpu"))
        self.ensemble_dir = os.path.join(
            config["output_dir"], "ensemble_members"
        )
        os.makedirs(self.ensemble_dir, exist_ok=True)
    
    def train_ensemble(self):
        """
        Train M independent models for the configured day.
        """
        self.logger.info(f"Training ensemble with {self.num_ensemble_members} members...")
        
        trained_models = []
        
        for member_id in range(self.num_ensemble_members):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training ensemble member {member_id + 1}/{self.num_ensemble_members}")
            self.logger.info(f"{'='*60}")
            
            # Create config for this member
            member_config = self.config.copy()
            
            # Unique seed for each member
            base_seed = self.config.get("random_seed", 42)
            member_seed = base_seed + member_id
            member_config["random_seed"] = member_seed
            
            # Setup seed deterministically
            self._setup_seed(member_seed)
            
            # Train one model
            trainer = self.base_trainer_class(member_config, self.logger)
            # trainer.finetune() already called in __init__
            
            # Save checkpoint with member ID
            member_checkpoint_path = os.path.join(
                self.ensemble_dir,
                f"ensemble_member_{member_id}.pth"
            )
            
            # Note: trainer saves its own models, but we may want to save ensemble metadata
            self.logger.info(f"Member {member_id} checkpoint would go to: {member_checkpoint_path}")
            
            trained_models.append(member_id)  # Track member IDs
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Ensemble training complete! Trained {len(trained_models)} models")
        self.logger.info(f"Ensemble directory: {self.ensemble_dir}")
        self.logger.info(f"{'='*60}")
        
        return trained_models
    
    @staticmethod
    def _setup_seed(seed):
        """Set up deterministic random seed."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        import random
        random.seed(seed)


class EnsembleInferencer:
    """
    Loads M trained models and performs ensemble inference.
    
    For each test sample:
    - Forward through all M models
    - Combine predictions and uncertainties
    - Return ensemble mean + variance
    """
    
    def __init__(self, ensemble_dir: str, device=None):
        """
        Args:
            ensemble_dir: Directory containing trained ensemble members
            device: torch.device for inference
        """
        self.ensemble_dir = ensemble_dir
        self.device = device or torch.device("cpu")
        self.models: List[torch.nn.Module] = []
        self._load_ensemble()
    
    def _load_ensemble(self):
        """Load all ensemble member checkpoints."""
        member_files = sorted([
            f for f in os.listdir(self.ensemble_dir)
            if f.startswith("ensemble_member_") and f.endswith(".pth")
        ])
        
        for member_file in member_files:
            checkpoint_path = os.path.join(self.ensemble_dir, member_file)
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            # Depending on your checkpoint format:
            model = checkpoint.get("model") or checkpoint  # Adapt as needed
            model.to(self.device)
            model.eval()
            self.models.append(model)
        
        logger.info(f"Loaded {len(self.models)} ensemble members from {self.ensemble_dir}")
    
    def forward_ensemble(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble, combine predictions.
        
        Args:
            X: Input batch [batch_size, 259]
            
        Returns:
            mu_e: Ensemble mean prediction [batch_size, 1]
            sig2_e: Ensemble variance [batch_size, 1]
        """
        mu_list = []
        d_list = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                mu_m, d_m, _ = model(X.to(self.device))
                mu_list.append(mu_m)
                d_list.append(d_m)
        
        # Stack: (batch_size, M)
        mu_stack = torch.cat(mu_list, dim=1)  # [batch_size, M]
        d_stack = torch.cat(d_list, dim=1)    # [batch_size, M]
        
        # Ensemble mean
        mu_e = mu_stack.mean(dim=1, keepdim=True)  # [batch_size, 1]
        
        # Ensemble variance (aleatoric + epistemic)
        # Aleatoric: average per-model variance
        var_aleatoric = (2.0 * (d_stack ** 2)).mean(dim=1, keepdim=True)
        
        # Epistemic: variance of ensemble means
        var_epistemic = (mu_stack ** 2).mean(dim=1, keepdim=True) - (mu_e ** 2)
        
        # Total variance
        sig2_e = var_aleatoric + var_epistemic
        
        return mu_e, sig2_e
```

### 3.2 Update Entry Point (Days 4–5)

**File:** `src/main.py`

```python
# In src/main.py - ADD at the end of main()

if config["mode"] == "finetune":
    # Check if ensemble training is enabled
    use_ensemble = config.get("ensemble", {}).get("enabled", False)
    
    if use_ensemble:
        from training.ensemble_trainer import EnsembleTrainer
        
        ensemble_trainer = EnsembleTrainer(
            config,
            logger,
            base_trainer_class=Finetuner  # Train M Finetuners
        )
        trained_members = ensemble_trainer.train_ensemble()
        
        logger.info(f"✓ Ensemble training completed: {len(trained_members)} models trained")
    
    else:
        # Single model training (existing behavior)
        Finetuner(config, logger)
```

**Config template:**

```yaml
# config_mao_et_al_2025.yaml

ensemble:
  enabled: true          # Enable ensemble training
  num_members: 10        # Per Mao et al.
```

**TODO:**
- [ ] Create `EnsembleTrainer` class in `training/ensemble_trainer.py`
- [ ] Create `EnsembleInferencer` class for combining predictions
- [ ] Update `main.py` to support ensemble mode
- [ ] Add config option `ensemble.enabled` and `ensemble.num_members`
- [ ] Test: train 2-member ensemble on small day subset (sanity check)
- [ ] Verify: each member gets unique seed, separate training loop
- [ ] Verify: ensemble variance formula (aleatoric + epistemic)

---

## Part 4: Configuration & Integration
### Goal: Single config file that ties everything together

### 4.1 Create Mao et al. Config Template (Day 5)

**File:** `config/config_mao_et_al_2025.yaml` (NEW)

```yaml
# ============================================================================
# Mao et al., 2025 - Daily VTEC Model Configuration
# ============================================================================
# Replication of:
#   Mao, X., et al. (2025). "NN-based global ionospheric mapping..."
#   Space Weather, 10.1029/2025SW004446
#
# Daily VTEC interpolation using:
#   - 259 features (256 SH spatial + 3 temporal)
#   - 3×90 MLP with tanh activation
#   - Laplacian NLL loss (robust to outliers)
#   - 10-member deep ensemble
#
# ============================================================================

# -------- GENERAL --------
mode: finetune                 # Single-day finetuning
year: '2024'
doy: '291'
random_seed: 42
target: vtec

# -------- INPUT DATA --------
data:
  GNSS_data_path: /path/to/GNSS/data
  scratch_dir: /scratch2/arrueegg/WP4/PNN_STEC/data/
  
  # Coordinate transforms (Mao et al. aligned)
  use_magnetic_coordinates: true    # Geographic → magnetic latitude
  use_sunfixed_longitude: true      # Geographic → sun-fixed longitude
  
  # Spherical harmonics spatial encoding
  SH_degree: 15              # Degree 15 → (15+1)² = 256 features
  compute_sh_on_load: true   # Compute SH embedding during data loading
  
  # Elevation cutoff (per Mao et al. spec)
  min_elevation: 15.0        # 15° as specified
  max_elevation: 90.0
  
  # Data subsets for training
  train_subset_size: 500000
  val_size: 100000
  test_size: 10000000  # Effectively all test data
  
  # SWI disabled (explicit per Mao et al.)
  use_SWI: false
  
  # Data options
  shuffle: true
  move_to_scratch: true
  use_agg_h5: false

# -------- FEATURES --------
# Note: Feature registry auto-configures based on above settings
# Features should total: 3 (temporal) + 256 (SH spatial) = 259
feature_control:
  # Only temporal features (explicit, others auto-managed by registry)
  sod_normalized: true
  sod_sin: true
  sod_cos: true
  
  # Note: Spatial SH features (256) handled by data loader
  # Note: All SWI removed per Mao et al. (trains daily, captures in observations)

# -------- MODEL ARCHITECTURE --------
model:
  model_type: MLP_LaplacianNLL    # New model class for this spec
  hidden_dim: 90                  # Per Mao et al. tuning
  num_layers: 3                   # Per Mao et al. tuning
  output_size: 2                  # μ (VTEC) + d (Laplacian scale)
  dropout_rate: 0.0               # Mao et al. doesn't mention dropout

# -------- TRAINING --------
training:
  # Loss function: Laplacian NLL (robust to outliers)
  loss_function: LaplacianNLLLoss
  
  # Optimizer (paper doesn't specify, so reasonable defaults)
  optimizer: Adam
  learning_rate: 0.0001
  weight_decay: 0.0               # Paper doesn't specify decay
  
  # Regularization
  target_weighting:
    enabled: false                # Not mentioned in paper
  kl_annealing:
    enabled: false                # Not used for MLP_LaplacianNLL
  
  # Other training options
  use_amp: true                   # Automatic mixed precision
  standardize_targets: false
  save_best_only: true
  log_space_point: mean

# -------- FINETUNING (Single Day) --------
finetune:
  learning_rate: 0.0001
  batchsize: 2048
  epochs: 150
  early_stopping: true
  patience: 10
  num_workers: 8
  prefetch_factor: 4
  scheduler: ReduceLROnPlateau
  scheduler_step_size: 100
  save_model_every_epoch: false
  
  # Daily training (no pretrain needed if finetune_from_scratch)
  finetune_from_scratch: true
  freeze_body: false

finetune_from_scratch: true       # Train each day independently

# -------- ENSEMBLE (Deep Ensemble) --------
ensemble:
  enabled: true                   # Enable M-member ensemble
  num_members: 10                 # Mao et al. uses 10

# -------- PRETRAINING (if needed - typically not for daily models) --------
pretrain:
  batchsize: 2048
  epochs: 150
  early_stopping: true
  patience: 10
  learning_rate: 0.0001
  num_workers: 4
  prefetch_factor: 4
  scheduler: ReduceLROnPlateau
  scheduler_step_size: 100

# -------- OUTPUT --------
output_dir: experiments/Mao_et_al_VTEC_2025_{year}_{doy}

# -------- EVALUATION --------
evaluation:
  enable_scenarios: false

# -------- DEBUGGING --------
debug: false
debug_single_batch: false
enable_timing: false

# -------- WANDB --------
project_name: PNN_VTEC_Mao_et_al_2025
wandb:
  offline: false

# -------- CLUSTER (Optional) --------
cluster: false
```

**TODO:**
- [ ] Create `config_mao_et_al_2025.yaml` template
- [ ] Document each key and its rationale vs standard configs
- [ ] Add validation: check SH_degree == 15, hidden_dim == 90, num_layers == 3
- [ ] Add validation: check use_SWI == false
- [ ] Add validation: check min_elevation >= 15.0
- [ ] Test config loading with this template

---

### 4.2 Add Configuration Validation (Day 5)

**File:** `src/utils/config_parser.py` (extend)

```python
def validate_mao_alignment(config):
    """
    Validate that config matches Mao et al., 2025 specification.
    
    Raises ValueError if misaligned, logs warnings for deviations.
    """
    logger = logging.getLogger(__name__)
    
    mao_spec = {
        "model.model_type": "MLP_LaplacianNLL",
        "model.hidden_dim": 90,
        "model.num_layers": 3,
        "model.output_size": 2,
        "training.loss_function": "LaplacianNLLLoss",
        "data.SH_degree": 15,
        "data.min_elevation": 15.0,
        "data.use_SWI": False,
        "ensemble.enabled": True,
        "ensemble.num_members": 10,
    }
    
    errors = []
    
    for key, expected_value in mao_spec.items():
        keys = key.split(".")
        value = config
        for k in keys:
            value = value.get(k)
        
        if value != expected_value:
            errors.append(
                f"  {key}: expected {expected_value}, got {value}"
            )
    
    if errors:
        logger.warning("⚠️  Config deviates from Mao et al., 2025 specification:")
        for error in errors:
            logger.warning(error)
        
        # For critical settings, raise error
        if "model.model_type" in str(errors):
            raise ValueError(
                "Model type must be MLP_LaplacianNLL for Mao et al. alignment"
            )
        if "training.loss_function" in str(errors):
            raise ValueError(
                "Loss must be LaplacianNLLLoss for Mao et al. alignment"
            )
        if "data.use_SWI" in str(errors):
            raise ValueError(
                "Mao et al. does not use SWI. Set use_SWI: false"
            )
    else:
        logger.info("✓ Config fully aligned with Mao et al., 2025 specification")


# Update parse_config() to call validation
def parse_config(config_path=None):
    # ... existing code to load YAML ...
    config = yaml.safe_load(...)
    
    # Optional: validate if config name suggests Mao alignment
    if config_path and "mao" in config_path.lower():
        validate_mao_alignment(config)
    
    return config
```

**TODO:**
- [ ] Add `validate_mao_alignment()` function to config parser
- [ ] Update `parse_config()` to call validation for Mao configs
- [ ] Add unit test: verify validation catches incorrect settings
- [ ] Add unit test: verify validation passes for correct settings

---

## Part 5: Testing & Validation
### Goal: End-to-end tests before full retraining

### 5.1 Unit Tests (Day 5)

**Create:** `tests/test_mao_alignment.py` (NEW)

```python
import unittest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

from src.utils.feature_registry import initialize_feature_registry, FeatureType
from src.model.model import MLP_LaplacianNLL
from src.utils.loss_function import LaplacianNLLLoss
from src.utils.coordinate_transforms import (
    geographic_to_magnetic_latitude,
    geographic_to_sunfixed_longitude
)


class TestMaoAlignment(unittest.TestCase):
    """Test suite for Mao et al., 2025 alignment."""
    
    def setUp(self):
        self.config = {
            "data": {
                "SH_degree": 15,
                "use_SWI": False,
                "use_magnetic_coordinates": True,
                "use_sunfixed_longitude": True,
                "compute_sh_on_load": True,
                "min_elevation": 15.0,
            },
            "feature_registry": None,
            "model": {
                "model_type": "MLP_LaplacianNLL",
                "hidden_dim": 90,
                "num_layers": 3,
            },
            "training": {
                "loss_function": "LaplacianNLLLoss",
            },
            "ensemble": {
                "enabled": True,
                "num_members": 10,
            },
        }
    
    def test_feature_count(self):
        """Test that feature registry produces 259 features."""
        registry = initialize_feature_registry(self.config)
        total_features = registry.get_total_features()
        self.assertEqual(
            total_features, 259,
            f"Expected 259 features, got {total_features}"
        )
    
    def test_feature_types(self):
        """Test that features are correctly categorized."""
        registry = initialize_feature_registry(self.config)
        
        temporal = registry.get_features_by_type(FeatureType.TEMPORAL)
        self.assertEqual(len(temporal), 3)
        self.assertIn("sod_normalized", temporal)
        self.assertIn("sod_sin", temporal)
        self.assertIn("sod_cos", temporal)
        
        spatial_sh = registry.get_features_by_type(FeatureType.SPATIAL_SH)
        self.assertEqual(len(spatial_sh), 1)
        self.assertEqual(spatial_sh[0], "sh_embedding_256")
    
    def test_model_output_shape(self):
        """Test MLP_LaplacianNLL output shapes."""
        model = MLP_LaplacianNLL(n_in=259, hidden_dim=90, num_layers=3)
        
        X = torch.randn(32, 259)  # Batch of 32 samples
        mu, d, variance = model(X)
        
        self.assertEqual(mu.shape, (32, 1))
        self.assertEqual(d.shape, (32, 1))
        self.assertEqual(variance.shape, (32, 1))
        
        # d should always be positive (softplus applied)
        self.assertTrue(torch.all(d > 0))
    
    def test_laplacian_nll_loss(self):
        """Test Laplacian NLL loss computation."""
        criterion = LaplacianNLLLoss()
        
        # Perfect prediction
        mu = torch.tensor([[10.0]])
        d = torch.tensor([[1.0]])
        y = torch.tensor([[10.0]])
        
        loss = criterion(mu, d, y)
        expected = np.log(1.0) + 0.0 / 1.0
        
        self.assertAlmostEqual(loss.item(), expected, places=5)
    
    def test_magnetic_latitude_transform(self):
        """Test magnetic latitude transformation (if spacepy available)."""
        lat_geo = np.array([45.0])
        lon_geo = np.array([0.0])
        year, doy = 2024, 291
        
        try:
            mag_lat = geographic_to_magnetic_latitude(lat_geo, lon_geo, year, doy)
            # Should return an array, not crash
            self.assertEqual(mag_lat.shape, lat_geo.shape)
        except ImportError:
            self.skipTest("spacepy not available")
    
    def test_sunfixed_longitude_transform(self):
        """Test sun-fixed longitude transformation."""
        lon_geo = np.array([0.0])
        year, doy, sod = 2024, 291, 43200.0  # Noon
        
        lon_sf = geographic_to_sunfixed_longitude(lon_geo, year, doy, np.array([sod]))
        
        # Should return valid longitude
        self.assertTrue(-180 <= lon_sf[0] <= 180)
        # At different SODs, lon_sf should differ
        sod_midnight = 0.0
        lon_sf_midnight = geographic_to_sunfixed_longitude(
            lon_geo, year, doy, np.array([sod_midnight])
        )
        self.assertNotAlmostEqual(lon_sf[0], lon_sf_midnight[0], places=1)
    
    def test_swd_removed(self):
        """Test that SWI features are not registered."""
        registry = initialize_feature_registry(self.config)
        
        all_features = registry.get_all_enabled_features()
        
        swood_features = ["Kp_index", "f107_index", "Dst-index,_nT"]
        for swd_feat in swood_features:
            self.assertNotIn(swd_feat, all_features)


if __name__ == "__main__":
    unittest.main()
```

**TODO:**
- [ ] Create `tests/test_mao_alignment.py`
- [ ] Run tests to verify 259 features, correct types, model outputs
- [ ] Run tests to verify coordinate transforms
- [ ] Run tests to verify SWI is removed
- [ ] Add integration test: load real data, verify shapes
- [ ] Add integration test: train one epoch on small batch

---

### 5.2 Integration Test (Day 5–6)

**Create:** `tests/test_mao_integration.py` (NEW)

```python
import unittest
import tempfile
import shutil
import torch
import numpy as np
from pathlib import Path

from src.utils.config_parser import parse_config
from src.utils.feature_registry import initialize_feature_registry
from src.model.model import get_model, MLP_LaplacianNLL
from src.training.ensemble_trainer import EnsembleTrainer
from src.finetune import Finetuner


class TestMaoIntegration(unittest.TestCase):
    """Integration tests for Mao et al. full pipeline."""
    
    def setUp(self):
        """Create minimal config for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            "mode": "finetune",
            "year": "2024",
            "doy": "291",
            "random_seed": 42,
            "device": torch.device("cpu"),
            "output_dir": self.test_dir,
            
            "data": {
                "SH_degree": 15,
                "use_SWI": False,
                "use_magnetic_coordinates": True,
                "use_sunfixed_longitude": True,
                "min_elevation": 15.0,
                "train_subset_size": 1000,
            },
            
            "model": {
                "model_type": "MLP_LaplacianNLL",
                "hidden_dim": 90,
                "num_layers": 3,
                "output_size": 2,
            },
            
            "training": {
                "loss_function": "LaplacianNLLLoss",
                "optimizer": "Adam",
                "learning_rate": 0.0001,
                "weight_decay": 0.0,
            },
            
            "finetune": {
                "epochs": 2,  # Minimal for testing
                "early_stopping": False,
                "batchsize": 32,
                "num_workers": 0,
            },
            
            "ensemble": {
                "enabled": False,  # Single model for quick test
                "num_members": 1,
            },
        }
        
        self.config["feature_registry"] = initialize_feature_registry(self.config)
    
    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.test_dir)
    
    def test_model_creation(self):
        """Test that model can be created."""
        model = get_model(self.config)
        self.assertIsInstance(model, MLP_LaplacianNLL)
    
    def test_forward_pass(self):
        """Test that model forward pass works."""
        model = get_model(self.config)
        X = torch.randn(32, 259)
        
        mu, d, var = model(X)
        
        self.assertEqual(mu.shape, (32, 1))
        self.assertEqual(d.shape, (32, 1))
        self.assertEqual(var.shape, (32, 1))
    
    def test_loss_computation(self):
        """Test that loss can be computed."""
        from src.utils.loss_function import get_criterion
        
        criterion = get_criterion(self.config)
        
        mu = torch.randn(32, 1)
        d = torch.nn.functional.softplus(torch.randn(32, 1)) + 1e-3
        y = torch.randn(32, 1)
        
        loss = criterion(mu, d, y)
        
        self.assertIsInstance(loss.item(), float)
        self.assertFalse(np.isnan(loss.item()))
    
    # Optionally skip this test if data files not available
    @unittest.skip("Requires real data files")
    def test_single_epoch_training(self):
        """Test that a single epoch of training runs."""
        # This would require mock data loaders
        # Skipping for now, can be implemented with synthetic data
        pass


if __name__ == "__main__":
    unittest.main()
```

**TODO:**
- [ ] Create `tests/test_mao_integration.py`
- [ ] Test model creation, forward pass, loss computation
- [ ] Test with synthetic data
- [ ] (Optional) Test with minimal real data if available

---

## Part 6: Implementation Timeline & Priorities

### Phase 1: Foundation (Days 1–2) — START HERE
**Priority: CRITICAL**

- [ ] Day 1: Add new features to registry (3 temporal + 256 SH)
- [ ] Day 1: Remove SWI features and update configs
- [ ] Day 2: Implement coordinate transforms (mag lat, sun-fixed lon)
- [ ] Day 2: Integrate SH embedding into data loader

**Outcome:** Feature pipeline matches Mao spec (259 features)

### Phase 2: Model & Loss (Days 2–3)
**Priority: CRITICAL**

- [ ] Day 2: Implement `MLP_LaplacianNLL` model class
- [ ] Day 3: Implement `LaplacianNLLLoss` loss class
- [ ] Day 3: Update training loop to use new model/loss
- [ ] Day 3: Create config template `config_mao_et_al_2025.yaml`

**Outcome:** Single model training works with Laplacian NLL

### Phase 3: Ensemble (Days 4–5)
**Priority: HIGH** (can delay if needed)

- [ ] Day 4: Create `EnsembleTrainer` and `EnsembleInferencer` classes
- [ ] Day 4: Integrate ensemble training into main pipeline
- [ ] Day 5: Test 2-member ensemble on small subset

**Outcome:** Can train 10 independent models per day and combine

### Phase 4: Testing & Validation (Day 5–6)
**Priority: MEDIUM**

- [ ] Day 5: Write unit tests for features, model, loss
- [ ] Day 5: Write integration tests
- [ ] Day 6: Dry-run on 1–2 days of real data
- [ ] Day 6: Verify output formats and metrics

**Outcome:** Confidence that implementation is correct

### Phase 5: Full Retraining (Day 6+)
**Priority: LOWER** (after validation)

- [ ] Day 6+: Retrain all daily VTEC models
- [ ] Collect metrics (MAE, RMSE, ensemble spread)
- [ ] Compare to official GIM products
- [ ] Archive checkpoints and results

---

## Part 7: Code Locations & File Changes Summary

### New Files to Create
```
src/utils/locationencoder/                  (exists)
  pe/spherical_harmonics.py                  (exists, use as-is)

src/training/ensemble_trainer.py             → NEW
tests/test_mao_alignment.py                  → NEW
tests/test_mao_integration.py                → NEW
config/config_mao_et_al_2025.yaml            → NEW
```

### Files to Modify
```
src/utils/feature_registry.py                → Update feature registry
src/utils/coordinate_transforms.py           → Add magnetic lat + sun-fixed lon
src/utils/loss_function.py                   → Add LaplacianNLLLoss
src/utils/preprocessing.py                   → Remove SWI processing (optional)
src/utils/config_parser.py                   → Add Mao alignment validation
src/model/model.py                           → Add MLP_LaplacianNLL, update get_model()
src/data_loader/datasets.py                  → Integrate SH embedding on-load
src/training/base_trainer.py                 → (Minor: handle new loss type)
src/training/train_manager.py                → Update loss computation for new model
src/finetune.py                              → (Minor: handle ensemble mode)
src/main.py                                  → Add ensemble entry point
config/config.yaml                           → Add new config options (template)
```

### No Changes Needed
```
src/pretrain.py                              (not used for daily models)
src/evaluation.py                            (works as-is with new models)
src/inference_map.py                         (can be adapted for gridding)
src/data_processing/add_split_indices.py     (not affected)
```

---

## Part 8: Checklist for Implementation

### Features (Checkpoint 1)
- [ ] Feature registry has exactly 259 features (3 + 256)
- [ ] All SWI features removed
- [ ] Spatial features (lat_ipp, lon_ipp, etc.) removed from registry
- [ ] New features registered: sod_normalized, sod_sin, sod_cos, sh_embedding_256
- [ ] Feature order is deterministic

### Coordinate Transforms (Checkpoint 2)
- [ ] `geographic_to_magnetic_latitude()` implemented (or fallback)
- [ ] `geographic_to_sunfixed_longitude()` implemented
- [ ] SH embedding computed and returns shape (256,)
- [ ] Data loader properly computes features: output shape (259,) ✓

### Model & Loss (Checkpoint 3)
- [ ] `MLP_LaplacianNLL` class implemented: 3×90 tanh
- [ ] Outputs: (μ, d, variance) shapes correct
- [ ] `LaplacianNLLLoss` implemented and registered
- [ ] Training loop updated to pass (μ, d, y) to loss
- [ ] Gradient flow verified (no NaN/Inf)

### Ensemble (Checkpoint 4)
- [ ] `EnsembleTrainer` can train M models with unique seeds
- [ ] `EnsembleInferencer` loads and combines models
- [ ] Ensemble variance formula correct: aleatoric + epistemic
- [ ] Can train 10-member ensemble on small subset

### Configuration (Checkpoint 5)
- [ ] `config_mao_et_al_2025.yaml` created and validated
- [ ] Config validation catches misalignments
- [ ] Can load config and initialize full pipeline

### Testing (Checkpoint 6)
- [ ] Unit tests pass: features, model, loss, transforms
- [ ] Integration tests pass: model creation, forward pass, loss
- [ ] Can train 1 epoch on synthetic data
- [ ] Can train 1 day on real data (with reduced epochs)

### Full Implementation (Final Checkpoint)
- [ ] All daily VTEC models retrained
- [ ] Metrics collected and compared to original
- [ ] Can produce GIM-like grids from NN predictions
- [ ] Results documented and archived

---

## Part 9: Troubleshooting Guide

### Common Issues

**Issue: "features don't sum to 259"**
- Check feature registry for duplicate/missing features
- Verify SH degree is 15 (should give 256 features)
- Verify 3 temporal features are registered

**Issue: "Spherical harmonics output wrong shape"**
- Check SH degree passed to SphericalHarmonics (should be 16 for degree 15)
- Verify forward() method returns [batch, 256]
- Check coordinate transforms return proper (lon, lat) order

**Issue: "Loss is NaN or Inf"**
- Check that `d` is positive (softplus should ensure this)
- Verify target `y` is not NaN/Inf
- Check loss formula: log(d) term should be stable
- Add numerical stability: `d = clamp(d, min=1e-6)`

**Issue: "Magnetic latitude transform fails"**
- spacepy may not be installed; check imports
- Fallback to geographic latitude (add warning)
- Consider simple dipole tilt if spacepy unavailable

**Issue: "Ensemble training too slow"**
- Each model trains independently: set to small test first
- Consider parallel training (not supported yet, POC)
- Can reduce epochs for testing

---

## Part 10: Reference Checklist for Gradual Rollout

```
✓ = Complete and tested
~ = Partial/in progress  
✗ = Not started

FEATURES:
  ✗ Feature registry: 259 features (3 temporal + 256 SH)
  ✗ Remove SWI
  ✗ Add magnetic latitude transform
  ✗ Add sun-fixed longitude transform
  ✗ Integrate SH embedding on-load
  ✗ Unit tests for features

MODEL:
  ✗ MLP_LaplacianNLL class (3×90 tanh)
  ✗ LaplacianNLLLoss class
  ✗ Training loop update
  ✗ Unit tests for model/loss

ENSEMBLE:
  ✗ EnsembleTrainer class
  ✗ EnsembleInferencer class
  ✗ Integration in main.py
  ✗ Unit tests for ensemble

CONFIG:
  ✗ config_mao_et_al_2025.yaml template
  ✗ Validation function
  ✗ Config tests

VALIDATION:
  ✗ Unit tests (test_mao_alignment.py)
  ✗ Integration tests (test_mao_integration.py)
  ✗ Single-day dry-run (real data)
  ✗ Multi-day validation
  ✗ Metrics comparison

FINAL:
  ✗ Full retraining (all days)
  ✗ GIM gridding/output
  ✗ Results archival
  ✗ Documentation & paper
```

---

## Summary

This plan provides a **practical, incremental path** to align your daily VTEC models with Mao et al., 2025:

1. **Reuse existing infrastructure** (feature registry, SH code, transforms)
2. **Extend modularly** (add new model class, loss, ensemble trainer)
3. **Test continuously** (unit → integration → real data)
4. **Validate at each checkpoint** before moving to next phase
5. **Document clearly** for reproducibility

**Total estimated effort:** ~7–10 days of focused development for a single developer, with parallelization possible.

**Next step:** Start with Part 1 (Features), which is the foundation for everything else.

