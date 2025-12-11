# GeomNet Architecture Improvements

## Problem Identified

The original GeomNet architecture was **suppressing latitude and azimuth dependencies** due to its mapping factor formula:

```python
# OLD (problematic):
mf = 1.0 + g * F.softplus(mf_raw)
# where g = 1.0 - sin(elevation)
```

**Why this was problematic:**
1. The `g` term (elevation-dependent scaling) **completely dominates** the output
2. Even if the backbone network learns latitude/azimuth patterns in `mf_raw`, they get scaled by `g`
3. `softplus` restricts `mf_raw` to positive values only, limiting expressiveness
4. Result: **Zero latitude variation** and **negligible azimuth variation** in predictions

## Physical Expectations

The mapping factor **should** depend on:

1. **Elevation** (primary): MF ≈ 1/sin(elevation) from thin-shell approximation
   - MF = 1.0 at zenith (90°)
   - MF increases as elevation decreases
   
2. **Station Latitude** (secondary): Due to:
   - Ionospheric shell height variations with latitude
   - Geomagnetic field geometry (especially in solar magnetic coordinates)
   - Different ionospheric structures at equator vs auroral zones (±60-70°)
   
3. **Azimuth** (tertiary): Should be minimal for thin-shell approximation
   - But could show small corrections for asymmetries
   - Model should learn this is negligible, not be forced to ignore it

## Solution Implemented

### 1. Improved GeomNet Architecture

```python
# NEW (improved):
mf = 1.0 + g_elev * (1.0 + torch.tanh(mf_raw))
# where g_elev = 1.0 - sin(elevation)
```

**Key improvements:**
- `tanh(mf_raw)` allows **both positive and negative corrections** in range [-1, 1]
- Formula becomes: `MF = 1 + g_elev * (1 + correction)`
- At 90° elevation: `g_elev = 0` → `MF = 1` ✓ (constraint preserved)
- At low elevation: `g_elev → 1` → `MF ≈ 2 + correction` (baseline + learned adjustments)
- The learned `mf_raw` can now express **additive corrections** for latitude/azimuth effects

**Why this works better:**
- Baseline behavior: `MF ≈ 1 + g_elev` gives ~1/sin(elev) approximation
- Learned corrections: `tanh(mf_raw)` adds latitude/azimuth-dependent adjustments
- The additive structure allows corrections to persist instead of being washed out

### 2. Fixed Validation Tests

**Old validation (incorrect):**
- Varied **IPP latitude** → affects VTECFieldNet, NOT GeomNet
- GeomNet only sees **station latitude** which was held constant
- Result: Always showed zero latitude dependence

**New validation (correct):**
- Varies **station latitude** → directly affects GeomNet input
- Tests azimuth dependence by varying azimuth angle
- Properly tests what GeomNet actually receives:
  - Station location features (lat_sta, lon_sta, sm_lat_sta, sm_lon_sta)
  - Direction vector (e_up, e_east, e_north) - encodes both elevation AND azimuth
  - Station SH embeddings

### 3. Updated Validation Script

The `validate_factorized_model.py` script now supports:

```python
create_test_samples_from_observations(
    observations, 
    elevation_angles,
    config,
    feature_splitter,
    station_latitudes=[... ],  # NEW: vary station latitude
    azimuth_angles_deg=[...]   # NEW: vary azimuth
)
```

The 2D analysis now correctly shows **elevation × station latitude** heatmaps.

## What Features Go Where?

Understanding the architecture:

```
Input Features → FeatureSplitter → Two Networks
                                    
VTEC features:                     GeomNet features:
- Temporal (year, doy, sod, LT)   - Station location (lat_sta, lon_sta, sm_*)
- IPP location (lat_ipp, lon_ipp) - Direction vector (e_up, e_east, e_north)
                                    * e_up = sin(elev)
- IPP SH embeddings                * e_east = cos(elev)*sin(azim)
- Space Weather Indices (SWI)      * e_north = cos(elev)*cos(azim)
                                   - Station SH embeddings
        ↓                                  ↓
   VTECFieldNet                        GeomNet
   (predicts VTEC ± σ)              (predicts MF)
        ↓                                  ↓
        └───────── STEC = MF × VTEC ──────┘
```

**Key insight:** 
- GeomNet receives **azimuth information** through the direction vector components (e_east, e_north)
- The old architecture just wasn't allowing it to use that information effectively

## Expected Results After Retraining

With the improved architecture, you should see:

1. **Elevation dependence**: ✓ (already working, slight improvement in accuracy)
   - MF decreases from ~1.5-2.0 at 10° to 1.0 at 90°
   
2. **Station latitude dependence**: 🆕 (now possible)
   - MF may vary by ~5-15% between equator and poles
   - Stronger variation near auroral zones (60-70°)
   
3. **Azimuth dependence**: 🆕 (now testable)
   - Should remain small (< 5%) for physical correctness
   - If larger, indicates the model learned important asymmetries

## Next Steps

1. **Retrain the model** with the improved GeomNet architecture:
   ```bash
   python src/main.py  # Uses updated model.py automatically
   ```

2. **Run validation** to see the improvements:
   ```bash
   python scripts/validate_factorized_model.py \
     --exp_path experiments/[new_experiment] \
     --num_angles 9 --num_observations 50
   ```

3. **Check the plots**:
   - `factorized_validation_elevation_response.png` - elevation dependence
   - `geomnet_2d_analysis_elevation_station_latitude.png` - NEW: latitude heatmap
   - `geomnet_azimuth_analysis_elev30.png` - azimuth dependence

## Important Notes

⚠️ **The currently trained model still uses the OLD architecture (softplus)**

To see the improvements:
1. Code changes are already in place in `src/model/model.py`
2. You need to **retrain** to get a model using the new architecture
3. The validation script is ready to test the new capabilities

## Files Modified

1. `/scratch2/arrueegg/WP4/PNN_STEC/src/model/model.py`
   - Updated `GeomNet.forward()` to use tanh-based corrections
   
2. `/scratch2/arrueegg/WP4/PNN_STEC/scripts/validate_factorized_model.py`
   - Updated `create_test_samples_from_observations()` to vary station_lat and azimuth
   - Updated `analyze_geom_net_2d()` to test station latitude (not IPP)
   - Existing azimuth analysis already correct

3. `/scratch2/arrueegg/WP4/PNN_STEC/scripts/test_geomnet_dependencies.py` (NEW)
   - Quick test script template for dependency validation
