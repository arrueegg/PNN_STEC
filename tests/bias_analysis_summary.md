# STEC Prediction Bias Analysis - Positioning Impact

## Executive Summary

The model shows **systematic biases** that significantly impact positioning quality. Stations with **GPS+Galileo** and moderate bias perform well, while **Galileo-only** stations or stations with large bias show poor positioning.

## Station Comparison

| Station | Lat     | Bias (TECU) | RMSE (TECU) | #Sats | Constellations | Positioning Quality |
|---------|---------|-------------|-------------|-------|----------------|---------------------|
| CHPG    | -22.68° | **+2.79**   | 7.71        | 23    | Galileo only   | ❌ **BAD**          |
| NNOR    | -31.05° | -2.11       | 3.11        | 57    | GPS + Galileo  | ✅ **GOOD**         |
| POAL    | -30.07° | -2.43       | 9.16        | 54    | GPS + Galileo  | ✅ **GOOD**         |
| SOLO    | -9.43°  | **-4.09**   | 9.86        | 25    | Galileo only   | ❌ **BAD**          |

## Key Findings

### 1. Bias is the Primary Issue, Not Random Error

**CHPG vs NNOR Comparison:**
- CHPG: RMSE=7.71 TECU, but **+2.79 TECU systematic bias**
- NNOR: RMSE=3.11 TECU, with -2.11 TECU bias
- Both have "acceptable" RMSE, but CHPG fails positioning due to **positive bias direction**

### 2. Geometry Analysis Shows Consistent Bias Across Elevations

**CHPG (Bad Positioning):**
- 10-20°: bias=+2.96 TECU
- 20-30°: bias=+2.93 TECU
- 30-45°: bias=+2.95 TECU
- 45-90°: bias=+2.05 TECU
- **Systematic over-prediction at ALL elevations**

**NNOR (Good Positioning):**
- 10-20°: bias=-2.75 TECU
- 20-30°: bias=-2.24 TECU
- 30-45°: bias=-2.06 TECU
- 45-90°: bias=-1.88 TECU
- **Consistent under-prediction, but GPS helps compensate**

### 3. Constellation Diversity Matters

**Stations with GPS + Galileo (NNOR, POAL):**
- 54-57 satellites total
- Better geometry → bias averages out across constellations
- **Good positioning despite moderate bias**

**Galileo-only Stations (CHPG, SOLO):**
- 23-25 satellites total
- Bias affects ALL satellites in same direction
- **Poor positioning due to systematic error accumulation**

### 4. Geographic/Ionospheric Effects

**SOLO (Equatorial, -9.43°):**
- Highest mean STEC: **70.90 TECU** (vs ~45 TECU for others)
- Largest absolute bias: **-4.09 TECU**
- Model struggles with equatorial anomaly region

**CHPG (Mid-latitude, -22.68°):**
- Only station with **positive bias** (+2.79 TECU)
- Highest STEC variance (std=42.38 TECU)
- Model over-compensates for ionospheric variability

## Why Bias Breaks Positioning

### Mathematical Impact

For positioning, the ionospheric correction is applied as:
```
pseudorange_corrected = pseudorange_raw - STEC_correction
```

A systematic bias means:
1. **All satellites** are corrected by the same offset
2. **No geometric averaging** can cancel the error
3. The position solution shifts systematically (typically vertical component)

### Example: CHPG with +2.79 TECU Bias

If CHPG has 20 satellites visible:
- Each pseudorange is over-corrected by ~2.79 × 0.163 = **0.45 meters** (at L1 frequency)
- With 20 satellites, this systematic error **compounds**
- Results in vertical position error of several meters

### Why NNOR Works Despite -2.11 TECU Bias

1. **GPS + Galileo**: 57 satellites with different orbital planes
2. **Better HDOP/VDOP**: More geometric diversity
3. **Constellation mixing**: GPS and Galileo biases may partially cancel
4. **Smaller RMSE**: Random errors are much lower (3.11 vs 7.71 TECU)

## Recommendations

### Short-term Solutions

1. **De-bias predictions before positioning:**
   ```python
   # Station-specific bias correction
   bias_corrections = {
       'CHPG': -2.79,  # Remove positive bias
       'SOLO': +4.09,  # Remove negative bias
       'NNOR': +2.11,  # Optional: remove bias
       'POAL': +2.43,  # Optional: remove bias
   }
   stec_corrected = stec_predicted + bias_corrections[station]
   ```

2. **Weight by uncertainty**: Use the predicted uncertainty to down-weight high-bias observations

3. **Constellation-aware filtering**: If GPS available, prioritize GPS satellites for positioning

### Medium-term Solutions

1. **Retrain with bias-aware loss:**
   - Add bias penalty term to loss function
   - Weight by positioning impact (elevation-dependent)

2. **Post-processing calibration:**
   - Use validation set to learn station-specific bias corrections
   - Apply polynomial correction as function of STEC magnitude

3. **Investigate feature importance:**
   - Why does CHPG have opposite bias sign?
   - Missing features for equatorial anomaly (SOLO)?

### Long-term Solutions

1. **Multi-task learning:**
   - Train jointly on STEC prediction AND positioning error
   - Direct positioning loss in addition to STEC loss

2. **Ensemble methods:**
   - Combine predictions from multiple models
   - May reduce systematic bias through averaging

3. **Geographic stratification:**
   - Train separate models for equatorial vs mid-latitude regions
   - Or add latitude-based features/embeddings

## Validation Checklist

To verify positioning quality for new stations:

1. ✅ Check **bias magnitude**: |bias| < 2.5 TECU preferred
2. ✅ Check **bias sign**: Negative bias generally safer than positive
3. ✅ Check **constellation diversity**: GPS+Galileo much better than single constellation
4. ✅ Check **number of satellites**: >40 satellites preferred
5. ✅ Check **RMSE**: Lower is better, but secondary to bias
6. ✅ Check **geographic location**: Equatorial regions are challenging

## Files Generated

- `tests/comparison_results_casdcb/` - CHPG detailed analysis
- `tests/comparison_results_NNOR/` - NNOR detailed analysis  
- `tests/comparison_results_POAL/` - POAL detailed analysis
- `tests/comparison_results_SOLO/` - SOLO detailed analysis

Each directory contains:
- `comparison_plots.png` - Visualization of predictions vs truth
- `detailed_results.csv` - Per-observation predictions and errors
- `metrics.txt` - Summary statistics
