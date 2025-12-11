# Evaluation Framework Implementation Tasks

This document contains a structured breakdown of tasks to build a comprehensive evaluation framework for the PNN_STEC project. Each task is self-contained and can be copy-pasted as a prompt.

**Important Guidelines:**
- Reuse existing code patterns from the codebase wherever possible
- Do NOT delete existing functionality unless explicitly obsolete
- Follow existing conventions (FeatureRegistry, config patterns, model loading)
- Each task builds incrementally on the infrastructure

---

## Task 1: Basic Test Set Evaluation Script

**Context:** The project currently has inference scripts for positioning evaluation (`src/inference_positioning.py`) and validation scripts for specific models (`scripts/validate_factorized_model.py`). We need a general-purpose test set evaluation script that works for all model types.

**Existing Infrastructure to Reuse:**
- `src/inference_positioning.py` lines 40-130: `initialize_output_indices_for_registry()` function for feature registry setup
- `src/utils/config_parser.py`: `load_config()` function to load experiment config
- `src/model/model.py`: `get_model()` function to instantiate models
- `src/data_loader/datasets.py`: `H5Dataset` class for loading test.h5
- `src/data_loader/collation.py`: `CollateWithSH` class for feature transformation
- `src/utils/metrics.py`: Existing metric functions (rmse, mae, mse, r2_score, mape)
- `src/training/base_trainer.py`: BaseTrainer class with `test_model()` method

**Task:**
Create a test set evaluation script at `scripts/evaluate_test_set.py` that:
1. Loads experiment config using `load_config()` from config_parser
2. Initializes FeatureRegistry using `initialize_feature_registry(config)`
3. Loads trained model checkpoint (search for `checkpoint_seed*.pth` in experiment folder)
4. Creates H5Dataset for test.h5 and DataLoader with CollateWithSH
5. Runs inference using model in eval mode with torch.no_grad()
6. Computes metrics using existing functions from `utils.metrics`: RMSE, MAE, R², bias (mean residual)
7. For uncertainty models: compute empirical coverage at 68%, 90%, 95% levels
8. Saves results to `experiment_folder/test_metrics.csv`
9. Prints formatted summary table to console

**Requirements:**
- Use argparse for CLI: `--experiment_folder` (required), `--seed` (default: 42)
- Model loading: Search for checkpoint file matching pattern `checkpoint_seed{seed}.pth`
- Handle model types: Check `config['model']['model_type']` to distinguish FactorizedSTEC vs others
- All models return (mean, variance) tuple - use variance for coverage if variance > 0
- Batch inference efficiently (use DataLoader batch_size from config)
- Include progress bar (tqdm) for test set inference
- Do NOT modify existing metric functions - import and use as-is

---

## Task 2: Stratified Metrics by Elevation

**Context:** Ionospheric STEC predictions degrade at low elevations due to longer ray paths and higher mapping function sensitivity. The existing validation script (`scripts/validate_factorized_model.py`) analyzes elevation response but doesn't compute stratified test metrics.

**Existing Infrastructure to Reuse:**
- Task 1 script (`scripts/evaluate_test_set.py`): Main evaluation infrastructure
- `scripts/validate_factorized_model.py`: Plotting style and figure layout conventions
- Feature registry output_indices: Use `registry.output_indices['e_up']` to extract elevation from Cartesian direction vectors
- Elevation computation: `elevation = arcsin(e_up)` where e_up is the vertical component of unit direction vector

**Task:**
Extend `scripts/evaluate_test_set.py` (from Task 1) to add stratified analysis by elevation:
- Elevation bins: [0-15°, 15-30°, 30-45°, 45-60°, 60-75°, 75-90°]
- Extract elevation from direction features: elevation_rad = arcsin(e_up), then convert to degrees
- For each bin compute: RMSE, MAE, R², bias, sample count, mean STEC value
- Save to: `experiment_folder/test_metrics_by_elevation.csv`
- Generate bar plot: RMSE vs elevation bin with error bars showing stderr
- Save plot to: `experiment_folder/plots/metrics_by_elevation.png`

**Requirements:**
- Add CLI flag: `--stratify-elevation` to enable this analysis
- Create `plots/` subdirectory if it doesn't exist
- Use matplotlib style consistent with validation scripts (use seaborn if available)
- Handle sparse bins: Print warning if any bin has N < 100 samples
- Preserve ALL existing functionality from Task 1 (don't remove base metrics)
- Extract e_up from feature vector using: `features[:, registry.output_indices['e_up']]`
- Include bin sample counts as text on bar plot for transparency

---

## Task 3: Stratified Metrics by Latitude & Solar Activity

**Context:** STEC varies significantly with latitude (equatorial anomaly, auroral zones) and solar activity (F10.7 index). Current evaluation doesn't stratify by these physical parameters.

**Existing Infrastructure to Reuse:**
- Task 1-2 evaluation framework
- Feature extraction: `registry.output_indices['lat_ipp_norm']` for IPP latitude
- Denormalization: Latitude range is [-90, 90] degrees (use registry.denormalize if available)
- SWI check: `config['data'].get('use_SWI', False)` to determine if F10.7 is available
- F10.7 extraction: `registry.output_indices['f107_index_norm']` from SWI features

**Task:**
Add two stratification dimensions to `scripts/evaluate_test_set.py`:

1. **Latitude stratification:**
   - Bins: [-90, -60], [-60, -30], [-30, 0], [0, 30], [30, 60], [60, 90] degrees
   - Extract IPP latitude from normalized features and denormalize to degrees
   - Compute RMSE/MAE/R²/bias/count per bin
   - Save to: `experiment_folder/test_metrics_by_latitude.csv`
   - Plot: bar chart of RMSE vs latitude bin (6 bars)

2. **Solar activity stratification:**
   - Only if `config['data']['use_SWI']` is True
   - Extract F10.7 from SWI features (check feature_control for 'f107_index' enabled)
   - Denormalize F10.7 to solar flux units (sfu)
   - Bins: Low (<100 sfu), Medium (100-150 sfu), High (>150 sfu)
   - Compute RMSE/MAE/R²/bias/count per bin
   - Save to: `experiment_folder/test_metrics_by_solar_activity.csv`
   - Plot: bar chart with 3 bars

**Requirements:**
- Add CLI flags: `--stratify-latitude`, `--stratify-solar` (independent toggles)
- If `--stratify-solar` but SWI not enabled: print info message and skip gracefully
- Extract from transformed feature vector (after CollateWithSH normalization)
- Denormalize using registry min/max values for interpretable bin boundaries
- Include sample count annotations on all bar plots
- Preserve ALL existing functionality (don't break Tasks 1-2)

---

## Task 4: Uncertainty Calibration Metrics

**Context:** Bayesian models (BNN, NLL, FactorizedSTEC) output predictive uncertainty as variance. We need to verify calibration quality - well-calibrated models have empirical coverage matching nominal confidence levels.

**Existing Infrastructure to Reuse:**
- All models return `(mean, variance)` tuple - check if `variance.mean() > 1e-6` for uncertainty-aware models
- Model types with uncertainty: BNN_NLL, ResNet_NLL, ResNet_BNN_NLL, BayesianResNetSTEC, FactorizedSTEC, AttentionMLP_NLL, Branch*_NLL
- Model types without: *_MSE models return zero variance
- Z-scores for confidence levels: 0.5→0.674, 0.68→1.0, 0.90→1.645, 0.95→1.96, 0.99→2.576

**Task:**
Add uncertainty calibration analysis to `scripts/evaluate_test_set.py`:
1. Check if model has meaningful variance: `var_mean = variance.mean().item()`; skip if < 1e-6
2. Compute prediction intervals for confidence levels [0.5, 0.68, 0.90, 0.95, 0.99]:
   - std = sqrt(variance)
   - interval = [mean - z*std, mean + z*std]
3. Calculate empirical coverage: fraction of true values within each interval
4. Compute calibration error: absolute difference between empirical and nominal coverage
5. Compute sharpness: mean interval width at 95% level (measure of uncertainty specificity)
6. Generate reliability diagram:
   - Scatter: empirical vs nominal coverage points
   - Reference: diagonal line (perfect calibration)
   - Annotate: calibration error for each point
7. Save metrics table to: `experiment_folder/test_uncertainty_calibration.csv`
8. Save plot to: `experiment_folder/plots/uncertainty_calibration.png`

**Requirements:**
- Add CLI flag: `--calibration-analysis`
- Auto-detect uncertainty capability from variance values (don't rely on model_type string)
- If no uncertainty: print "Model outputs deterministic predictions (zero variance), skipping calibration"
- Use scipy.stats.norm.ppf() for z-scores or hardcode values above
- CSV columns: [confidence_level, z_score, empirical_coverage, calibration_error, interval_width]
- Preserve ALL existing functionality (Tasks 1-3)

---

## Task 5: Baseline VTEC Model Training Script

**Context:** FactorizedSTEC (128x4 VTECNet + 32x2 GeomNet) learns both VTEC field and geometry-dependent corrections. To demonstrate this value, we need a baseline predicting only VTEC with standard thin-shell MF.

**Existing Infrastructure to Reuse:**
- `src/model/model.py`: Use existing `MLP_NLL` class (lines ~200-300) for VTEC predictor with uncertainty
- Alternative: Use `MLP_MSE` for deterministic baseline
- `src/training/base_trainer.py`: BaseTrainer class with full training loop (lines 200-535)
- `src/pretrain.py`: Pretrainer class structure (simple wrapper around BaseTrainer)
- `src/data_loader/datasets.py`: H5Dataset class already supports target='vtec' mode
- `config/config.yaml`: Copy and modify - change target from 'stec' to 'vtec'

**Key Changes Needed:**
1. Config: Set `data.target: vtec` instead of 'stec'
2. Config: Remove direction features (satazi, satele) - not needed for VTEC
3. Config: Keep temporal + IPP coordinates + SWI (if enabled)
4. Data loader: H5Dataset already computes VTEC from STEC using zenith angle when target='vtec'

**Task:**
Create `scripts/train_vtec_baseline.py` that:
1. Copies main.py structure but forces target='vtec'
2. Uses MLP_NLL architecture matching FactorizedSTEC VTECNet dimensions:
   - hidden_dim: 128 (same as vtec_hidden in config)
   - num_layers: 4 (same as vtec_layers)
   - Inputs: temporal (year, doy, sod, local_time) + IPP (lat, lon, sm_lat, sm_lon) + SH + SWI
3. Creates `config/config_vtec_baseline.yaml`:
   - Copy config.yaml
   - Set `data.target: "vtec"`
   - Set `model.model_type: "MLP_NLL"`
   - Set `feature_control.satazi: false` and `satele: false` (direction not needed)
   - Update experiment naming to include "VTEC" prefix
4. Training uses existing Pretrainer class (just import and run)
5. Saves to: `experiments/Pretrain_VTEC_MLP_NLL_h128_l4_<params>/checkpoint_seed{seed}.pth`

**Requirements:**
- Do NOT modify src/pretrain.py or src/main.py - create standalone script
- Reuse ALL existing infrastructure (dataset, trainer, model classes)
- H5Dataset already handles target='vtec' - it computes VTEC = STEC * cos(zenith) automatically
- Config must disable direction features since VTEC doesn't depend on viewing geometry
- Use same training hyperparameters as FactorizedSTEC for fair comparison

---

## Task 6: VTEC Baseline Inference with Standard MF

**Context:** After training VTEC baseline (Task 5), evaluate on test set. Baseline predicts VTEC → convert to STEC using thin-shell mapping function for comparison with FactorizedSTEC (which predicts STEC directly).

**Existing Infrastructure to Reuse:**
- Task 1 evaluation framework (`scripts/evaluate_test_set.py`)
- Thin-shell MF formula: `MF = 1 / sqrt(1 - (R_E/(R_E + h_ion) * cos(elevation))^2)`
  - R_E = 6371 km (Earth radius)
  - h_ion = 450 km (ionospheric shell height, typical value)
  - Simplified approximation: `MF ≈ 1 / cos(zenith)` where zenith = 90° - elevation
- Elevation extraction: Same as Task 2 - from direction vector e_up

**Task:**
Create `scripts/evaluate_vtec_baseline.py` that:
1. Loads VTEC baseline model from experiment folder
2. Verifies target type: check `config['data']['target'] == 'vtec'` (error if 'stec')
3. Loads test.h5 using existing H5Dataset + DataLoader infrastructure
4. Runs inference to predict VTEC (model outputs VTEC_mean, VTEC_variance)
5. Extracts elevation from direction features: `elevation_rad = arcsin(e_up)`
6. Computes thin-shell MF: `MF = 1 / cos(pi/2 - elevation_rad)` (vectorized)
7. Converts to STEC: `STEC = VTEC * MF` (element-wise multiplication)
8. Propagates uncertainty: `STEC_var = VTEC_var * MF^2` (variance transformation)
9. Computes same metrics as Task 1 using converted STEC values
10. Saves to: `experiment_folder/vtec_baseline_test_metrics.csv`

**Requirements:**
- Reuse metric computation and stratification code from Task 1
- Support all Task 1 flags: `--stratify-elevation`, `--stratify-latitude`, `--calibration-analysis`
- Warn for low elevations (<5°): MF becomes very large (>10x), print sample count
- Include MF statistics in output: mean, min, max, std across test set
- CSV should include row: 'mapping_function_mean', 'mapping_function_std' for traceability

---

## Task 7: IGS GIM Data Download and Processing

**Context:** International GNSS Service (IGS) provides Global Ionosphere Maps (GIMs) in IONEX format as the ionospheric community standard. Need to download and process for comparison.

**Data Specifications:**
- Product: CODE final GIM (Center for Orbit Determination in Europe)
- URL: `https://cddis.nasa.gov/archive/gnss/products/ionex/{YYYY}/{DDD}/`
- Filename: `codg{DDD}0.{YY}i.Z` (e.g., codg1830.24i.Z for 2024 DOY 183)
- Format: IONEX (IONosphere Map EXchange) - ASCII text with VTEC grid
- Grid: 2.5° lat × 5° lon resolution, 87.5°N to -87.5°S, 2-hour temporal resolution
- Alternative mirrors: ESA (ftp://igs-final.man.olsztyn.pl/pub/), IGS

**Existing Infrastructure:**
- Test dates: Read from `data/test.h5` timestamps (column: 'year', 'doy')
- HDF5 storage pattern: Similar to SWI data in `data/omni_hourly_2010-2025.h5`

**Task:**
Create `scripts/download_igs_gim.py` that:
1. Reads test.h5 to determine date range:
   - `with h5py.File('data/test.h5') as f: dates = f['data'][['year', 'doy']]`
   - Get unique (year, doy) pairs
2. Downloads CODE GIM for each date:
   - Construct URL with YYYY (4-digit year) and DDD (3-digit DOY)
   - Download `.Z` compressed file
   - Uncompress using `unlzw` or Python `lzma` module
3. Parses IONEX files:
   - Read header: grid dimensions, lat/lon arrays, epoch times
   - Extract VTEC maps (skip RMS maps if present)
   - VTEC units: 0.1 TECU in file → convert to TECU
4. Saves to HDF5: `data/igs_gim_YYYY_DDD.h5` with structure:
   ```
   /vtec_grid: (n_epochs, n_lat, n_lon) float32
   /lat_grid: (n_lat,) float32 [-87.5 to 87.5]
   /lon_grid: (n_lon,) float32 [-180 to 180]
   /times: (n_epochs,) float64 [hours since midnight UTC]
   ```
5. Handles errors:
   - Missing files: log warning, continue (some dates may not have final products yet)
   - Parse errors: skip file, log error message

**Requirements:**
- CLI args: `--start-date YYYY-MM-DD`, `--end-date YYYY-MM-DD`, `--output-dir` (default: data/)
- Use `requests` library for downloads with timeout (30s) and retries (3)
- Progress bar: tqdm for multi-day downloads
- IONEX parser: Use `georinex` library if available, else custom regex parser
- Verify grid: Check lat/lon dimensions match expected (71 lat × 73 lon typically)

---

## Task 8: IGS GIM Interpolation and Comparison

**Context:** IGS GIMs provide VTEC on 2.5° lat × 5° lon grid at 2-hour intervals. Must interpolate to arbitrary IPP locations for comparison.

**Existing Infrastructure to Reuse:**
- Task 1 evaluation framework and metric computation
- Task 6 MF computation for VTEC → STEC conversion
- `scipy.interpolate.RegularGridInterpolator` for spatial interpolation
- Feature extraction: `lat_ipp`, `lon_ipp` from test.h5

**Task:**
Create `scripts/evaluate_igs_gim.py` that:
1. Loads test.h5 with timestamps and IPP coordinates
2. For each test sample:
   - Extract: year, doy, hour, lat_ipp, lon_ipp, elevation
   - Load corresponding GIM file: `data/igs_gim_{year}_{doy:03d}.h5`
   - Find temporal neighbors: GIM epochs before/after observation hour
   - Spatial interpolation: Use RegularGridInterpolator(lat_grid, lon_grid, vtec_map)
   - Temporal interpolation: Linear between two nearest epochs
   - Result: GIM_VTEC at (lat_ipp, lon_ipp, time)
3. Convert GIM VTEC → STEC:
   - Use same thin-shell MF as Task 6: `MF = 1/cos(zenith)`
   - `GIM_STEC = GIM_VTEC * MF`
4. Computes residuals: `residual = STEC_true - GIM_STEC`
5. Computes metrics: RMSE, MAE, R², bias (using Task 1 functions)
6. Saves to: `experiments/igs_gim_comparison/test_metrics.csv`
7. Optionally runs stratifications (elevation, latitude) if flags provided

**Requirements:**
- Handle missing GIM files: Skip samples, log count of missing coverage
- Handle out-of-bounds: GIMs cover ±87.5° latitude only - skip high-latitude IPPs, log count
- Interpolation order: 'linear' for both spatial (bilinear) and temporal
- Wrap longitude: Handle ±180° discontinuity (GIM may use [0, 360] or [-180, 180])
- Performance: Load each GIM file once, cache in memory for batch processing
- CLI args: `--gim-data-dir` (default: data/), plus stratification flags from Task 1
- Log interpolation stats: samples/second, cache hit rate, missing data percentage

---

## Task 9: Model Comparison Summary Script

**Context:** We have evaluation metrics from multiple models/baselines (FactorizedSTEC, VTEC baseline, IGS GIM) in separate CSVs. Need unified comparison visualization.

**Existing Infrastructure to Reuse:**
- CSV format from Task 1: columns include RMSE, MAE, R2, bias, coverage metrics
- Plotting style from `scripts/validate_factorized_model.py`: matplotlib + seaborn
- Experiment name parsing: Extract model type from folder name

**Task:**
Create `scripts/compare_models.py` that:
1. Accepts multiple experiment folders as CLI arguments
2. For each folder:
   - Reads `test_metrics.csv` (overall metrics)
   - Reads `test_metrics_by_elevation.csv` (if exists)
   - Extracts model name from folder path (e.g., "FactorizedSTEC", "VTEC_baseline", "IGS_GIM")
3. Creates comparison table:
   - Rows: One per model
   - Columns: [Model, RMSE, MAE, R², Bias, Std, Coverage@95%, N_samples]
   - Sort by RMSE (ascending - best first)
4. Generates comparison plots:
   - **Plot 1 - Overall metrics bar chart**: Side-by-side bars for RMSE, MAE across models
   - **Plot 2 - Elevation profile**: Line plot of RMSE vs elevation bin, one line per model
   - **Plot 3 - Scatter comparison**: Prediction vs True for all models (subplot per model, shared axes)
   - **Plot 4 - Residual distributions**: Histograms of residuals overlaid with transparency
5. Statistical testing:
   - Paired t-test between each model pair (requires raw predictions, not just metrics)
   - Report p-values in comparison table (add columns: p_vs_baseline)
6. Saves outputs:
   - Table: `plots/model_comparison_table.csv` (CSV) and `*_table.tex` (LaTeX)
   - Plots: `plots/model_comparison_overall.png`, `*_elevation.png`, `*_scatter.png`, `*_residuals.png`

**Requirements:**
- CLI: `--experiments exp1 exp2 exp3 ...` (variable number), `--output-dir` (default: plots/)
- Handle missing metrics: If elevation stratification not run, skip elevation plot
- Statistical tests: Only if raw predictions available (may require re-loading test results)
- LaTeX table: Use booktabs format with \toprule, \midrule, \bottomrule
- Figure size: Use 10×6 inch for multi-panel plots, 300 DPI for publication quality
- Do NOT modify individual task scripts - only read their outputs

---

## Task 10: Residual Spatial Maps Visualization

**Context:** STEC errors may show geographic patterns (equatorial anomaly, auroral zones, ocean vs land). Need spatial visualization of error distribution.

**Existing Infrastructure to Reuse:**
- Task 1 prediction loading
- IPP coordinate extraction: `lat_ipp`, `lon_ipp` from feature registry
- Plotting: Use `cartopy` if available, else matplotlib with basemap-style plotting

**Task:**
Create `scripts/plot_residual_maps.py` that:
1. Loads test predictions and true values from experiment folder:
   - Option A: Re-run inference and save raw outputs
   - Option B: Load from cached predictions (if Task 1 saves them)
2. Extracts IPP coordinates (lat, lon) from test.h5 using FeatureRegistry
3. Computes residuals: `residual = true - predicted`
4. Bins data by IPP location:
   - Default grid: 2° lat × 5° lon (matches IGS GIM resolution)
   - Latitude bins: -90 to 90, longitude: -180 to 180
5. For each grid cell computes:
   - Mean residual (shows systematic bias: over/underprediction)
   - RMSE (shows error magnitude)
   - Sample count (for statistical reliability)
6. Generates two maps using cartopy:
   - **Map 1 - Mean Residual**: 
     - Colormap: RdBu_r (red=overpredict, blue=underpredict, white=zero)
     - Range: symmetric around zero (e.g., ±5 TECU)
   - **Map 2 - RMSE**:
     - Colormap: YlOrRd (white/yellow=low, red=high)
     - Range: 0 to percentile 95 of RMSE values
7. Map features:
   - Coastlines, country borders (gray, thin)
   - Gridlines every 30° with labels
   - Colorbar with units (TECU)
   - Mask cells with N < 50 samples (hatching pattern)
8. Saves to: `experiment_folder/plots/residual_spatial_maps.png` (two-panel figure)

**Requirements:**
- CLI args: `--experiment-folder`, `--grid-resolution` (default: "2x5" for 2° × 5°)
- Add `--compare-with <exp2>` to show difference map: (residual_exp1 - residual_exp2)
- Use cartopy.crs.PlateCarree() for equirectangular projection
- Figure size: 12×6 inches (side-by-side panels)
- Handle sparse regions gracefully: Print warning if >10% of cells have <50 samples
- Do NOT modify existing data loading - create standalone visualization script

---

## Task 11: Time Series Residual Analysis

**Context:** STEC has strong temporal variations (diurnal cycle, seasonal changes). Need to verify no systematic temporal biases (e.g., consistently worse at dawn or equinoxes).

**Existing Infrastructure to Reuse:**
- Task 1 prediction loading framework
- Temporal features from test.h5: year, doy, sod (seconds of day)
- Local time computation: Already in datasets.py - `compute_local_time_hours(sod, longitude)`

**Task:**
Create `scripts/plot_temporal_residuals.py` that:
1. Loads test predictions, true values, and timestamps
2. Extracts temporal features:
   - UTC hour: `hour_utc = sod // 3600`
   - DOY: already in test.h5
   - Local solar time: Use IPP longitude to compute LST = UTC + lon/15°
3. Computes residuals for each sample
4. Groups and aggregates:
   - **Diurnal**: Group by local solar time hour (0-23)
   - **Seasonal**: Group by DOY (1-366)
   - **Daily time series**: Group by date (year-doy combination)
   - **Hour-season interaction**: 2D bins (24 hours × 12 months)
5. Generates plots:
   - **Plot 1 - Diurnal pattern**: 
     - X: Local solar time (0-24h), Y: Mean residual
     - Error bands: ±1 std (shaded)
     - Highlights: Mark dawn (~6h), noon (~12h), dusk (~18h)
   - **Plot 2 - Seasonal pattern**:
     - X: DOY (1-366), Y: Mean residual
     - Error bands: ±1 std
     - Mark equinoxes (~80, ~265) and solstices (~172, ~355)
   - **Plot 3 - Daily RMSE time series**:
     - X: Date, Y: RMSE
     - Shows temporal trends in prediction quality
   - **Plot 4 - Hour-season heatmap**:
     - X: Month (1-12), Y: Hour (0-23), Color: Mean RMSE
     - Reveals interaction effects (e.g., worse at dawn in summer)
6. Statistical testing:
   - ANOVA: Test if diurnal/seasonal patterns are statistically significant
   - Report F-statistic and p-value on plots
7. Saves to: `experiment_folder/plots/temporal_residual_analysis.png` (4-panel figure)

**Requirements:**
- CLI args: `--experiment-folder`, `--time-resolution` (hourly|daily, default: hourly)
- Use consistent color scheme: Blues for time series, RdYlGn for residuals
- Figure layout: 2×2 grid, size 12×10 inches
- Include sample counts: Annotate sparse time bins (N < 100)
- Do NOT modify data loaders - work with existing test.h5 format

---

## Task 12: Ablation Study Configuration Files

**Context:** To understand which FactorizedSTEC components contribute to performance, need ablation studies. Current `config.yaml` is for full model with all features.

**Existing Infrastructure:**
- Config structure from `config/config.yaml`: nested YAML with sections for data, model, training, finetune
- Experiment naming in `src/utils/config_parser.py`: Automatically generates names from config parameters
- Feature control: `feature_control` section enables/disables individual features
- Model params: `vtec_hidden`, `vtec_layers`, `geom_hidden`, `geom_layers`, `SH_degree`, `use_SWI`

**Task:**
Create configuration files for ablation studies in `config/`:

**1. `config_factorized_no_swi.yaml`** - Remove space weather inputs:
```yaml
# Copy config.yaml, then modify:
data:
  use_SWI: false  # Disable space weather indices
feature_control:
  # Set all SWI features to false:
  Kp_index: false
  R_Sunspot_No: false
  Dst-index,_nT: false
  AE-index,_nT: false
  ap_index,_nT: false
  f107_index: false
# All other parameters identical
```
Purpose: Assess value of space weather context vs purely geometric/temporal features

**2. `config_factorized_sh3.yaml`** - Lower spherical harmonic order:
```yaml
data:
  SH_degree: 3  # Reduced from 5
# Reduces feature dimensionality: 9 SH basis functions vs 25
```
Purpose: Test if high-order spherical harmonics improve spatial representation

**3. `config_factorized_smaller_vtec.yaml`** - Reduced VTEC network capacity:
```yaml
model:
  vtec_hidden: 64   # Reduced from 128
  vtec_layers: 3     # Reduced from 4
  # geom_hidden and geom_layers unchanged
```
Purpose: Determine if VTECNet capacity is oversized (test for overparameterization)

**4. `config_factorized_no_geomnet.yaml`** - Standard MF baseline:
```yaml
model:
  geom_hidden: 0  # Disables GeomNet
  # OR add new flag:
  use_learned_mf: false  # Falls back to thin-shell MF
```
Purpose: Isolate GeomNet contribution - does learned MF outperform thin-shell?
**Note:** May require code modification in `model.py` to handle geom_hidden=0

**Requirements:**
- Each config: Add header comment block explaining ablation purpose and expected impact
- Maintain identical: random_seed, batch_size, epochs, optimizer, learning_rate
- Experiment naming: Will auto-generate with ablation suffix (e.g., "_noSWI", "_SH3")
- Validate: Test that config_parser.py loads each config without errors
- Do NOT modify config.yaml - keep as baseline reference
- Place in `config/` directory alongside existing sweep configs

---

## Task 13: Batch Training Script for Ablations

**Context:** Running ablation studies requires training multiple models sequentially. Manual execution is error-prone. Need automated batch script.

**Existing Infrastructure to Reuse:**
- Training entry point: `python src/main.py` (reads config from config/config.yaml by default)
- Config specification: Can use `--config <path>` flag if main.py supports it, or temporarily copy config
- Checkpoint pattern: Experiments save to `experiments/Pretrain_STEC_<model>_<params>/checkpoint_seed*.pth`
- Exit codes: Training returns 0 on success, non-zero on failure

**Task:**
Create `scripts/run_ablation_studies.sh` that:
1. Defines ablation config array:
```bash
ABLATIONS=(
  "config_factorized_no_swi"
  "config_factorized_sh3"
  "config_factorized_smaller_vtec"
  "config_factorized_no_geomnet"
)
```
2. Creates logs directory: `mkdir -p logs/`
3. For each ablation config:
   - Prints header: `echo "=== Training Ablation: $ablation_name ==="` with timestamp
   - Temporarily copies to config.yaml: `cp config/$config_file.yaml config/config.yaml`
   - Runs training: `python src/main.py 2>&1 | tee logs/ablation_${ablation_name}_${timestamp}.log`
   - Captures exit code: `exit_code=$?`
   - Restores original config: `git checkout config/config.yaml` (or keep backup)
   - Logs result: Success/failure with exit code
4. Estimates time:
   - Record start time of each ablation
   - Compute average training time after first ablation
   - Print: "Completed 2/4, estimated 3.5h remaining"
5. Final summary:
   - Print table: Ablation name, Exit code, Duration, Output folder
   - Print: "All ablations complete. Results in experiments/"

**Requirements:**
- Use `trap` for cleanup on Ctrl+C: `trap 'echo "Interrupted"; exit 1' INT TERM`
- Resumable: Check if experiment folder exists with `find experiments/ -name "*${ablation_tag}*"`
- Add `--skip-existing` flag: If experiment folder exists, skip training
- GPU memory monitoring: Log `nvidia-smi` output before each training
- Timestamp format: `$(date +%Y%m%d_%H%M%S)` for log filenames
- Do NOT modify src/main.py - use it as-is

---

## Task 14: Ablation Results Aggregation

**Context:** After training ablations (Task 13), metrics are scattered across experiment folders. Need to aggregate, compute deltas vs baseline, and assess component importance.

**Existing Infrastructure to Reuse:**
- Task 1 evaluation script: `scripts/evaluate_test_set.py`
- CSV parsing: pandas.read_csv() for metrics files
- Experiment discovery: `glob.glob('experiments/Pretrain_STEC_FactorizedSTEC*')` with pattern matching
- Statistical testing: scipy.stats.ttest_rel() for paired t-test

**Task:**
Create `scripts/aggregate_ablation_results.py` that:
1. Discovers experiments:
   - Baseline: Find folder matching `Pretrain_STEC_FactorizedSTEC_h1024_l4_nh4_v128x4_g32x2_*` (full config)
   - Ablations: Pattern match on modified parameters (e.g., `*_noSWI`, `*_SH3`, `*v64x3*`, `*g0x*`)
2. Runs evaluation on each (if not already done):
   - Check if `test_metrics.csv` exists in experiment folder
   - If not: `subprocess.run(['python', 'scripts/evaluate_test_set.py', '--experiment_folder', exp_path])`
   - Wait for completion, check return code
3. Loads and parses metrics:
   - Read CSV: `df = pd.read_csv(f'{exp_path}/test_metrics.csv')`
   - Extract: RMSE, MAE, R², bias from first row
4. Computes delta metrics vs baseline:
   - ΔRMSE = RMSE_ablation - RMSE_baseline (negative = improvement, positive = degradation)
   - Same for MAE, R²
5. Creates summary table:
   - Columns: [Ablation, RMSE, ΔRMSE, ΔRMSE%, MAE, ΔMAE, R², ΔR², p_value, Interpretation]
   - Sort by |ΔRMSE| descending (biggest impact first)
6. Statistical testing (if raw predictions available):
   - Load prediction arrays from both experiments
   - Paired t-test: `t, p = ttest_rel(residuals_baseline, residuals_ablation)`
   - Mark significant: p < 0.05 with asterisk (*)
7. Generates ablation bar chart:
   - Horizontal bars: Ablation name on Y-axis, ΔRMSE on X-axis
   - Color: Green (negative Δ = improvement shouldn't happen), Red (positive = degradation expected)
   - Sort by absolute impact
8. Saves outputs:
   - `experiments/ablation_study_summary.csv`
   - `experiments/ablation_study_summary.tex` (LaTeX booktabs format)
   - `plots/ablation_study_results.png`

**Requirements:**
- CLI args: `--baseline-folder`, `--ablation-pattern` (regex, default: `*FactorizedSTEC*`)
- Add `--recompute-metrics` flag: Force re-run evaluation even if CSV exists
- Handle missing experiments gracefully: Skip with warning if folder not found
- Interpretation column: Auto-generate text (e.g., "SWI critical: +1.2 TECU RMSE without")
- Do NOT modify Task 1 script - call it as subprocess
- Preserve individual CSVs - only aggregate, don't overwrite

---

## Task 15: Master Evaluation Pipeline Script

**Context:** We have many evaluation components (Tasks 1-14) that should run systematically. Master script orchestrates full pipeline for reproducible evaluation.

**Existing Infrastructure to Reuse:**
- All previous task scripts in `scripts/`
- Subprocess pattern: `subprocess.run(['python', 'script.py', '--arg', 'value'], check=True)`
- Progress tracking: Print timestamps and stage names

**Task:**
Create `scripts/run_full_evaluation.sh` that orchestrates evaluation pipeline:

**Pipeline stages:**
```bash
# Stage 1: Test Set Evaluation (Tasks 1-4)
echo "=== Stage 1/5: Test Set Evaluation ==="
python scripts/evaluate_test_set.py \
  --experiment-folder "$EXPERIMENT" \
  --stratify-elevation \
  --stratify-latitude \
  --stratify-solar \
  --calibration-analysis
# Time estimate: ~5-10 min

# Stage 2: Baseline Comparisons (Tasks 6, 8) - Optional
if [[ "$WITH_BASELINES" == "true" ]]; then
  echo "=== Stage 2/5: Baseline Comparisons ==="
  # VTEC baseline (if model exists)
  if [[ -d "$VTEC_BASELINE_PATH" ]]; then
    python scripts/evaluate_vtec_baseline.py --experiment-folder "$VTEC_BASELINE_PATH"
  fi
  # IGS GIM comparison
  python scripts/evaluate_igs_gim.py --gim-data-dir "$GIM_DIR"
fi
# Time estimate: ~10-15 min

# Stage 3: Visualizations (Tasks 10, 11)
echo "=== Stage 3/5: Spatial and Temporal Analysis ==="
python scripts/plot_residual_maps.py --experiment-folder "$EXPERIMENT"
python scripts/plot_temporal_residuals.py --experiment-folder "$EXPERIMENT"
# Time estimate: ~5 min

# Stage 4: Model Comparison (Task 9) - If multiple experiments provided
if [[ ${#EXPERIMENTS[@]} -gt 1 ]]; then
  echo "=== Stage 4/5: Model Comparison ==="
  python scripts/compare_models.py --experiments "${EXPERIMENTS[@]}" --output-dir plots/
fi
# Time estimate: ~2 min

# Stage 5: Summary Report
echo "=== Stage 5/5: Generating Summary ==="
cat "$EXPERIMENT/test_metrics.csv"
echo "All outputs saved to: $RESULTS_DIR"
ls -lh "$RESULTS_DIR"
```

**Features:**
1. Creates timestamped results folder: `experiments/evaluation_${exp_name}_${timestamp}/`
2. Copies all generated CSVs and plots to results folder for archival
3. Progress indicators: Print stage names with timestamps
4. Time estimation: Based on test set size (compute from test.h5)
5. Error handling: If stage fails, log error but continue to next stage
6. Final summary: Print results table and list output files

**Requirements:**
- Usage: `bash scripts/run_full_evaluation.sh <experiment_folder> [--with-baselines] [--gim-data-dir <path>]`
- Add `--quick` flag: Skip stratifications, use 10% test subset for rapid iteration
- Log all commands: `evaluation_commands.log` in results folder (for reproducibility)
- Time tracking: Print elapsed time for each stage and total duration
- Exit codes: Return 0 if all stages succeed, 1 if any critical stage fails
- Do NOT modify individual scripts - orchestrate via subprocess
- Preserve all existing outputs - copy, don't move

---

## Usage Notes

**Recommended execution order:**
1. **Tasks 1-4: Core evaluation infrastructure** (do first, builds incrementally)
   - Task 1: Basic test metrics (foundation for all others)
   - Task 2: Add elevation stratification
   - Task 3: Add latitude and solar activity stratification
   - Task 4: Add uncertainty calibration analysis
   - Result: Complete test set evaluation script with all metrics

2. **Task 5-6: VTEC baseline** (optional but recommended for ablation context)
   - Task 5: Train VTEC-only model (uses existing infrastructure)
   - Task 6: Evaluate with standard MF conversion
   - Purpose: Demonstrates value of learned mapping function (GeomNet)

3. **Task 7-8: IGS GIM comparison** (requires external data, run if internet available)
   - Task 7: Download IGS Global Ionosphere Maps
   - Task 8: Interpolate and compare against test set
   - Purpose: Benchmark against community standard ionospheric product

4. **Tasks 9-11: Visualization and comparison** (requires Tasks 1-4 complete)
   - Task 9: Multi-model comparison (works with 1+ experiments)
   - Task 10: Geographic error patterns (spatial maps)
   - Task 11: Temporal error patterns (diurnal/seasonal)
   - Result: Publication-ready figures and comparison tables

5. **Tasks 12-14: Ablation studies** (for detailed analysis, can run independently)
   - Task 12: Create ablation configs (quick, just YAML files)
   - Task 13: Train all ablations (longest step: ~hours per ablation)
   - Task 14: Aggregate results and generate comparison
   - Purpose: Quantify contribution of each model component

6. **Task 15: Master pipeline** (run after testing individual components)
   - Orchestrates Tasks 1-4, 6, 8-11 in sequence
   - Creates timestamped results archive
   - Best for final evaluation and reproducibility

**Dependencies between tasks:**
- **Task 2-4** extend Task 1 (incremental additions to same script)
- **Task 6** requires Task 5 (needs trained VTEC model)
- **Task 8** requires Task 7 (needs downloaded GIM data)
- **Task 9** requires Task 1 complete for at least one experiment (works with multiple)
- **Task 14** requires Task 13 (ablation models) + Task 1 (evaluation script)
- **Task 15** orchestrates Tasks 1-4, 6, 8, 10-11 (assumes components exist)

**Code reuse patterns across tasks:**
- **Model loading**: Tasks 1, 6, 8, 10, 11 all reuse same checkpoint loading pattern
- **Feature extraction**: Tasks 1-4, 6, 10, 11 use FeatureRegistry output_indices
- **Metric computation**: All evaluation tasks reuse functions from `src/utils/metrics.py`
- **Data loading**: H5Dataset + CollateWithSH pattern used in Tasks 1, 6, 8
- **Plotting style**: Matplotlib/seaborn conventions from validation scripts

**Copy-paste workflow:**
Each task block can be copied directly as a prompt:
```
[Copy entire task block from "## Task N:" through "**Requirements:**" section]
```

The AI assistant will:
- Understand the context and existing infrastructure
- Reuse specified code patterns without reinventing
- Preserve all existing functionality (no breaking changes)
- Follow project conventions (FeatureRegistry, config structure, naming)

**Time estimates (on typical hardware with ~500K test samples):**
- Task 1: 30-60 min implementation + 5-10 min execution
- Task 2-4: 20-30 min each (extensions to Task 1)
- Task 5: 45-60 min + hours for training (depends on epochs)
- Task 6: 20-30 min + 10 min execution
- Task 7: 30-45 min + variable download time (depends on date range)
- Task 8: 30-45 min + 15-20 min execution (interpolation is slow)
- Task 9: 30-45 min + 2-5 min execution
- Task 10-11: 30-45 min each + 5 min execution
- Task 12: 15-20 min (just config files)
- Task 13: 15-20 min + hours per ablation (4-6 hours total for 4 ablations)
- Task 14: 30-45 min + depends on number of ablations
- Task 15: 45-60 min (orchestration script, no new algorithms)
