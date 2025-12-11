# Tests Directory

This directory contains test scripts and utilities for validating model predictions.

## Available Scripts

### `compare_predictions_with_ground_truth.py`

Compares predicted STEC values from a CSV file with ground truth data from `test.h5`.

**Features:**
- Loads predictions from CSV and matches with ground truth by timestamp and satellite
- Computes comprehensive error metrics (MAE, RMSE, R², correlation, etc.)
- Evaluates uncertainty calibration for Bayesian models
- Generates visualization plots (scatter, error distribution, temporal analysis, calibration)
- Saves detailed results and metrics to files

**Usage:**

```bash
# Basic usage (station name inferred from CSV filename)
python tests/compare_predictions_with_ground_truth.py path/to/CHPG.csv

# Use CASDCB database instead of test.h5
python tests/compare_predictions_with_ground_truth.py path/to/CHPG.csv \
    --test-file /home/space/data/iono/STEC_DB_CASDCB/2024/122/ccl_2024122_30_5.h5 \
    --casdcb

# Specify station, year, and DOY explicitly
python tests/compare_predictions_with_ground_truth.py path/to/predictions.csv \
    --station CHPG --year 2024 --doy 122

# Save results and plots to output directory
python tests/compare_predictions_with_ground_truth.py path/to/CHPG.csv \
    --output-dir tests/comparison_results

# Full example with all options
python tests/compare_predictions_with_ground_truth.py \
    experiments/.../CHPG.csv \
    --station CHPG \
    --year 2024 \
    --doy 122 \
    --test-file data/test.h5 \
    --output-dir tests/comparison_results \
    --tolerance 1e-3
```

**Arguments:**
- `csv_file`: Path to CSV file with predictions (required)
- `--station`: Station name (default: inferred from filename)
- `--year`: Year (default: 2024)
- `--doy`: Day of year (default: 122)
- `--test-file`: Path to ground truth HDF5 file (default: data/test.h5)
- `--casdcb`: Use CASDCB database format instead of test.h5 format
- `--output-dir`: Directory to save plots and results (default: show plots only)
- `--tolerance`: Tolerance for matching IPP coordinates (default: 1e-3)

**Output Files (when --output-dir specified):**
- `comparison_plots.png`: Visualization of predictions vs ground truth
- `detailed_results.csv`: Merged predictions and ground truth with errors
- `metrics.txt`: Summary of all computed metrics

**Expected CSV Format:**
```csv
second_of_day,PRN,ipp_latitude,ipp_longitude,stec,uncertainty
0.0,E03,-17.2404,-50.7725,111.0998,25.0157
...
```

**Metrics Computed:**
- **Error metrics**: MAE, RMSE, bias, median AE, std, min/max error
- **Correlation**: R², Pearson, Spearman
- **Uncertainty calibration**: % within 1σ and 2σ (for Bayesian models)
**Example:**
```bash
# Compare predictions against test.h5
python tests/compare_predictions_with_ground_truth.py \
    experiments/Finetune_STEC_2024_122_BayesianResNetSTEC_h1024_l4_lr1e-4_bs512_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_lw1e-1_SWI/positioning/stec_corrections/2024122/CHPG.csv \
    --output-dir tests/comparison_results

# Compare predictions against CASDCB database (all observations)
python tests/compare_predictions_with_ground_truth.py \
    experiments/Finetune_STEC_2024_122_BayesianResNetSTEC_h1024_l4_lr1e-4_bs512_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_lw1e-1_SWI/positioning/stec_corrections/2024122/CHPG.csv \
    --test-file /home/space/data/iono/STEC_DB_CASDCB/2024/122/ccl_2024122_30_5.h5 \
    --casdcb \
    --output-dir tests/comparison_results_casdcb
``` --output-dir tests/comparison_results
```
