# Positioning Evaluation Pipeline Documentation

## Overview

The automated positioning evaluation pipeline allows you to:
1. Generate STEC corrections from your trained models
2. Download necessary GNSS products and RINEX files
3. Run PPP positioning with both your model's STEC and reference IGS GIM
4. Compare positioning accuracy between the two approaches
5. Generate daily performance metrics

## Quick Start

### Single Command Pipeline

```bash
bash scripts/run_positioning_pipeline.sh <experiment_name> <date>
```

**Example:**
```bash
bash scripts/run_positioning_pipeline.sh Finetune_STEC_2024_183_BayesianResNetSTEC 2024-07-01
```

### Step-by-Step Execution

If you prefer more control, run each step individually:

#### 1. Generate STEC Corrections
```bash
python src/inference_positioning.py \
    --experiment "Finetune_STEC_2024_183_BayesianResNetSTEC" \
    --date 2024-07-01
```

**Output:** CSV files in `experiments/<exp>/positioning_corrections/2024182/`
- Format: `<station>.csv` with columns: `second_of_day`, `PRN`, `ipp_latitude`, `ipp_longitude`, `stec`, `uncertainty`

#### 2. Run Positioning Evaluation
```bash
python src/positioning_eval/run_positioning_evaluation.py \
    --experiment "Finetune_STEC_2024_183_BayesianResNetSTEC" \
    --date 2024-07-01 \
    --all_test_stations \
    --parallel 4
```

**Options:**
- `--all_test_stations`: Process all stations from `test_station.list`
- `--stations ZIMM BRUS WTZR`: Process specific stations only
- `--parallel N`: Run N stations in parallel
- `--skip_downloads`: Skip product/RINEX downloads (use existing)
- `--pppx_path`: Path to pppx executable (default: `./src/stec_eval/pppx`)
- `--gim_base_path`: Path to IGS GIM directory

## Pipeline Components

### 1. STEC Correction Generation (`src/inference_positioning.py`)

Generates ionospheric corrections for each test station:
- Loads model checkpoint from experiment directory
- Filters observations by test stations only
- Runs model inference to predict STEC values
- Computes uncertainty estimates (for BNN models)
- Exports station-specific CSV files

### 2. Product Downloader (`src/positioning_eval/download_products.py`)

Downloads required GNSS products:
- Precise orbits (`.SP3`)
- Precise clocks (`.CLK`)
- Earth rotation parameters (`.ERP`)
- Satellite attitude (`.OBX`)
- CODE GIM (`.INX`)

### 3. RINEX Downloader (`src/positioning_eval/download_rinex.py`)

Downloads observation files from CDDIS:
- Supports both RINEX 2 and RINEX 3 formats
- Handles compressed files (`.Z`, `.gz`)
- Converts Hatanaka compressed to standard RINEX
- Automatic retry with multiple archive formats

### 4. INI Generator (`src/positioning_eval/generate_ini.py`)

Creates PPPx configuration files:
- Dynamic year/DOY insertion
- Configurable ionosphere source (CSV or IONEX)
- Experiment-specific output paths
- Product path resolution

### 5. Positioning Executor (`src/positioning_eval/run_positioning_evaluation.py`)

Orchestrates the complete workflow:
- Downloads products and RINEX files
- Generates INI files for each station/method
- Runs PPPx positioning:
  - Once with your model's STEC corrections
  - Once with IGS GIM reference
- Aggregates metrics across all stations
- Generates daily summary report

### 6. Metrics Module (`src/positioning_eval/metrics.py`)

Computes positioning accuracy:
- Parses PPPx `.pos` output files
- Converts ECEF to ENU coordinates
- Computes RMS, mean, std, 95th percentile errors
- Aggregates station-level metrics
- Generates comparative reports

## Output Structure

```
experiments/<experiment_name>/
├── positioning_corrections/
│   └── 2024182/
│       ├── ZIMM.csv           # STEC predictions for ZIMM
│       ├── BRUS.csv           # STEC predictions for BRUS
│       └── ...
├── positioning_eval/
│   └── 2024182/
│       ├── products/          # Downloaded GNSS products
│       │   ├── COD0OPSFIN_...SP3
│       │   ├── COD0OPSFIN_...CLK
│       │   └── ...
│       └── rinex/             # Downloaded RINEX files
│           ├── ZIMM00CHE_R_...rnx
│           └── ...
└── positioning_results/
    └── 2024182/
        ├── model/             # Results with your STEC
        │   ├── ZIMM/
        │   │   ├── pppx_model.ini
        │   │   ├── ZIMM_model.pos
        │   │   ├── ZIMM_model.log
        │   │   └── ZIMM_model.stat
        │   └── ...
        ├── gim/               # Results with IGS GIM
        │   ├── ZIMM/
        │   │   ├── pppx_gim.ini
        │   │   ├── ZIMM_gim.pos
        │   │   └── ...
        │   └── ...
        └── daily_summary.csv  # Aggregated metrics
```

## Performance Metrics

The `daily_summary.csv` file contains:

### Per-Station Metrics
- `station`: Station name
- `method`: "model" or "gim"
- `year`, `doy`: Date
- `n_epochs`: Number of positioning epochs
- `mean_nsat`: Average number of satellites

### Error Components (ENU)
- `e_mean`, `e_std`, `e_rms`: East component statistics
- `n_mean`, `n_std`, `n_rms`: North component statistics
- `u_mean`, `u_std`, `u_rms`: Up component statistics

### Overall Errors
- `error_2d_mean`, `error_2d_std`, `error_2d_rms`: Horizontal positioning
- `error_3d_mean`, `error_3d_std`, `error_3d_rms`: 3D positioning
- `error_2d_95th`, `error_3d_95th`: 95th percentile errors

## Advanced Usage

### Process Specific Stations Only
```bash
python src/positioning_eval/run_positioning_evaluation.py \
    --experiment "MyExperiment" \
    --date 2024-07-01 \
    --stations ZIMM BRUS WTZR GRAZ
```

### Skip Downloads (Use Cached Files)
```bash
python src/positioning_eval/run_positioning_evaluation.py \
    --experiment "MyExperiment" \
    --date 2024-07-01 \
    --all_test_stations \
    --skip_downloads
```

### Sequential Processing (No Parallelism)
```bash
python src/positioning_eval/run_positioning_evaluation.py \
    --experiment "MyExperiment" \
    --date 2024-07-01 \
    --all_test_stations \
    --parallel 1
```

## Requirements

### Software Dependencies
- Python packages: `torch`, `pandas`, `numpy`, `h5py`, `tqdm`
- System tools: `wget`, `gunzip`, `uncompress`, `CRX2RNX`
- PPPx executable (compiled positioning software)

### Data Requirements
- Trained model checkpoint in experiment directory
- Test station list: `src/data_processing/test_station.list`
- GNSS observation database (for STEC inference)
- IGS GIM archive (optional, for comparison)

### Access Requirements
- CDDIS access for RINEX downloads (may require NASA Earthdata credentials)
- CODE FTP access for product downloads

## Troubleshooting

### "No RINEX file found"
- Check CDDIS availability
- Verify station name formatting (4-char uppercase)
- Try manual download first

### "PPPx failed"
- Check PPPx executable path
- Verify product files are complete
- Review PPPx log files in output directory

### "IGS GIM not found"
- Verify `--gim_base_path` points to correct directory
- Check if GIM exists for requested date
- Pipeline will continue with model-only evaluation

### Download Timeouts
- Increase timeout values in download scripts
- Use `--skip_downloads` and download manually
- Check network connectivity

## Performance Tips

1. **Parallel Processing**: Use `--parallel 4` or higher for faster processing
2. **Caching**: After first run, use `--skip_downloads` to reprocess without re-downloading
3. **Subset Testing**: Test with `--stations ZIMM BRUS` before running all stations
4. **Batch Dates**: Write a bash loop to process multiple dates sequentially

## Example Workflow

```bash
# 1. First, run inference to generate STEC corrections
python src/inference_positioning.py \
    --experiment "Finetune_STEC_2024_183_BayesianResNetSTEC_h1024_l4" \
    --date 2024-07-01

# 2. Test with a single station first
python src/positioning_eval/run_positioning_evaluation.py \
    --experiment "Finetune_STEC_2024_183_BayesianResNetSTEC_h1024_l4" \
    --date 2024-07-01 \
    --stations ZIMM

# 3. If successful, run all test stations
python src/positioning_eval/run_positioning_evaluation.py \
    --experiment "Finetune_STEC_2024_183_BayesianResNetSTEC_h1024_l4" \
    --date 2024-07-01 \
    --all_test_stations \
    --parallel 8 \
    --skip_downloads  # Reuse already downloaded files

# 4. Analyze results
python -c "
import pandas as pd
df = pd.read_csv('experiments/<exp>/positioning_results/2024182/daily_summary.csv')
print(df.groupby('method')[['error_2d_rms', 'error_3d_rms']].mean())
"
```

## Citation

If you use this positioning evaluation pipeline, please cite the PNN_STEC project.
