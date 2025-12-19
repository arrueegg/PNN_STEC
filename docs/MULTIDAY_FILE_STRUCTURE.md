# Multiday Evaluation File Structure

## Overview
This document describes the complete file organization when running the multiday evaluation pipeline.

## Command
```bash
python cli.py multiday \
    --dates "2024-183:2024-189" \
    --stec_config config/config.yaml \
    --vtec_config config/config_vtec.yaml \
    --output_dir multiday_results/july_week1
```

## Complete File Structure

```
📁 Root Directory
│
├── 📁 experiments/
│   │   # Individual trained models for each day
│   │
│   ├── 📁 Pretrain_STEC_<MODEL>_<PARAMS>/
│   │   ├── config.yaml
│   │   ├── training.log
│   │   ├── 📁 model/
│   │   │   ├── best_model.pth
│   │   │   └── final_model.pth
│   │   └── 📁 plots/
│   │
│   ├── 📁 Finetune_STEC_2024_183_<MODEL>_<PARAMS>/
│   │   ├── config.yaml
│   │   ├── training.log
│   │   ├── 📁 model/
│   │   │   └── finetune_best_checkpoint.pth
│   │   ├── 📁 evaluation/
│   │   │   ├── 📁 own_vtec_gim/                    # Evaluation on own test set
│   │   │   │   ├── comparison_summary.txt         # Text report
│   │   │   │   ├── metrics_summary.csv            # ✅ CSV metrics (NEW!)
│   │   │   │   ├── detailed_predictions.csv       # All predictions
│   │   │   │   ├── scatter_comparison.png         # Publication plots
│   │   │   │   ├── residual_analysis.png
│   │   │   │   ├── error_distribution.png
│   │   │   │   ├── elevation_analysis.png
│   │   │   │   └── metrics_summary.png
│   │   │   │
│   │   │   └── 📁 madrigal_vtec_gim/              # Evaluation on Madrigal (independent)
│   │   │       ├── comparison_summary.txt
│   │   │       ├── metrics_summary.csv            # ✅ CSV metrics (NEW!)
│   │   │       ├── detailed_predictions.csv
│   │   │       └── [5 publication plots]
│   │   │
│   │   └── 📁 plots/
│   │
│   ├── 📁 Finetune_VTEC_2024_183_<MODEL>_<PARAMS>/
│   │   └── [same structure as STEC]
│   │
│   ├── 📁 Finetune_STEC_2024_184_<MODEL>_<PARAMS>/
│   │   └── [evaluation results for day 2]
│   │
│   └── ... [one STEC + one VTEC experiment per day]
│
└── 📁 multiday_results/july_week1/
    │   # Aggregate results across all days
    │
    ├── 📁 2024_DOY_183/
    │   ├── temp_config_stec_2024_183.yaml
    │   ├── temp_config_vtec_2024_183.yaml
    │   ├── training_stec_2024_183.log
    │   └── training_vtec_2024_183.log
    │
    ├── 📁 2024_DOY_184/
    │   └── [same structure for day 2]
    │
    ├── ... [one folder per day]
    │
    └── 📁 summary/
        ├── all_results.csv                         # All days combined
        ├── summary_statistics.csv                  # Mean/Std across days
        ├── rmse_by_date.png                        # Time series plot
        ├── metrics_boxplots.png                    # Distribution plots
        └── improvement_by_date.png                 # STEC vs baselines
```

## File Descriptions

### Per-Day Evaluation Results
Location: `experiments/Finetune_STEC_<DATE>_<MODEL>/evaluation/<dataset>_vtec_gim/`

#### `metrics_summary.csv` ✅ NEW
CSV file with metrics for easy aggregation across days:
```csv
Model,RMSE,MAE,R²,Bias,Std,Count
Direct STEC Model,2.1234,1.5678,0.9234,0.0123,2.0456,123456
VTEC + Mapping,3.4567,2.3456,0.8765,-0.1234,3.2345,123456
IGS GIM,4.5678,3.4567,0.7890,0.2345,4.3210,120000
```

#### `comparison_summary.txt`
Human-readable text report with:
- Experiment metadata (dates, configs, test size)
- Detailed metrics for each model
- Improvement percentages

#### `detailed_predictions.csv`
Full prediction results:
```csv
true_stec,stec_pred,elevation,vtec_model_stec,gim_stec
15.234,15.123,45.67,15.456,16.789
...
```

#### Publication-Ready Plots (5 PNG files)
1. **scatter_comparison.png** - Predictions vs truth with density colors
2. **residual_analysis.png** - Residuals vs predictions + Q-Q plot
3. **error_distribution.png** - Error histograms and CDFs
4. **elevation_analysis.png** - Performance by elevation angle
5. **metrics_summary.png** - Bar charts of RMSE, R², Bias

### Aggregate Summary Results
Location: `multiday_results/<OUTPUT_DIR>/summary/`

#### `all_results.csv`
Combined results from all days and datasets:
```csv
date,year,doy,dataset,Model,RMSE,MAE,R²,Bias,Std,Count
2024-183,2024,183,own_vtec_gim,Direct STEC Model,2.12,1.56,0.923,0.01,2.04,123456
2024-183,2024,183,own_vtec_gim,VTEC + Mapping,3.45,2.34,0.876,-0.12,3.23,123456
2024-183,2024,183,madrigal_vtec_gim,Direct STEC Model,2.34,1.67,0.912,0.02,2.15,87654
...
```

#### `summary_statistics.csv`
Mean and standard deviation across all days:
```csv
Dataset,Model,RMSE_mean,RMSE_std,MAE_mean,MAE_std,R2_mean,R2_std,Num_days
own_vtec_gim,Direct STEC Model,2.15,0.23,1.58,0.15,0.920,0.012,7
own_vtec_gim,VTEC + Mapping,3.42,0.31,2.36,0.19,0.875,0.018,7
own_vtec_gim,IGS GIM,4.56,0.45,3.45,0.28,0.785,0.025,7
madrigal_vtec_gim,Direct STEC Model,2.38,0.28,1.72,0.18,0.910,0.015,7
...
```

#### Aggregate Plots (3 PNG files)
1. **rmse_by_date.png** - Line plots showing RMSE evolution across days
2. **metrics_boxplots.png** - Box plots comparing model distributions
3. **improvement_by_date.png** - Bar charts showing % improvement over baselines

## Datasets Evaluated

### 1. Own Test Set
- **Source**: Test data from training splits (train.h5, test.h5)
- **Purpose**: Standard evaluation on held-out test data
- **Always evaluated**: Yes

### 2. Madrigal Independent Test Set
- **Source**: External Madrigal STEC database
- **Purpose**: Independent validation on unseen stations/conditions
- **Evaluated when**: Model is finetuned AND Madrigal data available
- **Location**: `/home/space/data/iono/Madrigal_STEC`

## Metrics Computed

For each model and dataset:
- **RMSE** (TECU) - Root Mean Square Error
- **MAE** (TECU) - Mean Absolute Error
- **R²** - Coefficient of Determination
- **Bias** (TECU) - Mean prediction error
- **Std** (TECU) - Standard deviation of residuals
- **Count** - Number of observations

## Models Compared

1. **Direct STEC Model** - Your trained neural network
2. **VTEC + Mapping** - Classical approach (VTEC model + mapping function)
3. **IGS GIM** - Operational baseline (GIM VTEC + mapping function)

## Storage Best Practices

### What Gets Kept
✅ **experiments/** - All trained models and evaluation results (PERMANENT)
✅ **multiday_results/summary/** - Aggregate statistics and plots (PERMANENT)

### What Can Be Cleaned
⚠️ **multiday_results/<DATE>_DOY_*/** - Temporary config/log files (can delete after successful run)
⚠️ **wandb/** - W&B logs (can clean periodically)

### Recommended Workflow
1. Run multiday evaluation
2. Verify results in `multiday_results/summary/`
3. Archive important experiment folders
4. Clean temporary date folders if needed

## Quick Access to Results

### For a single day:
```bash
# View metrics
cat experiments/Finetune_STEC_2024_183_*/evaluation/own_vtec_gim/comparison_summary.txt

# Open plots
eog experiments/Finetune_STEC_2024_183_*/evaluation/own_vtec_gim/*.png
```

### For aggregate results:
```bash
# View summary statistics
cat multiday_results/july_week1/summary/summary_statistics.csv

# Open aggregate plots
eog multiday_results/july_week1/summary/*.png
```

## Notes

- All results are automatically saved to both `experiments/` and `multiday_results/`
- CSV files use consistent column names for easy post-processing
- Publication plots are 300 DPI, suitable for papers
- Aggregate statistics compute both mean and std across days for robustness
