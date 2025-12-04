# Positioning Evaluation - Quick Reference

## Complete Pipeline (Recommended)

```bash
# Single command - runs everything
bash scripts/run_positioning_pipeline.sh "Finetune_STEC_2024_183_BayesianResNetSTEC" 2024-07-01
```

## Step-by-Step

### 1. Generate STEC Corrections
```bash
python src/inference_positioning.py \
    --experiment "Finetune_STEC_2024_183_BayesianResNetSTEC" \
    --date 2024-07-01
```

### 2. Run Positioning Evaluation
```bash
python src/positioning_eval/run_positioning_evaluation.py \
    --experiment "Finetune_STEC_2024_183_BayesianResNetSTEC" \
    --date 2024-07-01 \
    --all_test_stations \
    --parallel 4
```

## Key Options

### Process Specific Stations
```bash
--stations ZIMM BRUS WTZR
```

### Skip Downloads (Use Cached)
```bash
--skip_downloads
```

### Parallel Processing
```bash
--parallel 8  # Run 8 stations simultaneously
```

## Outputs

### STEC Corrections
```
experiments/<exp>/positioning_corrections/YYYYDDD/<station>.csv
```
Columns: `second_of_day`, `PRN`, `ipp_latitude`, `ipp_longitude`, `stec`, `uncertainty`

### Positioning Results
```
experiments/<exp>/positioning_results/YYYYDDD/
├── model/              # Your model results
│   └── <station>/
│       └── <station>_model.pos
├── gim/                # IGS GIM results
│   └── <station>/
│       └── <station>_gim.pos
└── daily_summary.csv   # Aggregated metrics
```

### Daily Summary Metrics
- `error_2d_rms`: Horizontal positioning RMS error
- `error_3d_rms`: 3D positioning RMS error
- `error_2d_95th`: 95th percentile horizontal error
- Per-station breakdown for both model and GIM

## Quick Analysis

```python
import pandas as pd

# Load summary
df = pd.read_csv('experiments/<exp>/positioning_results/YYYYDDD/daily_summary.csv')

# Compare methods
print(df.groupby('method')[['error_2d_rms', 'error_3d_rms']].mean())

# Best/worst stations
print(df[df['method']=='model'].nsmallest(5, 'error_2d_rms'))
print(df[df['method']=='model'].nlargest(5, 'error_2d_rms'))
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| RINEX download fails | Check CDDIS access, try `--skip_downloads` and download manually |
| PPPx fails | Check product files exist, review `.log` files |
| No IGS GIM found | Verify `--gim_base_path`, pipeline continues with model-only |
| Slow processing | Increase `--parallel` value |

## Full Documentation
See: `docs/positioning_evaluation_guide.md`
