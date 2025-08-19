# RAM Dataset System

This system provides high-performance, RAM-based datasets for cluster environments where abundant memory is available.

## Overview

When running on cluster environments with large amounts of RAM, loading datasets entirely into memory eliminates I/O bottlenecks and significantly speeds up training.

## Components

### H5RAMDataset

- Loads entire H5 aggregated datasets into RAM during initialization
- Supports SWI data integration (also loaded into RAM)
- Provides memory usage estimation and progress bars during data loading
- Located in `src/utils/data.py` alongside the regular `H5Dataset`

### Automatic Dataset Selection

The system automatically selects between regular and RAM-based datasets based on the cluster configuration:

```python
# Regular datasets (default for local development)
config['cluster'] = False  → Uses H5Dataset

# RAM datasets (automatic on cluster)
config['cluster'] = True   → Uses H5RAMDataset
```

**Note**: RAM datasets are only used for aggregated H5 files (`use_agg_h5: true`). For non-aggregated PyTables workflows, regular `PyTablesDatasetSplit` is always used regardless of cluster setting.

## Integration Points

### Configuration Files
- `config/config.yaml`: Local development (`cluster: false`)
- `hp_search/config_*.yaml`: Cluster configs (`cluster: true`)

### Hyperparameter Search
- `--cluster` flag automatically sets `cluster: true`
- Generated configs include the cluster flag for automatic RAM dataset usage

### Data Loading (`utils/data.py`)
- Modified `get_data_loaders()` function
- Automatic dataset class selection: `cluster=True` → `H5RAMDataset`, `cluster=False` → `H5Dataset`
- Logging indicates which dataset type is being used

## Usage

### Local Development
```yaml
# config/config.yaml
cluster: false  # Uses regular H5Dataset
```

### Cluster Execution
```yaml
# Automatically set by hyperparameter search with --cluster flag
cluster: true   # Uses H5RAMDataset
```

## Performance Benefits

### Expected Improvements on Cluster:
- **Data Loading**: Near-zero latency after initial load
- **Training Speed**: Significant speedup due to eliminated I/O waits
- **CPU Utilization**: Better GPU/CPU utilization without I/O blocking
- **Consistency**: Predictable performance without filesystem variations

### Memory Requirements:
- The system estimates and reports memory usage during loading
- Ensure cluster nodes have sufficient RAM for entire dataset
- Consider dataset size when requesting cluster resources

## Monitoring

The system provides detailed logging:
```
🚀 Loading train dataset into RAM from /path/to/train.h5...
📡 Loading SWI data into RAM from /path/to/swi.h5...
✅ train dataset loaded into RAM: 1,234,567 samples
📡 SWI data loaded: 98,765 time points
💾 Estimated RAM usage: 12.34 GB
```

## Architecture

```
┌─────────────────┬──────────────────┬─────────────────┐
│   Environment   │   Cluster Flag   │   Dataset Used  │
├─────────────────┼──────────────────┼─────────────────┤
│ Local Dev       │ cluster: false   │ H5Dataset       │
│ Cluster         │ cluster: true    │ H5RAMDataset    │
└─────────────────┴──────────────────┴─────────────────┘
```

The system is designed to be simple and automatic:
- **Single configuration flag**: Just set `cluster: true/false`
- **Automatic selection**: No need to manually choose dataset types
- **Backward compatible**: Existing code works unchanged
- **Optimized for aggregated data**: RAM datasets only activate for `use_agg_h5: true`

## Troubleshooting

### Out of Memory Errors
- Reduce dataset size or request more cluster memory
- Check actual vs. estimated memory usage
- RAM datasets are only used for aggregated H5 files, so ensure `use_agg_h5: true`

### Slow Initial Loading
- Normal behavior - initial loading takes time but subsequent access is fast
- Progress bars show loading status
- Consider the trade-off: slower start vs. faster training

### Missing Data Files
- Ensure all required H5 files exist in scratch directories
- Check SWI data availability
- Verify file paths in cluster configuration
