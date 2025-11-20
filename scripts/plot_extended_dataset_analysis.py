#!/usr/bin/env python3
"""
Extended analysis of train/val/test datasets with OMNI/SWI features and temporal splits.
Analyzes: STEC features + OMNI SWI indices, broken down by train/val/test splits and temporal periods.
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

# Configuration
DATA_DIR = "/scratch2/arrueegg/WP4/PNN_STEC/data"
OUTPUT_DIR = "plots/extended_analysis"
SPLITS = ["train", "val", "test"]

# STEC feature columns
STEC_COLUMNS = ["stec", "satele", "satazi", "sod", "lat_ipp", "lon_ipp"]

# OMNI/SWI feature indices (columns 0-24 in OMNI data)
# Common indices: Kp, Ap, F10.7, Dst, AE, etc.
OMNI_FEATURE_INDICES = {
    "kp": 14,          # Kp index (last column)
    "ap": 18,           # Ap index
    "f107": 19,         # F10.7 flux
    "dst": 16,          # Dst index
    "ae": 17,          # AE index
}

# Histogram bins - fixed equal-width bins for all features across all splits
# Using 50 bins to reduce noise while maintaining detail
BINS = {
    "stec": np.linspace(0, 250, 51),
    "satele": np.linspace(0, 90, 51),
    "satazi": np.linspace(0, 360, 51),
    "sod": np.linspace(0, 86400, 51),
    "lat_ipp": np.linspace(-90, 90, 51),
    "lon_ipp": np.linspace(-180, 180, 51),
    "kp": np.linspace(0, 9, 50),              # Kp ranges 0-9
    "ap": np.linspace(0, 400, 50),            # Ap ranges 0-400
    "f107": np.linspace(60, 280, 50),         # F10.7 ranges ~60-280
    "dst": np.linspace(-500, 100, 50),        # Dst ranges ~-500 to +100
    "ae": np.linspace(0, 2500, 50),           # AE ranges 0-2500
}

# Chunk size for reading
CHUNK_SIZE = 50_000
HISTOGRAM_SAMPLE_SIZE = 1_000_000


class RunningStats:
    """Welford accumulator for computing mean, std, min, max."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update_batch(self, x):
        """Update with a batch of values."""
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return
        batch_n = x.size
        batch_mean = x.mean()
        batch_M2 = ((x - batch_mean) ** 2).sum()
        self.min = min(self.min, x.min())
        self.max = max(self.max, x.max())

        if self.n == 0:
            self.n, self.mean, self.M2 = batch_n, batch_mean, batch_M2
        else:
            delta = batch_mean - self.mean
            total_n = self.n + batch_n
            self.M2 += batch_M2 + (delta**2) * self.n * batch_n / total_n
            self.mean = (self.n * self.mean + batch_n * batch_mean) / total_n
            self.n = total_n

    def finalize(self):
        """Return dict with statistics."""
        if self.n < 2:
            var = 0.0
        else:
            var = self.M2 / (self.n - 1)
        return {
            "count": int(self.n),
            "mean": float(self.mean),
            "std": float(np.sqrt(var)),
            "min": float(self.min),
            "max": float(self.max),
        }


def compute_quantile_bins(all_samples, all_columns, n_bins=50, quantile=0.95):
    """Compute bin ranges based on 95% quantiles of data from all splits."""
    global BINS
    
    print(f"\n🔢 Computing {quantile*100:.0f}% quantile ranges for bins...")
    
    for col in all_columns:
        # Collect all data for this feature across all splits
        all_data = []
        for split in SPLITS:
            if split in all_samples and col in all_samples[split]:
                data = np.array(all_samples[split][col])
                # Filter invalid values - preserve negative values for coordinates
                if col in ["lat_ipp", "lon_ipp"]:
                    data = data[data != 0]  # Keep negative values for coordinates
                else:
                    data = data[data > 0]  # Remove zero/negative for other features
                all_data.extend(data)
        
        if len(all_data) == 0:
            print(f"  ⚠ No data for {col}, keeping default bins")
            continue
        
        all_data = np.array(all_data)
        
        # Compute quantile range
        q_low = np.percentile(all_data, (1 - quantile) / 2 * 100)  # 2.5th percentile
        q_high = np.percentile(all_data, (1 + quantile) / 2 * 100)  # 97.5th percentile
        
        # Special handling for circular features
        if col == "satazi":
            q_low, q_high = 0, 360
        elif col == "sod":
            q_low, q_high = 0, 86400
        elif col == "satele":
            q_low, q_high = 0, 90
        elif col == "lon_ipp":
            q_low, q_high = -180, 180
        elif col == "lat_ipp":
            q_low, q_high = -90, 90
        
        # Create bins
        BINS[col] = np.linspace(q_low, q_high, n_bins + 1)
        print(f"  {col:12s}: [{q_low:10.2f}, {q_high:10.2f}]")
    
    return BINS


def get_omni_value(omni_h5, year, doy, hour, feature_col):
    """Get OMNI/SWI value for a specific time."""
    try:
        year_str = str(year)
        doy_str = f"{doy:03d}"
        
        if year_str not in omni_h5 or doy_str not in omni_h5[year_str]:
            return np.nan
        
        day_data = omni_h5[year_str][doy_str][:]
        if hour >= day_data.shape[0]:
            return np.nan
        
        return float(day_data[hour, feature_col])
    except (KeyError, IndexError):
        return np.nan


def analyze_split_with_omni(split_name, omni_h5, random_state=None):
    """Analyze a single split with OMNI/SWI data - 1M random samples."""
    file_path = os.path.join(DATA_DIR, f"{split_name}.h5")
    
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {file_path}")
        return None, None, None
    
    print(f"▶ Processing {split_name}...")
    
    with h5py.File(file_path, "r") as f:
        data_ds = f["data"]
        n_total = data_ds.shape[0]
        
        print(f"  Total records: {n_total:,}")
        print(f"  Randomly sampling {HISTOGRAM_SAMPLE_SIZE:,} records...")
        
        # Generate random indices using stride-based sampling for efficiency with large datasets
        rng = np.random.RandomState(random_state)
        stride = max(1, n_total // HISTOGRAM_SAMPLE_SIZE)
        base_indices = np.arange(0, n_total, stride)[:HISTOGRAM_SAMPLE_SIZE]
        # Add small random offsets to avoid systematic bias
        offsets = rng.randint(0, stride, size=len(base_indices))
        sample_indices = np.clip(base_indices + offsets, 0, n_total - 1)
        sample_indices = np.sort(sample_indices)
        
        # Initialize accumulators (STEC + OMNI features)
        all_columns = STEC_COLUMNS + list(OMNI_FEATURE_INDICES.keys())
        stats = {col: RunningStats() for col in all_columns}
        samples = {col: np.zeros(len(sample_indices), dtype=np.float32) for col in all_columns}
        temporal_periods = {}  # Track temporal distribution
        sample_ptr = 0
        
        # Process samples
        for sample_ptr, sample_idx in enumerate(tqdm(sample_indices, desc=f"  Sampling {split_name}")):
            row = data_ds[sample_idx]
            
            # Extract STEC features
            for col in STEC_COLUMNS:
                if col not in data_ds.dtype.names:
                    continue
                
                val = float(row[col])
                
                samples[col][sample_ptr] = val
                stats[col].update_batch(np.array([val]))
            
            # Extract OMNI/SWI features
            year = int(row["year"])
            doy = int(row["doy"])
            sod = int(row["sod"])
            hour = sod // 3600
            
            # Track temporal period
            date_key = f"{year}-{doy:03d}"
            temporal_periods[date_key] = temporal_periods.get(date_key, 0) + 1
            
            for feat_name, feat_col in OMNI_FEATURE_INDICES.items():
                omni_val = get_omni_value(omni_h5, year, doy, hour, feat_col)
                if not np.isnan(omni_val):
                    samples[feat_name][sample_ptr] = omni_val
                    stats[feat_name].update_batch(np.array([omni_val]))
        
        # Trim arrays to actual size
        for col in all_columns:
            samples[col] = samples[col][:sample_ptr]
    
    return stats, samples, temporal_periods


def main():
    """Main analysis function."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Open OMNI data once
    omni_file = os.path.join(DATA_DIR, "omni_hourly_2010-2025.h5")
    if not os.path.exists(omni_file):
        print(f"⚠ OMNI file not found: {omni_file}")
        return
    
    omni_h5 = h5py.File(omni_file, "r")
    
    try:
        # Analyze all splits
        all_stats = {}
        all_samples = {}
        all_temporal = {}
        all_columns = STEC_COLUMNS + list(OMNI_FEATURE_INDICES.keys())
        
        for split in SPLITS:
            stats, samples, temporal = analyze_split_with_omni(split, omni_h5, random_state=42)
            if stats is None:
                continue
            all_stats[split] = stats
            all_samples[split] = samples
            all_temporal[split] = temporal
        
        # Compute 95% quantile bins for all features
        compute_quantile_bins(all_samples, all_columns, n_bins=50, quantile=0.95)
        
        # Print summary statistics
        summary_file = os.path.join(OUTPUT_DIR, "extended_dataset_statistics.txt")
        print(f"\n📊 Writing summary to {summary_file}")
        
        with open(summary_file, "w") as f_out:
            f_out.write("=" * 90 + "\n")
            f_out.write("EXTENDED DATASET STATISTICS (STEC + OMNI/SWI)\n")
            f_out.write("=" * 90 + "\n\n")
            
            for split in SPLITS:
                if split not in all_stats:
                    continue
                
                header = f"\n{'=' * 90}\nSplit: {split.upper()}\n{'=' * 90}\n"
                print(header, end="")
                f_out.write(header)
                
                stats = all_stats[split]
                
                # STEC features
                f_out.write("\n[STEC FEATURES]\n")
                for col in STEC_COLUMNS:
                    if col not in stats:
                        continue
                    r = stats[col].finalize()
                    line = (
                        f"{col:12s}  count={r['count']:10d}  mean={r['mean']:10.4f}  "
                        f"std={r['std']:10.4f}  min={r['min']:10.4f}  max={r['max']:10.4f}\n"
                    )
                    print(line, end="")
                    f_out.write(line)
                
                # OMNI/SWI features
                f_out.write("\n[OMNI/SWI FEATURES]\n")
                for col in OMNI_FEATURE_INDICES.keys():
                    if col not in stats:
                        continue
                    r = stats[col].finalize()
                    line = (
                        f"{col:12s}  count={r['count']:10d}  mean={r['mean']:10.4f}  "
                        f"std={r['std']:10.4f}  min={r['min']:10.4f}  max={r['max']:10.4f}\n"
                    )
                    print(line, end="")
                    f_out.write(line)
                
                # Temporal coverage
                if split in all_temporal and all_temporal[split]:
                    f_out.write(f"\nTemporal coverage: {len(all_temporal[split])} unique days\n")
        
        # Create comparison table
        print("\n📊 Creating cross-split comparison...")
        comparison_file = os.path.join(OUTPUT_DIR, "extended_dataset_comparison.txt")
        
        with open(comparison_file, "w") as f_out:
            f_out.write("=" * 110 + "\n")
            f_out.write("CROSS-SPLIT COMPARISON (STEC + OMNI/SWI)\n")
            f_out.write("=" * 110 + "\n\n")
            
            for col in all_columns:
                f_out.write(f"\n{col.upper()}\n")
                f_out.write("-" * 110 + "\n")
                f_out.write(f"{'Split':<10} {'Count':<12} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}\n")
                f_out.write("-" * 110 + "\n")
                
                for split in SPLITS:
                    if split not in all_stats or col not in all_stats[split]:
                        continue
                    
                    r = all_stats[split][col].finalize()
                    f_out.write(
                        f"{split:<10} {r['count']:<12,} {r['mean']:<12.4f} "
                        f"{r['std']:<12.4f} {r['min']:<12.4f} {r['max']:<12.4f}\n"
                    )
        
        print(f"✓ Comparison written to {comparison_file}")
        
        # Plot individual histograms
        print("\n📈 Creating individual histograms...")
        
        for col in tqdm(all_columns, desc="  Histograms"):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f"Distribution of {col} (by split)", fontsize=14, fontweight="bold")
            
            # Get fixed bins for this feature from BINS dictionary
            bins = BINS.get(col, 50)
            
            for idx, split in enumerate(SPLITS):
                if split not in all_samples or col not in all_samples[split]:
                    continue
                
                samples_data = np.array(all_samples[split][col])
                if len(samples_data) == 0:
                    continue
                
                ax = axes[idx]
                ax.hist(samples_data, bins=bins, alpha=0.7, edgecolor='black')
                ax.set_title(f"{split.upper()} (n={len(samples_data):,})")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                ax.grid(axis="y", alpha=0.3)
            
            plt.tight_layout()
            hist_file = os.path.join(OUTPUT_DIR, f"histogram_{col}.png")
            plt.savefig(hist_file, dpi=100)
            plt.close()
        
        # Plot overlapping distributions
        print("\n📊 Creating overlapping comparison plots...")
        
        for col in tqdm(all_columns, desc="  Overlaps"):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
            
            # Get fixed bins for this feature from BINS dictionary
            bins = BINS.get(col, 50)
            
            for split in SPLITS:
                if split not in all_samples or col not in all_samples[split]:
                    continue
                
                samples_data = np.array(all_samples[split][col])
                # Filter invalid values - preserve negative values for coordinates
                if col in ["lat_ipp", "lon_ipp"]:
                    samples_data = samples_data[samples_data != 0]  # Keep negatives for coords
                else:
                    samples_data = samples_data[samples_data > 0]  # Remove zero/negative for others
                if len(samples_data) == 0:
                    continue
                
                ax.hist(samples_data, bins=bins, alpha=0.5, label=f"{split} (n={len(samples_data):,})", 
                       color=colors[split], density=True, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.set_title(f"Overlapping Distribution: {col}", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(axis="y", alpha=0.3)
            
            plt.tight_layout()
            overlap_file = os.path.join(OUTPUT_DIR, f"overlap_{col}.png")
            plt.savefig(overlap_file, dpi=100)
            plt.close()
        
        print(f"\n✅ Extended analysis complete! All files saved to {OUTPUT_DIR}/")
    
    finally:
        omni_h5.close()


if __name__ == "__main__":
    main()
