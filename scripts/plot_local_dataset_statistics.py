#!/usr/bin/env python3
"""
Analyze statistics for local train/test/val datasets in data/ folder.
Similar to plot_data_statistics.py but for the local .h5 files.
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATA_DIR = "/scratch2/arrueegg/WP4/PNN_STEC/data"
OUTPUT_DIR = "plots/local_data"
SPLITS = ["train", "val", "test"]

# Key columns to analyze
COLUMNS = ["stec", "satele", "satazi", "sod", "lat_ipp", "lon_ipp"]

# Histogram bins
BINS = {
    "stec": np.linspace(0, 150, 151),
    "satele": np.linspace(0, 90, 91),
    "satazi": np.linspace(0, 360, 361),
    "sod": np.linspace(0, 86400, 100),  # seconds of day
    "lat_ipp": np.linspace(-90, 90, 181),
    "lon_ipp": np.linspace(-180, 180, 361),
}

# Chunk size for reading large datasets
CHUNK_SIZE = 50_000
# Sample for histogram creation (for visualization efficiency)
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

    def merge(self, other):
        """Merge another RunningStats into this one."""
        if other.n == 0:
            return
        if self.n == 0:
            self.n, self.mean, self.M2 = other.n, other.mean, other.M2
            self.min, self.max = other.min, other.max
            return

        delta = other.mean - self.mean
        total_n = self.n + other.n
        M2 = self.M2 + other.M2 + (delta**2) * self.n * other.n / total_n
        new_mean = (self.n * self.mean + other.n * other.mean) / total_n

        self.n = total_n
        self.mean = new_mean
        self.M2 = M2
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)

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


def analyze_split(split_name, random_state=None):
    """Analyze a single split (train/val/test) - 1M random samples only."""
    file_path = os.path.join(DATA_DIR, f"{split_name}.h5")
    
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {file_path}")
        return None, None
    
    print(f"▶ Processing {split_name}...")
    
    with h5py.File(file_path, "r") as f:
        data_ds = f["data"]
        n_total = data_ds.shape[0]
        
        print(f"  Total records: {n_total:,}")
        print(f"  Randomly sampling {HISTOGRAM_SAMPLE_SIZE:,} records...")
        
        # Generate random indices
        rng = np.random.RandomState(random_state)
        sample_indices = np.sort(rng.choice(n_total, size=min(HISTOGRAM_SAMPLE_SIZE, n_total), replace=False))
        
        # Initialize accumulators
        stats = {col: RunningStats() for col in COLUMNS}
        samples = {col: np.zeros(len(sample_indices), dtype=np.float32) for col in COLUMNS}
        sample_ptr = 0
        
        # Process samples
        for sample_idx in sample_indices:
            row = data_ds[sample_idx]
            
            for col_idx, col in enumerate(COLUMNS):
                if col not in data_ds.dtype.names:
                    if sample_idx == 0:  # Only print once
                        print(f"  ⚠ Column '{col}' not found, skipping")
                    continue
                
                val = float(row[col])
                
                # Handle special column: convert sod from seconds to hours
                if col == "sod":
                    val = val / 3600.0
                
                # Store sample
                samples[col][sample_ptr] = val
                
                # Update stats
                stats[col].update_batch(np.array([val]))
            
            sample_ptr += 1
            
            if (sample_ptr + 1) % 100000 == 0:
                print(f"    Processed {sample_ptr:,} / {len(sample_indices):,} samples")
        
        # Trim arrays to actual size
        for col in COLUMNS:
            samples[col] = samples[col][:sample_ptr]
    
    return stats, samples


def main():
    """Main analysis function."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Analyze all splits
    all_stats = {}
    all_samples = {}
    
    for split in SPLITS:
        stats, samples = analyze_split(split, random_state=42)
        if stats is None:
            continue
        all_stats[split] = stats
        all_samples[split] = samples
    
    # Print summary statistics
    summary_file = os.path.join(OUTPUT_DIR, "local_dataset_statistics.txt")
    print(f"\n📊 Writing summary to {summary_file}")
    
    with open(summary_file, "w") as f_out:
        f_out.write("=" * 80 + "\n")
        f_out.write("LOCAL DATASET STATISTICS\n")
        f_out.write("=" * 80 + "\n\n")
        
        for split in SPLITS:
            if split not in all_stats:
                continue
            
            header = f"\n{'=' * 80}\nSplit: {split.upper()}\n{'=' * 80}\n"
            print(header, end="")
            f_out.write(header)
            
            stats = all_stats[split]
            
            for col in COLUMNS:
                if col not in stats:
                    continue
                
                r = stats[col].finalize()
                line = (
                    f"{col:12s}  "
                    f"count={r['count']:12d}  "
                    f"mean={r['mean']:10.4f}  "
                    f"std={r['std']:10.4f}  "
                    f"min={r['min']:10.4f}  "
                    f"max={r['max']:10.4f}\n"
                )
                print(line, end="")
                f_out.write(line)
    
    # Create comparison table
    print("\n📊 Creating cross-split comparison...")
    comparison_file = os.path.join(OUTPUT_DIR, "dataset_comparison.txt")
    
    with open(comparison_file, "w") as f_out:
        f_out.write("=" * 100 + "\n")
        f_out.write("CROSS-SPLIT COMPARISON\n")
        f_out.write("=" * 100 + "\n\n")
        
        for col in COLUMNS:
            f_out.write(f"\n{col.upper()}\n")
            f_out.write("-" * 100 + "\n")
            f_out.write(f"{'Split':<10} {'Count':<15} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}\n")
            f_out.write("-" * 100 + "\n")
            
            for split in SPLITS:
                if split not in all_stats or col not in all_stats[split]:
                    continue
                
                r = all_stats[split][col].finalize()
                f_out.write(
                    f"{split:<10} {r['count']:<15,} {r['mean']:<12.4f} "
                    f"{r['std']:<12.4f} {r['min']:<12.4f} {r['max']:<12.4f}\n"
                )
    
    print(f"✓ Comparison written to {comparison_file}")
    
    # Plot individual histograms for each column and split
    print("\n📈 Creating individual histograms...")
    
    for col in COLUMNS:
        # Combined histogram for all splits
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Distribution of {col} (by split)", fontsize=14, fontweight="bold")
        
        for idx, split in enumerate(SPLITS):
            if split not in all_samples or col not in all_samples[split]:
                continue
            
            samples = np.array(all_samples[split][col])
            if len(samples) == 0:
                continue
            
            ax = axes[idx]
            ax.hist(samples, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f"{split.upper()} (n={len(samples):,})")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        hist_file = os.path.join(OUTPUT_DIR, f"histogram_{col}.png")
        plt.savefig(hist_file, dpi=100)
        plt.close()
        print(f"  ✓ Saved {hist_file}")
    
    # Plot overlapping distributions for each column
    print("\n📊 Creating overlapping comparison plots...")
    
    for col in COLUMNS:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {'train': 'blue', 'val': 'orange', 'test': 'green'}
        
        for split in SPLITS:
            if split not in all_samples or col not in all_samples[split]:
                continue
            
            samples = np.array(all_samples[split][col])
            if len(samples) == 0:
                continue
            
            # Normalize histogram for fair comparison
            ax.hist(samples, bins=50, alpha=0.5, label=f"{split} (n={len(samples):,})", 
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
        print(f"  ✓ Saved {overlap_file}")
    
    print(f"\n✅ Analysis complete! All files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()