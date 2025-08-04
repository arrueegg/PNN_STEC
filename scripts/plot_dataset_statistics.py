#!/usr/bin/env python3
import os
import glob
import h5py
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt

# ← EDIT THESE
DATA_DIR       = "/home/space/data/IONO/STEC_DB_CASDCB/"
COLUMNS        = ["satele", "satazi", "stec", "sod"]
SPLITS         = ["train", "val", "test"]
IDX_CHUNK_SIZE = 5_000_000

# histogram bins (1-unit bins, adjust as needed)
BINS = {
    "satele": np.linspace(  0,  90,  91),
    "satazi": np.linspace(  0, 360, 361),
    "stec":   np.linspace(  0, 150, 151),
    "sod":    np.linspace(  0,  24,  25),
}

class RunningStats:
    """Welford accumulator with merge capability."""
    def __init__(self):
        self.n    = 0
        self.mean = 0.0
        self.M2   = 0.0
        self.min  = np.inf
        self.max  = -np.inf

    def update_batch(self, x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return
        batch_n    = x.size
        batch_mean = x.mean()
        batch_M2   = ((x - batch_mean)**2).sum()
        self.min   = min(self.min, x.min())
        self.max   = max(self.max, x.max())

        if self.n == 0:
            self.n, self.mean, self.M2 = batch_n, batch_mean, batch_M2
        else:
            delta   = batch_mean - self.mean
            total_n = self.n + batch_n
            self.M2 += batch_M2 + (delta**2) * self.n * batch_n / total_n
            self.mean = (self.n*self.mean + batch_n*batch_mean) / total_n
            self.n    = total_n

    def merge(self, other):
        """Merge another RunningStats into this one."""
        if other.n == 0:
            return
        if self.n == 0:
            # just copy
            self.n, self.mean, self.M2 = other.n, other.mean, other.M2
            self.min, self.max = other.min, other.max
            return

        delta   = other.mean - self.mean
        total_n = self.n + other.n
        M2 = self.M2 + other.M2 + (delta**2) * self.n * other.n / total_n
        new_mean = (self.n*self.mean + other.n*other.mean) / total_n

        self.n    = total_n
        self.mean = new_mean
        self.M2   = M2
        self.min  = min(self.min, other.min)
        self.max  = max(self.max, other.max)

    def finalize(self):
        if self.n < 2:
            var = 0.0
        else:
            var = self.M2 / (self.n - 1)
        return {
            "count": int(self.n),
            "mean":  float(self.mean),
            "std":   float(np.sqrt(var)),
            "min":   float(self.min),
            "max":   float(self.max),
        }

def worker(fn):
    """Process one file: return (stats_dict, hist_counts)."""
    # init local accumulators
    local_stats = {
        split: {col: RunningStats() for col in COLUMNS}
        for split in SPLITS
    }
    local_hist = {
        col: np.zeros(len(BINS[col]) - 1, dtype=np.int64)
        for col in COLUMNS
    }

    # infer HDF5 group path
    basename  = os.path.basename(fn)
    year_doy  = basename.split("_")[1]
    year, doy = year_doy[:4], year_doy[4:7]
    group_path = f"/{year}/{doy}"

    print(f"▶ Processing {basename}")
    with h5py.File(fn, "r") as f:
        grp     = f[group_path]
        data_ds = grp["all_data"]

        for split in SPLITS:
            idx_arr = grp.get(f"{split}_idx")
            if idx_arr is None:
                continue
            n_idx = idx_arr.shape[0]

            # stream in chunks
            for off in range(0, n_idx, IDX_CHUNK_SIZE):
                chunk_idx = idx_arr[off:off+IDX_CHUNK_SIZE]
                if chunk_idx.size == 0:
                    continue

                # 2) contiguous runs
                ids_sorted = np.sort(chunk_idx)
                # find break points where diff > 1
                breaks = np.where(np.diff(ids_sorted) > 1)[0] + 1
                runs = np.split(ids_sorted, breaks)

                for run in runs:
                    if run.size == 0:
                        continue
                    start, end = int(run[0]), int(run[-1]) + 1
                    block = data_ds[start:end]  # one sequential read
                    rel_idx = run - start
                    sel = block[rel_idx]       # rows we actually need

                    for col in COLUMNS:
                        if col not in sel.dtype.names:
                            continue
                        col_data = sel[col].astype(float)

                        # update stats
                        local_stats[split][col].update_batch(col_data)

                        # update histogram (with under/overflow)
                        hist, _ = np.histogram(col_data, bins=BINS[col])
                        hist[0]  += np.sum(col_data < BINS[col][0])
                        hist[-1] += np.sum(col_data >= BINS[col][-1])
                        local_hist[col] += hist

    return local_stats, local_hist

def main():
    pattern = os.path.join(DATA_DIR, "*/*/*.h5")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found in {DATA_DIR}")
        return

    # global accumulators
    global_stats = {
        split: {col: RunningStats() for col in COLUMNS}
        for split in SPLITS
    }
    global_hist = {
        col: np.zeros(len(BINS[col]) - 1, dtype=np.int64)
        for col in COLUMNS
    }

    # process in parallel
    with concurrent.futures.ProcessPoolExecutor() as exe:
        for local_stats, local_hist in exe.map(worker, files):
            # merge per-split stats
            for split in SPLITS:
                for col in COLUMNS:
                    global_stats[split][col].merge(local_stats[split][col])
            # merge histograms
            for col in COLUMNS:
                global_hist[col] += local_hist[col]

    # print numeric summary
    for split in SPLITS:
        print(f"\n=== Split: {split} ===")
        for col in COLUMNS:
            r = global_stats[split][col].finalize()
            print(
                f"{col:8s}  "
                f"count={r['count']:10d}  "
                f"mean={r['mean']:8.4f}  "
                f"std={r['std']:8.4f}  "
                f"min={r['min']:8.4f}  "
                f"max={r['max']:8.4f}"
            )

    # plot overall histograms
    for col in COLUMNS:
        edges  = BINS[col]
        counts = global_hist[col]
        total  = counts.sum()
        if total == 0:
            continue

        plt.figure()
        plt.bar(edges[:-1], counts, width=np.diff(edges), align="edge")
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
