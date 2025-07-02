import random
import bisect
import time
from pathlib import Path

import numpy as np
import h5py          # use h5py for direct HDF5 access
import duckdb       # DuckDB for Parquet
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader

# -----------------------------------------------------------------------------
# 1) True-Uniform HDF5 Dataset (yields numpy batches via h5py)
# -----------------------------------------------------------------------------
class UniformHDF5Dataset(IterableDataset):
    def __init__(self, base_path: str, total_rows: int, chunk_size: int = 512):
        super().__init__()
        self.total_rows = total_rows
        self.chunk_size = chunk_size
        # discover HDF5 shards and their row counts
        self.shards = []  # list of (path, node, n_rows)
        for p in Path(base_path).rglob("*.h5"):
            # derive station/group path
            parts = p.stem.split("_")
            if len(parts) < 2:
                continue
            sta = parts[1]
            node = f"/{sta}/all_data"
            try:
                with h5py.File(p, 'r') as f:
                    n = f[node].shape[0]
                    self.shards.append((str(p), node, n))
            except Exception:
                continue
        assert self.shards, "No HDF5 shards found"
        counts = [n for _, _, n in self.shards]
        self.cum_counts = np.cumsum(counts)
        self.total_records = int(self.cum_counts[-1])

    def __iter__(self):
        emitted = 0
        while emitted < self.total_rows:
            need = min(self.chunk_size, self.total_rows - emitted)
            # sample global indices uniformly
            globals_ = sorted(random.randrange(self.total_records) for _ in range(need))
            # map to shards
            shard_map = {}
            for g in globals_:
                i = bisect.bisect_right(self.cum_counts, g)
                prev = int(self.cum_counts[i-1]) if i > 0 else 0
                local = g - prev
                shard_map.setdefault(self.shards[i], []).append(local)

            # collect batch arrays
            batch_arrays = []
            for (path, node, _), locals_ in shard_map.items():
                try:
                    with h5py.File(path, 'r') as f:
                        ds = f[node]
                        arr = ds[sorted(locals_)]  # fancy indexing
                        # structured array to 2D numeric array
                        batch_arrays.append(np.vstack([row.tolist() for row in arr]))
                except Exception as e:
                    print(f"⚠️ HDF5 {Path(path).name}: {e}")
            if batch_arrays:
                batch = np.concatenate(batch_arrays, axis=0)
                emitted += batch.shape[0]
                yield batch

# -----------------------------------------------------------------------------
# 2) Efficient DuckDB Parquet Dataset (yields numpy batches via SQL LIMIT/OFFSET)
# -----------------------------------------------------------------------------
class UniformDuckDBParquetDataset(IterableDataset):
    def __init__(self, base_path: str, total_rows: int, chunk_size: int = 512):
        super().__init__()
        self.total_rows = total_rows
        self.chunk_size = chunk_size
        # discover Parquet shards and their row counts
        conn = duckdb.connect()
        self.shards = []  # list of (path, n_rows)
        for p in Path(base_path).rglob("*.parquet"):
            try:
                n = conn.execute(f"SELECT count(*) FROM read_parquet('{p}')").fetchone()[0]
                self.shards.append((str(p), n))
            except Exception:
                continue
        conn.close()
        assert self.shards, "No Parquet shards found"
        counts = [n for _, n in self.shards]
        self.cum_counts = np.cumsum(counts)
        self.total_records = int(self.cum_counts[-1])

    def __iter__(self):
        conn = duckdb.connect()
        emitted = 0
        while emitted < self.total_rows:
            need = min(self.chunk_size, self.total_rows - emitted)
            globals_ = sorted(random.randrange(self.total_records) for _ in range(need))
            # map to shards
            shard_map = {}
            for g in globals_:
                i = bisect.bisect_right(self.cum_counts, g)
                prev = int(self.cum_counts[i-1]) if i > 0 else 0
                local = g - prev
                shard_map.setdefault(self.shards[i][0], []).append(local)
            # collect batch arrays by fetching contiguous runs
            batch_arrays = []
            for path, locals_ in shard_map.items():
                locals_.sort()
                # identify contiguous runs
                runs = []
                start = prev_idx = locals_[0]
                length = 1
                for idx in locals_[1:]:
                    if idx == prev_idx + 1:
                        length += 1
                    else:
                        runs.append((start, length))
                        start, length = idx, 1
                    prev_idx = idx
                runs.append((start, length))
                # fetch each run
                for start, length in runs:
                    sql = f"SELECT * FROM read_parquet('{path}') LIMIT {length} OFFSET {start}"
                    try:
                        df = conn.execute(sql).df()
                        batch_arrays.append(df.values)
                    except Exception as e:
                        print(f"⚠️ DuckDB {Path(path).name}: {e}")
            if batch_arrays:
                batch = np.vstack(batch_arrays)
                emitted += batch.shape[0]
                yield batch
        conn.close()

# -----------------------------------------------------------------------------
# Benchmark harness
# -----------------------------------------------------------------------------
def benchmark_loader(loader, label):
    t0, n = time.perf_counter(), 0
    for batch in tqdm(loader, desc=label):
        n += batch.shape[0]
    dt = time.perf_counter() - t0
    print(f"✅ {label}: {n} rows in {dt:.2f}s → {n/dt:.1f} rows/sec")


def main():
    N = 100000
    WORKERS = 4
    H5_PATH = "/scratch2/arrueegg/WP4/PNN_STEC/h5_data"
    PQ_PATH = "/scratch2/arrueegg/WP4/PNN_STEC/parquet_data"

    datasets = {
        "HDF5": UniformHDF5Dataset(H5_PATH, N),
        "DuckDB Parquet": UniformDuckDBParquetDataset(PQ_PATH, N),
    }

    for name, ds in datasets.items():
        loader = DataLoader(ds, batch_size=None, num_workers=WORKERS)
        print(f"\n🔍 Benchmarking {name}:")
        benchmark_loader(loader, name)

if __name__ == "__main__":
    main()
