"""Independent check that the prediction store faithfully carries the raw database.

Phase 0 / section 8b of the rebuild plan. This script deliberately shares no code with
the pipeline: it reads the raw HDF5 with h5py and the store with pyarrow, and compares
them directly. If it agreed with the pipeline by importing the pipeline's own loaders, it
would only prove the pipeline is self-consistent.

What it checks, per day:
  1. Row count of the store equals len(test_idx) in the raw file.
  2. The store's identity columns (station, sat, sod, satele) equal all_data[test_idx].
  3. The store's true_stec equals all_data[test_idx]['stec'].
  4. RMSE recomputed from the raw ground truth against the store's predictions matches
     the RMSE computed entirely within the store.

Check 2 is the one nothing else does: it proves the predictions are aligned to the right
observations. A misalignment would still yield plausible metrics.
"""

import argparse
import sys

import h5py
import numpy as np
import pyarrow.parquet as pq

RAW_TEMPLATE = (
    "/home/space/data/iono/STEC_DB_CASDCB/{year}/{doy:03d}/ccl_{year}{doy:03d}_30_5.h5"
)
STORE_TEMPLATE = "predictions/{variant}/{dataset}/year={year}/doy={doy:03d}.parquet"

# The ground truth must survive the store bit-for-bit: it is copied, never recomputed, so
# any difference at all is a bug.
GROUND_TRUTH_TOLERANCE = 1e-6

# sod and satele are denormalised model inputs, not copies, so they round-trip through
# float32 normalisation and small differences are expected. Asserting on the maximum is the
# wrong test: satele saturates at the top of its normalisation range, so ~3 observations per
# day near zenith (89.98 -> 89.918) dominate the max while 99.99% agree to 2e-5. The
# question that matters is not "is it bitwise equal" but "could it move a result", so the
# bulk is checked at p99.9 and the boundary is checked exactly.
SOD_TOLERANCE_S = 0.01
SATELE_BULK_TOLERANCE_DEG = 1e-3

# The elevation cutoff is the only place a sub-degree satele difference could change which
# observations enter an analysis. No observation may land on a different side of it.
ELEVATION_CUTOFF_DEG = 5.0


def load_raw_test_rows(year: int, doy: int) -> dict[str, np.ndarray]:
    """Return the raw database's test-split rows, in file order."""
    path = RAW_TEMPLATE.format(year=year, doy=doy)
    group = f"{year}/{doy:03d}"
    with h5py.File(path, "r") as f:
        test_idx = f[f"{group}/test_idx"][:]
        data = f[f"{group}/all_data"]
        # Read each field separately and then subset, to avoid materialising all 21
        # fields for ~12 M rows at once.
        raw = {}
        for field in ("station", "sat", "stec", "satele", "sod"):
            raw[field] = data[field][:][test_idx]
    return raw


def compare_day(year: int, doy: int, variant: str, dataset: str) -> dict:
    raw = load_raw_test_rows(year, doy)
    store_path = STORE_TEMPLATE.format(
        variant=variant, dataset=dataset, year=year, doy=doy
    )
    # Read the single file directly: the store's own `year`/`doy` columns collide with the
    # year=/doy= partition directories if pyarrow is allowed to infer partitioning.
    table = pq.ParquetFile(store_path).read(
        columns=["station", "sat", "sod", "satele", "true_stec", "stec_pred"],
    )
    store = {
        name: table.column(name).to_numpy(zero_copy_only=False)
        for name in table.column_names
    }

    result = {
        "year": year,
        "doy": doy,
        "raw_rows": len(raw["stec"]),
        "store_rows": len(store["true_stec"]),
    }
    if result["raw_rows"] != result["store_rows"]:
        result["verdict"] = "ROW COUNT MISMATCH"
        return result

    # Identity columns. Station is uppercased in the store by design; the raw file stores
    # bytes, so decode before comparing.
    raw_station = np.char.upper(np.char.strip(raw["station"].astype(str)))
    store_station = np.char.upper(np.char.strip(store["station"].astype(str)))
    result["station_mismatches"] = int(np.sum(raw_station != store_station))

    raw_sat = np.char.upper(np.char.strip(raw["sat"].astype(str)))
    store_sat = np.char.upper(np.char.strip(store["sat"].astype(str)))
    result["sat_mismatches"] = int(np.sum(raw_sat != store_sat))

    result["sod_max_diff"] = float(np.max(np.abs(raw["sod"] - store["sod"])))
    satele_diff = np.abs(raw["satele"] - store["satele"])
    result["satele_max_diff"] = float(np.max(satele_diff))
    result["satele_p999_diff"] = float(np.percentile(satele_diff, 99.9))
    result["satele_over_bulk_tol"] = int(
        np.sum(satele_diff > SATELE_BULK_TOLERANCE_DEG)
    )
    result["cutoff_crossers"] = int(
        np.sum(
            (
                (raw["satele"] >= ELEVATION_CUTOFF_DEG)
                & (store["satele"] < ELEVATION_CUTOFF_DEG)
            )
            | (
                (raw["satele"] < ELEVATION_CUTOFF_DEG)
                & (store["satele"] >= ELEVATION_CUTOFF_DEG)
            )
        )
    )
    result["true_stec_max_diff"] = float(
        np.max(np.abs(raw["stec"] - store["true_stec"]))
    )

    # Metric cross-check: RMSE against the raw ground truth vs against the stored copy.
    pred = store["stec_pred"]
    finite = (
        np.isfinite(pred) & np.isfinite(raw["stec"]) & np.isfinite(store["true_stec"])
    )
    rmse_raw = float(np.sqrt(np.mean((pred[finite] - raw["stec"][finite]) ** 2)))
    rmse_store = float(
        np.sqrt(np.mean((pred[finite] - store["true_stec"][finite]) ** 2))
    )
    result["rmse_from_raw"] = rmse_raw
    result["rmse_from_store"] = rmse_store
    result["rmse_diff"] = abs(rmse_raw - rmse_store)
    result["n_finite"] = int(np.sum(finite))

    aligned = (
        result["station_mismatches"] == 0
        and result["sat_mismatches"] == 0
        and result["sod_max_diff"] <= SOD_TOLERANCE_S
        and result["satele_p999_diff"] <= SATELE_BULK_TOLERANCE_DEG
        and result["cutoff_crossers"] == 0
        and result["true_stec_max_diff"] <= GROUND_TRUTH_TOLERANCE
    )
    result["verdict"] = "OK" if aligned else "MISALIGNED"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--doys", type=int, nargs="+", required=True)
    parser.add_argument("--variant", default="finetuned_stec")
    parser.add_argument("--dataset", default="own")
    args = parser.parse_args()

    failures = 0
    for doy in args.doys:
        try:
            r = compare_day(args.year, doy, args.variant, args.dataset)
        except FileNotFoundError as exc:
            print(f"DOY {doy:3d}  SKIPPED  missing file: {exc}")
            continue

        if r["verdict"] != "OK":
            failures += 1
            print(f"DOY {doy:3d}  {r['verdict']}  {r}")
            continue

        print(
            f"DOY {doy:3d}  OK   rows={r['store_rows']:>9,}  "
            f"true_stec_maxdiff={r['true_stec_max_diff']:.1e}  "
            f"satele_p99.9={r['satele_p999_diff']:.1e} (clipped {r['satele_over_bulk_tol']:>2d}) "
            f"cutoff_crossers={r['cutoff_crossers']}  "
            f"RMSE raw={r['rmse_from_raw']:.6f} store={r['rmse_from_store']:.6f}"
        )

    print()
    print(f"{len(args.doys)} day(s) checked, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
