"""Independent re-derivation of Figures 5-8's per-bin MAE/RMSE, straight from the store.

`docs/revision/results_register.md` flags this as an open item (Item I): Figures 5-8
(`fig_residuals_elev`, `fig_residuals_lat`, `fig_residuals_localtime`,
`fig_residuals_year_month` in `stec.viz.manuscript_figures`) have only ever been checked
against `stec.analysis.pretrained_test_diagnostics`'s own per-observation cache
(`observations.parquet`) - the same cache the figures themselves read. Agreement between a
figure and that cache proves the figure aggregates the cache correctly; it does not prove
the cache faithfully reflects `predictions/pretrained_stec/own`, the raw 544-day-file
prediction store the cache was built from.

This module never touches that cache. It streams `predictions/pretrained_stec/own`
directly via `prediction_store.iter_days`/`store_path`, one day at a time, and
reimplements each figure's stated bin definition from scratch - read from
`stec/viz/manuscript_figures.py`, not imported from it - using running accumulators
(`stec.analysis.daily_metrics`'s reference pattern) rather than holding a multi-day array
in memory. That is a deliberately different computational strategy from the
cache-then-hold-everything approach the figures themselves use, so agreement between the
two is stronger evidence of correctness than re-running the same strategy twice.

Three of the four bin definitions are fixed and need one streaming pass:

* elevation - 17 bins, `np.linspace(5, 90, 18)` (`fig_residuals_elev`'s
  `_ELEVATION_BIN_RANGE`/`_ELEVATION_NUM_BINS` defaults).
* solar-magnetic latitude - 18 bins, `np.arange(-90, 91, 10)`, right-open
  (`fig_residuals_lat`'s `_GEOMAGNETIC_LAT_BIN_EDGES`), every bin kept even if empty.
* year-month - not a numeric bin at all: every row in one store day-file shares the same
  `(year, doy)`, so this is a direct dict accumulation keyed by the calendar year-month
  string `fig_residuals_year_month` derives from `(year, doy)`.

Local solar time is not fixed: `fig_residuals_localtime` calls `pd.cut(local_time_hours,
bins=24)` over the whole test set, and pandas' integer-`bins` mode computes edges from the
data's own min/max (padded by 0.1% of the range) - not clean [0, 24) hour boundaries.
Reproducing that needs two passes: pass 1 tracks the running min/max of an
independently-derived `local_time_hours` (`(sod/3600 + lon_ipp/15) % 24` - the same solar-
local-time formula `stratified_comparison.add_local_time` uses, reimplemented here rather
than imported, so this check does not depend on the helper it exists to verify against);
pass 2 bins with the edges those two numbers imply and accumulates. Feeding `pd.cut` an
array containing only the true min and max reproduces the exact edges `pd.cut` would
derive from the full 10,000,000-row series, because pandas' integer-`bins` algorithm
depends only on the input's min/max, never on the rest of its distribution - verified
directly against a synthetic full array before relying on it here.

Usage::

    python -m stec.analysis.pretrained_residuals_from_store
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import paths
from ..inference import prediction_store as ps

logger = logging.getLogger(__name__)

MODEL_VARIANT = "pretrained_stec"
DATASET = "own"

TRUTH_COLUMN = "true_stec"
PRED_COLUMN = "stec_pred"

# fig_residuals_elev's defaults: np.linspace(5.0, 90.0, 17 + 1).
ELEVATION_BIN_EDGES = np.linspace(5.0, 90.0, 18)

# fig_residuals_lat's _GEOMAGNETIC_LAT_BIN_EDGES.
LATITUDE_BIN_EDGES = np.arange(-90.0, 91.0, 10.0)

# fig_residuals_localtime's `pd.cut(..., bins=24)`.
N_LOCALTIME_BINS = 24

# Columns pass 1 needs: truth/prediction for the residual, satele/sm_lat_ipp for the
# elevation/latitude bins, sod/lon_ipp to derive local time where it is not already a
# column (every pretrained_stec/own day, per pretrained_test_diagnostics.py's docstring).
# year/doy are not read from the file at all - they come from the (year, doy) `iter_days`
# already parsed from the partition path, which is the day the caller actually asked for
# and is authoritative (see prediction_store.write_predictions's own comment on why the
# partition identity, not a row's denormalised model input, is the one to trust).
WANTED_COLUMNS = [
    TRUTH_COLUMN,
    PRED_COLUMN,
    "satele",
    "sm_lat_ipp",
    "local_time_hours",
    "sod",
    "lon_ipp",
]

DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir(
    "pretrained_residuals_from_store", rebuilt=True
)


def _wanted_columns(path: Path) -> list[str]:
    """Restrict the read to columns this day's file actually has (same reasoning as
    `daily_metrics._wanted_columns` and `pretrained_test_diagnostics._wanted_columns`)."""
    present = set(pq.ParquetFile(path).schema.names)
    return [column for column in WANTED_COLUMNS if column in present]


def _local_time_hours(frame: pd.DataFrame) -> np.ndarray | None:
    """Solar local time at the pierce point: UTC plus 15 degrees of longitude per hour.

    Reimplemented independently of `stratified_comparison.add_local_time` (same formula,
    fresh code) so this check does not call into the module it exists to verify against.
    """
    if "local_time_hours" in frame.columns:
        return frame["local_time_hours"].to_numpy(dtype=np.float64)
    if {"sod", "lon_ipp"} <= set(frame.columns):
        sod = frame["sod"].to_numpy(dtype=np.float64)
        lon_ipp = frame["lon_ipp"].to_numpy(dtype=np.float64)
        return (sod / 3600.0 + lon_ipp / 15.0) % 24.0
    return None


def _year_month_label(year: int, doy: int) -> str:
    """Calendar year-month for a `(year, doy)` pair, matching `fig_residuals_year_month`'s
    `pd.to_datetime(year) + pd.to_timedelta(doy - 1, "D")` -> `Period("M")` -> `str`,
    computed directly since every row in one store day-file shares the same `(year, doy)`.
    """
    date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
    return f"{date.year:04d}-{date.month:02d}"


def _bin_sums(
    values: np.ndarray, residual: np.ndarray, edges: np.ndarray, *, right: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One day's contribution to (sum |residual|, sum residual^2, count) per fixed bin.

    `pd.cut` on a plain ndarray returns a `Categorical` whose `.codes` already index the
    bin position directly (no string label to parse, no rounding of an `Interval`'s
    float edges) - -1 marks a value outside every bin, filtered out before `bincount`.
    """
    n_bins = len(edges) - 1
    codes = pd.cut(values, bins=edges, include_lowest=True, right=right).codes
    valid = codes >= 0
    sum_abs = np.bincount(
        codes[valid], weights=np.abs(residual[valid]), minlength=n_bins
    )
    sum_sq = np.bincount(codes[valid], weights=residual[valid] ** 2, minlength=n_bins)
    count = np.bincount(codes[valid], minlength=n_bins)
    return sum_abs, sum_sq, count


def _mae_rmse(
    sum_abs: np.ndarray, sum_sq: np.ndarray, count: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = np.where(count > 0, sum_abs / count, np.nan)
        rmse = np.where(count > 0, np.sqrt(sum_sq / count), np.nan)
    return mae, rmse


def collect(
    store_root: Path,
    years: list[int] | None = None,
    doys: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Stream `pretrained_stec/own` and return the four independently-recomputed tables.

    Keys: `residuals_elev`, `residuals_lat`, `residuals_year_month`, and
    `residuals_localtime` (omitted, with a warning, if no day carries `local_time_hours`
    or the `sod`/`lon_ipp` pair needed to derive it).
    """
    day_paths = ps.day_paths(
        MODEL_VARIANT, DATASET, years=years, doys=doys, root=store_root
    )
    if not day_paths:
        raise FileNotFoundError(
            f"No prediction files matched for {MODEL_VARIANT}/{DATASET} "
            f"(years={years}, doys={doys}) under {store_root}"
        )
    logger.info(f"{len(day_paths)} day(s) to read for {MODEL_VARIANT}/{DATASET}")

    n_elev = len(ELEVATION_BIN_EDGES) - 1
    n_lat = len(LATITUDE_BIN_EDGES) - 1
    elev_sum_abs = np.zeros(n_elev)
    elev_sum_sq = np.zeros(n_elev)
    elev_count = np.zeros(n_elev, dtype=np.int64)
    lat_sum_abs = np.zeros(n_lat)
    lat_sum_sq = np.zeros(n_lat)
    lat_count = np.zeros(n_lat, dtype=np.int64)
    month_sum_abs: dict[str, float] = {}
    month_sum_sq: dict[str, float] = {}
    month_count: dict[str, int] = {}

    local_time_min = np.inf
    local_time_max = -np.inf
    have_local_time = False

    days_read = 0
    rows_read = 0
    for path in day_paths:
        year = int(path.parent.name.split("=")[1])
        doy = int(path.stem.split("=")[1])
        wanted = _wanted_columns(path)
        if TRUTH_COLUMN not in wanted or PRED_COLUMN not in wanted:
            logger.warning(f"{year}-{doy:03d} is missing truth/prediction, skipping")
            continue

        frame = pd.read_parquet(path, columns=wanted)
        days_read += 1
        rows_read += len(frame)
        residual = frame[TRUTH_COLUMN].to_numpy(dtype=np.float64) - frame[
            PRED_COLUMN
        ].to_numpy(dtype=np.float64)

        if "satele" in frame.columns:
            sum_abs, sum_sq, count = _bin_sums(
                frame["satele"].to_numpy(dtype=np.float64),
                residual,
                ELEVATION_BIN_EDGES,
                right=True,
            )
            elev_sum_abs += sum_abs
            elev_sum_sq += sum_sq
            elev_count += count

        if "sm_lat_ipp" in frame.columns:
            sum_abs, sum_sq, count = _bin_sums(
                frame["sm_lat_ipp"].to_numpy(dtype=np.float64),
                residual,
                LATITUDE_BIN_EDGES,
                right=False,
            )
            lat_sum_abs += sum_abs
            lat_sum_sq += sum_sq
            lat_count += count

        label = _year_month_label(year, doy)
        month_sum_abs[label] = month_sum_abs.get(label, 0.0) + float(
            np.abs(residual).sum()
        )
        month_sum_sq[label] = month_sum_sq.get(label, 0.0) + float(np.sum(residual**2))
        month_count[label] = month_count.get(label, 0) + residual.size

        local_time = _local_time_hours(frame)
        if local_time is not None and local_time.size:
            local_time_min = min(local_time_min, float(np.nanmin(local_time)))
            local_time_max = max(local_time_max, float(np.nanmax(local_time)))
            have_local_time = True

    if days_read == 0:
        raise RuntimeError(
            f"no day under {store_root}/{MODEL_VARIANT}/{DATASET} carried "
            f"{TRUTH_COLUMN}/{PRED_COLUMN} - the store is empty or missing the columns "
            "Figures 5-8 need."
        )
    logger.info(f"pass 1 done: {days_read} day(s), {rows_read:,} row(s)")

    elev_mae, elev_rmse = _mae_rmse(elev_sum_abs, elev_sum_sq, elev_count)
    elevation_table = pd.DataFrame(
        {
            "bin_left": ELEVATION_BIN_EDGES[:-1],
            "bin_right": ELEVATION_BIN_EDGES[1:],
            "mae": elev_mae,
            "rmse": elev_rmse,
            "n": elev_count,
        }
    )

    lat_mae, lat_rmse = _mae_rmse(lat_sum_abs, lat_sum_sq, lat_count)
    latitude_table = pd.DataFrame(
        {
            "lat_bin_center": (LATITUDE_BIN_EDGES[:-1] + LATITUDE_BIN_EDGES[1:]) / 2,
            "mae": lat_mae,
            "rmse": lat_rmse,
            "n": lat_count,
        }
    )

    months = sorted(month_count)
    month_sum_abs_arr = np.array([month_sum_abs[m] for m in months])
    month_sum_sq_arr = np.array([month_sum_sq[m] for m in months])
    month_count_arr = np.array([month_count[m] for m in months], dtype=np.int64)
    month_mae, month_rmse = _mae_rmse(
        month_sum_abs_arr, month_sum_sq_arr, month_count_arr
    )
    year_month_table = pd.DataFrame(
        {
            "year_month": months,
            "mae": month_mae,
            "rmse": month_rmse,
            "n": month_count_arr,
        }
    )

    tables = {
        "residuals_elev": elevation_table,
        "residuals_lat": latitude_table,
        "residuals_year_month": year_month_table,
    }

    if not have_local_time:
        logger.warning(
            "no day carried local_time_hours or sod/lon_ipp - skipping residuals_localtime"
        )
        return tables

    # Pass 2: pd.cut on an array holding only the true min/max reproduces the exact edges
    # pd.cut(bins=24) would derive from the full 10 M-row series (verified against a
    # synthetic full array - see the module docstring), so this is not an approximation of
    # the figure's edges, it is the same edges.
    _, localtime_edges = pd.cut(
        np.array([local_time_min, local_time_max]), bins=N_LOCALTIME_BINS, retbins=True
    )
    lt_sum_abs = np.zeros(N_LOCALTIME_BINS)
    lt_sum_sq = np.zeros(N_LOCALTIME_BINS)
    lt_count = np.zeros(N_LOCALTIME_BINS, dtype=np.int64)

    for path in day_paths:
        year = int(path.parent.name.split("=")[1])
        doy = int(path.stem.split("=")[1])
        wanted = _wanted_columns(path)
        if TRUTH_COLUMN not in wanted or PRED_COLUMN not in wanted:
            continue
        frame = pd.read_parquet(path, columns=wanted)
        local_time = _local_time_hours(frame)
        if local_time is None:
            continue
        residual = frame[TRUTH_COLUMN].to_numpy(dtype=np.float64) - frame[
            PRED_COLUMN
        ].to_numpy(dtype=np.float64)
        # No include_lowest: matches fig_residuals_localtime's own `pd.cut(..., bins=24)`
        # call, which does not pass it either - the 0.1%-of-range padding baked into
        # `localtime_edges` is what captures the minimum without needing it.
        codes = pd.cut(local_time, bins=localtime_edges).codes
        valid = codes >= 0
        lt_sum_abs += np.bincount(
            codes[valid], weights=np.abs(residual[valid]), minlength=N_LOCALTIME_BINS
        )
        lt_sum_sq += np.bincount(
            codes[valid], weights=residual[valid] ** 2, minlength=N_LOCALTIME_BINS
        )
        lt_count += np.bincount(codes[valid], minlength=N_LOCALTIME_BINS)

    lt_mae, lt_rmse = _mae_rmse(lt_sum_abs, lt_sum_sq, lt_count)
    tables["residuals_localtime"] = pd.DataFrame(
        {"hour": range(N_LOCALTIME_BINS), "mae": lt_mae, "rmse": lt_rmse, "n": lt_count}
    )
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to these years; default is every year in the store (2014-2024).",
    )
    parser.add_argument(
        "--doys",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to these day-of-year values; default is every day in the store.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    started = time.monotonic()
    tables = collect(args.store_root, years=args.years, doys=args.doys)
    elapsed = time.monotonic() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)

    manifest = pd.DataFrame(
        [
            {"table": name, "rows": len(table), "elapsed_seconds": round(elapsed, 1)}
            for name, table in tables.items()
        ]
    )
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)

    logger.info(
        f"wrote {len(tables)} table(s) to {args.output_dir} in {elapsed:.1f}s: "
        + ", ".join(f"{name} ({len(table)} bins)" for name, table in tables.items())
    )


if __name__ == "__main__":
    main()
