#!/usr/bin/env python3
"""
Utilities to load Madrigal HDF5 STEC files and extract STEC values for observations.

The Madrigal HDF5 format may vary between users. This module implements a
robust, best-effort loader that tries a few common layouts and gracefully
returns NaNs when nothing usable is found.

Assumptions (reasonable defaults):
- Madrigal files are organized per-day (year/doy) or as single HDF5 files
  containing station groups or datasets with 'stec' in the name.
- Station identifiers in the Madrigal files match those in
  `src/data_processing/test_station.list` (case-sensitive). If not, you may
  need to adapt the matching logic.

Functions:
- load_split_lists: read test station and month lists used by the rest of the
  pipeline.
- find_madrigal_file: try common path patterns to find a madrigal HDF5 for a
  given date.
- extract_stec_for_date: given an open h5py.File and a small DataFrame of
  observations for the same date (columns: station, sod), returns an array of
  STEC values aligned with the observations.
"""

from pathlib import Path
from typing import Tuple, Optional
import logging
import h5py
import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def load_split_lists(base_dir: str = "./src/data_processing") -> Tuple[set, set]:
    """Load `test_station.list` and `test_dates.list`.

    Returns (test_stations, test_months) where test_months contains strings
    like 'YYYY-MM' to compare against year-month of observations.
    """
    base = Path(base_dir)
    st_file = base / "test_station.list"
    mon_file = base / "test_dates.list"

    if not st_file.exists() or not mon_file.exists():
        logger.warning("Split list files not found in %s", str(base))
        return set(), set()

    stations = set(np.loadtxt(str(st_file), dtype=str))
    months = set(np.loadtxt(str(mon_file), dtype=str))
    return stations, months


def find_madrigal_file(madrigal_path: str, date_obj) -> Optional[Path]:
    """Try a few common paths for a madrigal HDF5 file for the given date.

    date_obj: datetime.date or datetime-like
    """
    mp = Path(madrigal_path)
    year, month, day = date_obj.year, date_obj.month, date_obj.day

    # Check if file exists in year/doy structure
    filename = f"los_{year}{month:02d}{day:02d}_IGS.h5"
    filepath = mp / str(year) / filename

    # safe directory listing
    try:
        _ = list((mp / str(year)).iterdir())
    except Exception:
        # directory doesn't exist or unreadable
        if filepath.exists():
            logger.debug("Found madrigal file for %s: %s", str(date_obj), filepath)
            return filepath
        logger.debug("Year directory not found for %s", str(year))
        return None

    if filepath.exists():
        logger.debug("Found madrigal file for %s: %s", str(date_obj), filepath)
        return filepath

    logger.debug("No madrigal file found for %s in %s", str(date_obj), madrigal_path)
    return None


def extract_stec_for_date(h5path: Path, obs_df) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract STEC values for observations contained in obs_df from the HDF5 file.
    Integer-key optimized version using Polars for very fast matching.

    obs_df must contain at least 'lat_sta', 'lon_sta', and 'sod'.
    Returns (stec_values, success_mask) aligned with obs_df index.
    """
    # Ensure obs_df is Polars
    if not isinstance(obs_df, pl.DataFrame):
        obs_df = pl.from_pandas(obs_df)

    # --- Load Madrigal HDF5 table ---
    with h5py.File(h5path, "r") as h5f:
        table_data = h5f["Data"]["Table Layout"][:]

    # --- Convert Madrigal dataset to Polars ---
    df_mad = pl.DataFrame(
        {
            "lat_sta": table_data["gdlatr"],
            "lon_sta": table_data["gdlonr"],
            "sod": table_data["sod"],
            "satele": table_data["elm"],
            "satazi": table_data["azm"],
            "los_tec": table_data["los_tec"],
            "gnss_type": table_data["gnss_type"],
        }
    )

    # --- Define integer rounding/encoding (matches your tolerances) ---
    # Multiply lat/lon by 1e3 -> 0.001° bins; round sod to nearest second.
    df_mad = df_mad.with_columns(
        [
            (pl.col("lat_sta") * 1_000).round(0).cast(pl.Int32).alias("lat_i"),
            (pl.col("lon_sta") * 1_000).round(0).cast(pl.Int32).alias("lon_i"),
            pl.col("sod").round(0).cast(pl.Int32).alias("sod_i"),
            pl.col("satele").round(1).cast(pl.Int16).alias("satele_i"),
            pl.col("satazi").round(1).cast(pl.Int16).alias("satazi_i"),
        ]
    )

    obs_df = obs_df.with_columns(
        [
            (pl.col("lat_sta") * 1_000).round(0).cast(pl.Int32).alias("lat_i"),
            (pl.col("lon_sta") * 1_000).round(0).cast(pl.Int32).alias("lon_i"),
            pl.col("sod").round(0).cast(pl.Int32).alias("sod_i"),
        ]
    )

    # Add optional integer rounding if available
    if "satele" in obs_df.columns:
        obs_df = obs_df.with_columns(
            pl.col("satele").round(1).cast(pl.Int16).alias("satele_i")
        )
    else:
        obs_df = obs_df.with_columns(pl.lit(None).cast(pl.Int16).alias("satele_i"))

    if "satazi" in obs_df.columns:
        obs_df = obs_df.with_columns(
            pl.col("satazi").round(1).cast(pl.Int16).alias("satazi_i")
        )
    else:
        obs_df = obs_df.with_columns(pl.lit(None).cast(pl.Int16).alias("satazi_i"))

    # --- Join keys ---
    keys = ["lat_i", "lon_i", "sod_i", "satele_i", "satazi_i"]

    # --- Perform ultra-fast Polars join ---
    joined = obs_df.join(df_mad, on=keys, how="left")

    # --- Extract STEC values ---
    stec_out = joined["los_tec"].to_numpy()
    success = ~np.isnan(stec_out)

    print(
        f"Extracted STEC for {np.sum(success)}/{len(obs_df)} observations from {h5path.name}"
    )

    return stec_out, success


def build_madrigal_stec_for_testset(
    madrigal_path: str, test_df: pd.DataFrame, logger
) -> pd.DataFrame:
    """Main helper to add madrigal STEC to `test_df`.

    Filters `test_df` to test stations and months specified in split lists and
    attempts to fetch madrigal STEC values. Returns a copy of the dataframe
    with columns `madrigal_stec` and `madrigal_success` appended.
    """
    df = test_df.copy()
    stations, months = load_split_lists()

    # Add columns
    df["madrigal_stec"] = np.nan
    df["madrigal_success"] = False

    # Filter rows to only test stations/months
    if stations:
        mask_station = df["station"].isin(stations)
    else:
        mask_station = np.ones(len(df), dtype=bool)

    if months:
        # build year-month column
        ym = df["date"].dt.strftime("%Y-%m") if "date" in df.columns else None
        if ym is not None:
            mask_month = ym.isin(months)
        else:
            mask_month = np.ones(len(df), dtype=bool)
    else:
        mask_month = np.ones(len(df), dtype=bool)

    filtered_idx = df.index[mask_station & mask_month]
    if len(filtered_idx) == 0:
        logger.info("No testset rows matched Madrigal station/month filters")
        return df

    # Group by date (year,doy) if available
    if "year" in df.columns and "doy" in df.columns:

        def _date_from_row(r):
            from datetime import datetime

            # `year`/`doy` are denormalised model inputs: `doy` round-trips through
            # (doy-1)/365 and a float32 inverse, landing 26 days of the year just under
            # the integer (DOY 189 -> 188.99998). A truncating int() shifts those into
            # the previous day - the same defect repair_gim_baseline.py fixes for the
            # IGS GIM baseline (see CLAUDE.md's Gotchas). round() is required here.
            return (
                datetime(round(r["year"]), 1, 1)
                + pd.Timedelta(days=round(r["doy"]) - 1)
            ).date()

        df_sub = df.loc[filtered_idx]
        grouped = df_sub.groupby(df_sub.apply(_date_from_row, axis=1))
    elif "date" in df.columns:
        grouped = df.loc[filtered_idx].groupby(df.loc[filtered_idx]["date"].dt.date)
    else:
        # fallback: try grouping by year and doy if present
        grouped = {}
        logger.warning(
            "test_df lacks 'date' or 'year'/'doy' columns; Madrigal matching might fail"
        )
        return df

    for date_obj, group in grouped:
        h5file = find_madrigal_file(madrigal_path, date_obj)
        if h5file is None:
            logger.debug("Madrigal file not found for %s", str(date_obj))
            continue

        stec_vals, success = extract_stec_for_date(h5file, group)
        for i, idx in enumerate(group.index):
            df.at[idx, "madrigal_stec"] = stec_vals[i]
            df.at[idx, "madrigal_success"] = bool(success[i])

    return df


def sample_madrigal_observations(
    madrigal_path: str, n_samples: int = 1000, seed: int = 42, max_files: int = 200
) -> pd.DataFrame:
    """Randomly sample up to `n_samples` observations from Madrigal HDF5 files.

    This reads Madrigal .h5 files under `madrigal_path`, extracts the
    table at `Data`/`Table Layout` (if present), aggregates rows from up to
    `max_files` files (stopping early if enough rows are collected), and
    returns a Pandas DataFrame with the sampled rows.

    The returned DataFrame preserves the structured array column names found
    in the Madrigal files (for example: 'sod', 'gdlatr', 'gdlonr', 'elm',
    'azm', 'los_tec', etc.). A helper column '__source_file' indicates the
    originating file.
    """
    mp = Path(madrigal_path)
    all_files = sorted(mp.rglob("*.h5"))
    rng = np.random.default_rng(seed)

    frames = []
    collected = 0
    files_used = 0

    for f in all_files:
        if files_used >= max_files:
            break
        try:
            with h5py.File(f, "r") as h5f:
                if "Data" not in h5f or "Table Layout" not in h5f["Data"]:
                    continue
                table = h5f["Data"]["Table Layout"][:]
                if len(table) == 0:
                    continue
                df = pd.DataFrame(table)
                df["__source_file"] = str(f)
                frames.append(df)
                collected += len(df)
                files_used += 1
                if collected >= n_samples * 2:
                    break
        except Exception:
            # skip unreadable or malformed files
            continue

    if not frames:
        return pd.DataFrame()

    big = pd.concat(frames, ignore_index=True)
    k = min(n_samples, len(big))
    # Use pandas sample with random_state for reproducibility
    sampled = big.sample(n=k, random_state=seed).reset_index(drop=True)
    return sampled
