"""Read one day of raw observations out of the STEC database.

Produces the raw, un-normalised columns that `transforms.FeatureAssembler` consumes, keyed
by feature name. Keeping the read separate from the transform is what lets the assembler be
tested against the legacy collation without a database, and lets this be tested against the
database without a model.

Three details are not obvious from the schema and are wrong in quiet ways if guessed:

* **`year` and `doy` come from the file, not from the rows.** The raw table has no such
  columns; the legacy dataset takes them from the directory path. Deriving them from data
  is where the float32 round-trip problem starts.
* **`local_time_hours` is derived**, from second-of-day and the IPP *longitude* - not the
  station longitude.
* **Space weather is hourly and joined by `sod // 3600`**, from a separate file whose
  column order is recovered from an attribute and masked to drop YEAR, DOY and HR. The
  legacy dataset filled a registry SWI name the file did not carry with 0.0
  (`if in_idx is None: value = 0.0`); that fallback is **not** preserved here. A name
  `read_space_weather` cannot find in the file's own column list is simply absent from the
  returned dict, and `FeatureAssembler.assemble` then raises `KeyError` for it if the
  layout needs it, rather than silently training on a zero. In practice this path is dead:
  `data/omni_hourly_2010-2025.h5`'s column attributes carry all six registry SWI names.

The legacy dataset reads one row per `__getitem__`. This reads whole columns, because every
consumer here wants a day at a time and a 2 M-row day fetched row-wise is minutes rather
than seconds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from ..config import paths

logger = logging.getLogger(__name__)

# Columns read straight out of the observation table.
RAW_COLUMNS = (
    "sod",
    "satazi",
    "satele",
    "lat_sta",
    "lon_sta",
    "sm_lat_sta",
    "sm_lon_sta",
    "lat_ipp",
    "lon_ipp",
    "sm_lat_ipp",
    "sm_lon_ipp",
)

IDENTITY_COLUMNS = ("station", "sat", "slipc", "gfphase")
TARGET_COLUMN = "stec"

SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24

# Present in the space-weather table but not model inputs.
SWI_INDEX_COLUMNS = ("YEAR", "DOY", "HR")


def compute_local_time_hours(sod: np.ndarray, longitude_deg: np.ndarray) -> np.ndarray:
    """Local solar time at the pierce point, in hours, wrapped to [0, 24).

    Longitude is the *IPP* longitude, not the station's: local time is a property of where
    the ray pierces the ionosphere, which is what the model is being asked about.
    """
    return np.mod(sod / SECONDS_PER_HOUR + longitude_deg / 15.0, HOURS_PER_DAY)


def read_space_weather(
    year: int, doy: int, path: Path | None = None
) -> dict[str, np.ndarray]:
    """The 24 hourly space-weather rows for one day, keyed by registry column name."""
    path = Path(path) if path is not None else paths.OMNI_INDICES
    if not path.exists():
        logger.warning(f"No space-weather file at {path}; indices will be absent")
        return {}

    with h5py.File(path, "r") as handle:
        group = f"{year}/{doy:03d}"
        if group not in handle:
            logger.warning(f"No space-weather data for {year}-{doy:03d}")
            return {}
        table = handle[group][:]
        names = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in handle[group].attrs["columns"]
        ]

    keep = [name not in SWI_INDEX_COLUMNS for name in names]
    kept_names = [name for name, wanted in zip(names, keep, strict=True) if wanted]
    kept = table[:, np.asarray(keep)]
    return {name: kept[:, i] for i, name in enumerate(kept_names)}


def read_day(
    year: int,
    doy: int,
    split: str = "test",
    database_root: Path | None = None,
    space_weather: Path | None = None,
    with_identity: bool = False,
) -> dict[str, np.ndarray]:
    """Raw columns for one day's split, in file order.

    File order matters: the test path is deliberately sequential so index-based joins back
    to this table stay valid, so this must not reorder anything.
    """
    day_file = (
        paths.stec_database_day(year, doy)
        if database_root is None
        else Path(database_root)
        / str(year)
        / f"{doy:03d}"
        / f"ccl_{year}{doy:03d}_30_5.h5"
    )
    if not day_file.exists():
        raise FileNotFoundError(
            f"No STEC database file for {year}-{doy:03d}: {day_file}"
        )

    group = f"{year}/{doy:03d}"
    with h5py.File(day_file, "r") as handle:
        index_name = f"{group}/{split}_idx"
        if index_name not in handle:
            raise KeyError(f"No {split} split in {day_file}")
        selection = handle[index_name][:]
        table = handle[f"{group}/all_data"]

        columns: dict[str, np.ndarray] = {}
        for name in (*RAW_COLUMNS, TARGET_COLUMN):
            if name in table.dtype.names:
                columns[name] = table[name][:][selection]
        if with_identity:
            for name in IDENTITY_COLUMNS:
                if name in table.dtype.names:
                    columns[name] = table[name][:][selection]

    rows = len(selection)
    # Constant per file, and taken from the file rather than recovered from the data.
    columns["year"] = np.full(rows, float(year), dtype=np.float32)
    columns["doy"] = np.full(rows, float(doy), dtype=np.float32)

    if "sod" in columns and "lon_ipp" in columns:
        columns["local_time_hours"] = compute_local_time_hours(
            columns["sod"].astype(np.float64), columns["lon_ipp"].astype(np.float64)
        ).astype(np.float32)

    hourly = read_space_weather(year, doy, space_weather)
    if hourly and "sod" in columns:
        hour = np.clip(
            (columns["sod"] // SECONDS_PER_HOUR).astype(int), 0, HOURS_PER_DAY - 1
        )
        for name, values in hourly.items():
            columns[name] = values[hour].astype(np.float32)

    return columns
