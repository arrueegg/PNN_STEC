"""Madrigal HDF5 sample-build utilities, ported from `src/evaluation/madrigal_builder.py`.

Builds a small Madrigal-format HDF5 sample (matching the raw-database `PREPROC_DTYPE`
layout plus a `gnss_type` field) for inspection/tests. Used by
`scripts/build_madrigal_h5_sample.py`; unrelated to `stec.baselines.madrigal` (which reads
Madrigal's *own* `Table Layout` format to compare reference STEC, not this project's
`PREPROC_DTYPE` layout) or `stec.data.madrigal_reader` (which reads Madrigal as model
*input*, again in Madrigal's own column names).

Two adaptations from the source, both because the canonical locations moved during the
rebuild rather than because this port changes behaviour:

* File lookup goes through `stec.baselines.madrigal.find_madrigal_file`, which resolves
  `stec.config.paths.madrigal_day(year, month, day)` - the same canonical
  `MADRIGAL_ROOT/<year>/los_<YYYYMMDD>_IGS.h5` path CLAUDE.md documents - rather than
  globbing the `madrigal_path` argument by hand. `build_sample`'s `madrigal_path` parameter
  is kept for CLI-compatibility with `scripts/build_madrigal_h5_sample.py --madrigal_path`,
  but is no longer what selects the file; pass a `stec.config.paths.MADRIGAL_ROOT` override
  through the environment (see that module's docstring) if it needs to differ.
* Split date/station lists come from `stec.config.paths.date_list`/`station_list`
  (`stec/data/splits/`), not the pre-rebuild `src/data_processing/` location those files
  were `git mv`'d out of - see `positioning/scripts/generate_stec_corrections.py`'s
  `load_test_stations` docstring for the same correction.

`geographic_to_solar_magnetic` (`stec.data.coordinate_transforms`) is used for both station
and IPP coordinates here, faithfully reproducing the source's single-shell-height
approximation for both - see that module's docstring for why this is a known, deliberately
unfixed simplification rather than an oversight introduced by this port.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from functools import partial  # noqa: F401 - kept for parity with the source's imports
from multiprocessing import Pool, cpu_count
from pathlib import Path

import h5py
import numpy as np

from ..baselines.madrigal import find_madrigal_file
from ..config import paths
from .coordinate_transforms import geographic_to_solar_magnetic

logger = logging.getLogger(__name__)

# Structured dtype for the project's "one-table" HDF5 per split, copied from
# `src/utils/preprocessing.py::DTYPE` - that module (raw-database assembly) was not ported,
# so this is a standalone copy rather than an import, kept in sync by inspection since the
# raw layout has not changed since the rebuild started.
PREPROC_DTYPE = np.dtype(
    [
        ("station", "S8"),
        ("sat", "S4"),
        ("year", "i4"),
        ("doy", "i4"),
        ("stec", "f4"),
        ("vtec", "f4"),
        ("satele", "f4"),
        ("satazi", "f4"),
        ("lon_ipp", "f4"),
        ("lat_ipp", "f4"),
        ("sm_lat_ipp", "f4"),
        ("sm_lon_ipp", "f4"),
        ("sod", "f4"),
        ("lat_sta", "f4"),
        ("lon_sta", "f4"),
        ("sm_lat_sta", "f4"),
        ("sm_lon_sta", "f4"),
        ("gfphase", "f4"),
        ("slipc", "i4"),
    ]
)

MADRIGAL_DTYPE = np.dtype(PREPROC_DTYPE.descr + [("gnss_type", "i4")])

GNSS_TYPE_MAP = {
    b"GPS": 1,
    b"GLONASS": 2,
    b"GALILEO": 3,
    b"BEIDOU": 4,
    b"QZSS": 5,
    b"SBAS": 6,
    b"IRNSS": 7,
}


def parse_months(months_file: str) -> list[date]:
    months = sorted(set(np.loadtxt(months_file, dtype=str)))
    dates = []
    for month in months:
        year_i, mon_i = map(int, month.split("-"))
        start = datetime(year_i, mon_i, 1)
        end = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        d = start
        while d <= end:
            dates.append(d.date())
            d += timedelta(days=1)
    return dates


def _process_single_file(args):
    """Worker function to process a single Madrigal file (for `multiprocessing.Pool`).

    Args:
        args: (date_obj, madrigal_path, stations_bytes, tmp_dir, chunk_idx). `madrigal_path`
            is accepted for signature parity but unused - see the module docstring.
    """
    date_obj, _madrigal_path, stations_bytes, tmp_dir, chunk_idx = args

    h5file = find_madrigal_file(date_obj)
    if h5file is None:
        return None

    try:
        with h5py.File(h5file, "r") as mf:
            if "Data" not in mf:
                return None
            data_names = list(mf["Data"].keys())
            if not data_names:
                return None
            tabname = "Table Layout" if "Table Layout" in data_names else data_names[0]
            ds = mf["Data"][tabname]
            total_rows = ds.shape[0]

            # Sequential full read is much faster than fancy indexing at low selectivity.
            table_full = ds[:]

            mask = np.ones(total_rows, dtype=bool)

            if "sod" in table_full.dtype.names:
                mask &= table_full["sod"] % 300 == 0

            if stations_bytes is not None:
                station_field = None
                if "gps_site" in table_full.dtype.names:
                    station_field = "gps_site"
                elif "station" in table_full.dtype.names:
                    station_field = "station"

                if station_field is not None:
                    try:
                        station_lower = np.char.lower(
                            table_full[station_field].astype("S")
                        )
                    except Exception:
                        station_lower = table_full[station_field]
                    mask &= np.isin(station_lower, list(stations_bytes))

            if "los_tec" in table_full.dtype.names:
                stec_arr = table_full["los_tec"]
                mask &= ~np.isnan(stec_arr)
                mask &= stec_arr > 0

            if mask.sum() == 0:
                return None

            table = table_full[mask]

            n = len(table)
            block = np.zeros(n, dtype=MADRIGAL_DTYPE)

            if "gps_site" in table.dtype.names:
                block["station"] = table["gps_site"]
            elif "station" in table.dtype.names:
                block["station"] = table["station"]

            block["year"] = date_obj.year
            block["doy"] = date_obj.timetuple().tm_yday

            if "los_tec" in table.dtype.names:
                block["stec"] = table["los_tec"]
            else:
                block["stec"] = np.nan
            if "tec" in table.dtype.names:
                block["vtec"] = table["tec"]
            else:
                block["vtec"] = np.nan

            if "elm" in table.dtype.names:
                block["satele"] = table["elm"]
            if "azm" in table.dtype.names:
                block["satazi"] = table["azm"]

            if "gdlonr" in table.dtype.names:
                block["lon_sta"] = table["gdlonr"]
            elif "glon" in table.dtype.names:
                block["lon_sta"] = table["glon"]

            if "gdlatr" in table.dtype.names:
                block["lat_sta"] = table["gdlatr"]
            elif "gdlat" in table.dtype.names:
                block["lat_sta"] = table["gdlat"]

            if "gdlat" in table.dtype.names:
                block["lat_ipp"] = table["gdlat"]
            if "glon" in table.dtype.names:
                block["lon_ipp"] = table["glon"]

            if "sod" in table.dtype.names:
                block["sod"] = table["sod"]

            if "gnss_type" in table.dtype.names:
                gnss_raw = table["gnss_type"]
                gnss_stripped = np.char.strip(gnss_raw).astype("U10")

                gnss_int = np.zeros(len(gnss_stripped), dtype=np.int32)
                for gnss_name, gnss_id in GNSS_TYPE_MAP.items():
                    gnss_str = gnss_name.decode("utf-8")
                    gnss_int[gnss_stripped == gnss_str] = gnss_id
                block["gnss_type"] = gnss_int
            else:
                block["gnss_type"] = 0

            block["sm_lat_ipp"] = np.nan
            block["sm_lon_ipp"] = np.nan
            block["sm_lat_sta"] = np.nan
            block["sm_lon_sta"] = np.nan

            try:
                sod_selected = block["sod"].astype(float)
            except Exception:
                sod_selected = np.zeros(n, dtype=float)
            unique_sods, inverse_idx = np.unique(sod_selected, return_inverse=True)
            for u_idx, sod_val in enumerate(unique_sods):
                row_mask = inverse_idx == u_idx
                if not row_mask.any():
                    continue
                ts = datetime(date_obj.year, date_obj.month, date_obj.day) + timedelta(
                    seconds=float(sod_val)
                )
                lat_sta_arr = block["lat_sta"][row_mask]
                lon_sta_arr = block["lon_sta"][row_mask]
                lat_ipp_arr = block["lat_ipp"][row_mask]
                lon_ipp_arr = block["lon_ipp"][row_mask]
                try:
                    sm_lat_sta_arr, sm_lon_sta_arr = geographic_to_solar_magnetic(
                        lat_sta_arr, lon_sta_arr, ts
                    )
                    sm_lat_ipp_arr, sm_lon_ipp_arr = geographic_to_solar_magnetic(
                        lat_ipp_arr, lon_ipp_arr, ts
                    )
                except Exception:
                    sm_lat_sta_arr, sm_lon_sta_arr = lat_sta_arr, lon_sta_arr
                    sm_lat_ipp_arr, sm_lon_ipp_arr = lat_ipp_arr, lon_ipp_arr
                block["sm_lat_sta"][row_mask] = sm_lat_sta_arr
                block["sm_lon_sta"][row_mask] = sm_lon_sta_arr
                block["sm_lat_ipp"][row_mask] = sm_lat_ipp_arr
                block["sm_lon_ipp"][row_mask] = sm_lon_ipp_arr

            chunk_file = tmp_dir / f"madrigal_sample_chunk_{chunk_idx:04d}.h5"
            with h5py.File(chunk_file, "w") as cf:
                cf.create_dataset("data", data=block)

            return (
                str(chunk_file),
                n,
                str(h5file).split("/")[-1],
                total_rows,
                mask.sum(),
            )

    except Exception as e:
        logger.warning(
            "Failed processing date %s: %s", date_obj.strftime("%Y-%m-%d"), e
        )
        return None


def build_sample(
    madrigal_path: str,
    out_h5: str,
    split: str = "test",
    n_files: int = 50,
    tmp_dir: str | None = None,
    n_workers: int | None = None,
) -> str | None:
    """Build a small Madrigal HDF5 sample.

    Args:
        madrigal_path: accepted for CLI compatibility; actual file lookup goes through
            `stec.config.paths.madrigal_day` (see module docstring).
        out_h5: output HDF5 path.
        split: which split's date/station lists to use (train/val/test).
        n_files: maximum number of dates to process.
        tmp_dir: optional temporary chunk directory; default `data/temp_madrigal_sample`.
        n_workers: number of parallel workers; default uses all CPUs.

    Returns:
        Path to the created HDF5 (`out_h5`), or `None` if no data was found.
    """
    del madrigal_path  # retained only for CLI-signature compatibility; see docstring

    dates = parse_months(str(paths.date_list(split)))
    dates = dates[:n_files]

    logger.info(
        "Building Madrigal HDF5 sample: split=%s, max_dates=%s", split, len(dates)
    )
    logger.info("Dates discovered from %s: %d", paths.date_list(split), len(dates))

    stations_file = paths.station_list(split)
    if stations_file.exists():
        stations = np.loadtxt(str(stations_file), dtype=str)
        stations_bytes = set(s.strip().lower().encode("ascii") for s in stations)
        logger.info(
            "Loaded station filter (%s): %d stations",
            stations_file,
            len(stations_bytes),
        )
    else:
        stations_bytes = None

    out_path = Path(out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = (
        Path(tmp_dir) if tmp_dir is not None else Path("data") / "temp_madrigal_sample"
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n_workers = n_workers if n_workers is not None else cpu_count()
    logger.info("Using %d parallel workers", n_workers)

    worker_args = [
        (date_obj, None, stations_bytes, tmp_dir, idx)
        for idx, date_obj in enumerate(dates)
    ]

    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_process_single_file, worker_args)
    else:
        results = [_process_single_file(args) for args in worker_args]

    chunk_files = []
    for result in results:
        if result is not None:
            chunk_file, n_rows, filename, total_rows, filtered_rows = result
            chunk_files.append(chunk_file)
            logger.info(
                "Processed %s: %d -> %d rows, wrote %s",
                filename,
                total_rows,
                filtered_rows,
                Path(chunk_file).name,
            )

    if not chunk_files:
        logger.info("No chunks created, nothing to merge. Exiting.")
        return None

    total_records = 0
    for cf in chunk_files:
        with h5py.File(cf, "r") as f:
            total_records += f["data"].shape[0]

    logger.info(
        "Merging %d chunks into %s total_records=%d",
        len(chunk_files),
        out_path,
        total_records,
    )
    with h5py.File(out_path, "w") as final_f:
        final_dataset = final_f.create_dataset(
            "data",
            shape=(total_records,),
            dtype=MADRIGAL_DTYPE,
            chunks=(min(8192, max(1, total_records // 100)),),
            compression="gzip",
        )
        current_offset = 0
        for cf in chunk_files:
            with h5py.File(cf, "r") as f:
                chunk_data = f["data"][:]
                chunk_size = chunk_data.shape[0]
                final_dataset[current_offset : current_offset + chunk_size] = chunk_data
                current_offset += chunk_size

    for cf in chunk_files:
        try:
            os.remove(cf)
        except OSError:
            pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    logger.info("Wrote final file %s records=%d", out_path, total_records)
    return str(out_path)
