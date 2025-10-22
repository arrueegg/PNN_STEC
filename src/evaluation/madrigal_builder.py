"""Madrigal HDF5 build utilities.

This module exposes programmatic functions to build Madrigal-format HDF5 files
that match the project's preprocessing `DTYPE` layout plus an additional `gnss_type` field.
The sample builder was refactored out of the scripts folder so other parts of the codebase can
reuse it (production builder, tests, notebooks, etc.).

The implementation mirrors the behavior from the legacy `scripts/build_madrigal_h5_sample.py`:
- find per-day Madrigal files
- build light-weight masks (read `sod` and station fields first)
- read only selected rows and map them to `MADRIGAL_DTYPE` (PREPROC_DTYPE + gnss_type)
- compute precise solar-magnetic coordinates using `geographic_to_solar_magnetic`
- write temporary chunk H5 files then merge into a final root `data` dataset
- PARALLEL processing of multiple files for speed
"""
from pathlib import Path
import os
from datetime import datetime, timedelta
import numpy as np
import h5py
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

from src.evaluation.madrigal_loader import find_madrigal_file
from src.utils.preprocessing import DTYPE as PREPROC_DTYPE
from src.utils.coordinate_transforms import geographic_to_solar_magnetic

# Madrigal-specific extended DTYPE (adds gnss_type to standard PREPROC_DTYPE)
MADRIGAL_DTYPE = np.dtype(
    PREPROC_DTYPE.descr + [("gnss_type", "i4")]
)

# GNSS constellation name to integer mapping
GNSS_TYPE_MAP = {
    b'GPS': 1,
    b'GLONASS': 2,  # GLONASS
    b'GALILEO': 3,  # Galileo
    b'BEIDOU': 4,  # BeiDou
    b'QZSS': 5,  # QZSS
    b'SBAS': 6,
    b'IRNSS': 7,
}

logger = logging.getLogger(__name__)

def parse_months(months_file: str):
    months = sorted(set(np.loadtxt(months_file, dtype=str)))
    from datetime import datetime, timedelta

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
    """Worker function to process a single Madrigal file.
    
    This function is designed to be called by multiprocessing.Pool.
    
    Args:
        args: tuple of (date_obj, madrigal_path, stations_bytes, tmp_dir, chunk_idx)
    
    Returns:
        tuple of (chunk_file_path, num_rows) or None if processing failed
    """
    date_obj, madrigal_path, stations_bytes, tmp_dir, chunk_idx = args
    
    h5file = find_madrigal_file(madrigal_path, date_obj)
    if h5file is None:
        return None
    
    try:
        with h5py.File(h5file, 'r') as mf:
            if 'Data' not in mf:
                return None
            data_names = list(mf['Data'].keys())
            if not data_names:
                return None
            tabname = 'Table Layout' if 'Table Layout' in data_names else data_names[0]
            ds = mf['Data'][tabname]
            total_rows = ds.shape[0]

            # OPTIMIZATION: Read full dataset ONCE (much faster than fancy indexing for low selectivity)
            # Sequential read of 13M rows: ~4-5s vs fancy indexing: ~26s
            table_full = ds[:]
            
            # Build mask directly from full data
            mask = np.ones(total_rows, dtype=bool)
            
            # SOD filter
            if 'sod' in table_full.dtype.names:
                mask &= (table_full['sod'] % 300 == 0)
            
            # Station filter
            if stations_bytes is not None:
                station_field = None
                if 'gps_site' in table_full.dtype.names:
                    station_field = 'gps_site'
                elif 'station' in table_full.dtype.names:
                    station_field = 'station'
                
                if station_field is not None:
                    try:
                        station_lower = np.char.lower(table_full[station_field].astype('S'))
                    except Exception:
                        station_lower = table_full[station_field]
                    mask &= np.isin(station_lower, list(stations_bytes))
            
            # STEC filter
            if 'los_tec' in table_full.dtype.names:
                stec_arr = table_full['los_tec']
                mask &= ~np.isnan(stec_arr)
                mask &= (stec_arr > 0)

            if mask.sum() == 0:
                return None
            
            # Filter the full dataset in memory (very fast ~0.1s)
            table = table_full[mask]
            
            n = len(table)
            block = np.zeros(n, dtype=MADRIGAL_DTYPE)
            
            # Copy fields from table to block
            
            # Station field
            if 'gps_site' in table.dtype.names:
                block['station'] = table['gps_site']
            elif 'station' in table.dtype.names:
                block['station'] = table['station']
            
            # Date fields
            block['year'] = date_obj.year
            block['doy'] = date_obj.timetuple().tm_yday
            
            # TEC values
            if 'los_tec' in table.dtype.names:
                block['stec'] = table['los_tec']
            else:
                block['stec'] = np.nan
            if 'tec' in table.dtype.names:
                block['vtec'] = table['tec']
            else:
                block['vtec'] = np.nan
            
            # Angles
            if 'elm' in table.dtype.names:
                block['satele'] = table['elm']
            if 'azm' in table.dtype.names:
                block['satazi'] = table['azm']
            
            # Station coordinates
            if 'gdlonr' in table.dtype.names:
                block['lon_sta'] = table['gdlonr']
            elif 'glon' in table.dtype.names:
                block['lon_sta'] = table['glon']
                
            if 'gdlatr' in table.dtype.names:
                block['lat_sta'] = table['gdlatr']
            elif 'gdlat' in table.dtype.names:
                block['lat_sta'] = table['gdlat']
            
            # IPP coordinates  
            if 'gdlat' in table.dtype.names:
                block['lat_ipp'] = table['gdlat']
            if 'glon' in table.dtype.names:
                block['lon_ipp'] = table['glon']
            
            # SOD
            if 'sod' in table.dtype.names:
                block['sod'] = table['sod']
            
            # GNSS type conversion
            if 'gnss_type' in table.dtype.names:
                # Convert GNSS constellation names to integers (vectorized)
                gnss_raw = table['gnss_type']
                gnss_stripped = np.char.strip(gnss_raw).astype('U10')  # Strip and convert to unicode for comparison
                
                # Vectorized mapping
                gnss_int = np.zeros(len(gnss_stripped), dtype=np.int32)
                for gnss_name, gnss_id in GNSS_TYPE_MAP.items():
                    gnss_str = gnss_name.decode('utf-8')  # Convert bytes to string
                    gnss_int[gnss_stripped == gnss_str] = gnss_id
                block['gnss_type'] = gnss_int
            else:
                block['gnss_type'] = 0

            # SM coordinates initialized to NaN (computed below)
            block['sm_lat_ipp'] = np.nan
            block['sm_lon_ipp'] = np.nan
            block['sm_lat_sta'] = np.nan
            block['sm_lon_sta'] = np.nan

            # Optimized SM computation: compute once per unique timestamp
            try:
                sod_selected = block['sod'].astype(float)
            except Exception:
                sod_selected = np.zeros(n, dtype=float)
            unique_sods, inverse_idx = np.unique(sod_selected, return_inverse=True)
            for u_idx, sod_val in enumerate(unique_sods):
                row_mask = (inverse_idx == u_idx)
                if not row_mask.any():
                    continue
                ts = datetime(date_obj.year, date_obj.month, date_obj.day) + timedelta(seconds=float(sod_val))
                lat_sta_arr = block['lat_sta'][row_mask]
                lon_sta_arr = block['lon_sta'][row_mask]
                lat_ipp_arr = block['lat_ipp'][row_mask]
                lon_ipp_arr = block['lon_ipp'][row_mask]
                try:
                    sm_lat_sta_arr, sm_lon_sta_arr = geographic_to_solar_magnetic(lat_sta_arr, lon_sta_arr, ts)
                    sm_lat_ipp_arr, sm_lon_ipp_arr = geographic_to_solar_magnetic(lat_ipp_arr, lon_ipp_arr, ts)
                except Exception:
                    sm_lat_sta_arr, sm_lon_sta_arr = lat_sta_arr, lon_sta_arr
                    sm_lat_ipp_arr, sm_lon_ipp_arr = lat_ipp_arr, lon_ipp_arr
                block['sm_lat_sta'][row_mask] = sm_lat_sta_arr
                block['sm_lon_sta'][row_mask] = sm_lon_sta_arr
                block['sm_lat_ipp'][row_mask] = sm_lat_ipp_arr
                block['sm_lon_ipp'][row_mask] = sm_lon_ipp_arr

            # save chunk (no compression for speed - final file will be compressed)
            chunk_file = tmp_dir / f"madrigal_sample_chunk_{chunk_idx:04d}.h5"
            with h5py.File(chunk_file, 'w') as cf:
                cf.create_dataset('data', data=block)
            
            return (str(chunk_file), n, str(h5file).split('/')[-1], total_rows, mask.sum())

    except Exception as e:
        logger.warning("Failed processing date %s: %s", date_obj.strftime('%Y-%m-%d'), e)
        return None


def build_sample(madrigal_path, out_h5, split='test', n_files=50, tmp_dir=None, n_workers=None):
    """Build a small Madrigal HDF5 sample.

    Args:
        madrigal_path: root folder with Madrigal day files
        out_h5: output HDF5 path
        split: which split to use (train/val/test)
        n_files: maximum number of dates to process
        tmp_dir: optional temporary chunk dir; default 'data/temp_madrigal_sample'
        n_workers: number of parallel workers; default uses all CPUs

    Returns:
        Path to created HDF5 (out_h5) or None if no data was found.
    """
    months_file = Path('src/data_processing') / f"{split}_dates.list"
    if not months_file.exists():
        raise FileNotFoundError(f"Date list not found: {months_file}")
    dates = parse_months(str(months_file))
    
    # Limit to n_files
    dates = dates[:n_files]
    
    logger.info("Building Madrigal HDF5 sample: split=%s, max_dates=%s", split, len(dates))
    logger.info("Dates discovered from %s: %d", months_file, len(dates))

    stations_file = Path('src/data_processing') / f"{split}_station.list"
    if stations_file.exists():
        stations = np.loadtxt(str(stations_file), dtype=str)
        stations_bytes = set(s.strip().lower().encode('ascii') for s in stations)
        logger.info("Loaded station filter (%s): %d stations", stations_file, len(stations_bytes))
    else:
        stations_bytes = None

    out_path = Path(out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if tmp_dir is None:
        tmp_dir = Path('data') / 'temp_madrigal_sample'
    else:
        tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of workers
    if n_workers is None:
        n_workers = cpu_count()
    logger.info("Using %d parallel workers", n_workers)

    # Prepare arguments for parallel processing
    worker_args = [(date_obj, madrigal_path, stations_bytes, tmp_dir, idx) 
                   for idx, date_obj in enumerate(dates)]

    # Process files in parallel
    chunk_files = []
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_process_single_file, worker_args)
    else:
        # Sequential processing for debugging
        results = [_process_single_file(args) for args in worker_args]

    # Collect successful results
    for result in results:
        if result is not None:
            chunk_file, n_rows, filename, total_rows, filtered_rows = result
            chunk_files.append(chunk_file)
            logger.info("Processed %s: %d -> %d rows, wrote %s", filename, total_rows, filtered_rows, Path(chunk_file).name)

    if not chunk_files:
        logger.info("No chunks created, nothing to merge. Exiting.")
        return None

    # merge
    total_records = 0
    for cf in chunk_files:
        with h5py.File(cf, 'r') as f:
            total_records += f['data'].shape[0]

    logger.info("Merging %d chunks into %s total_records=%d", len(chunk_files), out_path, total_records)
    with h5py.File(out_path, 'w') as final_f:
        final_dataset = final_f.create_dataset('data', shape=(total_records,), dtype=MADRIGAL_DTYPE, chunks=(min(8192, max(1, total_records//100)),), compression='gzip')
        current_offset = 0
        for cf in chunk_files:
            with h5py.File(cf, 'r') as f:
                chunk_data = f['data'][:]
                chunk_size = chunk_data.shape[0]
                final_dataset[current_offset: current_offset + chunk_size] = chunk_data
                current_offset += chunk_size

    # cleanup chunk files
    for cf in chunk_files:
        try:
            os.remove(cf)
        except Exception:
            pass
    try:
        tmp_dir.rmdir()
    except Exception:
        pass

    logger.info("Wrote final file %s records=%d", out_path, total_records)
    return str(out_path)
