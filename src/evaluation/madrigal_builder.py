"""Madrigal HDF5 build utilities.

This module exposes programmatic functions to build Madrigal-format HDF5 files
that match the project's preprocessing `DTYPE` layout. The sample builder
was refactored out of the scripts folder so other parts of the codebase can
reuse it (production builder, tests, notebooks, etc.).

The implementation mirrors the behavior from the legacy `scripts/build_madrigal_h5_sample.py`:
- find per-day Madrigal files
- build light-weight masks (read `sod` and station fields first)
- read only selected rows and map them to `PREPROC_DTYPE`
- compute precise solar-magnetic coordinates using `geographic_to_solar_magnetic`
- write temporary chunk H5 files then merge into a final root `data` dataset
"""
from pathlib import Path
import os
from datetime import datetime, timedelta
import numpy as np
import h5py

from src.evaluation.madrigal_loader import find_madrigal_file
from src.utils.preprocessing import DTYPE as PREPROC_DTYPE
from src.utils.coordinate_transforms import geographic_to_solar_magnetic


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


def build_sample(madrigal_path, out_h5, split='test', n_files=50, tmp_dir=None):
    """Build a small Madrigal HDF5 sample.

    Args:
        madrigal_path: root folder with Madrigal day files
        out_h5: output HDF5 path
        split: which split to use (train/val/test)
        n_files: maximum number of dates to process
        tmp_dir: optional temporary chunk dir; default 'data/temp_madrigal_sample'

    Returns:
        Path to created HDF5 (out_h5) or None if no data was found.
    """
    months_file = Path('src/data_processing') / f"{split}_dates.list"
    if not months_file.exists():
        raise FileNotFoundError(f"Date list not found: {months_file}")
    dates = parse_months(str(months_file))

    stations_file = Path('src/data_processing') / f"{split}_station.list"
    if stations_file.exists():
        stations = np.loadtxt(str(stations_file), dtype=str)
        stations_bytes = set(s.strip().lower().encode('ascii') for s in stations)
    else:
        stations_bytes = None

    out_path = Path(out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if tmp_dir is None:
        tmp_dir = Path('data') / 'temp_madrigal_sample'
    else:
        tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = []
    processed_dates = 0
    chunk_idx = 0

    for date_obj in dates:
        if processed_dates >= n_files:
            break
        h5file = find_madrigal_file(madrigal_path, date_obj)
        if h5file is None:
            continue
        try:
            with h5py.File(h5file, 'r') as mf:
                if 'Data' not in mf:
                    continue
                data_names = list(mf['Data'].keys())
                if not data_names:
                    continue
                tabname = 'Table Layout' if 'Table Layout' in data_names else data_names[0]
                ds = mf['Data'][tabname]
                total_rows = ds.shape[0]

                # determine station field
                station_field = None
                if 'gps_site' in ds.dtype.names:
                    station_field = 'gps_site'
                elif 'station' in ds.dtype.names:
                    station_field = 'station'

                # read sod and station arrays only
                sod_arr = ds['sod'][:] if 'sod' in ds.dtype.names else None
                station_arr = ds[station_field][:] if station_field is not None else None

                # build mask: sod % 300 == 0
                mask = np.ones(total_rows, dtype=bool)
                if sod_arr is not None:
                    mask &= (sod_arr % 300 == 0)

                if stations_bytes is not None and station_arr is not None:
                    try:
                        station_lower = np.char.lower(station_arr.astype('S'))
                    except Exception:
                        station_lower = station_arr
                    mask &= np.isin(station_lower, list(stations_bytes))

                if mask.sum() == 0:
                    continue

                idx = np.nonzero(mask)[0]
                table = ds[idx]

                # convert to PREPROC_DTYPE
                n = len(table)
                block = np.zeros(n, dtype=PREPROC_DTYPE)
                if 'gps_site' in table.dtype.names:
                    block['station'] = table['gps_site']
                elif 'station' in table.dtype.names:
                    block['station'] = table['station']
                else:
                    block['station'] = b''
                block['year'] = date_obj.year
                block['doy'] = date_obj.timetuple().tm_yday
                if 'los_tec' in table.dtype.names:
                    block['stec'] = table['los_tec']
                elif 'tec' in table.dtype.names:
                    block['stec'] = table['tec']
                else:
                    block['stec'] = np.nan
                block['vtec'] = table['vtec'] if 'vtec' in table.dtype.names else np.nan
                block['satele'] = table['elm'] if 'elm' in table.dtype.names else np.nan
                block['satazi'] = table['azm'] if 'azm' in table.dtype.names else np.nan
                block['lon_sta'] = table['gdlonr'] if 'gdlonr' in table.dtype.names else (table['glon'] if 'glon' in table.dtype.names else np.nan)
                block['lat_sta'] = table['gdlatr'] if 'gdlatr' in table.dtype.names else (table['gdlat'] if 'gdlat' in table.dtype.names else np.nan)
                block['lat_ipp'] = table['gdlat'] if 'gdlat' in table.dtype.names else np.nan
                block['lon_ipp'] = table['glon'] if 'glon' in table.dtype.names else np.nan
                block['sm_lat_ipp'] = np.nan
                block['sm_lon_ipp'] = np.nan
                block['sm_lat_sta'] = np.nan
                block['sm_lon_sta'] = np.nan
                block['sod'] = table['sod'] if 'sod' in table.dtype.names else np.nan

                # precise SM computation grouped by unique sod
                try:
                    sod_selected = table['sod'] if 'sod' in table.dtype.names else np.zeros(n, dtype=float)
                except Exception:
                    sod_selected = np.zeros(n, dtype=float)
                sod_selected = sod_selected.astype(float)
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

                # save chunk
                chunk_file = tmp_dir / f"madrigal_sample_chunk_{chunk_idx:04d}.h5"
                with h5py.File(chunk_file, 'w') as cf:
                    cf.create_dataset('data', data=block, compression='gzip')
                chunk_files.append(str(chunk_file))
                chunk_idx += 1
                processed_dates += 1

        except Exception:
            continue

    if not chunk_files:
        return None

    # merge
    total_records = 0
    for cf in chunk_files:
        with h5py.File(cf, 'r') as f:
            total_records += f['data'].shape[0]

    with h5py.File(out_path, 'w') as final_f:
        final_dataset = final_f.create_dataset('data', shape=(total_records,), dtype=PREPROC_DTYPE, chunks=(min(8192, max(1, total_records//100)),), compression='gzip')
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

    return str(out_path)
