#!/usr/bin/env python3
import argparse
import yaml
import h5py
import numpy as np
import os
from datetime import datetime, timedelta
from tqdm import tqdm

# ——— Helper Functions ———

def filter_record(rec, config):
    """Return True if the record should be included."""
    # elevation filter
    if rec['satele'] < config['data']['min_elevation']:
        return False
    # plausible VTEC range
    if not (0 < rec['vtec'] <= 200):
        return False
    # DCB sanity check
    if abs(rec['dcbs']) <= 1e-3 or abs(rec['dcbr']) <= 1e-3:
        return False
    return True

def decode_station(raw):
    """Decode bytes→str if necessary."""
    return raw.decode('utf-8') if isinstance(raw, bytes) else raw

def generate_dates(months):
    dates = []
    for month in months:
        year, mon = map(int, month.split('-'))
        start = datetime(year, mon, 1)
        end   = (start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        cur = start
        while cur <= end:
            dates.append(cur)
            cur += timedelta(days=1)
    return dates

def get_file_list(config):
    """
    Returns a list of full HDF5 file paths for your train/val/test months.
    Assumes you have train_dates.list, etc., each listing YYYY-MM lines.
    """
    base = config['data']['GNSS_data_path']
    year = config['year']
    doy  = config['doy']  # string of the day-of-year

    # load your month-lists
    train_mon = sorted(set(np.loadtxt('./src/data_processing/train_dates.list', dtype=str)))
    val_mon   = sorted(set(np.loadtxt('./src/data_processing/val_dates.list',   dtype=str)))
    test_mon  = sorted(set(np.loadtxt('./src/data_processing/test_dates.list',  dtype=str)))

    # generate all dates, with optional subsampling for debugging
    train_dates = generate_dates(train_mon)
    val_dates   = generate_dates(val_mon)
    test_dates  = generate_dates(test_mon)

    # Combine and ensure the target date is included
    all_dates = train_dates + val_dates + test_dates

    #debugging: only use one day
    """all_dates = [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]
    ###############################################################################
    ###############################################################################
    ######################### Debugging: only use one day #########################"""

    target = datetime.strptime(f"{year}-{doy}", "%Y-%j")
    if target not in all_dates:
        all_dates.append(target)

    # Build file paths
    paths = []
    for dt in all_dates:
        fn = f"ccl_{dt.year}{dt.timetuple().tm_yday:03d}_30_5.h5"
        full = os.path.join(base, str(dt.year), f"{dt.timetuple().tm_yday:03d}", fn)
        if os.path.exists(full):
            paths.append(full)
    return paths

# ——— Main Function ———

def add_split_indices(config):
    # Load station-split lists
    train_stations = set(np.loadtxt('./src/data_processing/train_station.list', dtype=str))
    val_stations   = set(np.loadtxt('./src/data_processing/val_station.list',   dtype=str))
    test_stations  = set(np.loadtxt('./src/data_processing/test_station.list',  dtype=str))

    files = get_file_list(config)
    print(f"Found {len(files)} files to process.")

    for path in files:
        basename = os.path.basename(path)
        year = basename.split('_')[1][:4]
        doy  = basename.split('_')[1][4:]

        print(f"\nProcessing {basename}...")
        with h5py.File(path, 'r+') as h5f:
            grp_path = f"{year}/{doy}"
            if grp_path not in h5f:
                #print(f"  ▶ Skipping – group {grp_path} not found.")
                continue
            grp = h5f[grp_path]

            # skip if already done
            if all(x in grp for x in ('train_idx','val_idx','test_idx')):
                #print("  ▶ Indices already exist, skipping.")
                continue

            data = grp['all_data']
            nrows = data.shape[0]
            train_idx = []
            val_idx   = []
            test_idx  = []

            # iterate in chunks for speed
            CHUNK = 2_000_000
            for start in tqdm(range(0, nrows, CHUNK), desc="  ✂ chunk"):
                stop = min(start+CHUNK, nrows)
                block = data[start:stop]
                for j, rec in enumerate(block):
                    if not filter_record(rec, config):
                        continue
                    sta = decode_station(rec['station'])
                    idx = start + j
                    if   sta in train_stations: train_idx.append(idx)
                    elif sta in val_stations:   val_idx.append(idx)
                    elif sta in test_stations:  test_idx.append(idx)

            # write them back
            grp.create_dataset('train_idx', data=np.array(train_idx, dtype=np.int64))
            grp.create_dataset('val_idx',   data=np.array(val_idx,   dtype=np.int64))
            grp.create_dataset('test_idx',  data=np.array(test_idx,  dtype=np.int64))

            print(f"  ▶ Wrote indices: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

# ——— CLI Entrypoint ———

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Append train/val/test index lists to daily HDF5 files")
    p.add_argument('--config', required=True, help="YAML config with data paths and filters")
    p.add_argument('--year',   required=True, help="Target year (YYYY)")
    p.add_argument('--doy',    required=True, help="Target day-of-year (DDD)")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg['year'] = args.year
    cfg['doy']  = args.doy

    add_split_indices(cfg)
