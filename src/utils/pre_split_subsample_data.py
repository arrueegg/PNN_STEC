import tables
import h5py
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from tqdm import tqdm

# --- Modular Helper Functions ---

def filter_record(rec, config):
    """Return True if the record meets filtering criteria."""
    if rec['satele'] < config['data']['min_elevation']:
        return False
    if not (0 < rec['vtec'] <= 200):
        return False
    if abs(rec['dcbs']) <= 1e-3 or abs(rec['dcbr']) <= 1e-3:
        return False
    return True

def decode_field(rec, field):
    """Decode a field if it is in bytes."""
    return rec[field].decode('utf-8') if isinstance(rec[field], bytes) else rec[field]

def group_records_by_arc(records, config):
    """
    Groups records into arcs by (station, sat).
    Only records that pass the filtering criteria are included.
    """
    arcs = {}
    for rec in records:
        if not filter_record(rec, config):
            continue
        station = decode_field(rec, 'station')
        sat = decode_field(rec, 'sat')
        slipc = rec['slipc']
        key = (station, sat, slipc)
        arcs.setdefault(key, []).append(rec)
    return arcs

def subsample_arc(arc_records, sampling):
    """
    Subsamples records within an arc based on the average latitude.
    Lower latitudes (closer to equator) use a higher retention rate.
    """
    approach = 'random'  # random, latitude_dependent
    if approach == 'latitude_dependent':
        if sampling == "all":
            return arc_records
        elif sampling == "semi":
            subsampled_records = []
            for rec in arc_records:
                lat = rec['lat_sta']
                if abs(lat) < 30:
                    subsample_rate = 0.8
                elif abs(lat) < 60:
                    subsample_rate = 0.6
                else:
                    subsample_rate = 0.4
                if np.random.rand() < subsample_rate:
                    subsampled_records.append(rec)
            return subsampled_records
        elif sampling == "medium":
            subsampled_records = []
            for rec in arc_records:
                lat = rec['lat_sta']
                if abs(lat) < 30:
                    subsample_rate = 0.6
                elif abs(lat) < 60:
                    subsample_rate = 0.4
                else:
                    subsample_rate = 0.2
                if np.random.rand() < subsample_rate:
                    subsampled_records.append(rec)
            return subsampled_records
        elif sampling == "hard":
            subsampled_records = []
            for rec in arc_records:
                lat = rec['lat_sta']
                if abs(lat) < 30:
                    subsample_rate = 0.2
                elif abs(lat) < 60:
                    subsample_rate = 0.1
                else:
                    subsample_rate = 0.05
                if np.random.rand() < subsample_rate:
                    subsampled_records.append(rec)
            return subsampled_records
        elif sampling == "extreme":
            subsampled_records = []
            for rec in arc_records:
                lat = rec['lat_sta']
                if abs(lat) < 30:
                    subsample_rate = 0.1
                elif abs(lat) < 60:
                    subsample_rate = 0.05
                else:
                    subsample_rate = 0.025
                if np.random.rand() < subsample_rate:
                    subsampled_records.append(rec)
            return subsampled_records
        elif sampling == "ultra":
            subsampled_records = []
            for rec in arc_records:
                lat = rec['lat_sta']
                if abs(lat) < 30:
                    subsample_rate = 0.02
                elif abs(lat) < 60:
                    subsample_rate = 0.01
                else:
                    subsample_rate = 0.005
                if np.random.rand() < subsample_rate:
                    subsampled_records.append(rec)
            return subsampled_records
    elif approach == 'random':
        if sampling == "all":
            return arc_records
        elif sampling == "semi":
            return np.random.choice(arc_records, size=int(len(arc_records) * 0.5), replace=False).tolist()
        elif sampling == "medium":
            return np.random.choice(arc_records, size=int(len(arc_records) * 0.2), replace=False).tolist()
        elif sampling == "hard":
            return np.random.choice(arc_records, size=int(len(arc_records) * 0.05), replace=False).tolist()
        elif sampling == "extreme":
            return np.random.choice(arc_records, size=int(len(arc_records) * 0.01), replace=False).tolist()
        elif sampling == "ultra":
            return np.random.choice(arc_records, size=int(len(arc_records) * 0.005), replace=False).tolist()

def assign_arc_records(key, records, train_stations, val_stations, test_stations, 
                         train_records, val_records, test_records):
    """
    Assigns records for a given arc (identified by key) to the appropriate split.
    """
    station = key[0]
    if station in train_stations:
        train_records.extend(records)
    elif station in val_stations:
        val_records.extend(records)
    elif station in test_stations:
        test_records.extend(records)

def process_batch(batch_array, config, train_stations, val_stations, test_stations):
    """
    Processes a batch of records:
      - Groups records by arc,
      - Applies subsampling on each arc,
      - Assigns records to train/val/test splits.
    Returns three lists: train_records, val_records, and test_records.
    """
    train_records = []
    val_records = []
    test_records = []
    sampling = config['data']['sampling']
    arcs = group_records_by_arc(batch_array, config)
    for key, arc in arcs.items():
        subsampled = subsample_arc(arc, sampling)
        assign_arc_records(key, subsampled, train_stations, val_stations, test_stations,
                           train_records, val_records, test_records)
    return train_records, val_records, test_records

def append_split(records, dset, current_count, dtype):
    """
    Appends structured records to the HDF5 dataset.
    Returns the updated count.
    """
    if records:
        arr = np.array(records, dtype=dtype)
        n_new = len(arr)
        dset.resize((current_count + n_new,))
        dset[current_count:current_count + n_new] = arr
        return current_count + n_new
    return current_count


# --- Main Function for Filtering and Saving ---

def filter_and_save(config, h5_file_path, save_dir):
    """
    Reads an HDF5 file, filters and processes the data in batches,
    computes the IPPs, and saves the train, val, and test splits into a new HDF5 file.
    """
    h5_chunks = (256,)
    BATCH_SIZE = 5_000_000
    dtype = (
        [('station', 'S4'), ('sat', 'S3'), ('stec', 'f4'), ('vtec', 'f4'), 
         ('vtec_stddev', 'f4'), ('satres', 'f4'), ('satele', 'f4'), ('satazi', 'f4'), 
         ('dcbs', 'f4'), ('dcbr', 'f4'), 
         ('lon_ipp_450', 'f4'), ('lat_ipp_450', 'f4'), ('sm_lon_ipp_450', 'f4'), ('sm_lat_ipp_450', 'f4'),
         ('sod', 'f4'), ('lat_sta', 'f4'), ('lon_sta', 'f4'),
         ('sm_lat_sta', 'f4'), ('sm_lon_sta', 'f4'), ('slipc', 'f4'), ('gfphase', 'f4')]
    )

    basename = os.path.basename(h5_file_path)
    parts = basename.split('_')
    year = parts[1][:4]
    doy = parts[1][4:]

    sampling = config['data']['sampling']
    save_path = os.path.join(save_dir, f"Split_{year}{doy}_30_5_subsampled_{sampling}.h5")
    if os.path.exists(save_path):
        return
    else:
        print(f"Preprocessing {year} {doy}...")

    train_stations = set(np.loadtxt('./src/data_processing/train_station.list', dtype=str))
    val_stations = set(np.loadtxt('./src/data_processing/val_station.list', dtype=str))
    test_stations = set(np.loadtxt('./src/data_processing/test_station.list', dtype=str))

    with tables.open_file(h5_file_path, mode='r') as in_file, \
        h5py.File(save_path, mode='w') as out_file:

        data_node = in_file.get_node(f'/{year}/{doy}/all_data')
        out_group = out_file.require_group(f"{year}/{doy}")

        dset_train = out_group.create_dataset("train_data", shape=(0,), maxshape=(None,),
                                              dtype=dtype, chunks=h5_chunks)
        dset_val = out_group.create_dataset("val_data", shape=(0,), maxshape=(None,),
                                            dtype=dtype, chunks=h5_chunks)
        dset_test = out_group.create_dataset("test_data", shape=(0,), maxshape=(None,),
                                             dtype=dtype, chunks=h5_chunks)

        count_train = count_val = count_test = 0
        batch = []

        # Process rows in batches.
        for i, row in tqdm(enumerate(data_node.iterrows()), desc="Creating batches", total=len(data_node)):
            batch.append(row.fetch_all_fields())
            if len(batch) >= BATCH_SIZE or i == len(data_node) - 1:
                batch_array = np.array(batch)
                train_recs, val_recs, test_recs = process_batch(batch_array, config,
                                                                train_stations, val_stations, test_stations)
                count_train = append_split(train_recs, dset_train, count_train, dtype)
                count_val = append_split(val_recs, dset_val, count_val, dtype)
                count_test = append_split(test_recs, dset_test, count_test, dtype)
                batch = []

    print(f"Saved {year} {doy} to {save_path}")

def generate_dates(months):
        dates = []
        for month in months:
            year, month = map(int, month.split('-'))
            start_date = datetime(year, month, 1)
            end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            current_date = start_date
            while current_date <= end_date:
                dates.append(current_date)
                current_date += timedelta(days=1)
        return dates

def get_file_lists(config, year, doy):
    gnss_path = config['data']['GNSS_data_path']
    finetune_date = pd.to_datetime(f"{year}-{doy}", format="%Y-%j")

    train_months = sorted(set(np.loadtxt('./src/data_processing/train_dates.list', dtype=str)))
    val_months = sorted(set(np.loadtxt('./src/data_processing/val_dates.list', dtype=str)))
    test_months = sorted(set(np.loadtxt('./src/data_processing/test_dates.list', dtype=str)))

    train_dates = generate_dates(train_months)
    val_dates = generate_dates(val_months)
    test_dates = generate_dates(test_months)

    ####################################################
    # Debugging: only take a subset of dates for testing
    interval = 40
    train_dates = train_dates[::interval]
    val_dates = val_dates[::interval]
    test_dates = test_dates[::interval]
    ####################################################

    all_dates = pd.DatetimeIndex(train_dates + val_dates + test_dates)

    if finetune_date not in all_dates:
        all_dates = all_dates.append(pd.DatetimeIndex([finetune_date]))
    file_paths = []
    for date in all_dates:
        file_path = os.path.join(gnss_path, str(date.year), f'{date.dayofyear:03d}', 
                                 f'ccl_{date.year}{date.dayofyear:03d}_30_5.h5')
        if os.path.exists(file_path):
            file_paths.append(file_path)
    return file_paths

def split(config, save_dir):
    files = get_file_lists(config, config['year'], config['doy'])
    for file in files:
        filter_and_save(config, file, save_dir)
