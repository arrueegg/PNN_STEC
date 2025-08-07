import os
import shutil
import h5py
import torch
import numpy as np
import random
from torch.utils.data import RandomSampler, Dataset, DataLoader
from utils.locationencoder.pe import SphericalHarmonics
from data_processing.download_solar_indices import OmniDownloader
import tables
from tqdm import tqdm

import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

torch.multiprocessing.set_start_method('fork', force=True)
#spacepy.toolbox.update(leapsecs=True)

# Structured dtype for our “one‐table” HDF5 per split:
DTYPE = np.dtype([
    ('station',   'S8'),    # up to 8‐char ASCII
    ('year',      'i4'),
    ('doy',       'i4'),
    ('stec',      'f4'),
    ('vtec',      'f4'),
    ('satele',    'f4'),
    ('satazi',    'f4'),
    ('lon_ipp',   'f4'),
    ('lat_ipp',   'f4'),
    ('sm_lat_ipp','f4'),
    ('sm_lon_ipp','f4'),
    ('sod',       'f4'),
    ('lat_sta',   'f4'),
    ('lon_sta',   'f4'),
    ('sm_lat_sta','f4'),
    ('sm_lon_sta','f4'),
])

class H5Dataset(Dataset):
    def __init__(self, config, h5_path, split):
        self.config = config
        self.split  = split
        # open the aggregated split file
        self.file = h5py.File(h5_path, 'r')
        self.data = self.file['data']
        
        # SWI?
        self.use_SWI = config['data'].get('use_SWI', False)
        if self.use_SWI:
            swi_path = os.path.join(
                config['data']['SWI_data_path'],
                "omni_hourly_2010-2025.h5"
            )
            self.swi_file = h5py.File(swi_path, 'r')
            # build mask of SWI columns (drop YEAR, DOY, HR)
            # we can peek at any day, say first in first year
            yrs = list(self.swi_file.keys())
            days = list(self.swi_file[yrs[0]].keys())
            cols = [c.decode() for c in self.swi_file[yrs[0]][days[0]].attrs['columns']]
            self.swi_mask = [c not in ('YEAR','DOY','HR') for c in cols]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]
        # core features: DOY, SOD, station coords, az/el, IPP coords
        feat = torch.tensor([
            row['doy'],
            row['sod'],
            row['sm_lat_sta'],
            row['sm_lon_sta'],
            row['satazi'],
            row['satele'],
            row['lat_ipp'],
            row['lon_ipp']
        ], dtype=torch.float32)

        # SWI features?
        if self.use_SWI:
            year = str(int(row['year']))
            doy3 = f"{int(row['doy']):03d}"
            # read that day's SWI array and pick the hour
            daily = self.swi_file[year][doy3][:]
            hour = int(row['sod'] // 3600)
            swi_row = daily[hour]
            swi_feat = torch.tensor(swi_row[self.swi_mask], dtype=torch.float32)
            feat = torch.cat((feat, swi_feat), dim=0)

        # label = stec
        label = torch.tensor(row['stec'], dtype=torch.float32)

        # guard NaNs
        if torch.isnan(feat).any() or torch.isnan(label):
            raise ValueError(f"NaN in H5Dataset at idx {idx}")

        return feat, label

    def __del__(self):
        # close both files
        self.file.close()
        if self.use_SWI:
            self.swi_file.close()

class PyTablesDatasetSplit(Dataset):
    def __init__(self, config, h5_file_path, split):
        self.config = config
        self.h5_file_path = h5_file_path
        self.doy = self.h5_file_path.split('/')[-1].split('_')[1][4:]
        self.year = self.h5_file_path.split('/')[-1].split('_')[1][:4]
        self.split = split
        self.SWI_data = self.load_SWI()
        self.file = None
        with tables.open_file(self.h5_file_path, mode='r') as f:
            self.length = f.get_node(f'/{self.year}/{self.doy}/{self.split}_idx').shape[0]
        
    def load_SWI(self):
        if self.config['data']['use_SWI']:
            filename = os.path.join(self.config['data']['SWI_data_path'], f'omni_hourly_2010-2025.h5')
            if not os.path.exists(filename):
                downloader = OmniDownloader(self.config['data']['SWI_data_path'], "20100101", "20250625")
                downloader.run()
            with h5py.File(filename, "r") as f:
                data = f[str(self.year)][str(self.doy).zfill(3)][:]
                columns = [c.decode() for c in f[str(self.year)][str(self.doy).zfill(3)].attrs['columns']]
            # Convert to NumPy
            mask = [col not in ['YEAR', 'DOY', 'HR'] for col in columns]
            self.SWI_cols = mask
            return torch.tensor(np.array(data)[:, mask], dtype=torch.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = tables.open_file(self.h5_file_path, mode='r')
            self.data = self.file.get_node(f'/{self.year}/{self.doy}/all_data')
            self.indices = self.file.get_node(f'/{self.year}/{self.doy}/{self.split}_idx')
        row = self.data[self.indices[idx]]

        # Encode strings and combine with numeric features
        features = torch.tensor([
            int(self.doy),
            row['sod'],
            row['sm_lat_sta'],
            row['sm_lon_sta'],
            row['satazi'],
            row['satele'],
            row['lat_ipp'],
            row['lon_ipp']
            ], dtype=torch.float32)
        
        if self.config['data']['use_SWI']:
            # Add SWI features if available
            swi_idx = (row['sod'] / 3600).astype(int)  # Convert seconds to hours
            swi_features = self.SWI_data[swi_idx]
            features = torch.cat((features, swi_features), dim=0)
        
        # Return features and label separately
        label = torch.tensor(row['stec'], dtype=torch.float32)

        # Check for NaN or similar in features or label
        if torch.isnan(features).any() or torch.isnan(label):
            raise ValueError(f"NaN detected in features or label at index {idx}")

        return features, label

    def __del__(self):
        if self.file is not None:
            self.file.close()  # Ensure the file is closed properly

class CollateWithSH:
    def __init__(self, config):
        # SH degree and flag
        self.sh_degree  = config["data"].get("SH_degree", 0) or 0
        self.sh_enabled = self.sh_degree > 0
        if self.sh_enabled:
            self.sh_encoder = SphericalHarmonics(legendre_polys=self.sh_degree)

        self.num_core_features = 8

    def transform(self, features):
        """
        Normalize and transform the batch of features (one IPP point).
        """
        # raw feature indices
        DOY, SOD, LAT_STA, LON_STA, AZI, ELE, IPP_LAT, IPP_LON = range(self.num_core_features)

        # --- day of year ---
        doy_norm = ((features[:, DOY] - 1) / 365).unsqueeze(1)  # Normalize DOY to [0, 1]
        doy_sin = torch.sin(doy_norm * 2 * torch.pi)
        doy_cos = torch.cos(doy_norm * 2 * torch.pi)

        # --- time of day ---
        t      = features[:, SOD] / 86400 * (2 * torch.pi)
        sin_t  = torch.sin(t).unsqueeze(1)
        cos_t  = torch.cos(t).unsqueeze(1)
        norm_t = (2 * features[:, SOD] / 86400 - 1).unsqueeze(1)

        # --- azimuth / elevation ---
        a      = features[:, AZI] / 180 * torch.pi
        sin_a  = torch.sin(a).unsqueeze(1)
        cos_a  = torch.cos(a).unsqueeze(1)
        norm_e = (2 * features[:, ELE] / 90 - 1).unsqueeze(1)

        # --- station coords normalized to [-1,1] ---
        sm_lat_norm = ((features[:, LAT_STA] + 90) / 180 * 2 - 1).unsqueeze(1)
        sm_lon_norm = ((features[:, LON_STA] + 180) / 360 * 2 - 1).unsqueeze(1)

        # --- single IPP point normalized to [-1,1] ---
        ipp_lat_norm = ((features[:, IPP_LAT] + 90) / 180 * 2 - 1).unsqueeze(1)
        ipp_lon_norm = ((features[:, IPP_LON] + 180) / 360 * 2 - 1).unsqueeze(1)

        # concatenate core normalized features
        x_out = torch.cat([
            doy_sin, doy_cos, doy_norm,
            sin_t, cos_t, norm_t,
            sm_lat_norm, sm_lon_norm,
            sin_a, cos_a, norm_e,
            ipp_lat_norm, ipp_lon_norm
        ], dim=1)

        return x_out
    
    def norm_SWI(self, swi_features):
        """
        Normalize SWI features to [0, 1] range for each column by specified min/max values.
        """
        swi_features = swi_features.float()

        # Define min and max values as tensors (matching column order)
        min_max_values = [
            (2407, 3000),  # Bartels_rotation_number
            (0, 70),       # Scalar_B,_nT
            (0.0, 70),     # Vector_B_Magnitude,nT
            (-90, 90),     # Lat_Angle_of_B_GSE
            (0.0, 360.0),  # Long_Angle_of_B_GSE
            (-50, 35),     # BZ,_nT_GSE
            (-50, 35),     # BZ,_nT_GSM
            (240.0, 1100.0), # SW_Plasma_Speed,_km/s
            (0, 60),       # Flow_pressure
            (-20, 30),     # E_elecrtric_field
            (0, 120),      # Alfen_mach_number
            (0.0, 100.0),  # Kp_index
            (0.0, 300.0),  # R_Sunspot_No
            (-450, 100),   # Dst-index,_nT
            (0.0, 2500.0), # AE-index,_nT
            (0.0, 300.0),  # ap_index,_nT
            (62, 420),     # f107_index
            (-6, 16),      # pc-index
            (-2000.0, 20.0), # AL-index,_nT
            (-200.0, 1200.0), # AU-index,_nT
            (0, 15),       # Magnetosonic_Much_num
            (0, 0.015),    # Lyman_alpha
        ]

        # Convert min and max to tensors
        min_vals = torch.tensor([m[0] for m in min_max_values], dtype=torch.float32, device=swi_features.device)
        max_vals = torch.tensor([m[1] for m in min_max_values], dtype=torch.float32, device=swi_features.device)

        # Vectorized min-max normalization
        normalized_features = (swi_features - min_vals) / (max_vals - min_vals)

        # Concatenate normalized columns
        return normalized_features

    def SH_transform(self, raw, norm):
        """
        Append spherical-harmonic embeddings if enabled.
        `raw` is original features; `norm` is output from transform().
        """
        if not self.sh_enabled:
            return norm

        # SH embeddings for station (LON_STA, LAT_STA = cols 1,0)
        sta_lonlat = torch.stack([raw[:, 1], raw[:, 0]], dim=1)
        sh_sta = self.sh_encoder(sta_lonlat)

        # SH embeddings for IPP (IPP_LON, IPP_LAT = cols 6,5)
        ipp_lonlat = torch.stack([raw[:, 6], raw[:, 5]], dim=1)
        sh_ipp = self.sh_encoder(ipp_lonlat)

        # combine normalized + SH features
        return torch.cat([norm, sh_sta, sh_ipp], dim=1)
    
    def __call__(self, batch):
        """
        Process and collate a batch of (raw_features, labels).
        """
        feats, labels = zip(*batch)
        raw = torch.stack(feats, dim=0)
        y   = torch.stack(labels, dim=0)

        # Split core features and SWI features
        core = raw[:, :self.num_core_features]
        swi  = raw[:, self.num_core_features:]

        # Normalize features
        norm_core  = self.transform(core)
        x_out = self.SH_transform(core, norm_core)
        norm_swi = self.norm_SWI(swi) if swi.shape[1] > 0 else None

        # Append SWI features (raw or normalized if needed)
        if norm_swi is not None:
            x_out = torch.cat([norm_swi, x_out], dim=1)

        return x_out, y
    
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

def move_files_to_scratch(config, file_paths):
    """ Move data files to scratch storage for faster access.
    """
    file_paths_scratch = {
        'train': [],
        'val': [],
        'test': []
    }
    for split, paths in file_paths.items():
        for path in paths:
            year = path.split('/')[-3]
            doy = path.split('/')[-2]
            scratch_path = os.path.join(config['data']['scratch_dir'], year, doy, os.path.basename(path))
            os.makedirs(os.path.dirname(scratch_path), exist_ok=True)
            if not os.path.exists(scratch_path):
                shutil.copy(path, scratch_path)
            file_paths_scratch[split].append(scratch_path)
    return file_paths_scratch

def get_split_file_lists(config, year, doy):
    gnss_path = config['data']['GNSS_data_path']   
    sampling = config['data']['sampling']

    train_months = sorted(set(np.loadtxt('./src/data_processing/train_dates.list', dtype=str)))
    val_months = sorted(set(np.loadtxt('./src/data_processing/val_dates.list', dtype=str)))
    test_months = sorted(set(np.loadtxt('./src/data_processing/test_dates.list', dtype=str)))

    # Filter dates to be only after 2019 for debuging
    #train_months = [m for m in train_months if int(m.split('-')[0]) >= 2019]
    #val_months = [m for m in val_months if int(m.split('-')[0]) >= 2019]
    #test_months = [m for m in test_months if int(m.split('-')[0]) >= 2019]

    train_dates = generate_dates(train_months)
    val_dates = generate_dates(val_months)
    test_dates = generate_dates(test_months)

    # Debugging: only take a subset of dates for testing
    every_x_doy = config['data'].get('every_x_doy', 1)
    train_dates = train_dates[::every_x_doy]
    val_dates = val_dates[::every_x_doy]
    test_dates = test_dates[::every_x_doy]

    """train_dates = [datetime(2023, 1, 1)]  # For debugging, only use one day
    val_dates = [datetime(2023, 1, 2)]    # For debugging, only use one day
    test_dates = [datetime(2023, 1, 3)]   # For debugging, only use one day
    ######################### Debugging: only use one day #########################"""

    def get_file_paths(dates):
        file_paths = []
        for date in dates:
            #file_path = os.path.join(gnss_path, f'Split_{date.year}{date.timetuple().tm_yday:03d}_30_5_subsampled_{sampling}.h5')
            file_path = os.path.join(gnss_path, str(date.year), f'{date.timetuple().tm_yday:03d}', f'ccl_{date.year}{date.timetuple().tm_yday:03d}_30_5.h5')
            if os.path.exists(file_path):
                file_paths.append(file_path)
        return file_paths
    
    file_paths = {'train': get_file_paths(train_dates),
                  'val': get_file_paths(val_dates),
                  'test': get_file_paths(test_dates)
                  }
    
    move_to_scratch = config.get('move_to_scratch', True)
    if move_to_scratch:
        file_paths = move_files_to_scratch(config, file_paths)
    
    return file_paths

def build_split_h5(config):
    """
    Streams daily H5 files, applies your time+station splits,
    and writes train.h5 / val.h5 / test.h5 under config['data']['scratch_dir'].
    Expects date‐lists in YYYY-MM-DD format.
    """

    scratch = config['data']['scratch_dir']
    os.makedirs(scratch, exist_ok=True)

    # 1) load your date‐lists as YYYY-MM-DD strings
    train_months = sorted(set(np.loadtxt('./src/data_processing/train_dates.list', dtype=str)))
    val_months   = sorted(set(np.loadtxt('./src/data_processing/val_dates.list',   dtype=str)))
    test_months  = sorted(set(np.loadtxt('./src/data_processing/test_dates.list',  dtype=str)))

    train_dates = generate_dates(train_months)
    val_dates = generate_dates(val_months)
    test_dates = generate_dates(test_months)

    # Debugging: only take a subset of dates for testing
    every_x_doy = config['data'].get('every_x_doy', 1)
    train_dates = train_dates[::every_x_doy]
    val_dates = val_dates[::every_x_doy]
    test_dates = test_dates[::every_x_doy]

    # 2) load your station splits as ASCII‐bytes
    t_stns = [s.encode('ascii') for s in np.loadtxt('./src/data_processing/train_station.list', dtype=str)]
    v_stns = [s.encode('ascii') for s in np.loadtxt('./src/data_processing/val_station.list',   dtype=str)]
    e_stns = [s.encode('ascii') for s in np.loadtxt('./src/data_processing/test_station.list',  dtype=str)]
    splits = {
        'train': (t_stns, train_dates),
        'val':   (v_stns, val_dates),
        'test':  (e_stns, test_dates),
    }

    # 3) prepare one structured‐dtype dataset per split
    out = {}
    all_exist = True
    for sp in splits:
        fn = os.path.join(scratch, f'{sp}.h5')
        if os.path.exists(fn):
            out[sp] = {'file': None, 'dset': None, 'count': 0}
        else:
            all_exist = False
            f  = h5py.File(fn, 'w')
            d  = f.create_dataset(
                'data',
                shape=(0,),
                maxshape=(None,),
                dtype=DTYPE,
                chunks=True
            )
            out[sp] = {'file': f, 'dset': d, 'count': 0}
    if all_exist:
        return

    gnss_root = config['data']['GNSS_data_path']
    for year in tqdm(sorted(os.listdir(gnss_root)), desc="Processing years"):
        yp = os.path.join(gnss_root, year)
        for doy in tqdm(sorted(os.listdir(yp)), desc="Processing days"):
            dayfile = os.path.join(yp, doy, f'ccl_{year}{doy}_30_5.h5')
            if not os.path.isfile(dayfile):
                continue

            # build YYYY-MM-DD string
            dt = datetime.strptime(f"{year}{doy}", "%Y%j")

            # skip if this day isn't in any date‐list
            if dt not in train_dates + val_dates + test_dates:
                continue

            with tables.open_file(dayfile, 'r') as tbl:
                node = tbl.get_node(f'/{year}/{doy}/all_data')

                # for each split that uses this date
                for sp, (stn_bytes, date_set) in splits.items():
                    if dt not in date_set:
                        continue

                    # for each station in this split, read its contiguous block
                    for sb in stn_bytes:
                        # get all row‐indices for this station
                        idxs = node.get_where_list(f"station == b'{sb.decode()}'")
                        if len(idxs) == 0:
                            continue

                        # read one small recarray of only those rows
                        sub = node.read_coordinates(idxs)

                        # build an output block of shape (n,)
                        n   = len(sub)
                        block = np.zeros(n, dtype=DTYPE)
                        block['station']     = sub['station']
                        block['year']        = dt.year
                        block['doy']         = dt.timetuple().tm_yday
                        block['stec']        = sub['stec']
                        block['vtec']        = sub['vtec']
                        block['satele']      = sub['satele']
                        block['satazi']      = sub['satazi']
                        block['lon_ipp']     = sub['lon_ipp']
                        block['lat_ipp']     = sub['lat_ipp']
                        block['sm_lat_ipp']  = sub['sm_lat_ipp']
                        block['sm_lon_ipp']  = sub['sm_lon_ipp']
                        block['sod']         = sub['sod']
                        block['lat_sta']     = sub['lat_sta']
                        block['lon_sta']     = sub['lon_sta']
                        block['sm_lat_sta']  = sub['sm_lat_sta']
                        block['sm_lon_sta']  = sub['sm_lon_sta']

                        # append the entire block in one go
                        ds    = out[sp]['dset']
                        cnt   = out[sp]['count']
                        ds.resize(cnt + n, axis=0)
                        ds[cnt:cnt+n] = block
                        out[sp]['count'] += n

    # close all files
    for v in out.values():
        v['file'].close()

    print("✅ Built split H5 files at:", scratch)


def get_data_loaders(config):
    collate_fn = CollateWithSH(config)
    loaders = {}

    # build splits if requested
    if config['data'].get('use_agg_h5', False) and config['data'].get('build_agg_h5', True):
        build_split_h5(config)

    for split in ['train','val','test']:
        if config['data'].get('use_agg_h5', False):
            # move SWI data to scratch if needed
            swi_scratch_path = os.path.join(
                config['data']['scratch_dir'],
                "omni_hourly_2010-2025.h5"
            )
            if not os.path.exists(swi_scratch_path):
                swi_path = os.path.join(
                    config['data']['SWI_data_path'],
                    "omni_hourly_2010-2025.h5"
                )
                if not os.path.exists(swi_path):
                    downloader = OmniDownloader(config['data']['SWI_data_path'], "20100101", "20250625")
                    downloader.run()
                shutil.copy(swi_path, swi_scratch_path)
                config['data']['SWI_data_path'] = swi_scratch_path

            # load from our new single-file splits
            path = os.path.join(config['data']['scratch_dir'], f"{split}.h5")
            ds   = H5Dataset(config, path, split)
            sampler = None
            if split=='train' and config['data'].get('subset_per_epoch') < len(ds):
                sampler = RandomSampler(ds, replacement=False,
                                        num_samples=config['data']['subset_per_epoch'])
            loaders[split] = DataLoader(
                ds,
                batch_size   = config['pretrain']['batchsize'],
                num_workers  = config['pretrain']['num_workers'],
                prefetch_factor=2,
                shuffle      = (split=='train' and sampler is None),
                sampler      = sampler,
                collate_fn   = collate_fn,
                pin_memory   = False
            )
        else:
            # your original multi-file PyTables approach
            file_splits = get_split_file_lists(config, config['year'], config['doy'])
            datasets = [PyTablesDatasetSplit(config, p, split)
                        for p in tqdm(file_splits[split], desc=f"Loading {split}")]
            cd = torch.utils.data.ConcatDataset(datasets)
            sampler = None
            if config['data'].get('subset_per_epoch') < len(cd):
                sampler = RandomSampler(cd, replacement=False,
                                        num_samples=config['data']['subset_per_epoch'])
            loaders[split] = DataLoader(
                cd,
                batch_size      = config['pretrain']['batchsize'],
                num_workers     = config['pretrain']['num_workers'],
                prefetch_factor = 1,
                shuffle         = (split=='train' and sampler is None),
                sampler         = sampler,
                collate_fn      = collate_fn,
                pin_memory      = False
            )

    return loaders['train'], loaders['val'], loaders['test']