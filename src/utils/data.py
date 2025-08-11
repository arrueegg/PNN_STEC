import os
import shutil
import h5py
import torch
import numpy as np
import random
from torch.utils.data import RandomSampler, SequentialSampler, Dataset, DataLoader, Subset
from utils.locationencoder.pe import SphericalHarmonics
from data_processing.download_solar_indices import OmniDownloader
import tables
from tqdm import tqdm
from utils.feature_registry import FeatureType

import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

torch.multiprocessing.set_start_method('fork', force=True)

# Structured dtype for our "one‐table" HDF5 per split:
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
        
        # Get feature registry
        self.feature_registry = config.get('feature_registry')
        if not self.feature_registry:
            raise ValueError("Feature registry is required but not found in config")
        
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
        
        # Build features according to registry
        feature_vector = []
        
        # Add temporal features
        temporal_features = [
            row['year'],
            row['doy'], 
            row['sod']
        ]
        feature_vector.extend(temporal_features)
        
        # Add station features  
        station_features = [
            row['sm_lat_sta'],
            row['sm_lon_sta']
        ]
        feature_vector.extend(station_features)
        
        # Add direction features (azimuth, elevation)
        direction_features = [
            row['satazi'],
            row['satele']
        ]
        feature_vector.extend(direction_features)
        
        # Add IPP features
        ipp_features = [
            row['lat_ipp'],
            row['lon_ipp']
        ]
        feature_vector.extend(ipp_features)
        
        feat = torch.tensor(feature_vector, dtype=torch.float32)

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
        if hasattr(self, 'file') and self.file:
            self.file.close()
        if hasattr(self, 'swi_file') and self.use_SWI and self.swi_file:
            self.swi_file.close()

class PyTablesDatasetSplit(Dataset):
    def __init__(self, config, h5_file_path, split):
        self.config = config
        self.h5_file_path = h5_file_path
        self.doy = self.h5_file_path.split('/')[-1].split('_')[1][4:]
        self.year = self.h5_file_path.split('/')[-1].split('_')[1][:4]
        self.split = split
        
        # Get feature registry
        self.feature_registry = config.get('feature_registry')
        if not self.feature_registry:
            raise ValueError("Feature registry is required but not found in config")
            
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

        # Build features according to registry
        feature_vector = []
        
        # Add temporal features
        temporal_features = [
            int(self.year),
            int(self.doy),
            row['sod']
        ]
        feature_vector.extend(temporal_features)
        
        # Add station features
        station_features = [
            row['sm_lat_sta'],
            row['sm_lon_sta']
        ]
        feature_vector.extend(station_features)
        
        # Add direction features (azimuth, elevation)
        direction_features = [
            row['satazi'],
            row['satele']
        ]
        feature_vector.extend(direction_features)
        
        # Add IPP features  
        ipp_features = [
            row['lat_ipp'],
            row['lon_ipp']
        ]
        feature_vector.extend(ipp_features)
        
        features = torch.tensor(feature_vector, dtype=torch.float32)
        
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
        # Get feature registry
        self.feature_registry = config.get('feature_registry')
        self.feature_monitor = config.get('feature_monitor')
        
        if not self.feature_registry:
            raise ValueError("Feature registry is required but not found in config")
        
        # SH degree and flag
        self.sh_degree = config["data"].get("SH_degree", 0) or 0
        self.sh_enabled = self.sh_degree > 0
        if self.sh_enabled:
            self.sh_encoder = SphericalHarmonics(legendre_polys=self.sh_degree)

        # Pre-compute feature slices for efficiency
        self.slices = {
            'temporal': self.feature_registry.get_feature_slice(FeatureType.TEMPORAL),
            'station': self.feature_registry.get_feature_slice(FeatureType.STATION), 
            'direction': self.feature_registry.get_feature_slice(FeatureType.DIRECTION),
            'ipp': self.feature_registry.get_feature_slice(FeatureType.IPP),
            'swi': self.feature_registry.get_feature_slice(FeatureType.SWI) if config['data'].get('use_SWI', False) else slice(0, 0)
        }
        
        # Get specific feature indices from registry
        self.indices = self._compute_feature_indices()
        
        # Compute expected output dimensions
        self.expected_dim = self._compute_output_dim()

        # Initialize SWI normalizer
        self.swi_normalizer = SWINormalizer(self.feature_registry)

    def _compute_feature_indices(self):
        """Compute feature indices based on registry"""
        indices = {}
        
        # Get all feature names by type
        temporal_names = self.feature_registry.get_feature_names(FeatureType.TEMPORAL)
        station_names = self.feature_registry.get_feature_names(FeatureType.STATION)
        shared_names = self.feature_registry.get_feature_names(FeatureType.SHARED)
        ipp_names = self.feature_registry.get_feature_names(FeatureType.IPP)
        
        # Temporal features
        indices['year'] = temporal_names.index('year')
        indices['doy'] = temporal_names.index('doy') 
        indices['sod'] = temporal_names.index('sod')
        
        # Station features - use the solar magnetic coordinates primarily
        temporal_offset = len(temporal_names)
        indices['lat_sta'] = temporal_offset + station_names.index('sm_lat_sta')
        indices['lon_sta'] = temporal_offset + station_names.index('sm_lon_sta')
        
        # Add geographic station coordinates if needed for SH
        if 'lat_sta' in station_names and 'lon_sta' in station_names:
            indices['geo_lat_sta'] = temporal_offset + station_names.index('lat_sta')
            indices['geo_lon_sta'] = temporal_offset + station_names.index('lon_sta')
        else:
            # Fallback to solar magnetic if geographic not available
            indices['geo_lat_sta'] = indices['lat_sta']
            indices['geo_lon_sta'] = indices['lon_sta']
        
        # Shared features
        shared_offset = temporal_offset + len(station_names)
        indices['azi'] = shared_offset + shared_names.index('satazi')
        indices['ele'] = shared_offset + shared_names.index('satele')
        
        # IPP features
        ipp_offset = shared_offset + len(shared_names)
        indices['ipp_lat'] = ipp_offset + ipp_names.index('lat_ipp')
        indices['ipp_lon'] = ipp_offset + ipp_names.index('lon_ipp')
        
        return indices

    def _compute_output_dim(self):
        """Compute expected output dimension"""
        # Base normalized features
        base_dim = 14  # year, doy_sin, doy_cos, doy_norm, sin_t, cos_t, norm_t, lat_sta_norm, lon_sta_norm, sin_azi, cos_azi, norm_ele, ipp_lat_norm, ipp_lon_norm
        
        # Add SH embeddings if enabled
        if self.sh_enabled:
            sh_dim = self.sh_degree * self.sh_degree  # For both station and IPP
            base_dim += 2 * sh_dim
            
        # Add SWI features if enabled
        swi_slice = self.slices['swi']
        if swi_slice.stop > swi_slice.start:
            swi_dim = swi_slice.stop - swi_slice.start
            base_dim += swi_dim
            
        return base_dim

    def transform_temporal(self, features):
        """Transform temporal features"""
        year = features[:, self.indices['year']]
        doy = features[:, self.indices['doy']]
        sod = features[:, self.indices['sod']]
        
        # Normalize year
        year_norm = ((year - 2010) / 20).unsqueeze(1)
        
        # Day of year transformations
        doy_norm = ((doy - 1) / 365).unsqueeze(1)
        doy_sin = torch.sin(doy_norm * 2 * torch.pi)
        doy_cos = torch.cos(doy_norm * 2 * torch.pi)
        
        # Time of day transformations
        t = sod / 86400 * (2 * torch.pi)
        sin_t = torch.sin(t).unsqueeze(1)
        cos_t = torch.cos(t).unsqueeze(1)
        norm_t = (2 * sod / 86400 - 1).unsqueeze(1)
        
        return torch.cat([year_norm, doy_sin, doy_cos, doy_norm, sin_t, cos_t, norm_t], dim=1)

    def transform_station(self, features):
        """Transform station features"""
        lat_sta = features[:, self.indices['lat_sta']]
        lon_sta = features[:, self.indices['lon_sta']]
        
        # Normalize to [-1, 1]
        lat_norm = ((lat_sta + 90) / 180 * 2 - 1).unsqueeze(1)
        lon_norm = ((lon_sta + 180) / 360 * 2 - 1).unsqueeze(1)
        
        return torch.cat([lat_norm, lon_norm], dim=1)

    def transform_direction(self, features):
        """Transform direction features (azimuth, elevation)"""
        azi = features[:, self.indices['azi']]
        ele = features[:, self.indices['ele']]
        
        # Azimuth transformations
        a = azi / 180 * torch.pi
        sin_a = torch.sin(a).unsqueeze(1)
        cos_a = torch.cos(a).unsqueeze(1)
        
        # Elevation normalization
        norm_e = (2 * ele / 90 - 1).unsqueeze(1)
        
        return torch.cat([sin_a, cos_a, norm_e], dim=1)

    def transform_ipp(self, features):
        """Transform IPP features"""
        ipp_lat = features[:, self.indices['ipp_lat']]
        ipp_lon = features[:, self.indices['ipp_lon']]
        
        # Normalize to [-1, 1]
        lat_norm = ((ipp_lat + 90) / 180 * 2 - 1).unsqueeze(1)
        lon_norm = ((ipp_lon + 180) / 360 * 2 - 1).unsqueeze(1)
        
        return torch.cat([lat_norm, lon_norm], dim=1)

    def transform_swi(self, features):
        """Transform SWI features"""
        swi_slice = self.slices['swi']
        if swi_slice.stop <= swi_slice.start:
            return None
            
        swi_features = features[:, swi_slice]
        return self.swi_normalizer.normalize(swi_features)

    def compute_sh_embeddings(self, features):
        """Compute spherical harmonic embeddings for station and IPP"""
        if not self.sh_enabled:
            return None, None
            
        # Station SH embeddings (use geographic coordinates for SH)
        sta_lon = features[:, self.indices['geo_lon_sta']]
        sta_lat = features[:, self.indices['geo_lat_sta']]
        sta_lonlat = torch.stack([sta_lon, sta_lat], dim=1)
        sh_sta = self.sh_encoder(sta_lonlat)
        
        # IPP SH embeddings (geographic coordinates)
        ipp_lon = features[:, self.indices['ipp_lon']]
        ipp_lat = features[:, self.indices['ipp_lat']]
        ipp_lonlat = torch.stack([ipp_lon, ipp_lat], dim=1)
        sh_ipp = self.sh_encoder(ipp_lonlat)
        
        return sh_sta, sh_ipp

    def __call__(self, batch):
        """Process and collate a batch of (features, labels)"""
        feats, labels = zip(*batch)
        features = torch.stack(feats, dim=0)
        labels = torch.stack(labels, dim=0)

        # Transform different feature types
        temporal_transformed = self.transform_temporal(features)
        station_transformed = self.transform_station(features)
        direction_transformed = self.transform_direction(features)
        ipp_transformed = self.transform_ipp(features)
        swi_transformed = self.transform_swi(features)
        
        # Compute SH embeddings if enabled
        sh_sta, sh_ipp = self.compute_sh_embeddings(features)
        
        # Combine all transformed features
        output_features = [temporal_transformed, station_transformed, direction_transformed, ipp_transformed]
        
        # Add SH embeddings if computed
        if sh_sta is not None and sh_ipp is not None:
            output_features.extend([sh_sta, sh_ipp])
            
        # Add SWI features if available
        if swi_transformed is not None:
            output_features.append(swi_transformed)
        
        # Concatenate all features
        final_features = torch.cat(output_features, dim=1)
        
        # Monitor feature dimensions if enabled
        if self.feature_monitor:
            self.feature_monitor.log_batch_features(final_features, 'collated')
            
        return final_features, labels

# Rest of the utility functions remain the same...
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

    train_dates = generate_dates(train_months)
    val_dates = generate_dates(val_months)
    test_dates = generate_dates(test_months)

    # Debugging: only take a subset of dates for testing
    every_x_doy = config['data'].get('every_x_doy', 1)
    train_dates = train_dates[::every_x_doy]
    val_dates = val_dates[::every_x_doy]
    test_dates = test_dates[::every_x_doy]

    def get_file_paths(dates):
        file_paths = []
        for date in dates:
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
        if v['file']:
            v['file'].close()

    print("✅ Built split H5 files at:", scratch)

def get_fixed_subset_indices(ds, k, cache_path, seed=0):
    """
    Returns k unique indices for ds, chosen randomly but deterministically via seed.
    Caches to cache_path so future runs reuse the exact same subset.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Try load from cache
    if os.path.exists(cache_path):
        saved = torch.load(cache_path)
        if saved.get("len", None) == len(ds) and saved.get("k", None) == k:
            return saved["indices"]

    # Create new subset
    g = torch.Generator().manual_seed(seed)
    k = min(k, len(ds))
    # choose without replacement, deterministic by seed
    perm = torch.randperm(len(ds), generator=g)[:k]
    idx = perm.tolist()

    torch.save({"len": len(ds), "k": k, "seed": seed, "indices": idx}, cache_path)
    return idx

def get_data_loaders(config):
    collate_fn = CollateWithSH(config)
    loaders = {}

     # ---- config knobs ----
    train_subset = config['data'].get('train_subset_size', None)
    total_size = train_subset / 0.7 if train_subset else None
    val_subset = test_subset = int(total_size * 0.15) if total_size else None
    seed = int(config.get('seed', 42))
    bs   = config['pretrain']['batchsize']
    nw   = config['pretrain']['num_workers']
    use_agg_h5 = config['data'].get('use_agg_h5', False)
    build_agg_h5 = config['data'].get('build_agg_h5', True)

    # build splits if requested
    if use_agg_h5 and build_agg_h5:
        build_split_h5(config)

    for split in ['train','val','test']:
        if config['data'].get('use_agg_h5', False):
            # move SWI data to scratch if needed
            swi_scratch_path = os.path.join(config['data']['scratch_dir'], "omni_hourly_2010-2025.h5")
            if not os.path.exists(swi_scratch_path):
                swi_path = os.path.join(config['data']['SWI_data_path'], "omni_hourly_2010-2025.h5")
                if not os.path.exists(swi_path):
                    downloader = OmniDownloader(config['data']['SWI_data_path'], "20100101", "20250625")
                    downloader.run()
                shutil.copy(swi_path, swi_scratch_path)
                config['data']['SWI_data_path'] = os.path.dirname(swi_scratch_path)

            path = os.path.join(config['data']['scratch_dir'], f"{split}.h5")
            ds   = H5Dataset(config, path, split)
        else:
            file_splits = get_split_file_lists(config, config['year'], config['doy'])
            datasets = [PyTablesDatasetSplit(config, p, split) for p in tqdm(file_splits[split], desc=f"Loading {split}")]
            ds = torch.utils.data.ConcatDataset(datasets)

        # -----------------------------
        # Sampler / subset per split
        # -----------------------------
        if split == 'train':
            # If you want to scan full train each epoch, leave subset_per_epoch == 0
            if train_subset and train_subset < len(ds):
                # IMPORTANT: num_samples needs replacement=True
                g = torch.Generator().manual_seed(seed)  # re-seed per epoch in your train loop if desired
                sampler = RandomSampler(ds, replacement=True, num_samples=train_subset, generator=g)
                shuffle = False
            else:
                sampler = None
                shuffle = True  # classic full-epoch shuffle

            loaders[split] = DataLoader(
                ds,
                batch_size=bs,
                num_workers=nw,
                prefetch_factor=2,
                shuffle=shuffle,
                sampler=sampler,
                collate_fn=collate_fn,
                pin_memory=True,
            )

        else:
            # ---- fixed, random, deterministic subset for val/test ----
            subset_size = val_subset if split == 'val' else test_subset
            if subset_size:
                cache_dir = './val_test_subsets_idx'
                cache_path = os.path.join(config['data']['scratch_dir'], cache_dir, f"{split}_subset_idx.pt")
                idx = get_fixed_subset_indices(ds, subset_size, cache_path, seed=seed)
                ds = Subset(ds, idx)

            # Deterministic iteration for stable metrics
            sampler = SequentialSampler(ds)

            loaders[split] = DataLoader(
                ds,
                batch_size=bs,
                num_workers=nw,
                prefetch_factor=1,
                shuffle=False,
                sampler=sampler,
                collate_fn=collate_fn,
                pin_memory=True,
            )

    return loaders['train'], loaders['val'], loaders['test']


class SWINormalizer:
    """Centralized SWI feature normalization based on feature registry"""
    
    def __init__(self, feature_registry=None):
        self.feature_registry = feature_registry
        
        # SWI feature min/max values (matching column order from OmniDownloader)
        self.min_max_values = [
            (2407, 3000),     # Bartels_rotation_number
            (0, 70),          # Scalar_B,_nT
            (0.0, 70),        # Vector_B_Magnitude,nT
            (-90, 90),        # Lat_Angle_of_B_GSE
            (0.0, 360.0),     # Long_Angle_of_B_GSE
            (-50, 35),        # BZ,_nT_GSE
            (-50, 35),        # BZ,_nT_GSM
            (240.0, 1100.0),  # SW_Plasma_Speed,_km/s
            (0, 60),          # Flow_pressure
            (-20, 30),        # E_elecrtic_field
            (0, 120),         # Alfen_mach_number
            (0.0, 100.0),     # Kp_index
            (0.0, 300.0),     # R_Sunspot_No
            (-450, 100),      # Dst-index,_nT
            (0.0, 2500.0),    # AE-index,_nT
            (0.0, 300.0),     # ap_index,_nT
            (62, 420),        # f107_index
            (-6, 16),         # pc-index
            (-2000.0, 20.0),  # AL-index,_nT
            (-200.0, 1200.0), # AU-index,_nT
            (0, 15),          # Magnetosonic_Much_num
            (0, 0.015),       # Lyman_alpha
        ]
    
    def normalize(self, swi_features):
        """Normalize SWI features to [0, 1] range"""
        swi_features = swi_features.float()

        # Convert min and max to tensors
        min_vals = torch.tensor([m[0] for m in self.min_max_values], 
                               dtype=torch.float32, device=swi_features.device)
        max_vals = torch.tensor([m[1] for m in self.min_max_values], 
                               dtype=torch.float32, device=swi_features.device)

        # Vectorized min-max normalization
        normalized_features = (swi_features - min_vals) / (max_vals - min_vals)
        return normalized_features
    
    def get_expected_dim(self):
        """Get expected number of SWI features"""
        return len(self.min_max_values)
