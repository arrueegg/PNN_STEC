import os
import shutil
import h5py
import torch
from torch.utils.data import RandomSampler, SequentialSampler, Dataset, DataLoader, Subset
from utils.locationencoder.pe import SphericalHarmonics
from data_processing.download_solar_indices import OmniDownloader
import tables
from tqdm import tqdm
from utils.feature_registry import FeatureType
from utils.preprocessing import DTYPE, DataPreprocessor

import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

class EpochRandomSampler(RandomSampler):
    """
    RandomSampler that re-seeds itself each epoch to ensure different data sampling.
    """
    def __init__(self, data_source, replacement=False, num_samples=None, base_seed=42):
        # Initialize with a temporary generator
        temp_generator = torch.Generator().manual_seed(base_seed)
        super().__init__(data_source, replacement, num_samples, temp_generator)
        self.base_seed = base_seed
        self.epoch = 0
    
    def __iter__(self):
        # Re-seed the generator with base_seed + epoch for each epoch
        self.generator.manual_seed(self.base_seed + self.epoch)
        return super().__iter__()
    
    def set_epoch(self, epoch):
        """Call this method at the beginning of each epoch"""
        self.epoch = epoch

torch.multiprocessing.set_start_method('fork', force=True)

class H5Dataset(Dataset):
    def __init__(self, config, h5_path, split):
        self.config = config
        self.split = split
        # open the aggregated split file
        self.file = h5py.File(h5_path, 'r', swmr=True,
                              rdcc_nbytes=128*1024*1024,
                              rdcc_w0=0.75,
                              rdcc_nslots=100_003)
        self.data = self.file['data']
        
        # Get feature registry
        self.feature_registry = config.get('feature_registry')
        if not self.feature_registry:
            raise ValueError("Feature registry is required but not found in config")
        
        # Get enabled features (excluding target)
        all_features = self.feature_registry.get_all_enabled_features()
        self.target_feature = self.feature_registry.get_features_by_type(FeatureType.TARGET)[0]
        self.input_features = [f for f in all_features if f != self.target_feature]

        # Get target feature name
        if self.target_feature not in ['stec', 'vtec']:
            raise ValueError(f"Target feature {self.target_feature} is not valid. Expected 'stec' or 'vtec'.")

        # SWI setup
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
            
            # Get SWI feature names from registry for validation
            self.swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]
        
        # Build features according to registry order
        feature_vector = []
        
        for feature_name in self.input_features:
            # Get raw value
            if feature_name == 'year':
                value = float(row['year'])
            elif feature_name == 'doy':
                value = float(row['doy'])
            elif feature_name == 'sod':
                value = float(row['sod'])
            elif feature_name in ['lat_sta', 'lon_sta', 'sm_lat_sta', 'sm_lon_sta']:
                value = float(row[feature_name])
            elif feature_name in ['satazi', 'satele']:
                value = float(row[feature_name])
            elif feature_name in ['lat_ipp', 'lon_ipp', 'sm_lat_ipp', 'sm_lon_ipp']:
                value = float(row[feature_name])
            elif feature_name in self.swi_features:
                # SWI features will be handled separately below
                continue
            else:
                raise ValueError(f"Feature {feature_name} not found in data structure")
            feature_vector.append(value)
        
        feat = torch.tensor(feature_vector, dtype=torch.float32)

        # Add SWI features if enabled
        if self.use_SWI:
            year = str(int(row['year']))
            doy3 = f"{int(row['doy']):03d}"
            # read that day's SWI array and pick the hour
            daily = self.swi_file[year][doy3][:]
            hour = int(row['sod'] // 3600)
            swi_row = daily[hour]
            swi_values = swi_row[self.swi_mask]
            
            # Append raw SWI features without normalization
            swi_feat = torch.tensor(swi_values, dtype=torch.float32)
            feat = torch.cat((feat, swi_feat), dim=0)

        # Get target (label)
        label = torch.tensor(row[self.target_feature], dtype=torch.float32)

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

class H5RAMDataset(Dataset):
    """
    RAM-based dataset that loads entire H5 dataset into memory during initialization.
    This is ideal for cluster environments with abundant RAM where I/O elimination
    is critical for performance.
    """
    
    def __init__(self, config, h5_path, split):
        self.config = config
        self.split = split
        
        print(f"🚀 Loading {split} dataset into RAM from {h5_path}...")
        
        # Get feature registry
        self.feature_registry = config.get('feature_registry')
        if not self.feature_registry:
            raise ValueError("Feature registry is required but not found in config")
        
        # Get enabled features (excluding target)
        all_features = self.feature_registry.get_all_enabled_features()
        self.target_feature = self.feature_registry.get_features_by_type(FeatureType.TARGET)[0]
        self.input_features = [f for f in all_features if f != self.target_feature]
        
        # Get target feature name
        if self.target_feature not in ['stec', 'vtec']:
            raise ValueError(f"Target feature {self.target_feature} is not valid. Expected 'stec' or 'vtec'.")
        
        # SWI setup
        self.use_SWI = config['data'].get('use_SWI', False)
        self.swi_data = None
        self.swi_mask = None
        self.swi_features = None
        
        if self.use_SWI:
            swi_path = os.path.join(
                config['data']['SWI_data_path'],
                "omni_hourly_2010-2025.h5"
            )
            print(f"📡 Loading SWI data into RAM from {swi_path}...")
            self._load_swi_data(swi_path)
        
        # Load main dataset into RAM
        self._load_main_data(h5_path)
        
        print(f"✅ {split} dataset loaded into RAM: {len(self.data):,} samples")
        if self.use_SWI:
            print(f"📡 SWI data loaded: {len(self.swi_data):,} time points")
        
        # Estimate memory usage
        main_memory = self.data.nbytes if hasattr(self.data, 'nbytes') else 0
        swi_memory = 0
        if self.swi_data:
            # Traverse nested dictionary structure: year -> doy -> array
            for year_data in self.swi_data.values():
                for daily_array in year_data.values():
                    if hasattr(daily_array, 'nbytes'):
                        swi_memory += daily_array.nbytes
        total_memory = (main_memory + swi_memory) / (1024**3)  # Convert to GB
        print(f"💾 Estimated RAM usage: {total_memory:.2f} GB")
        print(f"💾 Estimated RAM usage: {main_memory / (1024**3):.2f} GB (main), {swi_memory / (1024**3):.2f} GB (SWI)")

    def _load_swi_data(self, swi_path):
        """Load all SWI data into a nested dictionary structure in RAM."""
        self.swi_data = {}
        
        with h5py.File(swi_path, 'r') as swi_file:
            # Get column mask (exclude YEAR, DOY, HR)
            years = list(swi_file.keys())
            if years:
                days = list(swi_file[years[0]].keys())
                if days:
                    cols = [c.decode() for c in swi_file[years[0]][days[0]].attrs['columns']]
                    self.swi_mask = [c not in ('YEAR', 'DOY', 'HR') for c in cols]
            
            # Get SWI feature names from registry
            self.swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI)
            
            # Load all SWI data with progress bar
            total_days = sum(len(list(swi_file[year].keys())) for year in years)
            pbar = tqdm(total=total_days, desc="Loading SWI data")
            
            for year in years:
                self.swi_data[year] = {}
                for doy in swi_file[year].keys():
                    # Load the entire day's data into RAM
                    daily_data = swi_file[year][doy][:]
                    self.swi_data[year][doy] = daily_data
                    pbar.update(1)
            
            pbar.close()
    
    def _load_main_data(self, h5_path):
        """Load the main H5 dataset into RAM."""
        with h5py.File(h5_path, 'r') as file:
            # Load all data into RAM as a numpy array
            print(f"📊 Loading main data array...")
            self.data = file['data'][:]  # Load entire dataset into memory
            print(f"📊 Main data loaded: shape {self.data.shape}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get item from RAM-cached data - should be very fast!"""
        row = self.data[idx]
        
        # Build features according to registry order
        feature_vector = []
        
        for feature_name in self.input_features:
            # Get raw value
            if feature_name == 'year':
                value = float(row['year'])
            elif feature_name == 'doy':
                value = float(row['doy'])
            elif feature_name == 'sod':
                value = float(row['sod'])
            elif feature_name in ['lat_sta', 'lon_sta', 'sm_lat_sta', 'sm_lon_sta']:
                value = float(row[feature_name])
            elif feature_name in ['satazi', 'satele']:
                value = float(row[feature_name])
            elif feature_name in ['lat_ipp', 'lon_ipp', 'sm_lat_ipp', 'sm_lon_ipp']:
                value = float(row[feature_name])
            elif feature_name in self.swi_features:
                # SWI features will be handled separately below
                continue
            else:
                raise ValueError(f"Feature {feature_name} not found in data structure")
            feature_vector.append(value)
        
        feat = torch.tensor(feature_vector, dtype=torch.float32)
        
        # Add SWI features if enabled - now from RAM!
        if self.use_SWI and self.swi_data:
            year = str(int(row['year']))
            doy3 = f"{int(row['doy']):03d}"
            hour = int(row['sod'] // 3600)
            
            # Access from RAM instead of disk
            if year in self.swi_data and doy3 in self.swi_data[year]:
                daily_data = self.swi_data[year][doy3]
                if hour < len(daily_data):
                    swi_row = daily_data[hour]
                    swi_values = swi_row[self.swi_mask]
                    
                    # Append raw SWI features without normalization
                    swi_feat = torch.tensor(swi_values, dtype=torch.float32)
                    feat = torch.cat((feat, swi_feat), dim=0)
                else:
                    # Handle edge case where hour is out of range
                    swi_feat = torch.zeros(sum(self.swi_mask), dtype=torch.float32)
                    feat = torch.cat((feat, swi_feat), dim=0)
            else:
                # Handle missing SWI data
                swi_feat = torch.zeros(sum(self.swi_mask), dtype=torch.float32)
                feat = torch.cat((feat, swi_feat), dim=0)
        
        # Get target (label)
        label = torch.tensor(row[self.target_feature], dtype=torch.float32)
        
        # Guard NaNs
        if torch.isnan(feat).any() or torch.isnan(label):
            raise ValueError(f"NaN in H5RAMDataset at idx {idx}")
        
        return feat, label

class PyTablesDatasetSplit(Dataset):
    def __init__(self, h5_file_path, year, doy, split, config):
        self.h5_file_path = h5_file_path
        self.year = year
        self.doy = doy
        self.split = split
        self.config = config
        self.file = None
        self.data = None
        self.indices = None
        
        # Get feature registry from config
        self.feature_registry = config.get('feature_registry')
        if not self.feature_registry:
            raise ValueError("Feature registry not found in config")
        
        # Get enabled features (excluding target)
        all_features = self.feature_registry.get_all_enabled_features()
        target_features = self.feature_registry.get_features_by_type(FeatureType.TARGET)
        self.target_feature = target_features[0]  # Get the single target feature name
        self.input_features = [f for f in all_features if f not in target_features]
        
        # Validate target feature name
        if self.target_feature not in ['stec', 'vtec']:
            raise ValueError(f"Target feature {self.target_feature} is not valid. Expected 'stec' or 'vtec'.")

    def __len__(self):
        if self.file is None:
            self.file = tables.open_file(self.h5_file_path, mode='r')
            self.indices = self.file.get_node(f'/{self.year}/{self.doy}/{self.split}_idx')
        return len(self.indices)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = tables.open_file(self.h5_file_path, mode='r')
            self.data = self.file.get_node(f'/{self.year}/{self.doy}/all_data')
            self.indices = self.file.get_node(f'/{self.year}/{self.doy}/{self.split}_idx')
        
        row = self.data[self.indices[idx]]

        # Build features according to registry order and normalize them
        feature_vector = []
        
        for feature_name in self.input_features:
            if feature_name == 'year':
                value = float(self.year)
            elif feature_name == 'doy':
                value = float(self.doy)
            elif feature_name in row.dtype.names:
                value = float(row[feature_name])
            else:
                raise ValueError(f"Feature {feature_name} not found in data")
            
            feature_vector.append(value)  # Add the value to the feature vector
        
        target_name = self.target_feature  # Now it's a string, not a list
        if target_name in row.dtype.names:
            target = float(row[target_name])
        else:
            raise ValueError(f"Target {target_name} not found in data")

        return torch.tensor(feature_vector, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    
    def __del__(self):
        if self.file is not None:
            self.file.close()

class CollateWithSH:
    def __init__(self, config):
        # Get feature registry
        self.feature_registry = config.get('feature_registry')
        
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
        
        # Get specific feature indices from registry (for raw input vector)
        self.input_indices = self._compute_input_feature_indices()
        
        # Compute and store output indices in the registry
        self.expected_dim = self._compute_and_store_output_indices()
        
        
    def _compute_input_feature_indices(self):
        """Compute feature indices for the RAW input vector (before transformation)"""
        indices = {}
        
        # Get all enabled features excluding target
        all_enabled_features = self.feature_registry.get_all_enabled_features()
        target_features = self.feature_registry.get_features_by_type(FeatureType.TARGET)
        
        # Remove target features from the input features list
        input_features = [f for f in all_enabled_features if f not in target_features]
        
        # Build indices mapping feature names to their positions in the RAW input vector
        idx = 0
        for feature_name in input_features:
            # Skip SWI features in the main feature vector - they're appended separately
            if feature_name in self.feature_registry.get_features_by_type(FeatureType.SWI):
                continue
            indices[feature_name] = idx
            idx += 1
        
        # Add SWI features at the end (they're concatenated after the main features)
        swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI)
        for feature_name in swi_features:
            indices[feature_name] = idx
            idx += 1
        
        return indices

    def _compute_and_store_output_indices(self):
        """Compute and store output indices in the feature registry"""
        output_indices = {}
        current_idx = 0
        
        # Temporal features
        temporal_features = self.feature_registry.get_feature_names(FeatureType.TEMPORAL)
        for feature_name in temporal_features:
            if feature_name == 'year':
                output_indices[f'{feature_name}_norm'] = current_idx
                current_idx += 1
            elif feature_name == 'doy' or feature_name == 'sod':
                output_indices[f'{feature_name}_sin'] = current_idx
                output_indices[f'{feature_name}_cos'] = current_idx + 1
                output_indices[f'{feature_name}_norm'] = current_idx + 2
                current_idx += 3
        
        # Station features
        station_features = self.feature_registry.get_feature_names(FeatureType.STATION)
        for feature_name in station_features:
            output_indices[f'{feature_name}_norm'] = current_idx
            current_idx += 1
        
        # Direction features
        direction_features = self.feature_registry.get_feature_names(FeatureType.DIRECTION)
        for feature_name in direction_features:
            if feature_name == 'satazi':
                output_indices[f'{feature_name}_norm'] = current_idx
                output_indices[f'{feature_name}_sin'] = current_idx + 1
                output_indices[f'{feature_name}_cos'] = current_idx + 2
                current_idx += 3
            elif feature_name == 'satele':
                output_indices[f'{feature_name}_norm'] = current_idx
                current_idx += 1
        
        # IPP features
        ipp_features = self.feature_registry.get_feature_names(FeatureType.IPP)
        for feature_name in ipp_features:
            output_indices[f'{feature_name}_norm'] = current_idx
            current_idx += 1
        
        # SH embeddings if enabled
        if self.sh_enabled:
            sh_dim = self.sh_degree * self.sh_degree
            
            # Check if station features are available (they're excluded for VTEC)
            has_station_features = len(self.feature_registry.get_feature_names(FeatureType.STATION)) > 0
            
            # Store ranges for SH embeddings - only include station SH if station features available
            if has_station_features:
                output_indices['sh_sta_geo'] = slice(current_idx, current_idx + sh_dim)
                current_idx += sh_dim
            else:
                output_indices['sh_sta_geo'] = None
            
            output_indices['sh_ipp_geo'] = slice(current_idx, current_idx + sh_dim)
            current_idx += sh_dim
            
            if has_station_features:
                output_indices['sh_sta_sm'] = slice(current_idx, current_idx + sh_dim)
                current_idx += sh_dim
            else:
                output_indices['sh_sta_sm'] = None
            
            output_indices['sh_ipp_sm'] = slice(current_idx, current_idx + sh_dim)
            current_idx += sh_dim
        
        # SWI features
        swi_features = self.feature_registry.get_feature_names(FeatureType.SWI)
        for feature_name in swi_features:
            output_indices[f'{feature_name}_norm'] = current_idx
            current_idx += 1
        
        # Store in registry
        self.feature_registry.set_output_indices(output_indices)

        return current_idx

    def transform_temporal(self, features):
        """Transform temporal features using feature registry"""
        temporal_features = self.feature_registry.get_feature_names(FeatureType.TEMPORAL)
        transformed_features = []

        for feature_name in temporal_features:
            feature_idx = self.input_indices[feature_name]
            feature_values = features[:, feature_idx]

            if feature_name == 'year':
                # Normalize year
                year_norm = self.feature_registry.normalize_feature(feature_name, feature_values).unsqueeze(1)
                transformed_features.extend([year_norm])
            elif feature_name == 'doy':
                # Day of year transformations
                doy_norm = self.feature_registry.normalize_feature(feature_name, feature_values).unsqueeze(1)
                doy_sin = torch.sin(doy_norm * 2 * torch.pi)
                doy_cos = torch.cos(doy_norm * 2 * torch.pi)
                transformed_features.extend([doy_sin, doy_cos, doy_norm])
            elif feature_name == 'sod':
                # Time of day transformations
                norm_sod = self.feature_registry.normalize_feature(feature_name, feature_values).unsqueeze(1)
                sin_sod = torch.sin(norm_sod * 2 * torch.pi)
                cos_sod = torch.cos(norm_sod * 2 * torch.pi)
                transformed_features.extend([sin_sod, cos_sod, norm_sod])
            else:
                raise ValueError(f"Unexpected temporal feature: {feature_name}")

        return torch.cat(transformed_features, dim=1)

    def transform_station(self, features):
        """Transform station features"""
        station_features = self.feature_registry.get_feature_names(FeatureType.STATION)
        
        # Return None if no station features are available (e.g., for VTEC)
        if not station_features:
            return None
            
        transformed_features = []

        for feature_name in station_features:
            feature_idx = self.input_indices[feature_name]
            feature_values = features[:, feature_idx]
            feature_norm = self.feature_registry.normalize_feature(feature_name, feature_values).unsqueeze(1)
            transformed_features.extend([feature_norm])

        return torch.cat(transformed_features, dim=1)

    def transform_direction(self, features):
        """Transform direction features (azimuth, elevation)"""
        direction_features = self.feature_registry.get_feature_names(FeatureType.DIRECTION)
        
        # Return None if no direction features are available (e.g., for VTEC)
        if not direction_features:
            return None
            
        transformed_features = []

        for feature_name in direction_features:
            feature_idx = self.input_indices[feature_name]
            feature_values = features[:, feature_idx]

            if feature_name == 'satazi':
                azi_norm = self.feature_registry.normalize_feature(feature_name, feature_values).unsqueeze(1)
                azi_sin = torch.sin(azi_norm * 2 * torch.pi)
                azi_cos = torch.cos(azi_norm * 2 * torch.pi)
                transformed_features.extend([azi_norm, azi_sin, azi_cos])
            elif feature_name == 'satele':
                ele_norm = self.feature_registry.normalize_feature(feature_name, feature_values).unsqueeze(1)
                transformed_features.append(ele_norm)
            else:
                raise ValueError(f"Unexpected direction feature: {feature_name}")

        return torch.cat(transformed_features, dim=1)

    def transform_ipp(self, features):
        """Transform IPP features"""
        ipp_features = self.feature_registry.get_feature_names(FeatureType.IPP)
        transformed_features = []

        for feature_name in ipp_features:
            feature_idx = self.input_indices[feature_name]
            feature_values = features[:, feature_idx]
            feature_norm = self.feature_registry.normalize_feature(feature_name, feature_values).unsqueeze(1)
            transformed_features.extend([feature_norm])

        return torch.cat(transformed_features, dim=1)

    def transform_swi(self, features):
        """Transform SWI features using feature registry"""
        swi_features = self.feature_registry.get_feature_names(FeatureType.SWI)
        
        if not swi_features:
            return None
        
        transformed_features = []

        for feature_name in swi_features:
            feature_idx = self.input_indices[feature_name]
            feature_values = features[:, feature_idx]
            feature_norm = self.feature_registry.normalize_feature(feature_name, feature_values).unsqueeze(1)
            transformed_features.append(feature_norm)

        return torch.cat(transformed_features, dim=1)

    def compute_sh_embeddings(self, features):
        """Compute spherical harmonic embeddings for station and IPP"""
        if not self.sh_enabled:
            return None, None, None, None

        # Check if station features are available (they're excluded for VTEC)
        has_station_features = 'lon_sta' in self.input_indices and 'lat_sta' in self.input_indices
        has_sm_station_features = 'sm_lon_sta' in self.input_indices and 'sm_lat_sta' in self.input_indices

        # Station SH embeddings (use geographic coordinates for SH) - only if station features available
        if has_station_features:
            sta_lon = features[:, self.input_indices['lon_sta']]
            sta_lat = features[:, self.input_indices['lat_sta']]
            sta_lonlat = torch.stack([sta_lon, sta_lat], dim=1)
            sh_sta_geo = self.sh_encoder(sta_lonlat)
        else:
            sh_sta_geo = None

        # IPP SH embeddings (use geographic coordinates for SH)
        ipp_lon = features[:, self.input_indices['lon_ipp']]
        ipp_lat = features[:, self.input_indices['lat_ipp']]
        ipp_lonlat = torch.stack([ipp_lon, ipp_lat], dim=1)
        sh_ipp_geo = self.sh_encoder(ipp_lonlat)

        # Station SH embedding (use solar magnetic coordinates for SH) - only if station features available
        if has_sm_station_features:
            sm_lon_sta = features[:, self.input_indices['sm_lon_sta']]
            sm_lat_sta = features[:, self.input_indices['sm_lat_sta']]
            sm_lonlat_sta = torch.stack([sm_lon_sta, sm_lat_sta], dim=1)
            sh_sta_sm = self.sh_encoder(sm_lonlat_sta)
        else:
            sh_sta_sm = None

        # IPP SH embedding (use solar magnetic coordinates for SH)
        sm_lon_ipp = features[:, self.input_indices['sm_lon_ipp']]
        sm_lat_ipp = features[:, self.input_indices['sm_lat_ipp']]
        sm_lonlat_ipp = torch.stack([sm_lon_ipp, sm_lat_ipp], dim=1)
        sh_ipp_sm = self.sh_encoder(sm_lonlat_ipp)

        # Return in the order expected by _compute_and_store_output_indices
        return sh_sta_geo, sh_ipp_geo, sh_sta_sm, sh_ipp_sm

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
        sh_sta_geo, sh_ipp_geo, sh_sta_sm, sh_ipp_sm = self.compute_sh_embeddings(features)

        # Combine all transformed features in the SAME ORDER as _compute_and_store_output_indices
        output_features = [temporal_transformed]
        
        # Only add station and direction features if they exist (excluded for VTEC)
        if station_transformed is not None:
            output_features.append(station_transformed)
        if direction_transformed is not None:
            output_features.append(direction_transformed)
            
        output_features.append(ipp_transformed)
        
        # Add SH embeddings if computed (in the exact order from _compute_and_store_output_indices)
        if self.sh_enabled:
            # Only add station SH embeddings if they exist (they're excluded for VTEC)
            if sh_sta_geo is not None:
                output_features.append(sh_sta_geo)
            output_features.append(sh_ipp_geo)
            if sh_sta_sm is not None:
                output_features.append(sh_sta_sm)
            output_features.append(sh_ipp_sm)

        # Add SWI features if available
        if swi_transformed is not None:
            output_features.append(swi_transformed)
        
        # Concatenate all features
        final_features = torch.cat(output_features, dim=1)
                
        return final_features, labels

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

def get_data_loaders(config, logger=None):
    collate_fn = CollateWithSH(config)
    loaders = {}

     # ---- config knobs ----
    train_subset = config['data'].get('train_subset_size', 50_000)
    total_size = train_subset / 0.7
    val_subset = max(int(total_size * 0.15), 200_000)
    test_subset = max(int(total_size * 0.15), 1_000_000)
    device = config['device']
    seed = int(config.get('seed', 42))
    bs   = config['pretrain']['batchsize']
    nw   = config['pretrain']['num_workers']
    pf   = config['pretrain']['prefetch_factor']
    use_agg_h5 = config['data'].get('use_agg_h5', False)
    build_agg_h5 = config['data'].get('build_agg_h5', True)
    
    # Add debug mode for single batch overfitting
    debug_single_batch = config.get('debug_single_batch', False)
    if debug_single_batch:
        train_subset = bs  # Use exactly one batch worth of data
        val_subset = bs
        test_subset = bs
        print(f"DEBUG MODE: Using single batch of size {bs} for all splits")

    # build splits if requested
    if use_agg_h5 and build_agg_h5:
        # Use the new class-based approach with resume capability
        preprocessor = DataPreprocessor(config, logger)
        success = preprocessor.build_split_h5()
        if not success:
            raise RuntimeError("Failed to build split H5 files")

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
            
            # Use RAM-based dataset if running on cluster
            if config.get('cluster', False):
                if logger:
                    logger.info(f"🚀 Using RAM-based dataset for {split} (cluster mode)")
                ds = H5RAMDataset(config, path, split)
            else:
                ds = H5Dataset(config, path, split)
        else:
            preprocessor = DataPreprocessor(config, logger)
            file_splits = preprocessor.get_split_file_lists()
            datasets = []
            
            for file_path in tqdm(file_splits[split], desc=f"Loading {split}"):
                # Extract year and doy from file path
                # Path format: .../{year}/{doy}/ccl_{year}{doy}_30_5.h5
                year = file_path.split('/')[-3]
                doy = file_path.split('/')[-2] 
                
                # Only use regular PyTablesDatasetSplit (no RAM version for this path)
                datasets.append(PyTablesDatasetSplit(file_path, year, doy, split, config))
            ds = torch.utils.data.ConcatDataset(datasets)

        # -----------------------------
        # Sampler / subset per split
        # -----------------------------
        if split == 'train':
            # Debug mode: use fixed subset for overfitting
            if debug_single_batch:
                cache_dir = './debug_subsets_idx'
                cache_path = os.path.join(config['data']['scratch_dir'], cache_dir, f"debug_train_subset_idx.pt")
                idx = get_fixed_subset_indices(ds, train_subset, cache_path, seed=seed)
                ds = Subset(ds, idx)
                # Use sequential sampler to get the same batch every time
                sampler = SequentialSampler(ds)
                shuffle = False
            # Regular training mode
            elif train_subset and train_subset < len(ds):
                # Use custom sampler that re-seeds each epoch for different data sampling without replacement
                sampler = EpochRandomSampler(ds, replacement=False, num_samples=train_subset, base_seed=seed)
                shuffle = False
            else:
                sampler = None
                shuffle = True  # classic full-epoch shuffle

            loaders[split] = DataLoader(
                ds,
                batch_size=bs,
                num_workers=nw,
                persistent_workers=True,
                prefetch_factor=pf,
                shuffle=shuffle,
                sampler=sampler,
                collate_fn=collate_fn,
                pin_memory=(device != 'cpu'),  # Only pin memory for GPU
            )

        else:
            # ---- fixed, random, deterministic subset for val/test ----
            subset_size = val_subset if split == 'val' else test_subset
            if subset_size:
                if debug_single_batch:
                    # In debug mode, use the same subset as training for consistency
                    cache_dir = './debug_subsets_idx'
                    cache_path = os.path.join(config['data']['scratch_dir'], cache_dir, f"debug_train_subset_idx.pt")
                else:
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
                prefetch_factor=pf,
                persistent_workers=True,
                shuffle=False,
                sampler=sampler,
                collate_fn=collate_fn,
                pin_memory=(device != 'cpu'),  # Only pin memory for GPU
            )

    return loaders['train'], loaders['val'], loaders['test']
