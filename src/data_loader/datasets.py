"""
Dataset Classes Module for PNN_STEC Data Loading

This module contains all dataset classes for handling different data formats:
- H5Dataset: Standard HDF5 dataset with on-demand loading
- H5RAMDataset: RAM-cached HDF5 dataset for cluster environments
- PyTablesDatasetSplit: PyTables-based dataset for split files

Extracted from the original data.py for better modularity and maintainability.
"""

import os
import h5py
import torch
import tables
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.feature_registry import FeatureType


class H5Dataset(Dataset):
    """Standard HDF5 dataset with on-demand loading for STEC data."""
    
    def __init__(self, config, h5_path, split):
        self.config = config
        self.split = split
        # open the aggregated split file
        self.file = h5py.File(h5_path, 'r', swmr=True)
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
        return len(self.data)

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
            hour = int(row['sod'] // 3600)
            # FIXED: Load only the specific hour, not the entire day
            swi_row = self.swi_file[year][doy3][hour]  # Load only one hour
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
    """PyTables-based dataset for handling split data files."""
    
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