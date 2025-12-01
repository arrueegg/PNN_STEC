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
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.feature_registry import FeatureType


def compute_local_time_hours(sod, longitude):
    """
    Compute local time in hours from UTC seconds of day and longitude.
    
    Args:
        sod: Seconds of day in UTC (0-86400)
        longitude: Longitude in degrees (-180 to 180)
    
    Returns:
        Local time in hours (0-24)
    """
    utc_hours = sod / 3600.0  # Convert seconds to hours
    # Longitude offset: each 15 degrees = 1 hour time zone
    longitude_offset = longitude / 15.0  
    local_time_hours = utc_hours + longitude_offset
    
    # Wrap to 0-24 hour range
    local_time_hours = local_time_hours % 24.0
    
    return local_time_hours


class H5Dataset(Dataset):
    """Standard HDF5 dataset with on-demand loading for STEC data."""

    def __init__(self, config, h5_path, split):
        self.config = config
        self.split = split
        # open the aggregated split file
        self.file = h5py.File(h5_path, "r", swmr=True)
        self.data = self.file["data"]

        # Get feature registry
        self.feature_registry = config.get("feature_registry")
        if not self.feature_registry:
            raise ValueError("Feature registry is required but not found in config")

        # Get enabled features (excluding target)
        all_features = self.feature_registry.get_all_enabled_features()
        self.target_feature = self.feature_registry.get_features_by_type(
            FeatureType.TARGET
        )[0]
        self.input_features = [f for f in all_features if f != self.target_feature]
        
        # Get SWI features for later filtering during iteration
        self.swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI) if self.feature_registry else []

        # Get target feature name
        if self.target_feature not in ["stec", "vtec"]:
            raise ValueError(
                f"Target feature {self.target_feature} is not valid. Expected 'stec' or 'vtec'."
            )
        
        # Check if we should return metadata (for dSTEC evaluation)
        self.return_metadata = config.get("return_metadata", False)
        self.metadata_fields = config.get("metadata_fields", ["station", "sat", "slipc", "gfphase"])

        # SWI setup
        self.use_SWI = config["data"].get("use_SWI", False)
        if self.use_SWI:
            swi_path = os.path.join(
                config["data"]["SWI_data_path"], "omni_hourly_2010-2025.h5"
            )
            self.swi_file = h5py.File(swi_path, "r")
            # build mask of SWI columns (drop YEAR, DOY, HR)
            # we can peek at any day, say first in first year
            yrs = list(self.swi_file.keys())
            days = list(self.swi_file[yrs[0]].keys())
            cols = [c.decode() for c in self.swi_file[yrs[0]][days[0]].attrs["columns"]]
            # store full column names and compute mask (keep YEAR/DOY/HR positions)
            self.swi_col_names = cols
            self.swi_mask = [c not in ("YEAR", "DOY", "HR") for c in cols]

            # Get SWI feature names from registry and compute mapping to masked file columns
            masked_names = [n for n, m in zip(cols, self.swi_mask) if m]
            self.swi_name_to_idx = {name: i for i, name in enumerate(masked_names)}
            # For each registry feature, record index in masked array (or None)
            self.swi_indices_in_file_order = [
                self.swi_name_to_idx.get(f, None) for f in self.swi_features
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        # Build features: non-SWI features first, then SWI features (old pretraining order)
        feature_vector = []

        # First, add all non-SWI features in registry order
        for feature_name in self.input_features:
            if feature_name not in self.swi_features:
                # Get raw value
                if feature_name == "year":
                    value = float(row["year"])
                elif feature_name == "doy":
                    value = float(row["doy"])
                elif feature_name == "sod":
                    value = float(row["sod"])
                elif feature_name == "local_time_hours":
                    # Compute local time from UTC seconds of day and IPP longitude
                    sod = float(row["sod"])
                    longitude = float(row["lon_ipp"])  # Use IPP longitude for local time
                    value = compute_local_time_hours(sod, longitude)
                elif feature_name in ["lat_sta", "lon_sta", "sm_lat_sta", "sm_lon_sta"]:
                    value = float(row[feature_name])
                elif feature_name in ["satazi", "satele"]:
                    value = float(row[feature_name])
                elif feature_name in ["lat_ipp", "lon_ipp", "sm_lat_ipp", "sm_lon_ipp"]:
                    value = float(row[feature_name])
                else:
                    raise ValueError(f"Feature {feature_name} not found in data structure")
                feature_vector.append(value)

        # Then append SWI features
        if self.use_SWI and self.swi_features:
            for feature_name in self.swi_features:
                year = str(int(row["year"]))
                doy3 = f"{int(row['doy']):03d}"
                hour = int(row["sod"] // 3600)
                swi_row = self.swi_file[year][doy3][hour]
                swi_values_masked = swi_row[self.swi_mask]

                # Find the index of this SWI feature within the masked array
                if hasattr(self, "swi_indices_in_file_order") and self.swi_indices_in_file_order:
                    # Get the position of this feature in the registry SWI features
                    swi_pos = self.swi_features.index(feature_name)
                    in_idx = self.swi_indices_in_file_order[swi_pos]
                    if in_idx is None:
                        value = 0.0
                    else:
                        value = float(swi_values_masked[in_idx])
                else:
                    value = 0.0
                feature_vector.append(value)

        feat = torch.tensor(feature_vector, dtype=torch.float32)

        # Get target (label)
        label = torch.tensor(row[self.target_feature], dtype=torch.float32)

        # guard NaNs
        if torch.isnan(feat).any() or torch.isnan(label):
            raise ValueError(f"NaN in H5Dataset at idx {idx}")

        # Return metadata if requested (for dSTEC evaluation)
        if self.return_metadata:
            metadata = {}
            for field in self.metadata_fields:
                value = row[field]
                # Decode bytes to string for text fields
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                metadata[field] = value
            return feat, label, metadata
        
        return feat, label

    def __del__(self):
        # close both files
        if hasattr(self, "file") and self.file:
            self.file.close()
        if hasattr(self, "swi_file") and self.use_SWI and self.swi_file:
            self.swi_file.close()


class H5RAMDataset(Dataset):
    """
    RAM-based dataset that loads entire H5 dataset into memory during initialization.
    This is ideal for cluster environments with abundant RAM where I/O elimination
    is critical for performance.
    """

    def __init__(self, config, h5_path, split):
        import logging

        logger = logging.getLogger(__name__)

        self.config = config
        self.split = split

        logger.info(f"🚀 Loading {split} dataset into RAM from {h5_path}...")

        # Get feature registry
        self.feature_registry = config.get("feature_registry")
        if not self.feature_registry:
            raise ValueError("Feature registry is required but not found in config")

        # Get enabled features (excluding target)
        all_features = self.feature_registry.get_all_enabled_features()
        self.target_feature = self.feature_registry.get_features_by_type(
            FeatureType.TARGET
        )[0]
        self.input_features = [f for f in all_features if f != self.target_feature]
        
        # Get SWI features for later filtering during iteration
        self.swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI) if self.feature_registry else []

        # Get target feature name
        if self.target_feature not in ["stec", "vtec"]:
            raise ValueError(
                f"Target feature {self.target_feature} is not valid. Expected 'stec' or 'vtec'."
            )
        
        # Check if we should return metadata (for dSTEC evaluation)
        self.return_metadata = config.get("return_metadata", False)
        self.metadata_fields = config.get("metadata_fields", ["station", "sat", "slipc", "gfphase"])

        # SWI setup
        self.use_SWI = config["data"].get("use_SWI", False)
        self.swi_data = None
        self.swi_mask = None

        if self.use_SWI:
            swi_path = os.path.join(
                config["data"]["SWI_data_path"], "omni_hourly_2010-2025.h5"
            )
            logger.info(f"📡 Loading SWI data into RAM from {swi_path}...")
            self._load_swi_data(swi_path)

        # Load main dataset into RAM
        self._load_main_data(h5_path)

        logger.info(f"✅ {split} dataset loaded into RAM: {len(self.data):,} samples")
        if self.use_SWI:
            logger.info(f"📡 SWI data loaded: {len(self.swi_data):,} time points")

        # Estimate memory usage
        main_memory = self.data.nbytes if hasattr(self.data, "nbytes") else 0
        swi_memory = 0
        if self.swi_data:
            # Traverse nested dictionary structure: year -> doy -> array
            for year_data in self.swi_data.values():
                for daily_array in year_data.values():
                    if hasattr(daily_array, "nbytes"):
                        swi_memory += daily_array.nbytes
        total_memory = (main_memory + swi_memory) / (1024**3)  # Convert to GB
        logger.info(f"💾 Estimated RAM usage: {total_memory:.2f} GB")
        logger.info(
            f"💾 Estimated RAM usage: {main_memory / (1024**3):.2f} GB (main), {swi_memory / (1024**3):.2f} GB (SWI)"
        )

    def _load_swi_data(self, swi_path):
        """Load all SWI data into a nested dictionary structure in RAM."""
        self.swi_data = {}

        with h5py.File(swi_path, "r") as swi_file:
            # Get column mask (exclude YEAR, DOY, HR)
            years = list(swi_file.keys())
            if years:
                days = list(swi_file[years[0]].keys())
                if days:
                    cols = [c.decode() for c in swi_file[years[0]][days[0]].attrs["columns"]]
                    # store full column names and compute mask
                    self.swi_col_names = cols
                    self.swi_mask = [c not in ("YEAR", "DOY", "HR") for c in cols]
                    masked_names = [n for n, m in zip(cols, self.swi_mask) if m]
                    # Map masked column names to indices
                    self.swi_name_to_idx = {name: i for i, name in enumerate(masked_names)}

                    # Map registry SWI features to indices (or None if missing)
                    self.swi_indices_in_file_order = [
                        self.swi_name_to_idx.get(f, None) for f in self.swi_features
                    ]

            # Load all SWI data with progress bar
            total_days = sum(len(list(swi_file[year].keys())) for year in years)
            pbar = tqdm(total=total_days, desc="Loading SWI data")

            for year in years:
                self.swi_data[year] = {}
                for doy in swi_file[year].keys():
                        # Load the entire day's data into RAM
                        daily_data = swi_file[year][doy][:]
                        # Store all hourly data with all SWI features (22)
                        hours = []
                        for hour_idx in range(len(daily_data)):
                            raw = daily_data[hour_idx]
                            masked = raw[self.swi_mask]  # All 22 SWI features
                            hours.append(masked)
                        self.swi_data[year][doy] = np.stack(hours, axis=0)
                        pbar.update(1)

            pbar.close()

    def _load_main_data(self, h5_path):
        """Load the main H5 dataset into RAM."""
        import logging

        logger = logging.getLogger(__name__)

        with h5py.File(h5_path, "r") as file:
            # Load all data into RAM as a numpy array
            logger.info("📊 Loading main data array...")
            self.data = file["data"][:]  # Load entire dataset into memory
            logger.info(f"📊 Main data loaded: shape {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get item from RAM-cached data - should be very fast!"""
        row = self.data[idx]

        # Build features: non-SWI features first, then SWI features (old pretraining order)
        feature_vector = []

        # First, add all non-SWI features in registry order
        for feature_name in self.input_features:
            if feature_name not in self.swi_features:
                # Get raw value
                if feature_name == "year":
                    value = float(row["year"])
                elif feature_name == "doy":
                    value = float(row["doy"])
                elif feature_name == "sod":
                    value = float(row["sod"])
                elif feature_name == "local_time_hours":
                    # Compute local time from UTC seconds of day and IPP longitude
                    sod = float(row["sod"])
                    longitude = float(row["lon_ipp"])  # Use IPP longitude for local time
                    value = compute_local_time_hours(sod, longitude)
                elif feature_name in ["lat_sta", "lon_sta", "sm_lat_sta", "sm_lon_sta"]:
                    value = float(row[feature_name])
                elif feature_name in ["satazi", "satele"]:
                    value = float(row[feature_name])
                elif feature_name in ["lat_ipp", "lon_ipp", "sm_lat_ipp", "sm_lon_ipp"]:
                    value = float(row[feature_name])
                else:
                    raise ValueError(f"Feature {feature_name} not found in data structure")
                feature_vector.append(value)

        # Then append SWI features
        if self.use_SWI and self.swi_features:
            for feature_name in self.swi_features:
                year = str(int(row["year"]))
                doy3 = f"{int(row['doy']):03d}"
                hour = int(row["sod"] // 3600)

                # Access from RAM instead of disk
                if year in self.swi_data and doy3 in self.swi_data[year]:
                    daily_data = self.swi_data[year][doy3]
                    if hour < len(daily_data):
                        swi_row = daily_data[hour]
                        # Find the index of this SWI feature within the masked array
                        if hasattr(self, "swi_indices_in_file_order") and self.swi_indices_in_file_order:
                            # Get the position of this feature in the registry SWI features
                            swi_pos = self.swi_features.index(feature_name)
                            in_idx = self.swi_indices_in_file_order[swi_pos]
                            if in_idx is None:
                                value = 0.0
                            else:
                                value = float(swi_row[in_idx])
                        else:
                            value = 0.0
                    else:
                        value = 0.0
                else:
                    value = 0.0
                feature_vector.append(value)

        feat = torch.tensor(feature_vector, dtype=torch.float32)

        # Get target (label)
        label = torch.tensor(row[self.target_feature], dtype=torch.float32)

        # Guard NaNs
        if torch.isnan(feat).any() or torch.isnan(label):
            raise ValueError(f"NaN in H5RAMDataset at idx {idx}")

        # Return metadata if requested (for dSTEC evaluation)
        if self.return_metadata:
            metadata = {}
            for field in self.metadata_fields:
                value = row[field]
                # Decode bytes to string for text fields
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                metadata[field] = value
            return feat, label, metadata

        return feat, label


class DayRAMDataset(Dataset):
    """RAM-loaded dataset for a single day's PyTables split.

    Loads the entire day's 'all_data' and the split indices into memory so that
    finetuning on a single day can be done without further disk I/O.
    """

    def __init__(self, h5_file_path, year, doy, split, config):
        self.h5_file_path = h5_file_path
        self.year = year
        self.doy = doy
        self.split = split
        self.config = config

        # Load file into RAM immediately
        with tables.open_file(self.h5_file_path, mode="r") as f:
            self.data = f.get_node(f"/{self.year}/{self.doy}/all_data")[:]
            self.indices = f.get_node(f"/{self.year}/{self.doy}/{self.split}_idx")[:]

        # Get feature registry from config
        self.feature_registry = config.get("feature_registry")
        if not self.feature_registry:
            raise ValueError("Feature registry not found in config")

        # Get enabled features (excluding target)
        all_features = self.feature_registry.get_all_enabled_features()
        target_features = self.feature_registry.get_features_by_type(FeatureType.TARGET)
        self.target_feature = target_features[0]
        self.input_features = [f for f in all_features if f not in target_features]
        
        # Get SWI features for later filtering during iteration
        self.swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI) if self.feature_registry else []

        if self.target_feature not in ["stec", "vtec"]:
            raise ValueError(
                f"Target feature {self.target_feature} is not valid. Expected 'stec' or 'vtec'."
            )
        
        # Check if we should return metadata (for dSTEC evaluation)
        self.return_metadata = config.get("return_metadata", False)
        self.metadata_fields = config.get("metadata_fields", ["station", "sat", "slipc", "gfphase"])
        
        # SWI setup: optionally load the specific day's SWI data into RAM
        self.use_SWI = config["data"].get("use_SWI", False)
        self.swi_day = None
        self.swi_mask = None
        if self.use_SWI:
            try:
                swi_path = os.path.join(config["data"]["SWI_data_path"], "omni_hourly_2010-2025.h5")
                if os.path.exists(swi_path):
                    with h5py.File(swi_path, "r") as swi_file:
                        y = str(self.year)
                        d = f"{int(self.doy):03d}"
                        if y in swi_file and d in swi_file[y]:
                            # get column names from attributes and compute mask
                            cols = [c.decode() for c in swi_file[y][d].attrs.get("columns", [])]
                            self.swi_mask = [c not in ("YEAR", "DOY", "HR") for c in cols]
                            self.swi_col_names = [c for c in cols if c not in ("YEAR", "DOY", "HR")]
                            masked_names = [n for n, m in zip(cols, self.swi_mask) if m]
                            # Map registry SWI features to indices
                            self.swi_name_to_idx = {name: i for i, name in enumerate(masked_names)}
                            self.swi_indices_in_file_order = [
                                self.swi_name_to_idx.get(f, None) for f in self.swi_features
                            ]
                            self.swi_day = swi_file[y][d][:]
                        else:
                            # missing SWI day - leave as None and handle later
                            self.swi_day = None
                else:
                    self.swi_day = None
            except Exception:
                self.swi_day = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.data[self.indices[idx]]

        # Build features: non-SWI features first, then SWI features (match H5RAMDataset order)
        feature_vector = []

        # First, add all non-SWI features in registry order
        for feature_name in self.input_features:
            if feature_name not in self.swi_features:
                # Get raw value
                if feature_name == "year":
                    value = float(self.year)
                elif feature_name == "doy":
                    value = float(self.doy)
                elif feature_name == "sod":
                    value = float(row["sod"]) if "sod" in row.dtype.names else 0.0
                elif feature_name == "local_time_hours":
                    sod = float(row["sod"]) if "sod" in row.dtype.names else 0.0
                    longitude = float(row["lon_ipp"]) if "lon_ipp" in row.dtype.names else 0.0
                    value = compute_local_time_hours(sod, longitude)
                elif feature_name in ["lat_sta", "lon_sta", "sm_lat_sta", "sm_lon_sta"]:
                    value = float(row[feature_name]) if feature_name in row.dtype.names else 0.0
                elif feature_name in ["satazi", "satele"]:
                    value = float(row[feature_name]) if feature_name in row.dtype.names else 0.0
                elif feature_name in ["lat_ipp", "lon_ipp", "sm_lat_ipp", "sm_lon_ipp"]:
                    value = float(row[feature_name]) if feature_name in row.dtype.names else 0.0
                else:
                    raise ValueError(f"Feature {feature_name} not found in data structure")
                feature_vector.append(value)

        # Then append SWI features
        if self.use_SWI and self.swi_features:
            for feature_name in self.swi_features:
                # SWI feature: fetch from loaded SWI day if available
                if self.swi_day is not None and self.swi_mask is not None:
                    hour = int(row["sod"] // 3600) if "sod" in row.dtype.names else 0
                    if hour < len(self.swi_day):
                        swi_row = self.swi_day[hour]
                        swi_values = swi_row[self.swi_mask]
                        # Find the index of this SWI feature
                        if hasattr(self, "swi_indices_in_file_order") and self.swi_indices_in_file_order:
                            swi_pos = self.swi_features.index(feature_name)
                            in_idx = self.swi_indices_in_file_order[swi_pos]
                            if in_idx is None:
                                value = 0.0
                            else:
                                value = float(swi_values[in_idx])
                        else:
                            value = 0.0
                    else:
                        value = 0.0
                else:
                    # SWI data not available, fallback to zeros
                    value = 0.0
                feature_vector.append(value)

        feat = torch.tensor(feature_vector, dtype=torch.float32)
        
        # Get target (label)
        target_name = self.target_feature
        if target_name in row.dtype.names:
            label = torch.tensor(float(row[target_name]), dtype=torch.float32)
        else:
            raise ValueError(f"Target {target_name} not found in data")
        
        # Guard NaNs
        if torch.isnan(feat).any() or torch.isnan(label):
            raise ValueError(f"NaN in DayRAMDataset at idx {idx}")
        
        # Return metadata if requested (for dSTEC evaluation)
        if self.return_metadata:
            metadata = {}
            for field in self.metadata_fields:
                if field in row.dtype.names:
                    value = row[field]
                    # Decode bytes to string for text fields
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    metadata[field] = value
            return feat, label, metadata
        
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
        self.feature_registry = config.get("feature_registry")
        if not self.feature_registry:
            raise ValueError("Feature registry not found in config")

        # Get enabled features (excluding target)
        all_features = self.feature_registry.get_all_enabled_features()
        target_features = self.feature_registry.get_features_by_type(FeatureType.TARGET)
        self.target_feature = target_features[0]  # Get the single target feature name
        self.input_features = [f for f in all_features if f not in target_features]
        
        # Get SWI features for later filtering during iteration
        self.swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI) if self.feature_registry else []

        # Validate target feature name
        if self.target_feature not in ["stec", "vtec"]:
            raise ValueError(
                f"Target feature {self.target_feature} is not valid. Expected 'stec' or 'vtec'."
            )
        
        # Check if we should return metadata (for dSTEC evaluation)
        self.return_metadata = config.get("return_metadata", False)
        self.metadata_fields = config.get("metadata_fields", ["station", "sat", "slipc", "gfphase"])

        # SWI setup
        self.use_SWI = config["data"].get("use_SWI", False)
        self.swi_file = None
        self.swi_mask = None
        if self.use_SWI:
            swi_path = os.path.join(config["data"]["SWI_data_path"], "omni_hourly_2010-2025.h5")
            if os.path.exists(swi_path):
                self.swi_file = h5py.File(swi_path, "r", swmr=True)
                # Get column mask (exclude YEAR, DOY, HR)
                years = list(self.swi_file.keys())
                if years:
                    days = list(self.swi_file[years[0]].keys())
                    if days:
                        cols = [c.decode() for c in self.swi_file[years[0]][days[0]].attrs["columns"]]
                        self.swi_mask = [c not in ("YEAR", "DOY", "HR") for c in cols]
                        masked_names = [n for n, m in zip(cols, self.swi_mask) if m]
                        # Map registry SWI features to indices
                        self.swi_name_to_idx = {name: i for i, name in enumerate(masked_names)}
                        self.swi_indices_in_file_order = [
                            self.swi_name_to_idx.get(f, None) for f in self.swi_features
                        ]

    def __len__(self):
        if self.file is None:
            self.file = tables.open_file(self.h5_file_path, mode="r")
            self.indices = self.file.get_node(
                f"/{self.year}/{self.doy}/{self.split}_idx"
            )
        return len(self.indices)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = tables.open_file(self.h5_file_path, mode="r")
            self.data = self.file.get_node(f"/{self.year}/{self.doy}/all_data")
            self.indices = self.file.get_node(
                f"/{self.year}/{self.doy}/{self.split}_idx"
            )

        row = self.data[self.indices[idx]]

        # Build features: non-SWI features first, then SWI features (match H5RAMDataset order)
        feature_vector = []

        # First, add all non-SWI features in registry order
        for feature_name in self.input_features:
            if feature_name not in self.swi_features:
                # Get raw value
                if feature_name == "year":
                    value = float(self.year)
                elif feature_name == "doy":
                    value = float(self.doy)
                elif feature_name == "sod":
                    value = float(row["sod"]) if "sod" in row.dtype.names else 0.0
                elif feature_name == "local_time_hours":
                    # Compute local time from UTC seconds of day and IPP longitude
                    sod = float(row["sod"]) if "sod" in row.dtype.names else 0.0
                    longitude = float(row["lon_ipp"]) if "lon_ipp" in row.dtype.names else 0.0
                    value = compute_local_time_hours(sod, longitude)
                elif feature_name in ["lat_sta", "lon_sta", "sm_lat_sta", "sm_lon_sta"]:
                    value = float(row[feature_name]) if feature_name in row.dtype.names else 0.0
                elif feature_name in ["satazi", "satele"]:
                    value = float(row[feature_name]) if feature_name in row.dtype.names else 0.0
                elif feature_name in ["lat_ipp", "lon_ipp", "sm_lat_ipp", "sm_lon_ipp"]:
                    value = float(row[feature_name]) if feature_name in row.dtype.names else 0.0
                else:
                    raise ValueError(f"Feature {feature_name} not found in data structure")
                feature_vector.append(value)

        # Then append SWI features
        if self.use_SWI and self.swi_features:
            for feature_name in self.swi_features:
                # SWI feature: fetch from SWI file
                if self.swi_file is not None and self.swi_mask is not None:
                    y = str(self.year)
                    d = f"{int(self.doy):03d}"
                    hour = int(row["sod"] // 3600) if "sod" in row.dtype.names else 0
                    if y in self.swi_file and d in self.swi_file[y]:
                        swi_row = self.swi_file[y][d][hour]
                        swi_values = swi_row[self.swi_mask]
                        # Find the index of this SWI feature
                        if hasattr(self, "swi_indices_in_file_order") and self.swi_indices_in_file_order:
                            swi_pos = self.swi_features.index(feature_name)
                            in_idx = self.swi_indices_in_file_order[swi_pos]
                            if in_idx is None:
                                value = 0.0
                            else:
                                value = float(swi_values[in_idx])
                        else:
                            value = 0.0
                    else:
                        value = 0.0
                else:
                    value = 0.0
                feature_vector.append(value)

        feat = torch.tensor(feature_vector, dtype=torch.float32)
        
        # Get target (label)
        target_name = self.target_feature
        if target_name in row.dtype.names:
            label = torch.tensor(float(row[target_name]), dtype=torch.float32)
        else:
            raise ValueError(f"Target {target_name} not found in data")
        
        # Guard NaNs
        if torch.isnan(feat).any() or torch.isnan(label):
            raise ValueError(f"NaN in PyTablesDatasetSplit at idx {idx}")
        
        # Return metadata if requested (for dSTEC evaluation)
        if self.return_metadata:
            metadata = {}
            for field in self.metadata_fields:
                if field in row.dtype.names:
                    value = row[field]
                    # Decode bytes to string for text fields
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    metadata[field] = value
            return feat, label, metadata
        
        return feat, label

    def __del__(self):
        if self.file is not None:
            self.file.close()
        if hasattr(self, "swi_file") and self.swi_file is not None:
            self.swi_file.close()
