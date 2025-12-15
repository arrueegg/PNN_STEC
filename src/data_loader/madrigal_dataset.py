"""
Dataset loader for Madrigal STEC data to enable direct model inference.

This allows using independent Madrigal observations as test data, providing
unbiased ground truth validation.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging
import warnings
import pandas as pd

# SpacePy for coordinate transformations
from spacepy.coordinates import Coords
from spacepy.time import Ticktock

# Suppress DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class MadrigalSTECDataset(Dataset):
    """
    Dataset for Madrigal STEC observations.
    
    Loads Madrigal data and formats it for model inference, matching the
    expected input structure of the STEC model.
    """
    
    def __init__(self, madrigal_path: str, year: int, doy: int, config: dict, 
                 elevation_threshold: float = 5.0, max_samples: Optional[int] = None,
                 station_list: Optional[list] = None):
        """
        Initialize Madrigal dataset.
        
        Args:
            madrigal_path: Path to Madrigal data directory
            year: Year
            doy: Day of year
            config: Model configuration dict (for feature settings)
            elevation_threshold: Minimum elevation angle (default: 5.0°)
            max_samples: Maximum number of samples to load (for testing)
            station_list: Optional list of station codes to filter by (e.g., test stations)
        """
        self.madrigal_path = Path(madrigal_path)
        self.year = year
        self.doy = doy
        self.config = config
        self.elevation_threshold = elevation_threshold
        self.station_list = [s.upper().strip() for s in station_list] if station_list else None
        
        # Get feature registry
        self.feature_registry = config.get("feature_registry")
        if not self.feature_registry:
            raise ValueError("Feature registry not found in config")
        
        # Setup SWI if needed
        self.use_SWI = config["data"].get("use_SWI", False)
        self.swi_file = None
        if self.use_SWI:
            import os
            swi_path = os.path.join(
                config["data"]["SWI_data_path"], "omni_hourly_2010-2025.h5"
            )
            self.swi_file = h5py.File(swi_path, "r")
            # Build mask of SWI columns (drop YEAR, DOY, HR)
            yrs = list(self.swi_file.keys())
            days = list(self.swi_file[yrs[0]].keys())
            cols = [c.decode() for c in self.swi_file[yrs[0]][days[0]].attrs["columns"]]
            self.swi_col_names = cols
            self.swi_mask = [c not in ("YEAR", "DOY", "HR") for c in cols]
            
            # Get SWI feature names from registry and compute mapping
            from utils.feature_registry import FeatureType
            masked_names = [n for n, m in zip(cols, self.swi_mask) if m]
            self.swi_name_to_idx = {name: i for i, name in enumerate(masked_names)}
            self.swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI)
            self.swi_indices_in_file_order = [
                self.swi_name_to_idx.get(feature_name) for feature_name in self.swi_features
            ]
        
        # Load Madrigal data
        self._load_data(max_samples)
        
        # Check if we should return metadata
        self.return_metadata = config.get("return_metadata", False)
        self.metadata_fields = config.get("metadata_fields", ["station", "sat", "satele", "satazi"])
        
    def _load_data(self, max_samples: Optional[int] = None):
        """Load and preprocess Madrigal data."""
        # Convert DOY to date
        date = datetime(self.year, 1, 1) + timedelta(days=self.doy - 1)
        date_str = date.strftime("%Y%m%d")
        
        # Construct file path
        file_path = self.madrigal_path / str(self.year) / f"los_{date_str}_IGS.h5"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Madrigal file not found: {file_path}")
        
        # Load data
        with h5py.File(file_path, 'r') as f:
            table = f['Data']['Table Layout']
            
            # Apply elevation filter
            elevations = table['elm'][:]
            valid_mask = elevations >= self.elevation_threshold
            
            # Apply station filter if provided
            if self.station_list:
                stations = np.array([s.decode().upper().strip()[:4] if isinstance(s, bytes) 
                                   else str(s).upper().strip()[:4] 
                                   for s in table['gps_site'][:]])
                station_mask = np.isin(stations, self.station_list)
                valid_mask = valid_mask & station_mask
            
            if max_samples:
                # Get indices of valid samples
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > max_samples:
                    # Randomly sample
                    np.random.seed(42)
                    sampled_indices = np.random.choice(valid_indices, max_samples, replace=False)
                    valid_mask = np.zeros(len(elevations), dtype=bool)
                    valid_mask[sampled_indices] = True
            
            # Load all required fields
            self.data = {
                'station': table['gps_site'][valid_mask].astype(str),
                'lat_sta': table['gdlatr'][valid_mask],  # Receiver latitude
                'lon_sta': table['gdlonr'][valid_mask],  # Receiver longitude
                'lat_ipp': table['gdlat'][valid_mask],   # IPP latitude
                'lon_ipp': table['glon'][valid_mask],     # IPP longitude
                'satazi': table['azm'][valid_mask],       # Azimuth (degrees)
                'satele': table['elm'][valid_mask],       # Elevation (degrees)
                'sod': table['sod'][valid_mask],          # Second of day
                'los_tec': table['los_tec'][valid_mask],  # Line-of-sight TEC (ground truth)
                'dlos_tec': table['dlos_tec'][valid_mask], # STEC uncertainty
            }
            
            # Add year and doy
            self.data['year'] = np.full(len(self.data['lat_sta']), self.year, dtype=np.float32)
            self.data['doy'] = np.full(len(self.data['lat_sta']), self.doy, dtype=np.float32)
            
            # Compute solar-magnetic coordinates if needed
            if any('sm_' in f for f in self.feature_registry.get_all_enabled_features()):
                self._add_sm_coordinates()
            
            # Compute local time if needed
            if 'local_time_hours' in self.feature_registry.get_all_enabled_features():
                self._add_local_time()
        
        self.length = len(self.data['lat_sta'])
        
    def _add_sm_coordinates(self):
        """Add solar-magnetic coordinates using SpacePy coordinate transformation."""
        # Prepare station coordinates
        unique_sta_coords = np.column_stack([
            np.full(len(self.data['lat_sta']), 1 + self.data.get('alt_sta', np.zeros(len(self.data['lat_sta']))) / 6371),
            self.data['lat_sta'],
            self.data['lon_sta']
        ])
        
        # Prepare IPP coordinates (at ionospheric height ~450km)
        unique_ipp_coords = np.column_stack([
            np.full(len(self.data['lat_ipp']), 1 + 450 / 6371),
            self.data['lat_ipp'],
            self.data['lon_ipp']
        ])
        
        # Create epochs for coordinate transformation
        date = datetime(self.year, 1, 1) + timedelta(days=self.doy - 1)
        epochs = [date + timedelta(seconds=float(sod)) for sod in self.data['sod']]
        
        # Transform station coordinates
        sta_coords = Coords(unique_sta_coords, 'GEO', 'sph')
        sta_coords.ticks = Ticktock(epochs, 'UTC')
        sta_sm = sta_coords.convert('SM', 'sph')
        
        self.data['sm_lat_sta'] = np.clip(sta_sm.lati.astype(np.float32), -90, 90)
        self.data['sm_lon_sta'] = ((sta_sm.long.astype(np.float32) + 180) % 360) - 180
        
        # Transform IPP coordinates
        ipp_coords = Coords(unique_ipp_coords, 'GEO', 'sph')
        ipp_coords.ticks = Ticktock(epochs, 'UTC')
        ipp_sm = ipp_coords.convert('SM', 'sph')
        
        self.data['sm_lat_ipp'] = np.clip(ipp_sm.lati.astype(np.float32), -90, 90)
        self.data['sm_lon_ipp'] = ((ipp_sm.long.astype(np.float32) + 180) % 360) - 180
        
    def _add_local_time(self):
        """Compute local time hours from longitude and SOD."""
        # Local time = UTC time + longitude offset
        utc_hours = self.data['sod'] / 3600.0
        lon_offset = self.data['lon_sta'] / 15.0  # 15 degrees per hour
        local_time = (utc_hours + lon_offset) % 24
        self.data['local_time_hours'] = local_time
    
    def __len__(self):
        return self.length
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'swi_file') and self.swi_file is not None:
            self.swi_file.close()
    
    def __getitem__(self, idx):
        """
        Get a single observation.
        
        Returns:
            Tuple of (features, target) where features match model input format
        """
        # Get all enabled features (excluding target)
        all_features = self.feature_registry.get_all_enabled_features()
        from utils.feature_registry import FeatureType
        target_features = self.feature_registry.get_features_by_type(FeatureType.TARGET)
        input_features = [f for f in all_features if f not in target_features]
        
        # Build feature vector (excluding SWI for now)
        feature_values = []
        for feature_name in input_features:
            # Skip SWI features - they'll be appended separately
            if feature_name in self.feature_registry.get_features_by_type(FeatureType.SWI):
                continue
            
            if feature_name in self.data:
                feature_values.append(self.data[feature_name][idx])
            else:
                # Feature not available in Madrigal data - use default/zero
                feature_values.append(0.0)
        
        # Add SWI features if enabled
        if self.config["data"].get("use_SWI", False):
            # Load SWI data for this time
            swi_values = self._get_swi_features(idx)
            feature_values.extend(swi_values)
        
        features = torch.tensor(feature_values, dtype=torch.float32)
        target = torch.tensor(self.data['los_tec'][idx], dtype=torch.float32)
        
        # Return metadata if requested
        if self.return_metadata:
            metadata = {}
            for field in self.metadata_fields:
                # Map metadata field names to data keys
                if field == 'station':
                    value = self.data['station'][idx] if 'station' in self.data else 'UNKNOWN'
                elif field == 'sat':
                    value = self.data['sat'][idx] if 'sat' in self.data else 'G00'
                elif field == 'satele':
                    value = self.data['satele'][idx] if 'satele' in self.data else self.data.get('elm', [0])[idx]
                elif field == 'satazi':
                    value = self.data['satazi'][idx] if 'satazi' in self.data else self.data.get('azm', [0])[idx]
                elif field in self.data:
                    value = self.data[field][idx]
                else:
                    value = 0.0
                
                # Convert to native Python types for consistency
                if isinstance(value, (np.ndarray, np.generic)):
                    value = value.item()
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                    
                metadata[field] = value
            return features, target, metadata
        
        return features, target
    
    def _get_swi_features(self, idx):
        """Get space weather indices for this observation."""
        if not self.use_SWI or not self.swi_file:
            return []
        
        year = str(int(self.data['year'][idx]))
        doy3 = f"{int(self.data['doy'][idx]):03d}"
        hour = int(self.data['sod'][idx] // 3600)
        
        try:
            swi_row = self.swi_file[year][doy3][hour]
            swi_values_masked = swi_row[self.swi_mask]
            
            # Extract values for each SWI feature
            swi_values = []
            for in_idx in self.swi_indices_in_file_order:
                if in_idx is None:
                    swi_values.append(0.0)
                else:
                    swi_values.append(float(swi_values_masked[in_idx]))
            return swi_values
        except (KeyError, IndexError):
            # If data not available, return zeros
            return [0.0] * len(self.swi_features)
    
    def get_metadata(self, idx):
        """Get metadata for an observation (for analysis)."""
        return {
            'station': self.data['station'][idx],
            'lat_sta': self.data['lat_sta'][idx],
            'lon_sta': self.data['lon_sta'][idx],
            'lat_ipp': self.data['lat_ipp'][idx],
            'lon_ipp': self.data['lon_ipp'][idx],
            'satazi': self.data['satazi'][idx],
            'satele': self.data['satele'][idx],
            'sod': self.data['sod'][idx],
            'year': self.data['year'][idx],
            'doy': self.data['doy'][idx],
        }


def get_madrigal_data_loader(
    madrigal_path: str,
    year: int,
    doy: int,
    config: dict,
    batch_size: int = 8192,
    num_workers: int = 4,
    elevation_threshold: float = 5.0,
    max_samples: Optional[int] = None,
    station_list: Optional[list] = None,
    logger: Optional[logging.Logger] = None
) -> DataLoader:
    """
    Create a DataLoader for Madrigal STEC data.
    
    Args:
        madrigal_path: Path to Madrigal data directory
        year: Year
        doy: Day of year
        config: Model configuration dict
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        elevation_threshold: Minimum elevation angle
        max_samples: Maximum number of samples (for testing)
        station_list: Optional list of station codes to filter by
        logger: Optional logger
        
    Returns:
        DataLoader for Madrigal data
    """
    # Import collation function
    from data_loader.collation import CollateWithSH
    collate_fn = CollateWithSH(config)
    
    # Create dataset
    dataset = MadrigalSTECDataset(
        madrigal_path=madrigal_path,
        year=year,
        doy=doy,
        config=config,
        elevation_threshold=elevation_threshold,
        max_samples=max_samples,
        station_list=station_list
    )
    
    if logger:
        logger.info(f"✅ Madrigal dataset created: {len(dataset):,} observations")
        logger.info(f"   Elevation threshold: {elevation_threshold}°")
        if station_list:
            logger.info(f"   Filtered to {len(station_list)} test stations")
        logger.info(f"   Date: {year} DOY {doy}")
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return loader, dataset
