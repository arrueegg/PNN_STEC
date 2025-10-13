"""
Multi-temporal inference dataset that reuses computations across timestamps.

This module provides a faster version of the grid inference dataset that:
1. Pre-loads all SWI data for the date range
2. Pre-computes coordinate transformations 
3. Reuses grid structures across timestamps
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import h5py

from utils.feature_registry import FeatureType
from utils.coordinate_transforms import calculate_ipp_coordinates, geographic_to_solar_magnetic


class MultiTemporalInferenceDataset(Dataset):
    """
    Efficient dataset for multi-temporal inference with caching optimizations.
    
    Pre-computes expensive operations and caches results for fast timestamp switching.
    """
    
    def __init__(self, config, lat_grid, lon_grid, elevation, azimuth, date_obj):
        """
        Initialize optimized dataset for the entire day.
        
        Args:
            config: Configuration dictionary
            lat_grid: 2D latitude grid (fixed for all timestamps)
            lon_grid: 2D longitude grid (fixed for all timestamps) 
            elevation: Fixed elevation angle
            azimuth: Fixed azimuth angle
            date_obj: Date object (for SWI data loading)
        """
        self.config = config
        self.elevation = elevation
        self.azimuth = azimuth
        self.date_obj = date_obj
        
        # Get feature registry
        self.feature_registry = config.get("feature_registry")
        if not self.feature_registry:
            raise ValueError("Feature registry is required but not found in config")
            
        # Get input features (excluding target and SWI - SWI handled separately)
        all_features = self.feature_registry.get_all_enabled_features()
        target_features = self.feature_registry.get_features_by_type(FeatureType.TARGET)
        swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI)
        self.input_features = [f for f in all_features if f not in target_features and f not in swi_features]
        
        # Flatten grid for processing
        self.lat_flat = lat_grid.flatten()
        self.lon_flat = lon_grid.flatten()
        self.n_points = len(self.lat_flat)
        self.grid_shape = lat_grid.shape
        
        # Pre-compute all coordinate transformations (expensive part!)
        print("Pre-computing coordinate transformations...")
        self._precompute_coordinates()
        
        # Pre-load SWI data for the entire day (if enabled)
        self.use_SWI = config["data"].get("use_SWI", False)
        if self.use_SWI:
            print("Pre-loading SWI data for the day...")
            self._preload_swi_data()
        
        print(f"Optimized dataset ready: {self.n_points} points")
    
    def _precompute_coordinates(self):
        """Pre-compute all coordinate transformations for the grid."""
        # Pre-compute IPP coordinates for all grid points  
        self.lat_ipp = np.zeros(self.n_points)
        self.lon_ipp = np.zeros(self.n_points)
        
        # Process in chunks to show progress for large grids
        chunk_size = 5000
        for start_idx in range(0, self.n_points, chunk_size):
            end_idx = min(start_idx + chunk_size, self.n_points)
            if self.n_points > 5000:  # Only show progress for large grids
                print(f"  IPP coordinates: {end_idx}/{self.n_points}")
                
            for i in range(start_idx, end_idx):
                lat_ipp, lon_ipp = calculate_ipp_coordinates(
                    self.lat_flat[i], self.lon_flat[i], self.azimuth, self.elevation
                )
                self.lat_ipp[i] = lat_ipp
                self.lon_ipp[i] = lon_ipp
        
        # Cache for solar magnetic coordinates (computed on demand)
        self.sm_cache = {}  # timestamp -> (sm_lat_sta, sm_lon_sta, sm_lat_ipp, sm_lon_ipp)
        
    def _preload_swi_data(self):
        """Pre-load all SWI data for the day to avoid repeated file I/O."""
        swi_file_path = self.config["data"].get("scratch_dir", "data/") + "omni_hourly_2010-2025.h5"
        
        self.swi_hourly_data = {}
        
        try:
            with h5py.File(swi_file_path, "r") as swi_file:
                year = str(self.date_obj.year)
                doy3 = f"{self.date_obj.timetuple().tm_yday:03d}"
                
                if year in swi_file and doy3 in swi_file[year]:
                    # Load entire day at once (24 hours)
                    daily_data = swi_file[year][doy3][:]
                    
                    # Build mask (same as H5Dataset approach)
                    cols = [c.decode() for c in swi_file[year][doy3].attrs["columns"]]
                    swi_mask = [c not in ("YEAR", "DOY", "HR") for c in cols]
                    
                    # Store all hourly data
                    for hour in range(24):
                        if hour < len(daily_data):
                            hourly_swi = daily_data[hour][swi_mask]
                            self.swi_hourly_data[hour] = hourly_swi
                        else:
                            # Fill missing hours with zeros
                            self.swi_hourly_data[hour] = np.zeros(sum(swi_mask))
                else:
                    # No data available - fill with zeros
                    print(f"Warning: No SWI data for {year}/{doy3}")
                    for hour in range(24):
                        self.swi_hourly_data[hour] = np.zeros(22)  # Default SWI size
                        
        except Exception as e:
            print(f"Error pre-loading SWI data: {e}")
            # Fallback - fill with zeros
            for hour in range(24):
                self.swi_hourly_data[hour] = np.zeros(22)
    
    def update_timestamp(self, timestamp: datetime):
        """
        Update the dataset for a new timestamp (fast operation).
        
        Args:
            timestamp: New timestamp to use
        """
        self.current_timestamp = timestamp
        
        # Update temporal features
        self.year = float(timestamp.year)
        self.doy = float(timestamp.timetuple().tm_yday)
        self.sod = float(timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second)
        
        # Update solar magnetic coordinates (with caching for speed)
        cache_key = timestamp.replace(minute=0, second=0)  # Round to nearest hour for caching
        
        if cache_key in self.sm_cache:
            # Use cached coordinates
            self.sm_lat_sta, self.sm_lon_sta, self.sm_lat_ipp, self.sm_lon_ipp = self.sm_cache[cache_key]
        else:
            # Compute coordinates and cache them
            self.sm_lat_sta = np.zeros(self.n_points)
            self.sm_lon_sta = np.zeros(self.n_points)
            self.sm_lat_ipp = np.zeros(self.n_points) 
            self.sm_lon_ipp = np.zeros(self.n_points)
            
            # Process in chunks for large grids
            chunk_size = 5000
            for start_idx in range(0, self.n_points, chunk_size):
                end_idx = min(start_idx + chunk_size, self.n_points)
                
                for i in range(start_idx, end_idx):
                    # Station SM coordinates
                    sm_lat_sta, sm_lon_sta = geographic_to_solar_magnetic(
                        self.lat_flat[i], self.lon_flat[i], cache_key
                    )
                    self.sm_lat_sta[i] = sm_lat_sta
                    self.sm_lon_sta[i] = sm_lon_sta
                    
                    # IPP SM coordinates  
                    sm_lat_ipp, sm_lon_ipp = geographic_to_solar_magnetic(
                        self.lat_ipp[i], self.lon_ipp[i], cache_key
                    )
                    self.sm_lat_ipp[i] = sm_lat_ipp
                    self.sm_lon_ipp[i] = sm_lon_ipp
            
            # Cache for reuse
            self.sm_cache[cache_key] = (
                self.sm_lat_sta.copy(), 
                self.sm_lon_sta.copy(), 
                self.sm_lat_ipp.copy(), 
                self.sm_lon_ipp.copy()
            )
        
        # Update SWI data for this hour
        if self.use_SWI:
            hour = timestamp.hour
            self.current_swi_values = self.swi_hourly_data.get(hour, np.zeros(22))
    
    def __len__(self):
        return self.n_points
    
    def __getitem__(self, idx):
        """Fast feature vector construction using pre-computed data."""
        # Build feature vector using pre-computed values
        feature_vector = []
        
        for feature_name in self.input_features:
            if feature_name == "year":
                value = self.year
            elif feature_name == "doy":
                value = self.doy
            elif feature_name == "sod":
                value = self.sod
            elif feature_name == "sm_lat_sta":
                value = float(self.sm_lat_sta[idx])
            elif feature_name == "sm_lon_sta":
                value = float(self.sm_lon_sta[idx])
            elif feature_name == "lat_sta":
                value = float(self.lat_flat[idx])
            elif feature_name == "lon_sta":
                value = float(self.lon_flat[idx])
            elif feature_name == "lat_ipp":
                value = float(self.lat_ipp[idx])
            elif feature_name == "lon_ipp":
                value = float(self.lon_ipp[idx])
            elif feature_name == "sm_lat_ipp":
                value = float(self.sm_lat_ipp[idx])
            elif feature_name == "sm_lon_ipp":
                value = float(self.sm_lon_ipp[idx])
            elif feature_name == "satazi":
                value = float(self.azimuth)
            elif feature_name == "satele":
                value = float(self.elevation)
            else:
                raise ValueError(f"Feature {feature_name} not supported in optimized dataset")
            
            feature_vector.append(value)
        
        feat = torch.tensor(feature_vector, dtype=torch.float32)
        
        # Add SWI features if enabled
        if self.use_SWI:
            swi_feat = torch.tensor(self.current_swi_values, dtype=torch.float32)
            feat = torch.cat((feat, swi_feat), dim=0)
        
        # Placeholder label
        label = torch.tensor(0.0, dtype=torch.float32)
        
        return feat, label
    
    def get_grid_shape(self):
        """Get the shape of the original grid for reshaping results."""
        return self.grid_shape


def create_multitemporal_inference_dataloader(
    config, lat_grid, lon_grid, elevation, azimuth, date_obj, batch_size
):
    """
    Create multi-temporal inference dataset and dataloader.
    
    This creates a multi-temporal dataset that pre-computes coordinates,
    caches SWI data, and reuses computations across timestamps.
    """
    from torch.utils.data import DataLoader
    from data_loader.collation import CollateWithSH
    
    # Create multi-temporal dataset
    dataset = MultiTemporalInferenceDataset(
        config, lat_grid, lon_grid, elevation, azimuth, date_obj
    )
    
    # Create collation function
    collate_fn = CollateWithSH(config)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Important: preserve order for grid reshaping
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing overhead for inference
        pin_memory=False
    )
    
    return dataset, dataloader