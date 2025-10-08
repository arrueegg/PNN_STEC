#!/usr/bin/env python3
"""
Global Map Inference Script for PNN_STEC Project

This script generates global STEC maps every hour for a specific date at fixed 
(but modifiable) elevation and azimuth angles. It reuses existing components 
from the codebase for consistent model loading and inference.

Key Features:
- Generates global grid of lat/lon points for inference
- Fixed elevation and azimuth angles (configurable)
- Hourly maps for a complete day (24 maps)
- Reuses BaseTrainer for consistent inference
- Saves results in the experiment folder of the model specified in config.yaml
- Supports both Bayesian (BNN) and standard (MLP) neural network models

Usage:
    python src/inference_map.py --date 2024-05-15 --elevation 30.0 --azimuth 180.0

Parameters:
    --date: Date for map generation (YYYY-MM-DD format)
    --elevation: Fixed elevation angle in degrees (default: 30.0)
    --azimuth: Fixed azimuth angle in degrees (default: 180.0)
    --lat_res: Latitude resolution in degrees (default: 2.5)
    --lon_res: Longitude resolution in degrees (default: 5.0)

Output:
- Hourly numpy files (.npz) with global STEC predictions and metadata
- Summary plots and statistics  
- All saved to: experiments/<experiment_name>/global_maps/

Note: This script uses numpy compressed files (.npz) for output. 
To use NetCDF format, install netCDF4: pip install netCDF4
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import h5py
from pathlib import Path
# Add the parent directory to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_parser import load_config, compute_exp_name
from utils.feature_registry import initialize_feature_registry, FeatureType
from training import BaseTrainer
from data_loader import CollateWithSH
from model.model import get_model
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from viz.base import FIGSIZE_WIDE
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

try:
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock
    SPACEPY_AVAILABLE = True
except ImportError:
    SPACEPY_AVAILABLE = False
    logger.warning("spacepy not available, will use geographic coordinates as placeholder")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate global STEC maps")
    parser.add_argument('--date', type=str, default='2024-07-01',
                       help='Date for map generation (YYYY-MM-DD format)')
    parser.add_argument('--elevation', type=float, default=90.0,
                       help='Fixed elevation angle in degrees (default: 90.0)')
    parser.add_argument('--azimuth', type=float, default=180.0,
                       help='Fixed azimuth angle in degrees (default: 180.0)')
    parser.add_argument('--lat_res', type=float, default=1.0,
                       help='Latitude resolution in degrees (default: 1.0)')
    parser.add_argument('--lon_res', type=float, default=1.0,
                       help='Longitude resolution in degrees (default: 1.0)')
    parser.add_argument('--config_path', type=str, default='config/config.yaml',
                       help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--create_gif', action='store_true', default=True,
                       help='Create a GIF animation from all hourly maps')
    parser.add_argument('--vmin', type=float, default=0,
                       help='Minimum value for STEC colorscale (default: 0)')
    parser.add_argument('--vmax', type=float, default=80,
                       help='Maximum value for STEC colorscale (default: 80)')
    return parser.parse_args()

def create_global_grid(lat_res=1.0, lon_res=1.0):
    """
    Create a global grid of latitude and longitude points.
    
    Args:
        lat_res: Latitude resolution in degrees
        lon_res: Longitude resolution in degrees
    
    Returns:
        tuple: (lat_grid, lon_grid) as 2D arrays
    """
    # Create 1D arrays
    lats = np.arange(-90, 90 + lat_res, lat_res)
    lons = np.arange(-180, 180 + lon_res, lon_res)
    
    # Create 2D meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    return lat_grid, lon_grid

def load_swi_data(timestamp):
    """
    Load Space Weather Index (SWI) data for a given timestamp.
    
    Args:
        timestamp: datetime object
        
    Returns:
        dict: Dictionary with SWI feature values, or None if data not available
    """
    swi_file_path = "/home/space/data/IONO/SWI/omni_hourly_2010-2025.h5"
    
    if not Path(swi_file_path).exists():
        logger.warning(f"SWI file not found at {swi_file_path}, using default values")
        return None
    
    try:
        with h5py.File(swi_file_path, 'r') as swi_file:
            year = timestamp.year
            doy = timestamp.timetuple().tm_yday
            hour = timestamp.hour
            
            # Format DOY with 3 digits as in the data structure
            doy3 = f"{doy:03d}"
            
            if str(year) not in swi_file:
                logger.warning(f"Year {year} not found in SWI file")
                return None
                
            if doy3 not in swi_file[str(year)]:
                logger.warning(f"DOY {doy3} not found for year {year} in SWI file")
                return None
            
            # Load the daily data (24 hours)
            daily_data = swi_file[str(year)][doy3][:]
            
            if hour >= len(daily_data):
                logger.warning(f"Hour {hour} not available for {year}-{doy3}")
                return None
            
            # Get data for the specific hour
            hourly_data = daily_data[hour]
            
            # Skip YEAR, DOY, HR columns (first 3) and map correctly to feature names
            # Based on the actual column structure in the HDF5 file
            swi_feature_mapping = [
                ('Bartels_rotation_number', 3),
                ('Scalar_B,_nT', 4),
                ('Vector_B_Magnitude,nT', 5),
                ('Lat_Angle_of_B_GSE', 6),
                ('Long_Angle_of_B_GSE', 7),
                ('BZ,_nT_GSE', 8),
                ('BZ,_nT_GSM', 9),
                ('SW_Plasma_Speed,_km/s', 10),
                ('Flow_pressure', 11),
                ('E_electric_field', 12),  # Note: file has typo 'E_elecrtric_field'
                ('Alfen_mach_number', 13),
                ('Kp_index', 14),
                ('R_Sunspot_No', 15),
                ('Dst-index,_nT', 16),
                ('AE-index,_nT', 17),
                ('ap_index,_nT', 18),
                ('f107_index', 19),
                ('pc-index', 20),
                ('AL-index,_nT', 21),
                ('AU-index,_nT', 22),
                ('Magnetosonic_Much_num', 23),
                ('Lyman_alpha', 24),
            ]
            
            swi_features = {
                name: hourly_data[idx] if len(hourly_data) > idx else 0.0
                for name, idx in swi_feature_mapping
            }
            
            return swi_features
            
    except Exception as e:
        logger.error(f"Error loading SWI data: {e}")
        return None

def calculate_ipp_coordinates(station_lat, station_lon, azimuth, elevation, ipp_height=450.0):
    """
    Calculate Ionospheric Pierce Point (IPP) coordinates.
    
    Args:
        station_lat: Station latitude in degrees
        station_lon: Station longitude in degrees  
        azimuth: Satellite azimuth angle in degrees (0=North, 90=East)
        elevation: Satellite elevation angle in degrees
        ipp_height: IPP height in km (default: 450 km)
        
    Returns:
        tuple: (ipp_lat, ipp_lon) in degrees
    """
    # Earth radius in km
    RE = 6371.0
    
    # Convert to radians
    lat_rad = np.deg2rad(station_lat)
    lon_rad = np.deg2rad(station_lon)
    az_rad = np.deg2rad(azimuth)
    el_rad = np.deg2rad(elevation)
    
    # Calculate the central angle (psi) to the IPP
    # Using thin shell approximation for ionosphere
    sin_psi = (RE / (RE + ipp_height)) * np.cos(el_rad)
    psi = np.arcsin(sin_psi)
    
    # Calculate IPP latitude
    ipp_lat_rad = np.arcsin(np.sin(lat_rad) * np.cos(psi) + 
                           np.cos(lat_rad) * np.sin(psi) * np.cos(az_rad))
    
    # Calculate IPP longitude
    delta_lon = np.arcsin(np.sin(psi) * np.sin(az_rad) / np.cos(ipp_lat_rad))
    ipp_lon_rad = lon_rad + delta_lon
    
    # Convert back to degrees
    ipp_lat = np.rad2deg(ipp_lat_rad)
    ipp_lon = np.rad2deg(ipp_lon_rad)
    
    # Normalize longitude to [-180, 180]
    ipp_lon = ((ipp_lon + 180) % 360) - 180
    
    return ipp_lat, ipp_lon

def coord_transform(input_type, output_type, lats, lons, epochs):
    """
    Transform coordinates using spacepy.
    
    Args:
        input_type: Input coordinate system (e.g., 'GEO')
        output_type: Output coordinate system (e.g., 'SM')
        lats: Array of latitudes in degrees
        lons: Array of longitudes in degrees
        epochs: Array of datetime objects
        
    Returns:
        Transformed coordinates object with .data attribute containing [[alt, lat, lon], ...]
    """
    if not SPACEPY_AVAILABLE:
        return None
        
    try:
        import numpy as np
        coords = np.array([[1 + 450 / 6371, lat, lon] for lat, lon in zip(lats, lons)], dtype=np.float64)
        geo_coords = coord.Coords(coords, input_type, 'sph')
        geo_coords.ticks = Ticktock(epochs, 'UTC')
        return geo_coords.convert(output_type, 'sph')
    except Exception as e:
        logger.warning(f"spacepy coordinate transformation failed: {e}")
        return None

def geographic_to_solar_magnetic(geo_lat, geo_lon, timestamp):
    """
    Convert geographic coordinates to solar magnetic coordinates using spacepy.
    
    Args:
        geo_lat: Geographic latitude in degrees (scalar or array)
        geo_lon: Geographic longitude in degrees (scalar or array)
        timestamp: datetime object
        
    Returns:
        tuple: (sm_lat, sm_lon) in degrees
    """
    if not SPACEPY_AVAILABLE:
        logger.warning("spacepy not available, using geographic coordinates as solar magnetic placeholder")
        if not hasattr(geo_lat, '__len__'):
            return float(geo_lat), float(geo_lon)
        else:
            return geo_lat, geo_lon
    
    try:
        # Handle scalar inputs
        if not hasattr(geo_lat, '__len__'):
            geo_lat = [geo_lat]
            geo_lon = [geo_lon]
            is_scalar = True
        else:
            is_scalar = False
            
        epochs = [timestamp] * len(geo_lat)
        
        # Transform coordinates
        sm_coords = coord_transform('GEO', 'SM', geo_lat, geo_lon, epochs)
        
        if sm_coords is not None:
            # Extract lat/lon from spacepy coords (format: [alt, lat, lon])
            sm_lat = sm_coords.data[:, 1]  # latitude is second column
            sm_lon = sm_coords.data[:, 2]  # longitude is third column
            
            if is_scalar:
                return float(sm_lat[0]), float(sm_lon[0])
            else:
                return sm_lat, sm_lon
        else:
            # Fallback to geographic coordinates
            logger.warning("Using geographic coordinates as solar magnetic placeholder")
            if is_scalar:
                return float(geo_lat[0]), float(geo_lon[0])
            else:
                return geo_lat, geo_lon
                
    except Exception as e:
        logger.warning(f"Coordinate transformation failed: {e}, using geographic coordinates")
        if not hasattr(geo_lat, '__len__'):
            return float(geo_lat), float(geo_lon)
        else:
            return geo_lat, geo_lon

def create_inference_data(date_str, hour, lat_grid, lon_grid, elevation, azimuth, config):
    """
    Create inference dataset for a specific hour and global grid using CollateWithSH directly.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        hour: Hour of day (0-23)
        lat_grid: 2D array of latitudes
        lon_grid: 2D array of longitudes  
        elevation: Fixed elevation angle in degrees
        azimuth: Fixed azimuth angle in degrees
        config: Configuration dictionary
    
    Returns:
        tuple: (torch.Tensor, grid_shape) - Feature tensor for inference and original grid shape
    """
    # Parse date
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    timestamp = date_obj.replace(hour=hour)
    year = date_obj.year
    doy = date_obj.timetuple().tm_yday
    sod = hour * 3600  # Convert hour to seconds of day
    
    # Flatten grids for vectorized operations
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    n_points = len(lat_flat)
    
    # Get feature registry
    feature_registry = config.get('feature_registry')
    if not feature_registry:
        raise ValueError("Feature registry is required but not found in config")
    
    # Get enabled input features in the EXACT same order as training data
    all_features = feature_registry.get_all_enabled_features()
    target_features = feature_registry.get_features_by_type(FeatureType.TARGET)
    target_feature = target_features[0] if target_features else 'stec'
    
    # Remove target from input features
    input_features = [f for f in all_features if f != target_feature]
    
    # Load SWI data for this timestamp
    swi_data_loaded = load_swi_data(timestamp)
    
    # Calculate all coordinate transformations once for efficiency
    logger.info("Processing global grid...")
    
    # Calculate IPP coordinates for all points
    ipp_lats = []
    ipp_lons = []
    for i in range(n_points):
        ipp_lat, ipp_lon = calculate_ipp_coordinates(lat_flat[i], lon_flat[i], azimuth, elevation)
        ipp_lats.append(ipp_lat)
        ipp_lons.append(ipp_lon)
    
    ipp_lats = np.array(ipp_lats)
    ipp_lons = np.array(ipp_lons)
    
    # Transform station coordinates to solar magnetic (vectorized)
    sm_lat_sta, sm_lon_sta = geographic_to_solar_magnetic(lat_flat, lon_flat, timestamp)
    
    # Transform IPP coordinates to solar magnetic (vectorized) 
    sm_lat_ipp, sm_lon_ipp = geographic_to_solar_magnetic(ipp_lats, ipp_lons, timestamp)
    
    # Ensure we have arrays, not scalars
    if not hasattr(sm_lat_sta, '__len__'):
        sm_lat_sta = np.full(n_points, sm_lat_sta)
        sm_lon_sta = np.full(n_points, sm_lon_sta)
    if not hasattr(sm_lat_ipp, '__len__'):
        sm_lat_ipp = np.full(n_points, sm_lat_ipp)
        sm_lon_ipp = np.full(n_points, sm_lon_ipp)
    
    # Create raw feature vectors in the EXACT order used during training
    feature_vectors = []
    
    for i in range(n_points):
        feature_vector = []
        
        # Build feature vector in the order expected by the model
        # This order MUST match the training data order exactly
        for feature_name in input_features:
            # Skip SWI features for now, they'll be added at the end
            if feature_name in feature_registry.get_features_by_type(FeatureType.SWI):
                continue
                
            if feature_name == 'year':
                value = float(year)
            elif feature_name == 'doy':
                value = float(doy)
            elif feature_name == 'sod':
                value = float(sod)
            elif feature_name == 'lat_ipp':
                value = float(ipp_lats[i])
            elif feature_name == 'lon_ipp':
                value = float(ipp_lons[i])
            elif feature_name == 'sm_lat_ipp':
                value = float(sm_lat_ipp[i])
            elif feature_name == 'sm_lon_ipp':
                value = float(sm_lon_ipp[i])
            elif feature_name == 'satazi':
                value = float(azimuth)
            elif feature_name == 'satele':
                value = float(elevation)
            elif feature_name == 'lat_sta':
                value = float(lat_flat[i])
            elif feature_name == 'lon_sta':
                value = float(lon_flat[i])
            elif feature_name == 'sm_lat_sta':
                value = float(sm_lat_sta[i])
            elif feature_name == 'sm_lon_sta':
                value = float(sm_lon_sta[i])
            else:
                logger.warning(f"Unknown non-SWI feature {feature_name}, using default value 0.0")
                value = 0.0
            
            feature_vector.append(value)
        
        # Add SWI features at the end (they're concatenated after main features in training)
        swi_features = feature_registry.get_features_by_type(FeatureType.SWI)
        for feature_name in swi_features:
            if swi_data_loaded is not None and feature_name in swi_data_loaded:
                value = float(swi_data_loaded[feature_name])
            else:
                value = 0.0  # Default value for SWI features
                if swi_data_loaded is None and i == 0:  # Only log once
                    logger.warning(f"Using default value for SWI feature {feature_name} (no SWI data loaded)")
            feature_vector.append(value)
        
        feature_vectors.append(feature_vector)
    
    # Convert to tensor (raw features)
    raw_features = torch.tensor(feature_vectors, dtype=torch.float32)
    
    # Create CollateWithSH instance to apply transformations exactly like in training
    collate_fn = CollateWithSH(config)
    
    # Apply transformations by creating a dummy batch
    # CollateWithSH expects a list of (features, labels) tuples
    dummy_labels = torch.zeros(n_points, 1)  # Dummy labels
    batch_data = [(raw_features[i], dummy_labels[i]) for i in range(n_points)]
    
    # Apply transformations
    transformed_features, _ = collate_fn(batch_data)
    
    return transformed_features, lat_grid.shape

def run_model_inference(model, features, config, batch_size=10000):
    """
    Run model inference on features in batches with proper target denormalization.
    
    Args:
        model: Trained model
        features: Input features tensor
        config: Configuration dictionary
        batch_size: Batch size for inference
    
    Returns:
        tuple: (predictions, uncertainties) if Bayesian, else (predictions, None)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get training configuration
    use_log_target = config['training'].get('log_target', False)
    use_target_standardization = config['training'].get('standardize_targets', False)
    feature_registry = config.get('feature_registry')
    target_name = config.get('target', 'stec')
    eps = 1e-6
    
    # Determine if model is Bayesian
    model_type = config['model']['model_type']
    is_bayesian = 'BNN' in model_type
    
    all_predictions = []
    all_uncertainties = [] if is_bayesian else None
    
    n_samples = features.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Inference"):
            batch_features = features[i:i+batch_size].to(device)
            
            if is_bayesian:
                # For Bayesian models, run multiple forward passes
                num_samples = 100
                batch_preds = []
                
                for _ in range(num_samples):
                    output = model(batch_features)
                    if isinstance(output, tuple):
                        pred_mean_raw, pred_var_raw = output[0], output[1]
                    else:
                        pred_mean_raw = output
                        pred_var_raw = torch.zeros_like(pred_mean_raw)
                    
                    # Apply transformations back to original space
                    pred_linear = apply_model_output_transforms(
                        pred_mean_raw.flatten(), pred_var_raw.flatten(), 
                        use_log_target, use_target_standardization, 
                        feature_registry, target_name, eps
                    )
                    batch_preds.append(pred_linear.cpu())
                
                # Calculate mean and uncertainty
                batch_preds = torch.stack(batch_preds)
                mean_pred = torch.mean(batch_preds, dim=0)
                std_pred = torch.std(batch_preds, dim=0)
                
                all_predictions.append(mean_pred)
                all_uncertainties.append(std_pred)
            else:
                # For standard models, single forward pass
                output = model(batch_features)
                if isinstance(output, tuple):
                    pred_mean_raw, pred_var_raw = output[0], output[1]
                else:
                    pred_mean_raw = output
                    pred_var_raw = torch.zeros_like(pred_mean_raw)
                
                # Apply transformations back to original space
                pred_linear = apply_model_output_transforms(
                    pred_mean_raw.flatten(), pred_var_raw.flatten(),
                    use_log_target, use_target_standardization,
                    feature_registry, target_name, eps
                )
                
                all_predictions.append(pred_linear.cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    uncertainties = torch.cat(all_uncertainties, dim=0) if is_bayesian else None
    
    return predictions, uncertainties

def apply_model_output_transforms(pred_mean_raw, pred_var_raw, use_log_target, 
                                use_target_standardization, feature_registry, 
                                target_name, eps):
    """
    Apply the inverse of training transformations to get predictions in original scale.
    
    Args:
        pred_mean_raw: Raw model output for mean
        pred_var_raw: Raw model output for variance
        use_log_target: Whether log-space training was used
        use_target_standardization: Whether target standardization was used
        feature_registry: Feature registry for denormalization
        target_name: Name of target variable
        eps: Small constant for numerical stability
        
    Returns:
        Predictions in original scale
    """
    if use_log_target:
        # Convert from log-space to linear space
        # Apply log-normal transformation
        point_standardized = torch.exp(pred_mean_raw + 0.5 * pred_var_raw) - eps
    else:
        # Already in linear space
        point_standardized = pred_mean_raw
    
    # Apply target denormalization if standardization was used during training
    if use_target_standardization and feature_registry:
        normalization_params = feature_registry.get_normalization_params(target_name)
        if normalization_params:
            min_val, max_val = normalization_params
            scale_factor = max_val - min_val
            point_original = point_standardized * scale_factor + min_val
            return point_original
    return point_standardized

def save_hourly_plot(lat_grid, lon_grid, stec_map, uncertainty_map,
                     output_path, date_str, hour, elevation, azimuth, vmin=0, vmax=80):
    """
    Save individual hourly STEC map as PNG file with GIM-style formatting.
    
    Args:
        lat_grid: 2D latitude grid
        lon_grid: 2D longitude grid
        stec_map: 2D STEC prediction array
        uncertainty_map: 2D uncertainty array (can be None)
        output_path: Base path to save files (without extension)
        date_str: Date string
        hour: Hour of day
        elevation: Elevation angle
        azimuth: Azimuth angle
        vmin: Minimum value for colorscale
        vmax: Maximum value for colorscale
    """
    # Create figure with cartographic projection matching GIM style
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), 
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Create the STEC map plot with fixed colorscale
    im = ax.pcolormesh(lon_grid, lat_grid, stec_map, 
                      cmap='gist_heat', shading='auto',
                      transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    
    # Add coastlines (white like in GIM plots)
    ax.coastlines(color='white')
    
    # Set title matching GIM style
    ax.set_title(f'PNN STEC for {date_str} {hour:02d}:00 UTC\nElevation: {elevation}°, Azimuth: {azimuth}°', 
                fontweight='bold', fontsize=16)
    
    # Set labels and formatting matching GIM style
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set global extent
    ax.set_global()
    
    # Add colorbar matching GIM style
    cbar = fig.colorbar(im, ax=ax, label='STEC (TECU)', shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('STEC (TECU)', fontsize=14)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_gif(image_paths, output_path, duration=500):
    """
    Create a GIF from a list of image paths.
    
    Parameters:
    image_paths (list): List of paths to PNG images
    output_path (str): Path for the output GIF file
    duration (int): Duration between frames in milliseconds
    """
    if not image_paths:
        logger.warning("No images found to create GIF")
        return
    
    # Load all images
    images = []
    for path in image_paths:
        if os.path.exists(path):
            img = Image.open(path)
            images.append(img)
    
    if images:
        # Save as GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        logger.info(f"GIF created: {output_path}")
    else:
        logger.warning("No valid images found to create GIF")

def find_experiment_directory(experiment_name, base_dir='experiments'):
    """Find the experiment directory that matches the given name."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory {base_dir} does not exist")
    
    # Look for exact match first
    exact_path = os.path.join(base_dir, experiment_name)
    if os.path.exists(exact_path):
        return exact_path
    
    # If no exact match, look for partial matches
    matches = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and experiment_name in item:
            matches.append(item_path)
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(f"Multiple experiment directories match '{experiment_name}': {matches}")
    else:
        raise FileNotFoundError(f"No experiment directory found matching '{experiment_name}'")

def find_model_checkpoint(experiment_dir):
    """Find the model checkpoint in the experiment directory."""
    model_dir = os.path.join(experiment_dir, 'model')
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Look for .pth files
    pth_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if len(pth_files) == 0:
        raise FileNotFoundError(f"No model checkpoint (.pth) files found in {model_dir}")
    elif len(pth_files) == 1:
        return os.path.join(model_dir, pth_files[0])
    else:
        # If multiple, prefer the one with 'pretrain' in the name
        pretrain_files = [f for f in pth_files if 'pretrain' in f.lower()]
        if pretrain_files:
            return os.path.join(model_dir, pretrain_files[0])
        else:
            return os.path.join(model_dir, pth_files[0])

def main():
    """Main function to generate global STEC maps."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config_path)
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize feature registry and add it to config
    feature_registry = initialize_feature_registry(config)
    config['feature_registry'] = feature_registry
    
    # Determine experiment name
    experiment_name = compute_exp_name(config)
    logger.info(f"Using experiment: {experiment_name}")
    
    # Find experiment directory
    experiment_dir = find_experiment_directory(experiment_name)
    
    # Find model checkpoint
    checkpoint_path = find_model_checkpoint(experiment_dir)
    
    # Create output directory with date subfolder
    output_dir = os.path.join(experiment_dir, 'global_maps', args.date)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create global grid
    logger.info(f"Creating global grid with resolution {args.lat_res}° x {args.lon_res}°")
    lat_grid, lon_grid = create_global_grid(args.lat_res, args.lon_res)
    logger.info(f"Grid shape: {lat_grid.shape}")
    
    # Load model
    logger.info("Loading model...")
    model = get_model(config).to(config['device'])
    checkpoint = torch.load(checkpoint_path, map_location=config['device'], weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate maps for each hour
    logger.info(f"Generating maps for {args.date}")
    hours = list(range(24))
    
    # List to store image paths for GIF creation
    image_paths = []
    
    for hour in hours:
        logger.info(f"Processing hour {hour:02d}:00 UTC")
        
        # Create inference data
        features, grid_shape = create_inference_data(
            args.date, hour, lat_grid, lon_grid, 
            args.elevation, args.azimuth, config
        )
        
        # Run inference
        predictions, uncertainties = run_model_inference(model, features, config)
        
        # Reshape to grid
        stec_map = predictions.squeeze().numpy().reshape(grid_shape)
        uncertainty_map = uncertainties.squeeze().numpy().reshape(grid_shape) if uncertainties is not None else None
        
        # Save hourly plot with GIM-style formatting
        base_filename = f'stec_map_{args.date}_{hour:02d}00'
        base_path = os.path.join(output_dir, base_filename)
        save_hourly_plot(lat_grid, lon_grid, stec_map, uncertainty_map,
                        base_path, args.date, hour, args.elevation, args.azimuth,
                        vmin=args.vmin, vmax=args.vmax)
        
        # Store image path for GIF creation
        image_paths.append(f"{base_path}.png")
        
        logger.info(f"Saved {base_filename}.png (STEC range: {np.nanmin(stec_map):.2f} - {np.nanmax(stec_map):.2f} TECU)")
    
    # Create GIF if requested
    if args.create_gif:
        gif_path = os.path.join(output_dir, f'stec_daily_{args.date}.gif')
        create_gif(image_paths, gif_path, duration=500)
    
    logger.info(f"Global map generation complete!")
    logger.info(f"Output saved to: {output_dir}")
    logger.info(f"Generated {len(hours)} hourly map plots")
    if args.create_gif:
        logger.info(f"Daily GIF saved to: {gif_path}")

if __name__ == "__main__":
    main()
