"""
Validation Script for FactorizedSTEC Model

This script performs sanity checks on the FactorizedSTEC model by:
1. Loading a trained model from experiments folder
2. Creating synthetic test cases with varying elevation angles
3. Running inference to get VTEC and MF predictions
4. Plotting the outputs to verify physical behavior:
   - MF should decrease with increasing elevation (approaches 1 at zenith)
   - VTEC should remain relatively stable across elevation changes
   - STEC = MF × VTEC should show expected variation

Usage:
    python scripts/validate_factorized_model.py --exp_path <path_to_experiment>
    
Example:
    python scripts/validate_factorized_model.py --exp_path experiments/Pretrain_STEC_FactorizedSTEC_h1024_l4_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_lw1e-1_SWI
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.model import FactorizedSTECModel, FactorizedSTECModelWrapper
from utils.feature_splitter import FeatureSplitter
from utils.feature_registry import initialize_feature_registry, FeatureType


def initialize_output_indices_for_registry(registry, config):
    """
    Initialize output indices for the feature registry.
    This mimics what CollateWithSH does but simplified for inference.
    """
    output_indices = {}
    current_idx = 0
    
    # Year (normalized)
    temporal_features = registry.get_feature_names(FeatureType.TEMPORAL)
    for feature_name in temporal_features:
        if feature_name == "year":
            output_indices[f"{feature_name}_norm"] = current_idx
            current_idx += 1
        elif feature_name in ["doy", "sod", "local_time_hours"]:
            # sin, cos, norm for cyclical features
            output_indices[f"{feature_name}_sin"] = current_idx
            output_indices[f"{feature_name}_cos"] = current_idx + 1
            output_indices[f"{feature_name}_norm"] = current_idx + 2
            current_idx += 3
    
    # Station features
    station_features = registry.get_feature_names(FeatureType.STATION)
    for feature_name in station_features:
        output_indices[f"{feature_name}_norm"] = current_idx
        current_idx += 1
    
    # Direction features - Cartesian unit vector
    direction_features = registry.get_feature_names(FeatureType.DIRECTION)
    if direction_features and "satazi" in direction_features and "satele" in direction_features:
        output_indices["e_up"] = current_idx
        output_indices["e_east"] = current_idx + 1
        output_indices["e_north"] = current_idx + 2
        current_idx += 3
    
    # IPP features
    ipp_features = registry.get_feature_names(FeatureType.IPP)
    for feature_name in ipp_features:
        output_indices[f"{feature_name}_norm"] = current_idx
        current_idx += 1
    
    # SH embeddings if enabled
    sh_degree = config.get('data', {}).get('SH_degree', 0)
    if sh_degree > 0:
        sh_dim = sh_degree * sh_degree
        has_station = len(station_features) > 0
        
        if has_station:
            output_indices["sh_sta_geo"] = slice(current_idx, current_idx + sh_dim)
            current_idx += sh_dim
        else:
            output_indices["sh_sta_geo"] = None
        
        output_indices["sh_ipp_geo"] = slice(current_idx, current_idx + sh_dim)
        current_idx += sh_dim
        
        if has_station:
            output_indices["sh_sta_sm"] = slice(current_idx, current_idx + sh_dim)
            current_idx += sh_dim
        else:
            output_indices["sh_sta_sm"] = None
        
        output_indices["sh_ipp_sm"] = slice(current_idx, current_idx + sh_dim)
        current_idx += sh_dim
    else:
        output_indices["sh_sta_geo"] = None
        output_indices["sh_ipp_geo"] = None
        output_indices["sh_sta_sm"] = None
        output_indices["sh_ipp_sm"] = None
    
    # SWI features
    swi_features = registry.get_feature_names(FeatureType.SWI)
    for feature_name in swi_features:
        output_indices[f"{feature_name}_norm"] = current_idx
        current_idx += 1
    
    registry.set_output_indices(output_indices)
    return current_idx


def load_model_and_config(exp_path):
    """Load trained FactorizedSTEC model and its configuration."""
    exp_path = Path(exp_path)
    
    # Load config
    config_path = exp_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize feature registry
    registry = initialize_feature_registry(config)
    
    # Initialize output indices for feature splitter
    total_features = initialize_output_indices_for_registry(registry, config)
    print(f"Total output features: {total_features}")
    
    # Find model checkpoint
    model_dir = exp_path / "model"
    model_files = list(model_dir.glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model checkpoint found in {model_dir}")
    
    model_path = model_files[0]  # Use first available checkpoint
    print(f"Loading model from: {model_path}")
    
    # Create feature splitter
    feature_splitter = FeatureSplitter(registry)
    
    # Get dimensions from splitter
    vtec_in_dim = feature_splitter.get_vtec_dim()
    geom_in_dim = feature_splitter.get_geom_dim()
    
    print(f"Feature dimensions - VTEC: {vtec_in_dim}, Geometry: {geom_in_dim}")
    
    # Create model architecture
    factorized_model = FactorizedSTECModel(
        vtec_in_dim=vtec_in_dim,
        geom_in_dim=geom_in_dim,
        vtec_hidden=config['model'].get('vtec_hidden', 128),
        geom_hidden=config['model'].get('geom_hidden', 64),
        vtec_layers=config['model'].get('vtec_layers', 4),
        geom_layers=config['model'].get('geom_layers', 3),
        activation=config['model'].get('activation', 'relu'),
        prior_sigma=config['model'].get('prior_sigma', 0.1)
    )
    
    # Wrap model
    model = FactorizedSTECModelWrapper(factorized_model, feature_splitter)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle legacy checkpoint with old naming convention (log_sigma_head -> variance_head)
    state_dict = checkpoint['model_state_dict']
    
    # Rename old keys if they exist
    legacy_mappings = {
        'model.vtec_net.log_sigma_head.weight_mu': 'model.vtec_net.variance_head.weight_mu',
        'model.vtec_net.log_sigma_head.weight_log_sigma': 'model.vtec_net.variance_head.weight_log_sigma',
        'model.vtec_net.log_sigma_head.bias_mu': 'model.vtec_net.variance_head.bias_mu',
        'model.vtec_net.log_sigma_head.bias_log_sigma': 'model.vtec_net.variance_head.bias_log_sigma',
    }
    
    for old_key, new_key in legacy_mappings.items():
        if old_key in state_dict:
            state_dict[new_key] = state_dict.pop(old_key)
    
    # Load state dict (strict=False to handle any other minor mismatches)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return model, config, feature_splitter


def load_real_observations(test_h5_path, num_samples=100, elevation_range=(10, 90)):
    """
    Load real observations from test.h5 dataset.
    
    Args:
        test_h5_path: Path to test.h5 file
        num_samples: Number of samples to load
        elevation_range: Tuple of (min_elev, max_elev) to filter samples
    
    Returns:
        raw_data: Structured array from h5 file
        indices: Selected indices
    """
    print(f"  Loading observations from {test_h5_path}...")
    
    with h5py.File(test_h5_path, 'r') as f:
        data = f['data'][:]
    
    # Filter by elevation range for more relevant samples
    min_elev, max_elev = elevation_range
    elev_mask = (data['satele'] >= min_elev) & (data['satele'] <= max_elev)
    
    # Also filter out any invalid data
    valid_mask = (data['stec'] > 0) & (data['vtec'] > 0) & np.isfinite(data['stec'])
    
    combined_mask = elev_mask & valid_mask
    valid_indices = np.where(combined_mask)[0]
    
    print(f"  Found {len(valid_indices)} valid observations in elevation range {min_elev}°-{max_elev}°")
    
    # Randomly sample
    np.random.seed(42)
    if len(valid_indices) > num_samples:
        selected_indices = np.random.choice(valid_indices, size=num_samples, replace=False)
    else:
        selected_indices = valid_indices[:num_samples]
    
    print(f"  Selected {len(selected_indices)} observations")
    
    return data[selected_indices], selected_indices


def create_test_samples_from_observations(raw_observations, elevation_angles_deg, config, feature_splitter, 
                                         ipp_latitudes=None):
    """
    Create test samples by modifying real observations to have varying elevation angles
    and optionally varying IPP latitudes.
    
    For each observation, create multiple versions with different elevation angles and/or latitudes.
    This allows testing how the model responds to elevation and latitude changes for the same
    underlying ionospheric conditions.
    
    Args:
        raw_observations: Structured array from h5 file
        elevation_angles_deg: Array of elevation angles to test
        config: Experiment config
        feature_splitter: FeatureSplitter instance
        ipp_latitudes: Optional array of IPP latitudes to test (if None, use original)
    
    Returns:
        test_data: Tensor of transformed features
        elevation_labels: Array of elevation angles for each sample
        ground_truth: Dictionary with original STEC and VTEC values
        latitude_labels: Array of IPP latitudes for each sample (if ipp_latitudes provided)
    """
    registry = config['feature_registry']
    output_indices = registry._output_indices
    
    # Determine total feature count
    max_idx = 0
    for key, val in output_indices.items():
        if isinstance(val, slice):
            max_idx = max(max_idx, val.stop)
        elif val is not None:
            max_idx = max(max_idx, val + 1)
    
    total_features = max_idx
    
    test_samples = []
    elev_labels = []
    lat_labels = []
    original_stec = []
    original_vtec = []
    original_elev = []
    original_lat_ipp = []
    
    # Determine if we're varying latitudes
    vary_latitude = ipp_latitudes is not None
    if not vary_latitude:
        ipp_latitudes = [None]  # Dummy list for iteration
    
    for obs in raw_observations:
        # Store original ground truth
        orig_stec = float(obs['stec'])
        orig_vtec = float(obs['vtec'])
        orig_elev = float(obs['satele'])
        orig_lat_ipp = float(obs['lat_ipp'])
        
        for lat_deg in ipp_latitudes:
            for elev_deg in elevation_angles_deg:
                elev_rad = np.deg2rad(elev_deg)
                
                # Create feature vector with transformed features
                sample = np.zeros(total_features)
                
                # ===== TEMPORAL FEATURES (from observation) =====
                if 'year_norm' in output_indices:
                    year = int(obs['year'])
                    sample[output_indices['year_norm']] = (year - 2000) / 30
                
                if 'doy_norm' in output_indices:
                    doy = int(obs['doy'])
                    doy_norm = (doy - 1) / 365
                    sample[output_indices['doy_sin']] = np.sin(doy_norm * 2 * np.pi)
                    sample[output_indices['doy_cos']] = np.cos(doy_norm * 2 * np.pi)
                    sample[output_indices['doy_norm']] = doy_norm
                
                if 'sod_norm' in output_indices:
                    sod = float(obs['sod'])
                    sod_norm = sod / 86400
                    sample[output_indices['sod_sin']] = np.sin(sod_norm * 2 * np.pi)
                    sample[output_indices['sod_cos']] = np.cos(sod_norm * 2 * np.pi)
                    sample[output_indices['sod_norm']] = sod_norm
                
                if 'local_time_hours_norm' in output_indices:
                    local_time = sod / 3600
                    lt_norm = local_time / 24
                    sample[output_indices['local_time_hours_sin']] = np.sin(lt_norm * 2 * np.pi)
                    sample[output_indices['local_time_hours_cos']] = np.cos(lt_norm * 2 * np.pi)
                    sample[output_indices['local_time_hours_norm']] = lt_norm
                
                # ===== STATION FEATURES (from observation, normalized) =====
                if 'lat_sta_norm' in output_indices:
                    lat_sta = float(obs['lat_sta'])
                    sample[output_indices['lat_sta_norm']] = lat_sta / 90.0
                if 'lon_sta_norm' in output_indices:
                    lon_sta = float(obs['lon_sta'])
                    sample[output_indices['lon_sta_norm']] = lon_sta / 180.0
                if 'sm_lat_sta_norm' in output_indices:
                    sm_lat_sta = float(obs['sm_lat_sta'])
                    sample[output_indices['sm_lat_sta_norm']] = sm_lat_sta / 90.0
                if 'sm_lon_sta_norm' in output_indices:
                    sm_lon_sta = float(obs['sm_lon_sta'])
                    sample[output_indices['sm_lon_sta_norm']] = sm_lon_sta / 180.0
                
                # ===== DIRECTION FEATURES (MODIFIED with new elevation) =====
                # Use original azimuth but NEW elevation
                if 'e_up' in output_indices:
                    azimuth_deg = float(obs['satazi'])
                    azimuth_rad = np.deg2rad(azimuth_deg)
                    
                    # Cartesian unit vector with MODIFIED elevation
                    sample[output_indices['e_up']] = np.sin(elev_rad)
                    sample[output_indices['e_east']] = np.cos(elev_rad) * np.sin(azimuth_rad)
                    sample[output_indices['e_north']] = np.cos(elev_rad) * np.cos(azimuth_rad)
                
                # ===== IPP FEATURES (from observation or modified, normalized) =====
                if 'lat_ipp_norm' in output_indices:
                    lat_ipp = lat_deg if vary_latitude else float(obs['lat_ipp'])
                    sample[output_indices['lat_ipp_norm']] = lat_ipp / 90.0
                if 'lon_ipp_norm' in output_indices:
                    lon_ipp = float(obs['lon_ipp'])
                    sample[output_indices['lon_ipp_norm']] = lon_ipp / 180.0
                if 'sm_lat_ipp_norm' in output_indices:
                    sm_lat_ipp = float(obs['sm_lat_ipp'])
                    sample[output_indices['sm_lat_ipp_norm']] = sm_lat_ipp / 90.0
                if 'sm_lon_ipp_norm' in output_indices:
                    sm_lon_ipp = float(obs['sm_lon_ipp'])
                    sample[output_indices['sm_lon_ipp_norm']] = sm_lon_ipp / 180.0
                
                # ===== SPHERICAL HARMONICS (compute from locations) =====
                for sh_key in ['sh_sta_geo', 'sh_ipp_geo', 'sh_sta_sm', 'sh_ipp_sm']:
                    if sh_key in output_indices and output_indices[sh_key] is not None:
                        sh_slice = output_indices[sh_key]
                        # Use observation-specific seed for consistency
                        obs_seed = int(obs['year']) * 1000 + int(obs['doy'])
                        rng = np.random.RandomState(obs_seed)
                        sample[sh_slice] = rng.randn(sh_slice.stop - sh_slice.start) * 0.1
                
                # ===== SPACE WEATHER INDICES =====
                if 'Kp_index_norm' in output_indices:
                    sample[output_indices['Kp_index_norm']] = 2.0 / 9.0
                if 'R_Sunspot_No_norm' in output_indices:
                    sample[output_indices['R_Sunspot_No_norm']] = 50.0 / 300.0
                if 'Dst-index,_nT_norm' in output_indices:
                    sample[output_indices['Dst-index,_nT_norm']] = (10.0 - (-500.0)) / 600.0
                if 'AE-index,_nT_norm' in output_indices:
                    sample[output_indices['AE-index,_nT_norm']] = 100.0 / 2000.0
                if 'ap_index,_nT_norm' in output_indices:
                    sample[output_indices['ap_index,_nT_norm']] = 10.0 / 400.0
                if 'f107_index_norm' in output_indices:
                    sample[output_indices['f107_index_norm']] = 100.0 / 300.0
                
                test_samples.append(sample)
                elev_labels.append(elev_deg)
                lat_labels.append(lat_deg if vary_latitude else orig_lat_ipp)
                original_stec.append(orig_stec)
                original_vtec.append(orig_vtec)
                original_elev.append(orig_elev)
                original_lat_ipp.append(orig_lat_ipp)
    
    test_data = torch.FloatTensor(np.array(test_samples))
    elev_labels = np.array(elev_labels)
    lat_labels = np.array(lat_labels)
    
    ground_truth = {
        'stec': np.array(original_stec),
        'vtec': np.array(original_vtec),
        'original_elevation': np.array(original_elev),
        'original_lat_ipp': np.array(original_lat_ipp)
    }
    
    if vary_latitude:
        return test_data, elev_labels, ground_truth, lat_labels
    else:
        return test_data, elev_labels, ground_truth
    return test_data, elev_labels, ground_truth


def run_inference(model, test_data, elevation_labels):
    """
    Run inference on test data to get VTEC, MF, and STEC predictions.
    
    Args:
        model: The trained model
        test_data: Test samples
        elevation_labels: Expected elevation angles for verification
    
    Returns:
        results: Dictionary with arrays of predictions
    """
    model.eval()
    
    # Verify that elevation angles in the data match expectations
    # Extract e_up and compute elevation from it
    registry = model.splitter.feature_registry
    output_indices = registry._output_indices
    
    if 'e_up' in output_indices:
        e_up_values = test_data[:, output_indices['e_up']].numpy()
        # e_up = sin(elevation), so elevation = arcsin(e_up)
        extracted_elevations = np.arcsin(np.clip(e_up_values, -1, 1)) * 180 / np.pi
        
        # Compare with expected elevations
        elevation_diff = np.abs(extracted_elevations - elevation_labels)
        max_diff = elevation_diff.max()
        mean_diff = elevation_diff.mean()
        
        print(f"  Elevation angle verification:")
        print(f"    Expected range: {elevation_labels.min():.1f}° to {elevation_labels.max():.1f}°")
        print(f"    Extracted range: {extracted_elevations.min():.1f}° to {extracted_elevations.max():.1f}°")
        print(f"    Max difference: {max_diff:.6f}°")
        print(f"    Mean difference: {mean_diff:.6f}°")
        
        if max_diff > 0.01:
            print(f"  WARNING: Elevation angles in test data don't match expected values!")
    
    with torch.no_grad():
        # Use forward_detailed to get all intermediate outputs
        detailed_output = model.forward_detailed(test_data)
    
    # Convert to numpy
    results = {
        'vtec_mean': detailed_output['vtec_mean'].cpu().numpy().flatten(),
        'vtec_std': detailed_output['sigma_v'].cpu().numpy().flatten(),
        'mf': detailed_output['mf'].cpu().numpy().flatten(),
        'stec_mean': detailed_output['mu_stec'].cpu().numpy().flatten(),
        'stec_std': detailed_output['sigma_stec'].cpu().numpy().flatten(),
    }
    
    return results


def analyze_geom_net_2d(model, config, feature_splitter, raw_observations, output_dir):
    """
    Analyze GeomNet behavior across 2D parameter space (elevation × latitude).
    
    Creates heatmaps showing MF predictions as a function of elevation angle and IPP latitude.
    This helps understand:
    - How MF varies with elevation (expected: MF decreases toward 1 as elevation increases)
    - How MF varies with latitude (latitude-dependent ionospheric behavior)
    - Combined elevation-latitude effects
    
    Args:
        model: Trained FactorizedSTEC model
        config: Experiment config
        feature_splitter: FeatureSplitter instance
        raw_observations: Sample observations for creating test cases
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("GEOMNET 2D ANALYSIS: Elevation × Latitude")
    print("="*70)
    
    # Define parameter grid
    elevation_grid = np.linspace(10, 90, 17)  # 17 elevations from 10° to 90°
    latitude_grid = np.linspace(-60, 60, 13)  # 13 latitudes from -60° to 60°
    
    print(f"Parameter grid: {len(elevation_grid)} elevations × {len(latitude_grid)} latitudes")
    print(f"Total samples: {len(elevation_grid) * len(latitude_grid) * len(raw_observations)}")
    
    # Use a subset of observations for efficiency
    n_obs = min(10, len(raw_observations))
    test_obs = raw_observations[:n_obs]
    
    # Create test samples varying both elevation and latitude
    test_data, elev_labels, ground_truth, lat_labels = create_test_samples_from_observations(
        test_obs, elevation_grid, config, feature_splitter, ipp_latitudes=latitude_grid
    )
    
    print(f"Generated {len(test_data)} test samples")
    print("Running inference...")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        detailed_output = model.forward_detailed(test_data)
    
    # Extract results
    mf_values = detailed_output['mf'].cpu().numpy().flatten()
    vtec_values = detailed_output['vtec_mean'].cpu().numpy().flatten()
    stec_values = detailed_output['mu_stec'].cpu().numpy().flatten()
    
    # Reshape into grids for each observation
    n_elev = len(elevation_grid)
    n_lat = len(latitude_grid)
    
    # Average across observations
    mf_grid = np.zeros((n_lat, n_elev))
    vtec_grid = np.zeros((n_lat, n_elev))
    stec_grid = np.zeros((n_lat, n_elev))
    
    for i in range(n_obs):
        start_idx = i * n_elev * n_lat
        obs_data = mf_values[start_idx:start_idx + n_elev * n_lat].reshape(n_lat, n_elev)
        mf_grid += obs_data
        
        obs_vtec = vtec_values[start_idx:start_idx + n_elev * n_lat].reshape(n_lat, n_elev)
        vtec_grid += obs_vtec
        
        obs_stec = stec_values[start_idx:start_idx + n_elev * n_lat].reshape(n_lat, n_elev)
        stec_grid += obs_stec
    
    mf_grid /= n_obs
    vtec_grid /= n_obs
    stec_grid /= n_obs
    
    # Compute theoretical MF for comparison
    Re = 6371  # Earth radius in km
    h_shell = 350  # Shell height in km
    theoretical_mf_grid = np.zeros((n_lat, n_elev))
    for i, elev in enumerate(elevation_grid):
        elev_rad = np.deg2rad(elev)
        theoretical_mf = 1 / np.sqrt(1 - ((Re / (Re + h_shell)) * np.cos(elev_rad))**2)
        theoretical_mf_grid[:, i] = theoretical_mf
    
    # Create comprehensive 2D visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # === Plot 1: MF Heatmap ===
    ax = fig.add_subplot(gs[0, 0])
    im = ax.contourf(elevation_grid, latitude_grid, mf_grid, levels=20, cmap='viridis')
    ax.contour(elevation_grid, latitude_grid, mf_grid, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=11)
    ax.set_ylabel('IPP Latitude (degrees)', fontsize=11)
    ax.set_title('Predicted Mapping Factor (MF)', fontweight='bold', fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MF', fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # === Plot 2: Theoretical MF Heatmap ===
    ax = fig.add_subplot(gs[0, 1])
    im = ax.contourf(elevation_grid, latitude_grid, theoretical_mf_grid, levels=20, cmap='viridis')
    ax.contour(elevation_grid, latitude_grid, theoretical_mf_grid, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=11)
    ax.set_ylabel('IPP Latitude (degrees)', fontsize=11)
    ax.set_title('Theoretical MF (Thin Shell)', fontweight='bold', fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MF', fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # === Plot 3: MF Difference (Predicted - Theoretical) ===
    ax = fig.add_subplot(gs[0, 2])
    mf_diff = mf_grid - theoretical_mf_grid
    vmax = max(abs(mf_diff.min()), abs(mf_diff.max()))
    im = ax.contourf(elevation_grid, latitude_grid, mf_diff, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.contour(elevation_grid, latitude_grid, mf_diff, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=11)
    ax.set_ylabel('IPP Latitude (degrees)', fontsize=11)
    ax.set_title('MF Difference (Pred - Theory)', fontweight='bold', fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('ΔMF', fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # === Plot 4: MF vs Elevation at different latitudes ===
    ax = fig.add_subplot(gs[1, 0])
    # Select representative latitudes
    lat_indices = [0, n_lat//4, n_lat//2, 3*n_lat//4, n_lat-1]
    for idx in lat_indices:
        lat = latitude_grid[idx]
        ax.plot(elevation_grid, mf_grid[idx, :], 'o-', label=f'{lat:.0f}° lat', linewidth=2, markersize=4)
    
    # Add theoretical for comparison
    ax.plot(elevation_grid, theoretical_mf_grid[0, :], 'k--', label='Theoretical', linewidth=2, alpha=0.7)
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=11)
    ax.set_ylabel('Mapping Factor (MF)', fontsize=11)
    ax.set_title('MF vs Elevation at Different Latitudes', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.9, max(mf_grid.max(), theoretical_mf_grid.max()) * 1.05])
    
    # === Plot 5: MF vs Latitude at different elevations ===
    ax = fig.add_subplot(gs[1, 1])
    # Select representative elevations
    elev_indices = [0, n_elev//4, n_elev//2, 3*n_elev//4, n_elev-1]
    for idx in elev_indices:
        elev = elevation_grid[idx]
        ax.plot(latitude_grid, mf_grid[:, idx], 'o-', label=f'{elev:.0f}° elev', linewidth=2, markersize=4)
    
    ax.set_xlabel('IPP Latitude (degrees)', fontsize=11)
    ax.set_ylabel('Mapping Factor (MF)', fontsize=11)
    ax.set_title('MF vs Latitude at Different Elevations', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # === Plot 6: Relative Error Heatmap ===
    ax = fig.add_subplot(gs[1, 2])
    relative_error = 100 * (mf_grid - theoretical_mf_grid) / theoretical_mf_grid
    vmax = max(abs(relative_error.min()), abs(relative_error.max()))
    im = ax.contourf(elevation_grid, latitude_grid, relative_error, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.contour(elevation_grid, latitude_grid, relative_error, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=11)
    ax.set_ylabel('IPP Latitude (degrees)', fontsize=11)
    ax.set_title('MF Relative Error (%)', fontweight='bold', fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Error (%)', fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # === Plot 7: VTEC Heatmap ===
    ax = fig.add_subplot(gs[2, 0])
    im = ax.contourf(elevation_grid, latitude_grid, vtec_grid, levels=20, cmap='plasma')
    ax.contour(elevation_grid, latitude_grid, vtec_grid, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=11)
    ax.set_ylabel('IPP Latitude (degrees)', fontsize=11)
    ax.set_title('VTEC Predictions (should be ~constant with elevation)', fontweight='bold', fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('VTEC (TECU)', fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # === Plot 8: STEC Heatmap ===
    ax = fig.add_subplot(gs[2, 1])
    im = ax.contourf(elevation_grid, latitude_grid, stec_grid, levels=20, cmap='plasma')
    ax.contour(elevation_grid, latitude_grid, stec_grid, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=11)
    ax.set_ylabel('IPP Latitude (degrees)', fontsize=11)
    ax.set_title('STEC Predictions (STEC = MF × VTEC)', fontweight='bold', fontsize=12)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('STEC (TECU)', fontsize=10)
    ax.grid(True, alpha=0.2)
    
    # === Plot 9: Statistics Summary ===
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    
    # Calculate statistics
    stats_text = "GeomNet 2D Analysis Statistics\n" + "="*35 + "\n\n"
    stats_text += f"Mapping Factor (MF):\n"
    stats_text += f"  Range: {mf_grid.min():.3f} - {mf_grid.max():.3f}\n"
    stats_text += f"  Mean: {mf_grid.mean():.3f} ± {mf_grid.std():.3f}\n"
    stats_text += f"  At zenith (90°): {mf_grid[:, -1].mean():.4f}\n\n"
    
    stats_text += f"Theoretical MF:\n"
    stats_text += f"  Range: {theoretical_mf_grid.min():.3f} - {theoretical_mf_grid.max():.3f}\n"
    stats_text += f"  At zenith (90°): {theoretical_mf_grid[0, -1]:.4f}\n\n"
    
    stats_text += f"MF Error:\n"
    stats_text += f"  MAE: {np.abs(mf_diff).mean():.3f}\n"
    stats_text += f"  RMSE: {np.sqrt((mf_diff**2).mean()):.3f}\n"
    stats_text += f"  Mean Rel Error: {relative_error.mean():.2f}%\n"
    stats_text += f"  Max Rel Error: {np.abs(relative_error).max():.2f}%\n\n"
    
    stats_text += f"VTEC:\n"
    stats_text += f"  Mean: {vtec_grid.mean():.2f} ± {vtec_grid.std():.2f} TECU\n"
    stats_text += f"  Elevation variation: {vtec_grid.max() - vtec_grid.min():.3f} TECU\n\n"
    
    stats_text += f"STEC:\n"
    stats_text += f"  Mean: {stec_grid.mean():.2f} ± {stec_grid.std():.2f} TECU\n"
    stats_text += f"  Range: {stec_grid.min():.2f} - {stec_grid.max():.2f} TECU\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('FactorizedSTEC GeomNet Analysis: Elevation × Latitude Dependencies',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plot_path = output_dir / 'geomnet_2d_analysis_elevation_latitude.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n2D analysis plot saved to: {plot_path}")
    
    # Print numerical summary
    print("\n" + "="*70)
    print("GEOMNET 2D ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nMapping Factor Statistics:")
    print(f"  Predicted MF range: {mf_grid.min():.3f} to {mf_grid.max():.3f}")
    print(f"  MF at zenith (90°, mean across latitudes): {mf_grid[:, -1].mean():.4f} ± {mf_grid[:, -1].std():.4f}")
    print(f"  MF at low elevation (10°, mean across latitudes): {mf_grid[:, 0].mean():.3f} ± {mf_grid[:, 0].std():.3f}")
    print(f"  Theoretical at 10°: {theoretical_mf_grid[0, 0]:.3f}")
    
    print(f"\nLatitude Dependence:")
    # Check if MF varies with latitude at fixed elevation
    mid_elev_idx = n_elev // 2
    mid_elev = elevation_grid[mid_elev_idx]
    lat_variation = mf_grid[:, mid_elev_idx].max() - mf_grid[:, mid_elev_idx].min()
    print(f"  MF variation across latitudes at {mid_elev:.0f}°: {lat_variation:.4f}")
    print(f"  MF at equator (0°, {mid_elev:.0f}°): {mf_grid[n_lat//2, mid_elev_idx]:.3f}")
    print(f"  MF at poles (±60°, {mid_elev:.0f}°): {(mf_grid[0, mid_elev_idx] + mf_grid[-1, mid_elev_idx])/2:.3f}")
    
    print(f"\nAccuracy vs Theoretical:")
    print(f"  Mean Absolute Error: {np.abs(mf_diff).mean():.3f}")
    print(f"  RMSE: {np.sqrt((mf_diff**2).mean()):.3f}")
    print(f"  Mean Relative Error: {relative_error.mean():.2f}%")
    print(f"  Max Absolute Relative Error: {np.abs(relative_error).max():.2f}%")
    
    print(f"\nVTEC Consistency Check:")
    print(f"  VTEC should remain ~constant across elevations")
    print(f"  Mean VTEC: {vtec_grid.mean():.2f} TECU")
    print(f"  VTEC std across all conditions: {vtec_grid.std():.3f} TECU")
    print(f"  Max VTEC variation with elevation: {np.max([vtec_grid[:, i].max() - vtec_grid[:, i].min() for i in range(n_elev)]):.3f} TECU")
    
    print("="*70)
    
    plt.show()


def plot_results(elevation_angles, results, ground_truth, output_dir):
    """
    Create visualization plots for validation with ground truth comparison.
    
    Args:
        elevation_angles: Array of elevation angles for each sample
        results: Dictionary with prediction arrays
        ground_truth: Dictionary with original STEC, VTEC values
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Compute statistics per elevation bin
    unique_elevs = np.unique(elevation_angles)
    
    stats = {
        'elev': unique_elevs,
        'vtec_mean': [],
        'vtec_std_mean': [],
        'vtec_std_std': [],
        'mf_mean': [],
        'mf_std': [],
        'stec_mean': [],
        'stec_std_mean': [],
        'stec_std_std': [],
    }
    
    for elev in unique_elevs:
        mask = elevation_angles == elev
        stats['vtec_mean'].append(results['vtec_mean'][mask].mean())
        stats['vtec_std_mean'].append(results['vtec_std'][mask].mean())
        stats['vtec_std_std'].append(results['vtec_std'][mask].std())
        stats['mf_mean'].append(results['mf'][mask].mean())
        stats['mf_std'].append(results['mf'][mask].std())
        stats['stec_mean'].append(results['stec_mean'][mask].mean())
        stats['stec_std_mean'].append(results['stec_std'][mask].mean())
        stats['stec_std_std'].append(results['stec_std'][mask].std())
    
    # Convert to arrays
    for key in stats:
        if key != 'elev':
            stats[key] = np.array(stats[key])
    
    # Theoretical MF (simple thin shell approximation)
    Re = 6371  # Earth radius in km
    h_shell = 450  # Shell height in km
    theoretical_mf = 1 / np.sqrt(1 - ((Re / (Re + h_shell)) * np.cos(np.deg2rad(unique_elevs)))**2)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('FactorizedSTEC Model Validation: Elevation Angle Response', fontsize=16, fontweight='bold')
    
    # Plot 1: VTEC vs Elevation
    ax = axes[0, 0]
    ax.plot(stats['elev'], stats['vtec_mean'], 'o-', linewidth=2, markersize=6, label='Mean VTEC')
    ax.fill_between(stats['elev'], 
                     stats['vtec_mean'] - stats['vtec_std_mean'],
                     stats['vtec_mean'] + stats['vtec_std_mean'],
                     alpha=0.3, label='±1σ (uncertainty)')
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('VTEC (TECU)', fontsize=12)
    ax.set_title('VTEC Prediction vs Elevation\n(Should be relatively stable)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: MF vs Elevation with theoretical comparison
    ax = axes[0, 1]
    ax.plot(stats['elev'], stats['mf_mean'], 'o-', linewidth=2, markersize=6, label='Predicted MF', color='C1')
    ax.fill_between(stats['elev'],
                     stats['mf_mean'] - stats['mf_std'],
                     stats['mf_mean'] + stats['mf_std'],
                     alpha=0.3, color='C1')
    ax.plot(stats['elev'], theoretical_mf, '--', linewidth=2, label='Theoretical MF (thin shell)', color='red')
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('Mapping Factor (MF)', fontsize=12)
    ax.set_title('Mapping Factor vs Elevation\n(Should decrease toward 1 at zenith)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.5, label='MF=1 (zenith)')
    
    # Plot 3: STEC vs Elevation
    ax = axes[0, 2]
    ax.plot(stats['elev'], stats['stec_mean'], 'o-', linewidth=2, markersize=6, label='Mean STEC', color='C2')
    ax.fill_between(stats['elev'],
                     stats['stec_mean'] - stats['stec_std_mean'],
                     stats['stec_mean'] + stats['stec_std_mean'],
                     alpha=0.3, color='C2', label='±1σ (uncertainty)')
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('STEC (TECU)', fontsize=12)
    ax.set_title('STEC Prediction vs Elevation\n(STEC = MF × VTEC)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: VTEC Uncertainty vs Elevation
    ax = axes[1, 0]
    ax.plot(stats['elev'], stats['vtec_std_mean'], 'o-', linewidth=2, markersize=6, color='C3')
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('VTEC Uncertainty σ_v (TECU)', fontsize=12)
    ax.set_title('VTEC Uncertainty vs Elevation', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: MF relative error from theoretical
    ax = axes[1, 1]
    relative_error = 100 * (stats['mf_mean'] - theoretical_mf) / theoretical_mf
    ax.plot(stats['elev'], relative_error, 'o-', linewidth=2, markersize=6, color='C4')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('Relative Error (%)', fontsize=12)
    ax.set_title('MF Prediction Error\n(vs Theoretical Thin Shell)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: STEC Uncertainty vs Elevation
    ax = axes[1, 2]
    ax.plot(stats['elev'], stats['stec_std_mean'], 'o-', linewidth=2, markersize=6, color='C5')
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('STEC Uncertainty σ_s (TECU)', fontsize=12)
    ax.set_title('STEC Uncertainty vs Elevation\n(σ_s = |MF| × σ_v)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'factorized_validation_elevation_response.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Create a second plot: scatter plots showing distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('FactorizedSTEC Model: Output Distributions', fontsize=16, fontweight='bold')
    
    # Scatter: VTEC vs Elevation (all samples)
    ax = axes[0]
    scatter = ax.scatter(elevation_angles, results['vtec_mean'], 
                        c=results['vtec_std'], cmap='viridis', 
                        alpha=0.5, s=10)
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('VTEC (TECU)', fontsize=12)
    ax.set_title('VTEC Distribution Across All Samples', fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('VTEC Uncertainty σ_v (TECU)', fontsize=10)
    
    # Scatter: MF vs Elevation (all samples)
    ax = axes[1]
    scatter = ax.scatter(elevation_angles, results['mf'], 
                        c=results['stec_std'], cmap='plasma',
                        alpha=0.5, s=10)
    ax.plot(unique_elevs, theoretical_mf, 'r--', linewidth=2, label='Theoretical')
    ax.set_xlabel('Elevation Angle (degrees)', fontsize=12)
    ax.set_ylabel('Mapping Factor (MF)', fontsize=12)
    ax.set_title('MF Distribution Across All Samples', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('STEC Uncertainty σ_s (TECU)', fontsize=10)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'factorized_validation_distributions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Print numerical summary
    print("\n" + "="*70)
    print("NUMERICAL VALIDATION SUMMARY")
    print("="*70)
    print(f"\nElevation Range: {unique_elevs.min():.1f}° to {unique_elevs.max():.1f}°")
    print(f"Number of observations: {len(ground_truth['stec']) // len(unique_elevs)}")
    
    print(f"\nVTEC Statistics:")
    print(f"  Mean VTEC across all elevations: {results['vtec_mean'].mean():.2f} ± {results['vtec_mean'].std():.2f} TECU")
    print(f"  VTEC variation with elevation: {stats['vtec_mean'].max() - stats['vtec_mean'].min():.2f} TECU")
    print(f"  Average VTEC uncertainty: {results['vtec_std'].mean():.2f} TECU")
    
    # Ground truth comparison
    print(f"\nGround Truth Comparison:")
    unique_orig_elevs = np.unique(ground_truth['original_elevation'])
    print(f"  Original elevation range in data: {unique_orig_elevs.min():.1f}° to {unique_orig_elevs.max():.1f}°")
    print(f"  Original STEC: {ground_truth['stec'].mean():.2f} ± {ground_truth['stec'].std():.2f} TECU (range: {ground_truth['stec'].min():.2f}-{ground_truth['stec'].max():.2f})")
    print(f"  Original VTEC: {ground_truth['vtec'].mean():.2f} ± {ground_truth['vtec'].std():.2f} TECU (range: {ground_truth['vtec'].min():.2f}-{ground_truth['vtec'].max():.2f})")
    
    # For samples at their original elevation, compare predictions vs ground truth
    orig_elev_matches = []
    for i, (elev, orig_elev) in enumerate(zip(elevation_angles, ground_truth['original_elevation'])):
        if np.abs(elev - orig_elev) < 1.0:  # Within 1 degree
            orig_elev_matches.append(i)
    
    if len(orig_elev_matches) > 0:
        pred_stec_at_orig = results['stec_mean'][orig_elev_matches]
        true_stec_at_orig = ground_truth['stec'][orig_elev_matches]
        pred_vtec_at_orig = results['vtec_mean'][orig_elev_matches]
        true_vtec_at_orig = ground_truth['vtec'][orig_elev_matches]
        
        stec_mae = np.abs(pred_stec_at_orig - true_stec_at_orig).mean()
        vtec_mae = np.abs(pred_vtec_at_orig - true_vtec_at_orig).mean()
        
        print(f"\nPrediction Accuracy at Original Elevations ({len(orig_elev_matches)} samples):")
        print(f"  STEC MAE: {stec_mae:.2f} TECU")
        print(f"  VTEC MAE: {vtec_mae:.2f} TECU")
        print(f"  STEC predicted: {pred_stec_at_orig.mean():.2f} ± {pred_stec_at_orig.std():.2f} TECU")
        print(f"  STEC true: {true_stec_at_orig.mean():.2f} ± {true_stec_at_orig.std():.2f} TECU")
    
    print(f"\nMapping Factor Statistics:")
    print(f"  MF at lowest elevation ({unique_elevs.min():.1f}°): {stats['mf_mean'][0]:.3f}")
    print(f"  MF at highest elevation ({unique_elevs.max():.1f}°): {stats['mf_mean'][-1]:.3f}")
    print(f"  Theoretical MF at {unique_elevs.min():.1f}°: {theoretical_mf[0]:.3f}")
    print(f"  Theoretical MF at {unique_elevs.max():.1f}°: {theoretical_mf[-1]:.3f}")
    print(f"  Mean absolute error: {np.abs(stats['mf_mean'] - theoretical_mf).mean():.3f}")
    print(f"  Mean relative error: {np.abs(relative_error).mean():.2f}%")
    
    print(f"\nSTEC Statistics:")
    print(f"  STEC range: {results['stec_mean'].min():.2f} to {results['stec_mean'].max():.2f} TECU")
    print(f"  Average STEC uncertainty: {results['stec_std'].mean():.2f} TECU")
    print("="*70)
    
    plt.show()


def analyze_azimuth_dependence(model, config, feature_splitter, raw_observations, output_dir, 
                                num_azimuth=18, elevation_deg=30.0):
    """
    Analyze GeomNet's mapping factor dependence on azimuth angle at fixed elevation.
    
    Physically, MF should NOT depend on azimuth (only elevation matters for thin shell).
    This test checks for spurious azimuth dependencies in the learned GeomNet.
    
    Args:
        model: Trained FactorizedSTEC model
        config: Model configuration
        feature_splitter: Feature splitter for routing features to subnetworks
        raw_observations: Array of observations from test.h5
        output_dir: Directory to save plots
        num_azimuth: Number of azimuth angles to test (default: 18, every 20°)
        elevation_deg: Fixed elevation angle in degrees (default: 30°)
    """
    print("\n" + "="*70)
    print(f"GEOMNET AZIMUTH ANALYSIS: Fixed Elevation = {elevation_deg}°")
    print("="*70)
    print(f"Testing {num_azimuth} azimuth angles (0° to 360°)")
    print(f"Using {len(raw_observations)} observations")
    
    # Get output indices from registry
    registry = config['feature_registry']
    total_features = initialize_output_indices_for_registry(registry, config)
    output_indices = registry._output_indices
    
    # Generate azimuth angles
    azimuth_angles_deg = np.linspace(0, 360, num_azimuth, endpoint=False)  # 0-340 for 18 points
    elev_rad = np.deg2rad(elevation_deg)
    
    # Generate test samples
    test_samples = []
    azimuth_labels = []
    
    for obs in raw_observations[:10]:  # Use subset for speed
        for azim_deg in azimuth_angles_deg:
            azim_rad = np.deg2rad(azim_deg)
            
            sample = np.zeros(total_features)
            
            # Copy temporal features
            if 'year_norm' in output_indices:
                sample[output_indices['year_norm']] = (int(obs['year']) - 2000) / 30
            if 'doy_norm' in output_indices:
                doy_norm = (int(obs['doy']) - 1) / 365
                sample[output_indices['doy_sin']] = np.sin(doy_norm * 2 * np.pi)
                sample[output_indices['doy_cos']] = np.cos(doy_norm * 2 * np.pi)
                sample[output_indices['doy_norm']] = doy_norm
            if 'sod_norm' in output_indices:
                sod_norm = float(obs['sod']) / 86400
                sample[output_indices['sod_sin']] = np.sin(sod_norm * 2 * np.pi)
                sample[output_indices['sod_cos']] = np.cos(sod_norm * 2 * np.pi)
                sample[output_indices['sod_norm']] = sod_norm
            if 'local_time_hours_norm' in output_indices:
                lt_norm = (float(obs['sod']) / 3600) / 24
                sample[output_indices['local_time_hours_sin']] = np.sin(lt_norm * 2 * np.pi)
                sample[output_indices['local_time_hours_cos']] = np.cos(lt_norm * 2 * np.pi)
                sample[output_indices['local_time_hours_norm']] = lt_norm
            
            # Copy station features
            if 'lat_sta_norm' in output_indices:
                sample[output_indices['lat_sta_norm']] = float(obs['lat_sta']) / 90.0
            if 'lon_sta_norm' in output_indices:
                sample[output_indices['lon_sta_norm']] = float(obs['lon_sta']) / 180.0
            if 'sm_lat_sta_norm' in output_indices:
                sample[output_indices['sm_lat_sta_norm']] = float(obs['sm_lat_sta']) / 90.0
            if 'sm_lon_sta_norm' in output_indices:
                sample[output_indices['sm_lon_sta_norm']] = float(obs['sm_lon_sta']) / 180.0
            
            # Direction features - VARY AZIMUTH, FIX ELEVATION
            if 'e_up' in output_indices:
                sample[output_indices['e_up']] = np.sin(elev_rad)
                sample[output_indices['e_east']] = np.cos(elev_rad) * np.sin(azim_rad)
                sample[output_indices['e_north']] = np.cos(elev_rad) * np.cos(azim_rad)
            
            # Copy IPP features
            if 'lat_ipp_norm' in output_indices:
                sample[output_indices['lat_ipp_norm']] = float(obs['lat_ipp']) / 90.0
            if 'lon_ipp_norm' in output_indices:
                sample[output_indices['lon_ipp_norm']] = float(obs['lon_ipp']) / 180.0
            if 'sm_lat_ipp_norm' in output_indices:
                sample[output_indices['sm_lat_ipp_norm']] = float(obs['sm_lat_ipp']) / 90.0
            if 'sm_lon_ipp_norm' in output_indices:
                sample[output_indices['sm_lon_ipp_norm']] = float(obs['sm_lon_ipp']) / 180.0
            
            # Spherical harmonics
            for sh_key in ['sh_sta_geo', 'sh_ipp_geo', 'sh_sta_sm', 'sh_ipp_sm']:
                if sh_key in output_indices and output_indices[sh_key] is not None:
                    sh_slice = output_indices[sh_key]
                    obs_seed = int(obs['year']) * 1000 + int(obs['doy'])
                    rng = np.random.RandomState(obs_seed)
                    sample[sh_slice] = rng.randn(sh_slice.stop - sh_slice.start) * 0.1
            
            # Space weather indices
            if 'Kp_index_norm' in output_indices:
                sample[output_indices['Kp_index_norm']] = 2.0 / 9.0
            if 'R_Sunspot_No_norm' in output_indices:
                sample[output_indices['R_Sunspot_No_norm']] = 50.0 / 300.0
            if 'Dst-index,_nT_norm' in output_indices:
                sample[output_indices['Dst-index,_nT_norm']] = (10.0 - (-500.0)) / 600.0
            if 'AE-index,_nT_norm' in output_indices:
                sample[output_indices['AE-index,_nT_norm']] = 100.0 / 2000.0
            if 'ap_index,_nT_norm' in output_indices:
                sample[output_indices['ap_index,_nT_norm']] = 10.0 / 400.0
            if 'f107_index_norm' in output_indices:
                sample[output_indices['f107_index_norm']] = 100.0 / 300.0
            
            test_samples.append(sample)
            azimuth_labels.append(azim_deg)
    
    test_tensor = torch.tensor(np.array(test_samples), dtype=torch.float32)
    azimuth_labels = np.array(azimuth_labels)
    
    print(f"Generated {len(test_samples)} test samples")
    print("Running inference...")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        detailed_output = model.forward_detailed(test_tensor)
    
    # Extract results
    vtec_mean = detailed_output['vtec_mean'].cpu().numpy().flatten()
    vtec_std = np.sqrt(detailed_output['vtec_variance'].cpu().numpy().flatten())
    mf = detailed_output['mf'].cpu().numpy().flatten()
    
    # Analyze results
    mf_by_azimuth = {}
    vtec_by_azimuth = {}
    
    for azim_deg in np.unique(azimuth_labels):
        mask = azimuth_labels == azim_deg
        mf_by_azimuth[azim_deg] = mf[mask]
        vtec_by_azimuth[azim_deg] = vtec_mean[mask]
    
    # Statistics
    mf_means = [mf_by_azimuth[az].mean() for az in sorted(mf_by_azimuth.keys())]
    mf_stds = [mf_by_azimuth[az].std() for az in sorted(mf_by_azimuth.keys())]
    vtec_means = [vtec_by_azimuth[az].mean() for az in sorted(vtec_by_azimuth.keys())]
    azimuth_sorted = sorted(mf_by_azimuth.keys())
    
    mf_variation = np.max(mf_means) - np.min(mf_means)
    vtec_variation = np.max(vtec_means) - np.min(vtec_means)
    mf_mean_overall = np.mean(mf_means)
    theoretical_mf = 1.0 / np.sin(elev_rad)
    
    # Create plot
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'GeomNet Azimuth Dependence Analysis (Elevation = {elevation_deg}°)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: MF vs Azimuth
    ax = axes[0, 0]
    ax.errorbar(azimuth_sorted, mf_means, yerr=mf_stds, fmt='o-', capsize=5, 
                linewidth=2, markersize=6, label='Predicted MF')
    ax.axhline(theoretical_mf, color='r', linestyle='--', linewidth=2, 
               label=f'Theoretical MF = {theoretical_mf:.3f}')
    ax.axhline(mf_mean_overall, color='g', linestyle=':', linewidth=2, 
               label=f'Mean MF = {mf_mean_overall:.3f}')
    ax.set_xlabel('Azimuth Angle (degrees)', fontsize=12)
    ax.set_ylabel('Mapping Factor (MF)', fontsize=12)
    ax.set_title('Mapping Factor vs Azimuth\n(Should be constant)', fontweight='bold')
    ax.set_xlim(-10, 370)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: VTEC vs Azimuth
    ax = axes[0, 1]
    ax.plot(azimuth_sorted, vtec_means, 'o-', linewidth=2, markersize=6, color='C1')
    ax.axhline(np.mean(vtec_means), color='r', linestyle='--', linewidth=2, 
               label=f'Mean = {np.mean(vtec_means):.2f} TECU')
    ax.set_xlabel('Azimuth Angle (degrees)', fontsize=12)
    ax.set_ylabel('VTEC (TECU)', fontsize=12)
    ax.set_title('VTEC vs Azimuth\n(Should be constant)', fontweight='bold')
    ax.set_xlim(-10, 370)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Polar plot of MF
    ax = axes[1, 0]
    ax = plt.subplot(2, 2, 3, projection='polar')
    azim_rad = np.deg2rad(azimuth_sorted)
    ax.plot(azim_rad, mf_means, 'o-', linewidth=2, markersize=6)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('MF Polar Plot\n(Circle = no azimuth dependence)', fontweight='bold', pad=20)
    ax.set_ylim(np.min(mf_means) * 0.95, np.max(mf_means) * 1.05)
    
    # Plot 4: MF variation statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    AZIMUTH DEPENDENCE STATISTICS
    {'='*40}
    
    Fixed Elevation: {elevation_deg}°
    Number of observations: {len(raw_observations[:10])}
    Number of azimuth angles: {num_azimuth}
    
    Mapping Factor:
      Mean MF: {mf_mean_overall:.4f}
      MF range: {np.min(mf_means):.4f} to {np.max(mf_means):.4f}
      MF variation: {mf_variation:.4f} ({mf_variation/mf_mean_overall*100:.2f}%)
      Std dev: {np.mean(mf_stds):.4f}
      
    Theoretical:
      MF at {elevation_deg}°: {theoretical_mf:.4f}
      Error: {np.abs(mf_mean_overall - theoretical_mf):.4f}
      Relative error: {np.abs(mf_mean_overall - theoretical_mf)/theoretical_mf*100:.2f}%
    
    VTEC:
      Mean VTEC: {np.mean(vtec_means):.2f} TECU
      VTEC variation: {vtec_variation:.4f} TECU
      
    INTERPRETATION:
      {'✓ PASS' if mf_variation < 0.05 else '✗ FAIL'}: MF azimuth variation < 5%
      {'✓ PASS' if vtec_variation < 1.0 else '✗ FAIL'}: VTEC azimuth variation < 1 TECU
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    plot_path = output_dir / f'geomnet_azimuth_analysis_elev{int(elevation_deg)}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nAzimuth analysis plot saved to: {plot_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("AZIMUTH DEPENDENCE SUMMARY")
    print("="*70)
    print(f"Elevation angle: {elevation_deg}°")
    print(f"\nMapping Factor Statistics:")
    print(f"  Mean MF: {mf_mean_overall:.4f} (Theoretical: {theoretical_mf:.4f})")
    print(f"  MF range: {np.min(mf_means):.4f} to {np.max(mf_means):.4f}")
    print(f"  MF variation across azimuth: {mf_variation:.4f} ({mf_variation/mf_mean_overall*100:.2f}%)")
    print(f"  Mean std dev within azimuth bins: {np.mean(mf_stds):.4f}")
    print(f"\nVTEC Statistics:")
    print(f"  Mean VTEC: {np.mean(vtec_means):.2f} TECU")
    print(f"  VTEC variation across azimuth: {vtec_variation:.4f} TECU")
    print(f"\nPhysical Expectation: MF should NOT depend on azimuth")
    if mf_variation < 0.05:
        print(f"  ✓ PASS: Azimuth variation ({mf_variation:.4f}) is negligible")
    else:
        print(f"  ✗ WARNING: Azimuth variation ({mf_variation:.4f}) may indicate model issues")
    print("="*70)


def analyze_high_dimensional_mf(model, config, feature_splitter, raw_observations, output_dir):
    """
    High-dimensional analysis of Mapping Factor across multiple parameters.
    
    Analyzes MF behavior in 3D parameter space (elevation × latitude × azimuth) to identify
    complex interaction patterns and potential spurious dependencies.
    
    Args:
        model: Trained FactorizedSTEC model
        config: Experiment config
        feature_splitter: FeatureSplitter instance
        raw_observations: Sample observations for creating test cases
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("HIGH-DIMENSIONAL MF ANALYSIS: Elevation × Latitude × Azimuth")
    print("="*70)
    
    # Define parameter grids
    elevation_grid = np.array([15, 30, 45, 60, 75, 90])  # 6 elevations
    latitude_grid = np.linspace(-60, 60, 9)  # 9 latitudes
    azimuth_grid = np.linspace(0, 315, 8)  # 8 azimuths (0-315 by 45°)
    
    print(f"Parameter grid:")
    print(f"  Elevations: {len(elevation_grid)} points - {elevation_grid}")
    print(f"  Latitudes: {len(latitude_grid)} points - {latitude_grid[0]:.0f}° to {latitude_grid[-1]:.0f}°")
    print(f"  Azimuths: {len(azimuth_grid)} points - {azimuth_grid[0]:.0f}° to {azimuth_grid[-1]:.0f}°")
    print(f"Total combinations: {len(elevation_grid) * len(latitude_grid) * len(azimuth_grid)}")
    
    # Get output indices from registry
    registry = config['feature_registry']
    total_features = initialize_output_indices_for_registry(registry, config)
    output_indices = registry._output_indices
    
    # Use subset of observations
    n_obs = min(5, len(raw_observations))
    test_obs = raw_observations[:n_obs]
    
    # Generate test samples
    test_samples = []
    elev_labels = []
    lat_labels = []
    azim_labels = []
    
    print(f"Generating test samples from {n_obs} observations...")
    
    for obs in test_obs:
        for lat_deg in latitude_grid:
            for azim_deg in azimuth_grid:
                for elev_deg in elevation_grid:
                    elev_rad = np.deg2rad(elev_deg)
                    azim_rad = np.deg2rad(azim_deg)
                    
                    sample = np.zeros(total_features)
                    
                    # Temporal features
                    if 'year_norm' in output_indices:
                        sample[output_indices['year_norm']] = (int(obs['year']) - 2000) / 30
                    if 'doy_norm' in output_indices:
                        doy_norm = (int(obs['doy']) - 1) / 365
                        sample[output_indices['doy_sin']] = np.sin(doy_norm * 2 * np.pi)
                        sample[output_indices['doy_cos']] = np.cos(doy_norm * 2 * np.pi)
                        sample[output_indices['doy_norm']] = doy_norm
                    if 'sod_norm' in output_indices:
                        sod_norm = float(obs['sod']) / 86400
                        sample[output_indices['sod_sin']] = np.sin(sod_norm * 2 * np.pi)
                        sample[output_indices['sod_cos']] = np.cos(sod_norm * 2 * np.pi)
                        sample[output_indices['sod_norm']] = sod_norm
                    if 'local_time_hours_norm' in output_indices:
                        lt_norm = (float(obs['sod']) / 3600) / 24
                        sample[output_indices['local_time_hours_sin']] = np.sin(lt_norm * 2 * np.pi)
                        sample[output_indices['local_time_hours_cos']] = np.cos(lt_norm * 2 * np.pi)
                        sample[output_indices['local_time_hours_norm']] = lt_norm
                    
                    # Station features
                    if 'lat_sta_norm' in output_indices:
                        sample[output_indices['lat_sta_norm']] = float(obs['lat_sta']) / 90.0
                    if 'lon_sta_norm' in output_indices:
                        sample[output_indices['lon_sta_norm']] = float(obs['lon_sta']) / 180.0
                    if 'sm_lat_sta_norm' in output_indices:
                        sample[output_indices['sm_lat_sta_norm']] = float(obs['sm_lat_sta']) / 90.0
                    if 'sm_lon_sta_norm' in output_indices:
                        sample[output_indices['sm_lon_sta_norm']] = float(obs['sm_lon_sta']) / 180.0
                    
                    # Direction features - VARY ALL
                    if 'e_up' in output_indices:
                        sample[output_indices['e_up']] = np.sin(elev_rad)
                        sample[output_indices['e_east']] = np.cos(elev_rad) * np.sin(azim_rad)
                        sample[output_indices['e_north']] = np.cos(elev_rad) * np.cos(azim_rad)
                    
                    # IPP features - VARY LATITUDE
                    if 'lat_ipp_norm' in output_indices:
                        sample[output_indices['lat_ipp_norm']] = lat_deg / 90.0
                    if 'lon_ipp_norm' in output_indices:
                        sample[output_indices['lon_ipp_norm']] = float(obs['lon_ipp']) / 180.0
                    if 'sm_lat_ipp_norm' in output_indices:
                        sample[output_indices['sm_lat_ipp_norm']] = float(obs['sm_lat_ipp']) / 90.0
                    if 'sm_lon_ipp_norm' in output_indices:
                        sample[output_indices['sm_lon_ipp_norm']] = float(obs['sm_lon_ipp']) / 180.0
                    
                    # Spherical harmonics
                    for sh_key in ['sh_sta_geo', 'sh_ipp_geo', 'sh_sta_sm', 'sh_ipp_sm']:
                        if sh_key in output_indices and output_indices[sh_key] is not None:
                            sh_slice = output_indices[sh_key]
                            obs_seed = int(obs['year']) * 1000 + int(obs['doy'])
                            rng = np.random.RandomState(obs_seed)
                            sample[sh_slice] = rng.randn(sh_slice.stop - sh_slice.start) * 0.1
                    
                    # Space weather
                    if 'Kp_index_norm' in output_indices:
                        sample[output_indices['Kp_index_norm']] = 2.0 / 9.0
                    if 'R_Sunspot_No_norm' in output_indices:
                        sample[output_indices['R_Sunspot_No_norm']] = 50.0 / 300.0
                    if 'Dst-index,_nT_norm' in output_indices:
                        sample[output_indices['Dst-index,_nT_norm']] = (10.0 - (-500.0)) / 600.0
                    if 'AE-index,_nT_norm' in output_indices:
                        sample[output_indices['AE-index,_nT_norm']] = 100.0 / 2000.0
                    if 'ap_index,_nT_norm' in output_indices:
                        sample[output_indices['ap_index,_nT_norm']] = 10.0 / 400.0
                    if 'f107_index_norm' in output_indices:
                        sample[output_indices['f107_index_norm']] = 100.0 / 300.0
                    
                    test_samples.append(sample)
                    elev_labels.append(elev_deg)
                    lat_labels.append(lat_deg)
                    azim_labels.append(azim_deg)
    
    test_tensor = torch.tensor(np.array(test_samples), dtype=torch.float32)
    elev_labels = np.array(elev_labels)
    lat_labels = np.array(lat_labels)
    azim_labels = np.array(azim_labels)
    
    print(f"Generated {len(test_samples)} test samples")
    print("Running inference...")
    
    # Run inference
    model.eval()
    with torch.no_grad():
        detailed_output = model.forward_detailed(test_tensor)
    
    mf_values = detailed_output['mf'].cpu().numpy().flatten()
    vtec_values = detailed_output['vtec_mean'].cpu().numpy().flatten()
    
    # Compute theoretical MF for each sample
    Re = 6371  # Earth radius in km
    h_shell = 350  # Shell height in km
    theoretical_mf = np.zeros_like(elev_labels)
    for i, elev in enumerate(elev_labels):
        elev_rad = np.deg2rad(elev)
        theoretical_mf[i] = 1 / np.sqrt(1 - ((Re / (Re + h_shell)) * np.cos(elev_rad))**2)
    
    # Average across observations
    n_combos = len(elevation_grid) * len(latitude_grid) * len(azimuth_grid)
    mf_averaged = np.zeros(n_combos)
    vtec_averaged = np.zeros(n_combos)
    theoretical_averaged = np.zeros(n_combos)
    elev_final = np.zeros(n_combos)
    lat_final = np.zeros(n_combos)
    azim_final = np.zeros(n_combos)
    
    idx = 0
    for i_lat, lat in enumerate(latitude_grid):
        for i_azim, azim in enumerate(azimuth_grid):
            for i_elev, elev in enumerate(elevation_grid):
                mask = (elev_labels == elev) & (lat_labels == lat) & (azim_labels == azim)
                mf_averaged[idx] = mf_values[mask].mean()
                vtec_averaged[idx] = vtec_values[mask].mean()
                theoretical_averaged[idx] = theoretical_mf[mask].mean()
                elev_final[idx] = elev
                lat_final[idx] = lat
                azim_final[idx] = azim
                idx += 1
    
    # Calculate errors
    mf_error = mf_averaged - theoretical_averaged
    mf_rel_error = 100 * mf_error / theoretical_averaged
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)
    
    # === 3D Scatter Plot: MF in parameter space ===
    ax = fig.add_subplot(gs[0:2, 0:2], projection='3d')
    scatter = ax.scatter(elev_final, lat_final, azim_final, c=mf_averaged, 
                        cmap='viridis', s=50, alpha=0.6)
    ax.set_xlabel('Elevation (°)', fontsize=10)
    ax.set_ylabel('Latitude (°)', fontsize=10)
    ax.set_zlabel('Azimuth (°)', fontsize=10)
    ax.set_title('MF in 3D Parameter Space\n(Elevation × Latitude × Azimuth)', 
                 fontweight='bold', fontsize=11)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('MF', fontsize=9)
    
    # === 3D Scatter Plot: MF Error ===
    ax = fig.add_subplot(gs[0:2, 2:4], projection='3d')
    vmax = max(abs(mf_error.min()), abs(mf_error.max()))
    scatter = ax.scatter(elev_final, lat_final, azim_final, c=mf_error, 
                        cmap='RdBu_r', s=50, alpha=0.6, vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Elevation (°)', fontsize=10)
    ax.set_ylabel('Latitude (°)', fontsize=10)
    ax.set_zlabel('Azimuth (°)', fontsize=10)
    ax.set_title('MF Error in 3D Parameter Space\n(Predicted - Theoretical)', 
                 fontweight='bold', fontsize=11)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('ΔMF', fontsize=9)
    
    # === Heatmap: MF vs Elevation & Latitude (averaged over azimuth) ===
    ax = fig.add_subplot(gs[2, 0])
    mf_elev_lat = np.zeros((len(latitude_grid), len(elevation_grid)))
    for i_lat, lat in enumerate(latitude_grid):
        for i_elev, elev in enumerate(elevation_grid):
            mask = (elev_final == elev) & (lat_final == lat)
            mf_elev_lat[i_lat, i_elev] = mf_averaged[mask].mean()
    
    im = ax.imshow(mf_elev_lat, aspect='auto', cmap='viridis', origin='lower',
                   extent=[elevation_grid[0], elevation_grid[-1], 
                          latitude_grid[0], latitude_grid[-1]])
    ax.set_xlabel('Elevation (°)', fontsize=10)
    ax.set_ylabel('Latitude (°)', fontsize=10)
    ax.set_title('MF: Elev × Lat\n(avg over azimuth)', fontweight='bold', fontsize=10)
    plt.colorbar(im, ax=ax)
    
    # === Heatmap: MF vs Elevation & Azimuth (averaged over latitude) ===
    ax = fig.add_subplot(gs[2, 1])
    mf_elev_azim = np.zeros((len(azimuth_grid), len(elevation_grid)))
    for i_azim, azim in enumerate(azimuth_grid):
        for i_elev, elev in enumerate(elevation_grid):
            mask = (elev_final == elev) & (azim_final == azim)
            mf_elev_azim[i_azim, i_elev] = mf_averaged[mask].mean()
    
    im = ax.imshow(mf_elev_azim, aspect='auto', cmap='viridis', origin='lower',
                   extent=[elevation_grid[0], elevation_grid[-1], 
                          azimuth_grid[0], azimuth_grid[-1]])
    ax.set_xlabel('Elevation (°)', fontsize=10)
    ax.set_ylabel('Azimuth (°)', fontsize=10)
    ax.set_title('MF: Elev × Azim\n(avg over latitude)', fontweight='bold', fontsize=10)
    plt.colorbar(im, ax=ax)
    
    # === Heatmap: MF vs Latitude & Azimuth (averaged over elevation) ===
    ax = fig.add_subplot(gs[2, 2])
    mf_lat_azim = np.zeros((len(azimuth_grid), len(latitude_grid)))
    for i_azim, azim in enumerate(azimuth_grid):
        for i_lat, lat in enumerate(latitude_grid):
            mask = (lat_final == lat) & (azim_final == azim)
            mf_lat_azim[i_azim, i_lat] = mf_averaged[mask].mean()
    
    im = ax.imshow(mf_lat_azim, aspect='auto', cmap='viridis', origin='lower',
                   extent=[latitude_grid[0], latitude_grid[-1], 
                          azimuth_grid[0], azimuth_grid[-1]])
    ax.set_xlabel('Latitude (°)', fontsize=10)
    ax.set_ylabel('Azimuth (°)', fontsize=10)
    ax.set_title('MF: Lat × Azim\n(avg over elevation)', fontweight='bold', fontsize=10)
    plt.colorbar(im, ax=ax)
    
    # === Variance Analysis ===
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    
    # Calculate variance contributions
    mf_var_total = np.var(mf_averaged)
    
    # Variance by elevation (marginalizing others)
    mf_by_elev = [mf_averaged[elev_final == e].mean() for e in elevation_grid]
    var_elev = np.var(mf_by_elev)
    
    # Variance by latitude
    mf_by_lat = [mf_averaged[lat_final == lat].mean() for lat in latitude_grid]
    var_lat = np.var(mf_by_lat)
    
    # Variance by azimuth
    mf_by_azim = [mf_averaged[azim_final == az].mean() for az in azimuth_grid]
    var_azim = np.var(mf_by_azim)
    
    var_text = "Variance Decomposition\n" + "="*30 + "\n\n"
    var_text += f"Total MF variance: {mf_var_total:.4f}\n\n"
    var_text += f"Main Effects:\n"
    var_text += f"  Elevation: {var_elev:.4f}\n"
    var_text += f"    ({var_elev/mf_var_total*100:.1f}% of total)\n"
    var_text += f"  Latitude: {var_lat:.4f}\n"
    var_text += f"    ({var_lat/mf_var_total*100:.1f}% of total)\n"
    var_text += f"  Azimuth: {var_azim:.4f}\n"
    var_text += f"    ({var_azim/mf_var_total*100:.1f}% of total)\n\n"
    
    var_text += f"Expected:\n"
    var_text += f"  Elevation: dominant\n"
    var_text += f"  Latitude: small (physical)\n"
    var_text += f"  Azimuth: ~0 (spurious)\n"
    
    ax.text(0.05, 0.95, var_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # === Distribution plots ===
    ax = fig.add_subplot(gs[3, 0])
    ax.hist(mf_error, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('MF Error (Pred - Theory)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Error Distribution', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[3, 1])
    ax.hist(mf_rel_error, bins=30, alpha=0.7, edgecolor='black', color='C1')
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Relative Error (%)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Relative Error Distribution', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # === Interaction effects ===
    ax = fig.add_subplot(gs[3, 2])
    # Plot MF range for each elevation across all lat/azim combinations
    mf_ranges = []
    for elev in elevation_grid:
        mask = elev_final == elev
        mf_ranges.append([mf_averaged[mask].min(), mf_averaged[mask].max()])
    mf_ranges = np.array(mf_ranges)
    
    ax.plot(elevation_grid, mf_ranges[:, 0], 'o-', label='Min MF', linewidth=2)
    ax.plot(elevation_grid, mf_ranges[:, 1], 's-', label='Max MF', linewidth=2)
    ax.fill_between(elevation_grid, mf_ranges[:, 0], mf_ranges[:, 1], alpha=0.3)
    ax.set_xlabel('Elevation (°)', fontsize=10)
    ax.set_ylabel('MF Range', fontsize=10)
    ax.set_title('MF Variability at Each Elevation\n(across all lat/azim)', 
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # === Statistics summary ===
    ax = fig.add_subplot(gs[3, 3])
    ax.axis('off')
    
    stats_text = "Statistics Summary\n" + "="*30 + "\n\n"
    stats_text += f"MF Statistics:\n"
    stats_text += f"  Range: {mf_averaged.min():.3f} - {mf_averaged.max():.3f}\n"
    stats_text += f"  Mean: {mf_averaged.mean():.3f}\n"
    stats_text += f"  Std: {mf_averaged.std():.3f}\n\n"
    
    stats_text += f"Accuracy:\n"
    stats_text += f"  MAE: {np.abs(mf_error).mean():.3f}\n"
    stats_text += f"  RMSE: {np.sqrt((mf_error**2).mean()):.3f}\n"
    stats_text += f"  Mean bias: {mf_error.mean():.3f}\n"
    stats_text += f"  Rel error: {mf_rel_error.mean():.2f}%\n\n"
    
    stats_text += f"VTEC Consistency:\n"
    stats_text += f"  Mean: {vtec_averaged.mean():.2f} TECU\n"
    stats_text += f"  Std: {vtec_averaged.std():.2f} TECU\n"
    stats_text += f"  Range: {vtec_averaged.max()-vtec_averaged.min():.2f}\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('High-Dimensional MF Analysis: Full 3D Parameter Space', 
                 fontsize=14, fontweight='bold')
    
    plot_path = output_dir / 'geomnet_high_dimensional_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nHigh-dimensional analysis plot saved to: {plot_path}")
    
    # Print detailed summary
    print("\n" + "="*70)
    print("HIGH-DIMENSIONAL ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nParameter Space:")
    print(f"  Elevation: {len(elevation_grid)} levels")
    print(f"  Latitude: {len(latitude_grid)} levels")
    print(f"  Azimuth: {len(azimuth_grid)} levels")
    print(f"  Total combinations: {len(elevation_grid) * len(latitude_grid) * len(azimuth_grid)}")
    
    print(f"\nVariance Decomposition (main effects):")
    print(f"  Total variance: {mf_var_total:.4f}")
    print(f"  Elevation contribution: {var_elev:.4f} ({var_elev/mf_var_total*100:.1f}%)")
    print(f"  Latitude contribution: {var_lat:.4f} ({var_lat/mf_var_total*100:.1f}%)")
    print(f"  Azimuth contribution: {var_azim:.4f} ({var_azim/mf_var_total*100:.1f}%)")
    
    # Interaction analysis
    print(f"\nInteraction Analysis:")
    for elev in elevation_grid:
        mask = elev_final == elev
        mf_range = mf_averaged[mask].max() - mf_averaged[mask].min()
        print(f"  At {elev:.0f}°: MF range = {mf_range:.4f} (variation across lat/azim)")
    
    print(f"\nPhysical Interpretation:")
    if var_azim / mf_var_total < 0.01:
        print(f"  ✓ Azimuth dependence is negligible (<1% of variance)")
    else:
        print(f"  ⚠ Azimuth contributes {var_azim/mf_var_total*100:.1f}% - may indicate spurious dependence")
    
    if var_elev / mf_var_total > 0.90:
        print(f"  ✓ Elevation is dominant factor (>{var_elev/mf_var_total*100:.1f}%)")
    else:
        print(f"  ⚠ Elevation only accounts for {var_elev/mf_var_total*100:.1f}% of variance")
    
    print("="*70)
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Validate FactorizedSTEC model with elevation angle tests')
    parser.add_argument('--exp_path', type=str, required=True,
                       help='Path to experiment folder containing trained model')
    parser.add_argument('--min_elev', type=float, default=10.0,
                       help='Minimum elevation angle in degrees (default: 10)')
    parser.add_argument('--max_elev', type=float, default=90.0,
                       help='Maximum elevation angle in degrees (default: 90)')
    parser.add_argument('--num_angles', type=int, default=9,
                       help='Number of elevation angles to test (default: 9)')
    parser.add_argument('--num_observations', type=int, default=100,
                       help='Number of real observations to load from test.h5 (default: 100)')
    parser.add_argument('--test_h5_path', type=str, default='data/test.h5',
                       help='Path to test.h5 dataset (default: data/test.h5)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: exp_path/validation)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.exp_path, 'validation')
    
    print("="*70)
    print("FactorizedSTEC Model Validation Script")
    print("="*70)
    print(f"Experiment: {args.exp_path}")
    print(f"Test dataset: {args.test_h5_path}")
    print(f"Elevation range: {args.min_elev}° to {args.max_elev}°")
    print(f"Number of angles: {args.num_angles}")
    print(f"Number of observations: {args.num_observations}")
    print("="*70)
    
    # Load model
    print("\n[1/5] Loading model and configuration...")
    model, config, feature_splitter = load_model_and_config(args.exp_path)
    
    # Load real observations
    print("\n[2/5] Loading real observations from test dataset...")
    raw_observations, _ = load_real_observations(
        args.test_h5_path, 
        num_samples=args.num_observations,
        elevation_range=(args.min_elev, args.max_elev)
    )
    
    # Create test samples with modified elevations
    print("\n[3/5] Creating test samples with varying elevation angles...")
    elevation_angles = np.linspace(args.min_elev, args.max_elev, args.num_angles)
    test_data, elev_labels, ground_truth = create_test_samples_from_observations(
        raw_observations, elevation_angles, config, feature_splitter
    )
    print(f"Generated {len(test_data)} test samples ({len(raw_observations)} obs × {args.num_angles} elevations)")
    
    # Run inference
    print("\n[4/5] Running inference...")
    results = run_inference(model, test_data, elev_labels)
    print(f"Inference complete!")
    
    # Plot results
    print("\n[5/5] Creating validation plots...")
    plot_results(elev_labels, results, ground_truth, args.output_dir)
    
    # Additional 2D analysis of GeomNet
    print("\n[BONUS] Running 2D GeomNet analysis (elevation × latitude)...")
    analyze_geom_net_2d(model, config, feature_splitter, raw_observations, args.output_dir)
    
    # Azimuth dependence analysis
    print("\n[BONUS] Running azimuth dependence analysis...")
    analyze_azimuth_dependence(model, config, feature_splitter, raw_observations, args.output_dir,
                              num_azimuth=18, elevation_deg=30.0)
    
    # High-dimensional analysis
    print("\n[BONUS] Running high-dimensional analysis (Elevation × Latitude × Azimuth)...")
    analyze_high_dimensional_mf(model, config, feature_splitter, raw_observations, args.output_dir)
    
    print(f"\n{'='*70}")
    print("Validation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
