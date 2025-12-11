#!/usr/bin/env python
"""
Quick test script to validate GeomNet's learned dependencies on elevation, station latitude, and azimuth.

This script tests the IMPROVED GeomNet architecture with tanh activation that should allow
latitude and azimuth dependencies to emerge.
"""

import numpy as np
import torch
import sys
import yaml
from pathlib import Path

sys.path.append('src')
from model.model import FactorizedSTECModelWrapper
from utils.feature_registry import FeatureRegistry, initialize_feature_registry

def create_synthetic_features(n_samples, sta_lat=45.0, azimuth=0.0, elevation=30.0):
    """
    Create synthetic feature tensors for testing GeomNet.
    
    Returns a tensor with proper feature ordering matching the collated output.
    """
    # Simplified feature creation - just the key geometry features
    # In reality this would go through the full CollateWithSH transformation
    
    features = []
    for _ in range(n_samples):
        sample = []
        
        # Temporal (dummy values)
        sample.extend([0.5, 0.0, 1.0, 0.0])  # year, doy_sin/cos/norm
        sample.extend([0.5, 0.5, 0.5])  # sod_sin/cos/norm  
        sample.extend([0.5, 0.5, 0.5])  # local_time_sin/cos/norm
        
        # Station location (KEY: varies with sta_lat)
        sample.append(sta_lat / 90.0)  # lat_sta_norm
        sample.append(0.0)  # lon_sta_norm (keep constant)
        sample.append(sta_lat / 90.0)  # sm_lat_sta_norm (simplified)
        sample.append(0.0)  # sm_lon_sta_norm
        
        # Direction vector (KEY: varies with elevation and azimuth)
        elev_rad = np.deg2rad(elevation)
        azim_rad = np.deg2rad(azimuth)
        e_up = np.sin(elev_rad)
        e_east = np.cos(elev_rad) * np.sin(azim_rad)
        e_north = np.cos(elev_rad) * np.cos(azim_rad)
        sample.extend([e_up, e_east, e_north])
        
        # IPP location (dummy - goes to VTEC net)
        sample.extend([0.5, 0.0, 0.5, 0.0])  # lat/lon/sm_lat/sm_lon
        
        # SH embeddings (dummy - 25 dims each for degree 5)
        sample.extend([0.0] * 25)  # sh_sta_geo
        sample.extend([0.0] * 25)  # sh_ipp_geo
        sample.extend([0.0] * 25)  # sh_sta_sm
        sample.extend([0.0] * 25)  # sh_ipp_sm
        
        # SWI (dummy)
        sample.extend([0.2, 0.15, 0.5, 0.05, 0.025, 0.33])
        
        features.append(sample)
    
    return torch.tensor(features, dtype=torch.float32)


def test_latitude_dependence(model):
    """Test if MF varies with station latitude at fixed elevation/azimuth."""
    print("\n" + "="*70)
    print("TEST 1: Station Latitude Dependence")
    print("="*70)
    print("Fixed: elevation=30°, azimuth=0° (north)")
    print("Varying: station latitude from -60° to +60°")
    
    latitudes = np.linspace(-60, 60, 13)
    mf_values = []
    
    for lat in latitudes:
        x = create_synthetic_features(1, sta_lat=lat, azimuth=0.0, elevation=30.0)
        with torch.no_grad():
            output = model.forward_detailed(x)
            mf = output['mf'].item()
            mf_values.append(mf)
    
    mf_values = np.array(mf_values)
    variation = mf_values.max() - mf_values.min()
    
    print(f"\nResults:")
    print(f"  MF range: {mf_values.min():.4f} to {mf_values.max():.4f}")
    print(f"  MF variation: {variation:.4f} ({variation/mf_values.mean()*100:.2f}%)")
    print(f"  MF at equator (0°): {mf_values[len(latitudes)//2]:.4f}")
    print(f"  MF at high lat (60°): {mf_values[-1]:.4f}")
    
    if variation < 0.001:
        print(f"  ✗ ISSUE: Latitude variation is negligible (< 0.1%)")
    elif variation < 0.01:
        print(f"  ⚠ WARNING: Latitude variation is small (< 1%)")
    else:
        print(f"  ✓ GOOD: Latitude dependence detected")
    
    return variation


def test_azimuth_dependence(model):
    """Test if MF varies with azimuth at fixed elevation/station_lat."""
    print("\n" + "="*70)
    print("TEST 2: Azimuth Dependence")
    print("="*70)
    print("Fixed: elevation=30°, station_lat=45°")
    print("Varying: azimuth from 0° to 360°")
    
    azimuths = np.linspace(0, 360, 19)[:-1]  # 0-340 in 20° steps
    mf_values = []
    
    for azim in azimuths:
        x = create_synthetic_features(1, sta_lat=45.0, azimuth=azim, elevation=30.0)
        with torch.no_grad():
            output = model.forward_detailed(x)
            mf = output['mf'].item()
            mf_values.append(mf)
    
    mf_values = np.array(mf_values)
    variation = mf_values.max() - mf_values.min()
    
    print(f"\nResults:")
    print(f"  MF range: {mf_values.min():.4f} to {mf_values.max():.4f}")
    print(f"  MF variation: {variation:.4f} ({variation/mf_values.mean()*100:.2f}%)")
    print(f"  MF at azimuth 0° (north): {mf_values[0]:.4f}")
    print(f"  MF at azimuth 90° (east): {mf_values[len(azimuths)//4]:.4f}")
    print(f"  MF at azimuth 180° (south): {mf_values[len(azimuths)//2]:.4f}")
    
    # Physical expectation: MF should NOT strongly depend on azimuth (thin shell)
    # But model COULD learn small corrections for asymmetries
    if variation < 0.01:
        print(f"  ✓ GOOD: Azimuth variation is small (as expected for thin shell)")
    elif variation < 0.05:
        print(f"  ⚠ NOTICE: Model learned moderate azimuth dependence")
    else:
        print(f"  ✗ WARNING: Strong azimuth dependence (> 5%) - may indicate issues")
    
    return variation


def test_elevation_dependence(model):
    """Test if MF varies correctly with elevation."""
    print("\n" + "="*70)
    print("TEST 3: Elevation Dependence")
    print("="*70)
    print("Fixed: station_lat=45°, azimuth=0°")
    print("Varying: elevation from 10° to 90°")
    
    elevations = np.linspace(10, 90, 9)
    mf_values = []
    mf_theoretical = 1.0 / np.sin(np.deg2rad(elevations))
    
    for elev in elevations:
        x = create_synthetic_features(1, sta_lat=45.0, azimuth=0.0, elevation=elev)
        with torch.no_grad():
            output = model.forward_detailed(x)
            mf = output['mf'].item()
            mf_values.append(mf)
    
    mf_values = np.array(mf_values)
    errors = np.abs(mf_values - mf_theoretical)
    relative_errors = errors / mf_theoretical * 100
    
    print(f"\nResults:")
    print(f"  MF at 10°: {mf_values[0]:.3f} (theoretical: {mf_theoretical[0]:.3f}, error: {relative_errors[0]:.1f}%)")
    print(f"  MF at 30°: {mf_values[2]:.3f} (theoretical: {mf_theoretical[2]:.3f}, error: {relative_errors[2]:.1f}%)")
    print(f"  MF at 90°: {mf_values[-1]:.3f} (theoretical: {mf_theoretical[-1]:.3f}, error: {relative_errors[-1]:.1f}%)")
    print(f"  Mean absolute error: {errors.mean():.3f}")
    print(f"  Mean relative error: {relative_errors.mean():.1f}%")
    
    if np.abs(mf_values[-1] - 1.0) < 0.01:
        print(f"  ✓ GOOD: MF ≈ 1 at zenith (90°)")
    else:
        print(f"  ✗ ISSUE: MF ≠ 1 at zenith (constraint violated)")
    
    return relative_errors.mean()


def main():
    print("="*70)
    print("GeomNet Dependency Test - Improved Architecture")
    print("="*70)
    print("Testing if the improved GeomNet (with tanh) learns:")
    print("  1. Station latitude dependence")
    print("  2. Azimuth dependence (should be minimal)")
    print("  3. Correct elevation dependence")
    
    # Load trained model
    exp_path = "experiments/Pretrain_STEC_FactorizedSTEC_h1024_l4_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_lw1e-1_SWI"
    
    print(f"\nLoading model from: {exp_path}")
    
    # Load config
    config_path = Path(exp_path) / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize feature registry
    config = initialize_feature_registry(config)
    
    # Load model (simplified - just for architecture testing)
    from model.model import load_model_and_config
    
    print("\nNOTE: This test uses synthetic features, so absolute MF values may differ")
    print("      from real data. We're testing for PRESENCE of dependencies, not accuracy.")
    
    print("\n⚠ IMPORTANT: The current trained model uses the OLD architecture (softplus).")
    print("   To see improvements, you need to RETRAIN with the new tanh architecture!")
    
    lat_var = test_latitude_dependence(None)  # Would need actual model
    azim_var = test_azimuth_dependence(None)
    elev_error = test_elevation_dependence(None)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Station latitude variation: {lat_var:.4f}")
    print(f"Azimuth variation: {azim_var:.4f}")
    print(f"Elevation error: {elev_error:.1f}%")
    print("\nTo enable latitude/azimuth dependencies:")
    print("  1. Retrain model with updated GeomNet (tanh architecture)")
    print("  2. Ensure sufficient training on diverse geographic locations")
    print("  3. Re-run this validation script")
    print("="*70)


if __name__ == '__main__':
    print("\nThis is a template for testing GeomNet dependencies.")
    print("The actual trained model still uses the old architecture.")
    print("\nTo fully test improvements, you need to:")
    print("  1. Retrain the FactorizedSTEC model")
    print("  2. Run the full validation: python scripts/validate_factorized_model.py")
    print("\nThe validation script has been updated to test:")
    print("  - Station latitude dependence (not IPP latitude)")
    print("  - Azimuth dependence")
    print("  - Elevation dependence")
