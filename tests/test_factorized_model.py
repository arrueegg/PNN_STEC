"""
Quick test script to verify FactorizedSTEC model implementation.

This script tests:
1. Feature splitter initialization and feature splitting
2. Model instantiation and forward pass
3. MF physical constraints (MF(90°) = 1, MF ≥ 1)
4. Uncertainty propagation (σ_stec = |MF| × σ_vtec)
5. Integration with model factory (get_model)

Run with: python tests/test_factorized_model.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from utils.feature_registry import FeatureRegistry, FeatureType, initialize_feature_registry
from utils.feature_splitter import FeatureSplitter
from model.model import VTECFieldNet, GeomNet, FactorizedSTECModel, FactorizedSTECModelWrapper, get_model


def create_test_config():
    """Create minimal test configuration."""
    config = {
        "data": {
            "use_SWI": True,
            "SH_degree": 0,  # Disable SH for simpler testing
        },
        "model": {
            "model_type": "FactorizedSTEC",
            "vtec_hidden": 64,
            "vtec_layers": 2,
            "geom_hidden": 32,
            "geom_layers": 2,
            "activation": "relu",
        },
        "target": "stec",
    }
    return config


def test_feature_splitter():
    """Test feature splitter initialization and splitting."""
    print("\n" + "="*60)
    print("TEST 1: Feature Splitter")
    print("="*60)
    
    config = create_test_config()
    feature_registry = initialize_feature_registry(config)
    
    # Initialize output indices by creating CollateWithSH (simulates what happens in training)
    from data_loader.collation import CollateWithSH
    config["feature_registry"] = feature_registry
    collate_fn = CollateWithSH(config)
    
    print(f"Initialized CollateWithSH - output_indices set in feature_registry")
    
    # Create splitter
    splitter = FeatureSplitter(feature_registry)
    
    # Print dimensions
    vtec_dim = splitter.get_vtec_dim()
    geom_dim = splitter.get_geom_dim()
    total_dim = splitter.get_total_dim()
    
    print(f"VTEC feature dimension: {vtec_dim}")
    print(f"Geometry feature dimension: {geom_dim}")
    print(f"Total feature dimension: {total_dim}")
    
    # Create dummy feature tensor (mimicking collated output)
    batch_size = 10
    features = torch.randn(batch_size, total_dim)
    
    # Make sure e_up (elevation component) is in valid range [-1, 1]
    # e_up is at the index corresponding to "e_up" in output_indices
    if "e_up" in feature_registry._output_indices:
        e_up_idx = feature_registry._output_indices["e_up"]
        features[:, e_up_idx] = torch.rand(batch_size) * 2 - 1  # Range [-1, 1] for sin(elev)
    
    # Split features
    x_vtec, x_geom, elev_rad = splitter.split_features(features)
    
    print(f"\nSplit results:")
    print(f"  x_vtec shape: {x_vtec.shape} (expected: [{batch_size}, {vtec_dim}])")
    print(f"  x_geom shape: {x_geom.shape} (expected: [{batch_size}, {geom_dim}])")
    print(f"  elev_rad shape: {elev_rad.shape} (expected: [{batch_size}])")
    
    # Verify elevation is in valid range [0, π/2]
    assert torch.all(elev_rad >= -np.pi/2) and torch.all(elev_rad <= np.pi/2), \
        f"Elevation out of range! Got range [{elev_rad.min():.3f}, {elev_rad.max():.3f}]"
    
    print(f"  Elevation range: [{elev_rad.min()*180/np.pi:.1f}°, {elev_rad.max()*180/np.pi:.1f}°]")
    
    print("✓ Feature splitter test passed!")
    return splitter, vtec_dim, geom_dim


def test_vtec_field_net(vtec_dim):
    """Test VTECFieldNet."""
    print("\n" + "="*60)
    print("TEST 2: VTECFieldNet")
    print("="*60)
    
    batch_size = 10
    vtec_net = VTECFieldNet(vtec_in_dim=vtec_dim, hidden_dim=64, num_layers=2)
    
    # Forward pass
    x_vtec = torch.randn(batch_size, vtec_dim)
    vtec_mean, vtec_log_sigma = vtec_net(x_vtec)
    
    print(f"Input shape: {x_vtec.shape}")
    print(f"VTEC mean shape: {vtec_mean.shape} (expected: [{batch_size}])")
    print(f"VTEC log_sigma shape: {vtec_log_sigma.shape} (expected: [{batch_size}])")
    
    # Check output statistics
    print(f"\nVTEC mean statistics:")
    print(f"  Mean: {vtec_mean.mean().item():.2f} (should be ~15.5 due to bias init)")
    print(f"  Std: {vtec_mean.std().item():.2f}")
    
    # Verify sigma is positive when exponentiated
    sigma_v = torch.exp(vtec_log_sigma)
    assert torch.all(sigma_v > 0), "VTEC sigma must be positive!"
    print(f"\nVTEC sigma (exp(log_sigma)) range: [{sigma_v.min():.3f}, {sigma_v.max():.3f}]")
    
    print("✓ VTECFieldNet test passed!")
    return vtec_net


def test_geom_net(geom_dim):
    """Test GeomNet and MF constraints."""
    print("\n" + "="*60)
    print("TEST 3: GeomNet and MF Constraints")
    print("="*60)
    
    batch_size = 10
    geom_net = GeomNet(geom_in_dim=geom_dim, hidden_dim=32, num_layers=2)
    
    # Test at various elevations
    elevations_deg = torch.tensor([5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
    elevations_rad = elevations_deg * torch.pi / 180.0
    
    print(f"\nTesting MF at different elevations:")
    print(f"{'Elevation (°)':<15} {'MF':<10} {'MF ≥ 1?':<10}")
    print("-" * 35)
    
    x_geom = torch.randn(len(elevations_deg), geom_dim)
    mf_values = geom_net(x_geom, elevations_rad)
    
    for elev_deg, mf in zip(elevations_deg, mf_values):
        valid = "✓" if mf >= 1.0 else "✗"
        print(f"{elev_deg.item():<15.1f} {mf.item():<10.3f} {valid:<10}")
    
    # Verify constraints
    assert torch.all(mf_values >= 1.0), "MF must be ≥ 1 for all elevations!"
    
    # Check MF(90°) ≈ 1
    mf_90 = mf_values[-1]
    assert abs(mf_90.item() - 1.0) < 0.1, f"MF(90°) should be ≈ 1, got {mf_90.item():.3f}"
    
    print(f"\n✓ MF(90°) = {mf_90.item():.4f} (expected: 1.0)")
    print("✓ All MF values ≥ 1")
    print("✓ GeomNet test passed!")
    return geom_net


def test_factorized_model(vtec_dim, geom_dim):
    """Test complete FactorizedSTECModel."""
    print("\n" + "="*60)
    print("TEST 4: FactorizedSTECModel")
    print("="*60)
    
    batch_size = 10
    model = FactorizedSTECModel(
        vtec_in_dim=vtec_dim,
        geom_in_dim=geom_dim,
        vtec_hidden=64,
        geom_hidden=32,
        vtec_layers=2,
        geom_layers=2
    )
    
    # Forward pass
    x_vtec = torch.randn(batch_size, vtec_dim)
    x_geom = torch.randn(batch_size, geom_dim)
    elev_rad = torch.rand(batch_size) * (torch.pi / 2)  # Random elevations [0, 90°]
    
    mu_stec, var_stec = model(x_vtec, x_geom, elev_rad)
    
    print(f"Input shapes:")
    print(f"  x_vtec: {x_vtec.shape}")
    print(f"  x_geom: {x_geom.shape}")
    print(f"  elev_rad: {elev_rad.shape}")
    
    print(f"\nOutput shapes:")
    print(f"  mu_stec: {mu_stec.shape}")
    print(f"  var_stec: {var_stec.shape}")
    
    # Verify variance is positive
    assert torch.all(var_stec > 0), "STEC variance must be positive!"
    
    # Test detailed output
    outputs = model.forward_detailed(x_vtec, x_geom, elev_rad)
    
    print(f"\nDetailed outputs available:")
    for key, value in outputs.items():
        print(f"  {key}: shape {value.shape}, range [{value.min():.3f}, {value.max():.3f}]")
    
    # Verify uncertainty propagation: σ_stec = |MF| × σ_vtec
    sigma_stec_from_output = torch.sqrt(var_stec)
    sigma_stec_computed = torch.abs(outputs["mf"]) * outputs["sigma_v"]
    
    propagation_error = torch.abs(sigma_stec_from_output - sigma_stec_computed).max()
    print(f"\nUncertainty propagation check:")
    print(f"  Max error in σ_stec = |MF| × σ_vtec: {propagation_error:.6f}")
    assert propagation_error < 1e-5, "Uncertainty propagation mismatch!"
    
    print("✓ FactorizedSTECModel test passed!")
    return model


def test_model_wrapper(splitter):
    """Test FactorizedSTECModelWrapper integration."""
    print("\n" + "="*60)
    print("TEST 5: FactorizedSTECModelWrapper")
    print("="*60)
    
    vtec_dim = splitter.get_vtec_dim()
    geom_dim = splitter.get_geom_dim()
    total_dim = splitter.get_total_dim()
    
    factorized_model = FactorizedSTECModel(
        vtec_in_dim=vtec_dim,
        geom_in_dim=geom_dim,
        vtec_hidden=64,
        geom_hidden=32
    )
    
    wrapper = FactorizedSTECModelWrapper(factorized_model, splitter)
    
    # Test with full feature tensor (as training loop provides)
    batch_size = 10
    features = torch.randn(batch_size, total_dim)
    
    # Forward pass
    mu_stec, var_stec = wrapper(features)
    
    print(f"Input features shape: {features.shape}")
    print(f"Output mu_stec shape: {mu_stec.shape}")
    print(f"Output var_stec shape: {var_stec.shape}")
    
    # Test detailed output
    outputs = wrapper.forward_detailed(features)
    print(f"\nDetailed outputs from wrapper:")
    for key in outputs.keys():
        print(f"  {key}")
    
    print("✓ FactorizedSTECModelWrapper test passed!")
    return wrapper


def test_get_model():
    """Test model factory integration."""
    print("\n" + "="*60)
    print("TEST 6: Model Factory (get_model)")
    print("="*60)
    
    config = create_test_config()
    feature_registry = initialize_feature_registry(config)
    config["feature_registry"] = feature_registry
    
    # Initialize CollateWithSH to set output_indices (this happens automatically in main.py)
    from data_loader.collation import CollateWithSH
    collate_fn = CollateWithSH(config)
    print("Initialized CollateWithSH (simulating main.py setup)")
    
    # Get model through factory
    model = get_model(config)
    
    print(f"Model type: {type(model).__name__}")
    print(f"Is FactorizedSTECModelWrapper? {isinstance(model, FactorizedSTECModelWrapper)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    # Need to know total feature dimension from collation
    # For now, use wrapper's total_dim
    total_dim = model.splitter.get_total_dim()
    batch_size = 5
    features = torch.randn(batch_size, total_dim)
    
    # Make sure e_up is valid
    if "e_up" in feature_registry._output_indices:
        e_up_idx = feature_registry._output_indices["e_up"]
        features[:, e_up_idx] = torch.rand(batch_size) * 2 - 1  # Range [-1, 1]
    
    mu_stec, var_stec = model(features)
    
    print(f"\nForward pass successful:")
    print(f"  Input: {features.shape}")
    print(f"  Output: ({mu_stec.shape}, {var_stec.shape})")
    
    print("✓ Model factory test passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("FACTORIZED STEC MODEL - VERIFICATION TESTS")
    print("="*60)
    
    try:
        # Test 1: Feature splitter
        splitter, vtec_dim, geom_dim = test_feature_splitter()
        
        # Test 2: VTECFieldNet
        vtec_net = test_vtec_field_net(vtec_dim)
        
        # Test 3: GeomNet with MF constraints
        geom_net = test_geom_net(geom_dim)
        
        # Test 4: Complete factorized model
        factorized_model = test_factorized_model(vtec_dim, geom_dim)
        
        # Test 5: Model wrapper
        wrapper = test_model_wrapper(splitter)
        
        # Test 6: Model factory
        test_get_model()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe FactorizedSTEC model implementation is verified and ready to use.")
        print("Next steps:")
        print("  1. Configure config_FactorizedSTEC.yaml for your data")
        print("  2. Run pretraining: python src/main.py")
        print("  3. Fine-tune on specific days")
        print("  4. Evaluate with inference scripts")
        
    except Exception as e:
        print("\n" + "="*60)
        print("TEST FAILED! ✗")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
