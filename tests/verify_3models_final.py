"""
Final verification of three key models:
1. BayesianResNetSTEC
2. AttentionMLP_BNN_NLL
3. FactorizedSTEC

Checks:
- Model architecture correctness
- Forward pass functionality
- Output format (mean, variance)
- Variance positivity
- Parameter counts
- Bayesian component identification
"""

import torch
import yaml
import sys
sys.path.insert(0, 'src')

from model.model import get_model
from utils.feature_registry import FeatureRegistry, initialize_feature_registry

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def count_bayesian_parameters(model):
    """Count parameters in Bayesian layers (BayesLinear)"""
    import torchbnn as bnn
    bayes_params = 0
    for module in model.modules():
        if isinstance(module, bnn.BayesLinear):
            bayes_params += sum(p.numel() for p in module.parameters())
    return bayes_params

def verify_model(config_path, model_name):
    """Verify a single model"""
    print(f"\n{'='*80}")
    print(f"VERIFYING: {model_name}")
    print(f"{'='*80}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize feature registry
    feature_registry = initialize_feature_registry(config)
    
    # Build model
    print(f"\n📦 Building model from {config_path}...")
    model = get_model(config)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    bayesian_params = count_bayesian_parameters(model)
    bayesian_pct = (bayesian_params / total_params * 100) if total_params > 0 else 0
    
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Bayesian parameters:  {bayesian_params:,} ({bayesian_pct:.1f}%)")
    
    # Test forward pass
    print(f"\n🧪 Testing forward pass...")
    model.eval()
    
    batch_size = 32
    n_features = feature_registry.get_total_features()
    x_dummy = torch.randn(batch_size, n_features)
    
    with torch.no_grad():
        mean, variance = model(x_dummy)
    
    # Verify output shapes
    assert mean.shape == (batch_size,), f"❌ Mean shape {mean.shape} != ({batch_size},)"
    assert variance.shape == (batch_size,), f"❌ Variance shape {variance.shape} != ({batch_size},)"
    print(f"   ✅ Output shapes correct: mean={mean.shape}, variance={variance.shape}")
    
    # Verify variance positivity
    assert (variance > 0).all(), f"❌ Negative variances detected!"
    print(f"   ✅ All variances positive")
    
    # Check variance range
    var_min = variance.min().item()
    var_max = variance.max().item()
    var_mean = variance.mean().item()
    print(f"   Variance range: [{var_min:.4f}, {var_max:.4f}], mean={var_mean:.4f}")
    
    # Check mean range
    mean_min = mean.min().item()
    mean_max = mean.max().item()
    mean_mean = mean.mean().item()
    print(f"   Mean range: [{mean_min:.4f}, {mean_max:.4f}], mean={mean_mean:.4f}")
    
    # Test multiple forward passes for epistemic uncertainty (Bayesian models)
    if bayesian_params > 0:
        print(f"\n🎲 Testing epistemic uncertainty (5 forward passes)...")
        model.train()  # Enable dropout/sampling
        predictions = []
        for _ in range(5):
            with torch.no_grad():
                mean_sample, _ = model(x_dummy[:5])  # Test on 5 samples
                predictions.append(mean_sample)
        
        predictions = torch.stack(predictions)  # (5, 5)
        epistemic_std = predictions.std(dim=0).mean().item()
        print(f"   Epistemic uncertainty (std): {epistemic_std:.4f} TECU")
        
        if epistemic_std > 0.01:
            print(f"   ✅ Epistemic uncertainty present")
        else:
            print(f"   ⚠️  Low epistemic uncertainty (might be deterministic backbone)")
    
    print(f"\n✅ {model_name} VERIFIED SUCCESSFULLY!")
    return True

def main():
    """Verify all three models"""
    models = [
        ("config/config_cluster_BayesianResNetSTEC.yaml", "BayesianResNetSTEC"),
        ("config/config_cluster_AttentionMLP_BNN_NLL.yaml", "AttentionMLP_BNN_NLL"),
        ("config/config_cluster_FactorizedSTEC.yaml", "FactorizedSTEC"),
    ]
    
    print("="*80)
    print("FINAL VERIFICATION OF THREE KEY MODELS")
    print("="*80)
    
    all_passed = True
    for config_path, model_name in models:
        try:
            verify_model(config_path, model_name)
        except Exception as e:
            print(f"\n❌ {model_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL MODELS VERIFIED SUCCESSFULLY!")
        print("="*80)
        print("\n✅ Models are ready for hyperparameter tuning on cluster")
        print("   Run: bash scripts/launch_full_sweep.sh 4")
    else:
        print("❌ SOME MODELS FAILED VERIFICATION")
        print("="*80)
        sys.exit(1)

if __name__ == "__main__":
    main()
