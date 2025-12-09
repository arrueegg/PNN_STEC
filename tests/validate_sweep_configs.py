#!/usr/bin/env python3
"""
Test W&B sweep configurations to ensure they're valid.
This validates the sweep YAML files can be parsed and contain required fields.
"""

import yaml
import sys
from pathlib import Path

def validate_sweep_config(config_path):
    """Validate a W&B sweep configuration file."""
    print(f"\nValidating: {config_path.name}")
    print("=" * 70)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['program', 'method', 'metric', 'parameters']
        missing_fields = [f for f in required_fields if f not in config]
        
        if missing_fields:
            print(f"❌ FAILED: Missing required fields: {missing_fields}")
            return False
        
        # Validate program path
        program_path = Path(config['program'])
        if not program_path.name.endswith('.py'):
            print(f"⚠️  WARNING: Program '{config['program']}' doesn't end with .py")
        
        if config['program'] != 'src/main.py':
            print(f"⚠️  WARNING: Program should be 'src/main.py', got '{config['program']}'")
            print(f"   (If using wrapper, sweep won't work correctly with --override)")
        
        # Validate method
        valid_methods = ['bayes', 'grid', 'random']
        if config['method'] not in valid_methods:
            print(f"❌ FAILED: Invalid method '{config['method']}'. Must be one of {valid_methods}")
            return False
        
        # Validate metric
        if 'goal' not in config['metric'] or 'name' not in config['metric']:
            print("❌ FAILED: Metric must have 'goal' and 'name'")
            return False
        
        if config['metric']['goal'] not in ['minimize', 'maximize']:
            print(f"❌ FAILED: Metric goal must be 'minimize' or 'maximize'")
            return False
        
        # Check parameters
        if not config['parameters']:
            print("❌ FAILED: No parameters defined")
            return False
        
        # Validate parameter format
        for param_name, param_config in config['parameters'].items():
            if not isinstance(param_config, dict):
                print(f"❌ FAILED: Parameter '{param_name}' must be a dict")
                return False
            
            # Check if it has values or distribution
            if 'values' not in param_config and 'min' not in param_config:
                print(f"❌ FAILED: Parameter '{param_name}' must have 'values' or 'min'/'max'")
                return False
        
        # Print summary
        print(f"✅ VALID")
        print(f"   Program: {config['program']}")
        print(f"   Method: {config['method']}")
        print(f"   Metric: {config['metric']['goal']} {config['metric']['name']}")
        print(f"   Parameters: {len(config['parameters'])} defined")
        print(f"   Run cap: {config.get('run_cap', 'unlimited')}")
        
        # Check model type
        if 'model.model_type' in config['parameters']:
            model_types = config['parameters']['model.model_type'].get('values', [])
            print(f"   Model types: {model_types}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Validate all cluster sweep configurations."""
    print("=" * 70)
    print("W&B Sweep Configuration Validation")
    print("=" * 70)
    
    config_dir = Path(__file__).parent.parent / 'config'
    
    # Find all cluster sweep configs
    sweep_configs = sorted(config_dir.glob('wandb_sweep_config_*_cluster.yaml'))
    
    if not sweep_configs:
        print("\n❌ No cluster sweep configs found!")
        return 1
    
    print(f"\nFound {len(sweep_configs)} sweep configurations to validate\n")
    
    results = {}
    for config_path in sweep_configs:
        results[config_path.name] = validate_sweep_config(config_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\n⚠️  Some sweep configurations are invalid!")
        return 1
    else:
        print("\n✅ All sweep configurations are valid!")
        return 0

if __name__ == '__main__':
    sys.exit(main())
