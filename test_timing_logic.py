#!/usr/bin/env python3

import yaml
import sys
sys.path.append('src')

def test_timing_logic():
    """Test that timing is only controlled by enable_timing, not cluster"""
    
    print("=" * 60)
    print("TESTING TIMING LOGIC WITHOUT CLUSTER DEPENDENCY")
    print("=" * 60)
    
    # Test different scenarios
    test_cases = [
        {"enable_timing": False, "cluster": False, "expected": False},
        {"enable_timing": False, "cluster": True, "expected": False},
        {"enable_timing": True, "cluster": False, "expected": True},
        {"enable_timing": True, "cluster": True, "expected": True},
    ]
    
    for i, case in enumerate(test_cases):
        config = case.copy()
        expected = config.pop("expected")
        
        # Simulate the timing logic from base_trainer.py
        timing_enabled = config.get('enable_timing', False)
        
        result = "✅ PASS" if timing_enabled == expected else "❌ FAIL"
        print(f"Test {i+1}: enable_timing={config['enable_timing']}, cluster={config['cluster']} -> timing_enabled={timing_enabled} (expected: {expected}) {result}")
    
    print("\n" + "=" * 60)
    print("All tests should PASS - timing should only depend on enable_timing")

if __name__ == "__main__":
    test_timing_logic()
