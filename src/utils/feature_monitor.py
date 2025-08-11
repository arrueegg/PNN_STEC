import torch
import logging
from typing import Dict, List, Optional
from .feature_registry import FeatureRegistry, FeatureType

class FeatureMonitor:
    """Monitor feature consistency across pipeline stages."""
    
    def __init__(self, feature_registry: FeatureRegistry, logger: Optional[logging.Logger] = None):
        self.registry = feature_registry
        self.logger = logger or logging.getLogger(__name__)
        self.stage_checks: Dict[str, Dict] = {}
        
    def check_stage(self, stage_name: str, tensor: torch.Tensor, 
                   expected_features: Optional[int] = None) -> bool:
        """Check feature tensor at a pipeline stage."""
        expected = expected_features or self.registry.get_total_features()
        actual = tensor.shape[-1]
        
        if actual != expected:
            self.logger.error(f"Stage {stage_name}: Expected {expected} features, got {actual}")
            self.stage_checks[stage_name] = {
                'expected': expected,
                'actual': actual,
                'tensor_shape': tensor.shape,
                'status': 'FAIL'
            }
            raise ValueError(f"Feature mismatch at {stage_name}: expected {expected}, got {actual}")
        
        self.stage_checks[stage_name] = {
            'expected': expected,
            'actual': actual,
            'tensor_shape': tensor.shape,
            'status': 'PASS'
        }
        
        self.logger.debug(f"Stage {stage_name}: Feature check PASSED ({actual} features)")
        return True
        
    def log_feature_summary(self):
        """Log summary of all feature checks."""
        self.logger.info("=== Feature Pipeline Summary ===")
        if not self.stage_checks:
            self.logger.info("No stages checked yet")
            return
            
        for stage, check in self.stage_checks.items():
            status = check['status']
            shape = check['tensor_shape']
            expected = check['expected']
            actual = check['actual']
            self.logger.info(f"{stage}: {status} - Expected: {expected}, Actual: {actual}, Shape: {shape}")
            
    def validate_feature_order(self, tensor: torch.Tensor, 
                              feature_names: List[str]) -> bool:
        """Validate that features are in expected order."""
        if len(feature_names) != tensor.shape[-1]:
            self.logger.error(f"Feature name count ({len(feature_names)}) doesn't match tensor dimension ({tensor.shape[-1]})")
            return False
            
        # Check if all features are registered and in correct order
        registry_features = self.registry.get_feature_order()
        missing_features = []
        extra_features = []
        
        for name in feature_names:
            if name not in self.registry._features:
                missing_features.append(name)
                
        for name in registry_features:
            if name not in feature_names and self.registry._features[name]['enabled']:
                extra_features.append(name)
                
        if missing_features:
            self.logger.warning(f"Features not in registry: {missing_features}")
            
        if extra_features:
            self.logger.warning(f"Expected features missing from input: {extra_features}")
            
        return len(missing_features) == 0 and len(extra_features) == 0
    
    def check_feature_consistency(self, tensor: torch.Tensor, stage_name: str) -> bool:
        """Comprehensive feature consistency check."""
        try:
            # Check tensor dimensions
            if not self.check_stage(stage_name, tensor):
                return False
                
            # Log tensor statistics for debugging
            self.logger.debug(f"Stage {stage_name} tensor stats: "
                            f"min={tensor.min():.4f}, max={tensor.max():.4f}, "
                            f"mean={tensor.mean():.4f}, std={tensor.std():.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Feature consistency check failed at {stage_name}: {str(e)}")
            return False
    
    def get_stage_status(self, stage_name: str) -> Optional[str]:
        """Get status of a specific stage."""
        return self.stage_checks.get(stage_name, {}).get('status')
    
    def has_failures(self) -> bool:
        """Check if any stage has failed."""
        return any(check['status'] == 'FAIL' for check in self.stage_checks.values())