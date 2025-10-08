"""
DEPRECATED: Backward compatibility module for data.py.

This module maintains the original interface while delegating to the new modular
components in the src/data/ package. This ensures existing code continues to work
without modification.

⚠️  DEPRECATED: This module is deprecated and will be removed in a future version.
    Please update your imports to use the new modular data package:
    
    # Old (deprecated):
    from utils.data import get_data_loaders, H5Dataset, CollateWithSH
    
    # New (recommended):
    from data import get_data_loaders, H5Dataset, CollateWithSH
    
The new modular structure provides:
- Better separation of concerns
- Improved maintainability  
- Enhanced testability
- Cleaner dependencies

For more details, see REFACTORING_SUMMARY_data.md
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "utils.data is deprecated. Use 'from data import ...' instead of 'from utils.data import ...'",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all functionality from the new modular data package for backward compatibility
try:
    from data_loader.datasets import H5Dataset, H5RAMDataset, PyTablesDatasetSplit  
    from data_loader.samplers import EpochRandomSampler, get_fixed_subset_indices
    from data_loader.collation import CollateWithSH
    from data_loader.loaders import get_data_loaders, get_test_data_loader
except ImportError as e:
    # Fallback error message if the new modular structure isn't available
    raise ImportError(
        f"Failed to import from new modular data package: {e}\n"
        "Please ensure the src/data/ package is properly installed."
    )

# Maintain backward compatibility for any existing imports
__all__ = [
    'H5Dataset',
    'H5RAMDataset', 
    'PyTablesDatasetSplit',
    'EpochRandomSampler',
    'get_fixed_subset_indices',
    'CollateWithSH',
    'get_data_loaders',
    'get_test_data_loader'
]