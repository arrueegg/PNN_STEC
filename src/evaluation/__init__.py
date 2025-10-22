"""
STEC Evaluation Package

This package provides streamlined evaluation tools for STEC model evaluation:
- gim_mapper: GIM VTEC to STEC mapping functionality
- plotter: Comprehensive plotting system combining basic and enhanced analysis
- utils: Lightweight CSV saving and statistics utilities

The package provides a complete evaluation workflow from model inference 
to comprehensive visualization and statistical analysis.
"""

# Use lazy imports to avoid loading heavy dependencies when not needed
def __getattr__(name):
    """Lazy import for optional heavy dependencies."""
    if name == 'GIMMapper':
        from .gim_mapper import GIMMapper
        return GIMMapper
    elif name == 'STECPlotter':
        from .plotter import STECPlotter
        return STECPlotter
    elif name == 'create_stec_plots':
        from .plotter import create_stec_plots
        return create_stec_plots
    elif name == 'save_results_csv':
        from .utils import save_results_csv
        return save_results_csv
    elif name == 'print_and_save_statistics':
        from .utils import print_and_save_statistics
        return print_and_save_statistics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'GIMMapper',
    'STECPlotter', 
    'create_stec_plots',
    'save_results_csv',
    'print_and_save_statistics',
]