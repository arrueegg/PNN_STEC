"""
STEC Evaluation Package

This package provides streamlined evaluation tools for STEC model evaluation:
- gim_mapper: GIM VTEC to STEC mapping functionality
- plotter: Comprehensive plotting system combining basic and enhanced analysis
- utils: Lightweight CSV saving and statistics utilities

The package provides a complete evaluation workflow from model inference 
to comprehensive visualization and statistical analysis.
"""

from .gim_mapper import GIMMapper
from .plotter import STECPlotter, create_stec_plots
from .utils import save_results_csv, print_and_save_statistics

__all__ = [
    'GIMMapper',
    'STECPlotter', 
    'create_stec_plots',
    'save_results_csv',
    'print_and_save_statistics',
]