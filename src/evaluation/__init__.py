"""
Evaluation subpackage for GNSS-based STEC ML model assessment.

This package contains modules for comparing STEC model predictions against
various observation sources in STEC space to evaluate mapping function errors
and model performance.

Modules:
- adapters: Observation source adapters (testset, madrigal, grid, vgosdb)
- stec_eval: Core STEC comparison logic and workflow coordination
"""

from .adapters import get_adapter, DatasetAdapter
from .stec_eval import run_stec_evaluation

__all__ = ['get_adapter', 'DatasetAdapter', 'run_stec_evaluation']