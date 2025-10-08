"""
Analysis module for data processing and statistical calculations.

This module provides utilities for data transformation, statistical analysis,
and preparation of data for visualization.
"""

from .metrics import (
    modify_df,
    get_default_bin_ranges,
    calc_rmse,
    calculate_temporal_statistics,
    calculate_spatial_statistics,
    prepare_uncertainty_data,
    calculate_performance_metrics,
)

__all__ = [
    "modify_df",
    "get_default_bin_ranges",
    "calc_rmse",
    "calculate_temporal_statistics",
    "calculate_spatial_statistics",
    "prepare_uncertainty_data",
    "calculate_performance_metrics",
]
