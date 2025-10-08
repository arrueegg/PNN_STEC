"""
Main plotting interface that provides easy access to all visualization functions.

This module aggregates plotting functions from all visualization modules
and provides high-level functions for common plotting workflows.
"""

import pandas as pd
from typing import Optional, Any

# Import all visualization modules
from .distributions import (
    plot_binned_boxplot,
    plot_binned_boxplot_clipped,
    plot_histogram_of_residuals,
    plot_mae_vs_doy,
    plot_residuals_vs_feature,
    plot_residuals_vs_feature_clipped,
)
from .performance import (
    plot_prediction_scatter,
    plot_az_el_heatmap,
    plot_residuals_vs_date,
)
from .spatial import (
    plot_spatial_error_map,
    plot_spatial_error_map_by_local_time,
    plot_solar_magnetic_ipp_error_map,
)
from .uncertainty import (
    plot_uncertainty_calibration_binned,
    plot_coverage_probability,
    plot_binned_uncertainty_analysis,
    plot_uncertainty_calibration,
)
from .base import (
    FIGSIZE_SQUARE,
    FIGSIZE_WIDE,
    FIGSIZE_HISTOGRAM,
    FIGSIZE_HEATMAP,
    ensure_dir,
    get_scientific_label,
    configure_plotting,
)

# Import analysis utilities
from analysis.metrics import (
    modify_df,
    get_default_bin_ranges,
    calc_rmse,
    calculate_performance_metrics,
    prepare_uncertainty_data,
)

# Export all functions and constants
__all__ = [
    # Main interface
    "plot_test_metrics",
    "plot_comprehensive_uncertainty_analysis",
    # Distribution plots
    "plot_binned_boxplot",
    "plot_binned_boxplot_clipped",
    "plot_histogram_of_residuals",
    "plot_mae_vs_doy",
    "plot_residuals_vs_feature",
    "plot_residuals_vs_feature_clipped",
    # Performance plots
    "plot_prediction_scatter",
    "plot_az_el_heatmap",
    "plot_residuals_vs_date",
    # Spatial plots
    "plot_spatial_error_map",
    "plot_spatial_error_map_by_local_time",
    "plot_solar_magnetic_ipp_error_map",
    # Uncertainty plots
    "plot_uncertainty_calibration_binned",
    "plot_coverage_probability",
    "plot_binned_uncertainty_analysis",
    "plot_uncertainty_calibration",
    # Constants and utilities
    "FIGSIZE_SQUARE",
    "FIGSIZE_WIDE",
    "FIGSIZE_HISTOGRAM",
    "FIGSIZE_HEATMAP",
    "ensure_dir",
    "get_scientific_label",
    "configure_plotting",
    # Analysis utilities
    "modify_df",
    "get_default_bin_ranges",
    "calc_rmse",
    "calculate_performance_metrics",
    "prepare_uncertainty_data",
    # Legacy aliases
    "plot_binned_uncertainty_analysis_lines_only",
    "plot_prediction_quality",
    "plot_spatial_analysis",
    "plot_uncertainty_analysis",
]


def plot_test_metrics(
    test_df: pd.DataFrame,
    output_dir: str = "plots",
    feature_registry: Optional[Any] = None,
) -> None:
    """
    Generate comprehensive test metrics plots.

    This is the main interface function that creates all standard evaluation plots.

    Args:
        test_df: DataFrame with test results
        output_dir: Directory to save plots
        feature_registry: Optional feature registry for binning ranges
    """
    # Ensure output directory exists
    ensure_dir(output_dir)

    # Prepare data
    df = modify_df(test_df)

    # Get binning ranges
    bin_range_dict = get_default_bin_ranges(feature_registry)

    print(f"Generating plots for {len(df):,} test samples...")

    # Core performance plots
    print("Creating prediction scatter plot...")
    plot_prediction_scatter(df, output_dir)

    print("Creating residual histogram...")
    plot_histogram_of_residuals(df, output_dir)

    # Spatial analysis
    print("Creating spatial error maps...")
    if "lat_ipp" in df.columns and "lon_ipp" in df.columns:
        plot_spatial_error_map(df, output_dir)

        if "time" in df.columns:
            plot_spatial_error_map_by_local_time(df, output_dir)

    if "sm_lat_ipp" in df.columns and "sm_lon_ipp" in df.columns:
        plot_solar_magnetic_ipp_error_map(df, output_dir)

    # Angular analysis
    if "satazi" in df.columns and "satele" in df.columns:
        print("Creating azimuth-elevation heatmaps...")
        plot_az_el_heatmap(df, output_dir, metric="residual")
        plot_az_el_heatmap(df, output_dir, metric="mae")

    # Temporal analysis
    if "doy" in df.columns:
        print("Creating temporal analysis plots...")
        plot_mae_vs_doy(df, output_dir)
        plot_residuals_vs_feature(
            df, "doy", output_dir=output_dir, bin_range_dict=bin_range_dict
        )

    if any(col in df.columns for col in ["datetime", "year"]):
        plot_residuals_vs_date(df, output_dir)

    # Feature-specific analysis
    features_to_analyze = ["time", "satele", "satazi", "kp", "f107", "dst"]
    for feature in features_to_analyze:
        if feature in df.columns:
            print(f"Creating residual analysis for {feature}...")
            plot_residuals_vs_feature(
                df, feature, output_dir=output_dir, bin_range_dict=bin_range_dict
            )

    # Uncertainty analysis (if uncertainty data available)
    uncertainty_cols = [col for col in df.columns if "unc" in col.lower()]
    if uncertainty_cols:
        print("Creating uncertainty analysis plots...")
        df_unc = prepare_uncertainty_data(df)

        plot_uncertainty_calibration_binned(df_unc, output_dir)
        plot_coverage_probability(df_unc, output_dir)
        plot_binned_uncertainty_analysis(df_unc, output_dir)
        plot_uncertainty_calibration(df_unc, output_dir)

    # Calculate and save performance metrics
    metrics = calculate_performance_metrics(df)

    # Save metrics to file
    import os

    metrics_file = os.path.join(output_dir, "performance_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Model Performance Metrics\n")
        f.write("=" * 30 + "\n\n")
        for key, value in metrics.items():
            if "pval" in key:
                f.write(f"{key}: {value:.2e}\n")
            else:
                f.write(f"{key}: {value:.6f}\n")

    print(f"All plots saved to: {output_dir}")
    print(f"Performance metrics saved to: {metrics_file}")


def plot_comprehensive_uncertainty_analysis(
    df: pd.DataFrame, output_dir: str = "plots"
) -> None:
    """
    Create comprehensive uncertainty analysis plots.

    Args:
        df: DataFrame with uncertainty data
        output_dir: Directory to save plots
    """
    ensure_dir(output_dir)
    df_unc = prepare_uncertainty_data(df)

    print("Creating comprehensive uncertainty analysis...")
    plot_binned_uncertainty_analysis(df_unc, output_dir)
    plot_uncertainty_calibration_binned(df_unc, output_dir)
    plot_coverage_probability(df_unc, output_dir)


# Legacy function aliases for backward compatibility
def plot_binned_uncertainty_analysis_lines_only(
    df: pd.DataFrame, output_dir: str = "plots"
) -> None:
    """Legacy alias for uncertainty analysis."""
    plot_binned_uncertainty_analysis(df, output_dir)


# Aliases for commonly used functions
plot_prediction_quality = plot_prediction_scatter
plot_spatial_analysis = plot_spatial_error_map
plot_uncertainty_analysis = plot_binned_uncertainty_analysis
