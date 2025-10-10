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
    plot_box_by_date,
    plot_residuals_vs_local_time,
    plot_residuals_vs_solar_indices,
)
from .performance import (
    plot_prediction_scatter,
    plot_prediction_density,
    plot_az_el_heatmap,
    plot_residuals_vs_date,
)
from .spatial import (
    plot_spatial_error_map,
    plot_spatial_error_map_by_local_time,
    plot_solar_magnetic_ipp_error_map,
    plot_box_by_lat,
)
from .uncertainty import (
    plot_uncertainty_calibration_binned,
    plot_coverage_probability,
    plot_binned_uncertainty_analysis,
    plot_uncertainty_calibration,
    plot_sigma_coverage_comparison,
    plot_uncertainty_distributions,
    plot_binned_uncertainty_error_analysis,
)
from .base import (
    FIGSIZE_SQUARE,
    FIGSIZE_WIDE,
    FIGSIZE_HISTOGRAM,
    FIGSIZE_HEATMAP,
    ensure_dir,
    get_scientific_label,
    configure_plotting,
    create_temporal_metrics_summaries,
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
    "plot_box_by_date",
    "plot_residuals_vs_local_time",
    "plot_residuals_vs_solar_indices",
    # Performance plots
    "plot_prediction_scatter",
    "plot_prediction_density",
    "plot_az_el_heatmap",
    "plot_residuals_vs_date",
    # Spatial plots
    "plot_spatial_error_map",
    "plot_spatial_error_map_by_local_time",
    "plot_solar_magnetic_ipp_error_map",
    "plot_box_by_lat",
    # Uncertainty plots
    "plot_uncertainty_calibration_binned",
    "plot_coverage_probability",
    "plot_binned_uncertainty_analysis",
    "plot_uncertainty_calibration",
    "plot_sigma_coverage_comparison",
    "plot_uncertainty_distributions",
    "plot_binned_uncertainty_error_analysis",
    # Constants and utilities
    "FIGSIZE_SQUARE",
    "FIGSIZE_WIDE",
    "FIGSIZE_HISTOGRAM",
    "FIGSIZE_HEATMAP",
    "ensure_dir",
    "get_scientific_label",
    "configure_plotting",
    "create_temporal_metrics_summaries",
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
    Generate comprehensive test metrics plots organized in subfolders.

    This is the main interface function that creates all standard evaluation plots
    and organizes them into structured subfolders.

    Args:
        test_df: DataFrame with test results
        output_dir: Directory to save plots
        feature_registry: Optional feature registry for binning ranges
    """
    import logging

    logger = logging.getLogger(__name__)

    # Create organized subdirectories
    test_metrics_dir = f"{output_dir}/test_metrics"
    prediction_dir = f"{test_metrics_dir}/prediction_analysis"
    uncertainty_dir = f"{test_metrics_dir}/uncertainty_analysis"
    spatial_dir = f"{test_metrics_dir}/spatial_analysis"
    temporal_dir = f"{test_metrics_dir}/temporal_analysis"
    feature_dir = f"{test_metrics_dir}/feature_analysis"

    # Ensure all directories exist
    ensure_dir(test_metrics_dir)
    ensure_dir(prediction_dir)
    ensure_dir(uncertainty_dir)
    ensure_dir(spatial_dir)
    ensure_dir(temporal_dir)
    ensure_dir(feature_dir)

    # Prepare data
    df = modify_df(test_df)

    # Get binning ranges
    bin_range_dict = get_default_bin_ranges(feature_registry)

    logger.info(f"Generating plots for {len(df):,} test samples...")

    # Core performance plots in prediction_analysis/
    logger.info("Creating prediction scatter plot...")
    plot_prediction_scatter(df, prediction_dir)

    logger.info("Creating prediction density plot...")
    plot_prediction_density(df, prediction_dir)

    # Spatial analysis in spatial_analysis/
    logger.info("Creating spatial error maps...")
    if "lat_ipp" in df.columns and "lon_ipp" in df.columns:
        plot_spatial_error_map(df, spatial_dir)

        if "time" in df.columns:
            spatial_time_dir = f"{spatial_dir}/by_local_time"
            ensure_dir(spatial_time_dir)
            plot_spatial_error_map_by_local_time(df, spatial_time_dir)

    if "sm_lat_ipp" in df.columns and "sm_lon_ipp" in df.columns:
        plot_solar_magnetic_ipp_error_map(df, spatial_dir)

    # Angular analysis in spatial_analysis/
    if "satazi" in df.columns and "satele" in df.columns:
        logger.info("Creating azimuth-elevation heatmaps...")
        plot_az_el_heatmap(df, spatial_dir, metric="residual")
        plot_az_el_heatmap(df, spatial_dir, metric="mae")

    # Temporal analysis in temporal_analysis/
    if "doy" in df.columns:
        logger.info("Creating temporal analysis plots...")
        plot_residuals_vs_feature(
            df, "doy", output_dir=temporal_dir, bin_range_dict=bin_range_dict
        )

    if any(col in df.columns for col in ["datetime", "year"]):
        plot_residuals_vs_date(df, temporal_dir)

    # Plot summary by date if year and doy are available
    if "year" in df.columns and "doy" in df.columns:
        logger.info("Creating temporal summary plots...")
        plot_box_by_date(df, temporal_dir)

    # Plot summary by magnetic latitude if available
    if "sm_lat_ipp" in df.columns:
        logger.info("Creating latitude band analysis...")
        plot_box_by_lat(df, spatial_dir)

    # Feature-specific analysis in feature_analysis/
    features_to_analyze = [
        "time",
        "satele",
        "satazi",
        "kp",
        "f107",
        "dst",
        "target_stec",
        "pred_stec",
    ]
    for feature in features_to_analyze:
        if feature in df.columns:
            logger.info(f"Creating residual analysis for {feature}...")

            # Determine number of bins based on feature
            if feature == "time":
                num_bins = 24
            elif feature == "doy":
                num_bins = 24
            elif feature == "satele":
                num_bins = 17
            elif feature == "satazi":
                num_bins = 24
            elif feature in ["kp", "kp_binned"]:
                num_bins = 9
            elif feature in ["dst", "f107", "sunspot"]:
                num_bins = 20 if feature == "dst" else 10
            else:
                num_bins = 20  # default for target_stec, pred_stec

            plot_residuals_vs_feature(
                df,
                feature,
                num_bins=num_bins,
                output_dir=feature_dir,
                bin_range_dict=bin_range_dict,
            )

            # Create clipped version for target_stec
            if feature == "target_stec":
                plot_residuals_vs_feature_clipped(
                    df,
                    feature,
                    num_bins=num_bins,
                    output_dir=feature_dir,
                    bin_range_dict=bin_range_dict,
                    x_limits=(0.5, 10.5),
                    y_limits=(-50, 100),
                )

    # Additional feature analysis plots - Local time and solar indices
    logger.info("Creating local time and solar index residual analysis...")
    plot_residuals_vs_local_time(df, feature_dir)
    plot_residuals_vs_solar_indices(df, feature_dir)

    # Uncertainty analysis in uncertainty_analysis/ subfolder
    uncertainty_cols = [col for col in df.columns if "unc" in col.lower()]
    if uncertainty_cols:
        logger.info("Creating uncertainty analysis plots...")
        df_unc = prepare_uncertainty_data(df)

        plot_coverage_probability(df_unc, uncertainty_dir)
        plot_binned_uncertainty_analysis(df_unc, uncertainty_dir)
        plot_sigma_coverage_comparison(df_unc, uncertainty_dir)
        plot_uncertainty_distributions(df_unc, uncertainty_dir)
        plot_binned_uncertainty_error_analysis(df_unc, uncertainty_dir)

    # Calculate and save performance metrics
    metrics = calculate_performance_metrics(df)

    # Save metrics to file
    import os

    metrics_file = os.path.join(test_metrics_dir, "performance_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Model Performance Metrics\n")
        f.write("=" * 30 + "\n\n")
        for key, value in metrics.items():
            if "pval" in key:
                f.write(f"{key}: {value:.2e}\n")
            else:
                f.write(f"{key}: {value:.6f}\n")

    # Create temporal metrics summaries
    logger.info("Creating temporal metrics summaries...")
    create_temporal_metrics_summaries(df, test_metrics_dir)

    if uncertainty_cols:
        logger.info(
            f"Uncertainty analysis: {len([f for f in os.listdir(uncertainty_dir) if f.endswith('.png')])} plots"
        )
    if "time" in df.columns and "lat_ipp" in df.columns:
        logger.info(f"Spatial by time analysis: {spatial_time_dir}")


def plot_comprehensive_uncertainty_analysis(
    df: pd.DataFrame, output_dir: str = "plots"
) -> None:
    """
    Create comprehensive uncertainty analysis plots.

    Args:
        df: DataFrame with uncertainty data
        output_dir: Directory to save plots
    """
    import logging

    logger = logging.getLogger(__name__)

    ensure_dir(output_dir)
    df_unc = prepare_uncertainty_data(df)

    logger.info("Creating comprehensive uncertainty analysis...")
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
