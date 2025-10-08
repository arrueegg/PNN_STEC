"""
Data analysis and transformation utilities.

This module provides data processing functions for visualization preparation,
statistical calculations, and data transformations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


def modify_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard data modifications for visualization.

    Args:
        df: Input DataFrame

    Returns:
        Modified DataFrame with additional columns
    """
    df = df.copy()

    # Calculate basic metrics if not present
    if (
        "residual" not in df.columns
        and "target_stec" in df.columns
        and "pred_stec" in df.columns
    ):
        df["residual"] = df["target_stec"] - df["pred_stec"]

    if (
        "mae" not in df.columns
        and "target_stec" in df.columns
        and "pred_stec" in df.columns
    ):
        df["mae"] = np.abs(df["target_stec"] - df["pred_stec"])

    if "abs_residual" not in df.columns and "residual" in df.columns:
        df["abs_residual"] = np.abs(df["residual"])

    # Convert datetime if needed
    if "datetime" not in df.columns and "year" in df.columns and "doy" in df.columns:

        df["datetime"] = pd.to_datetime(df["year"], format="%Y") + pd.to_timedelta(
            df["doy"] - 1, unit="D"
        )

    return df


def get_default_bin_ranges(
    feature_registry: Optional[Any] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Get default binning ranges for common features.

    Args:
        feature_registry: Optional feature registry object

    Returns:
        Dictionary mapping feature names to (min, max) ranges
    """
    default_ranges = {
        "doy": (1, 366),
        "sod": (0, 86400),
        "time": (0, 24),
        "satele": (5, 90),
        "satazi": (0, 360),
        "lat_ipp": (-90, 90),
        "lon_ipp": (-180, 180),
        "sm_lat_ipp": (-90, 90),
        "sm_lon_ipp": (-180, 180),
        "kp": (0, 9),
        "dst": (-200, 100),
        "f107": (50, 300),
        "sunspot": (0, 300),
        "year": (2010, 2025),
    }

    # If feature registry is provided, could extend with registry-specific ranges
    if feature_registry is not None:
        # Add any feature registry specific ranges here
        pass

    return default_ranges


def calc_rmse(values: np.ndarray) -> float:
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean(values**2))


def calculate_temporal_statistics(
    df: pd.DataFrame, group_by: str = "date"
) -> pd.DataFrame:
    """
    Calculate temporal statistics for residuals and errors.

    Args:
        df: DataFrame with temporal data
        group_by: Temporal grouping ('date', 'doy', 'hour', etc.)

    Returns:
        DataFrame with temporal statistics
    """
    df = modify_df(df)

    if group_by == "date":
        if "datetime" in df.columns:
            df["date"] = df["datetime"].dt.date
        else:
            # Create date from year/doy

            df["date"] = pd.to_datetime(df["year"], format="%Y") + pd.to_timedelta(
                df["doy"] - 1, unit="D"
            )
            df["date"] = df["date"].dt.date

        # Group by date
        grouped = df.groupby("date")
    elif group_by == "doy":
        grouped = df.groupby("doy")
    elif group_by == "hour":
        if "time" in df.columns:
            df["hour"] = df["time"].astype(int)
        else:
            df["hour"] = (df["sod"] / 3600).astype(int)
        grouped = df.groupby("hour")
    else:
        grouped = df.groupby(group_by)

    # Calculate statistics
    stats = grouped.agg(
        {
            "residual": ["mean", "std", "count"],
            "mae": "mean",
            "target_stec": "mean",
            "pred_stec": "mean",
        }
    ).reset_index()

    # Flatten column names
    stats.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in stats.columns
    ]

    # Rename for clarity
    rename_map = {
        "residual_mean": "mean_residual",
        "residual_std": "std_residual",
        "residual_count": "count",
        "mae_mean": "mean_mae",
        "target_stec_mean": "mean_target",
        "pred_stec_mean": "mean_pred",
    }
    stats = stats.rename(columns=rename_map)

    return stats


def calculate_spatial_statistics(
    df: pd.DataFrame,
    lat_col: str = "lat_ipp",
    lon_col: str = "lon_ipp",
    bin_size: float = 5.0,
) -> pd.DataFrame:
    """
    Calculate spatial statistics for errors.

    Args:
        df: DataFrame with spatial data
        lat_col: Latitude column name
        lon_col: Longitude column name
        bin_size: Spatial bin size in degrees

    Returns:
        DataFrame with spatial statistics
    """
    df = modify_df(df)

    # Create spatial bins
    lat_bins = np.arange(-90, 91, bin_size)
    lon_bins = np.arange(-180, 181, bin_size)

    df["lat_bin"] = pd.cut(df[lat_col], bins=lat_bins, include_lowest=True)
    df["lon_bin"] = pd.cut(df[lon_col], bins=lon_bins, include_lowest=True)

    # Calculate statistics per bin
    spatial_stats = (
        df.groupby(["lat_bin", "lon_bin"])
        .agg(
            {
                "mae": ["mean", "std", "count"],
                "residual": ["mean", "std"],
                "target_stec": "mean",
                "pred_stec": "mean",
            }
        )
        .reset_index()
    )

    # Flatten column names
    spatial_stats.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in spatial_stats.columns
    ]

    # Add bin centers
    spatial_stats["lat_center"] = spatial_stats["lat_bin"].apply(lambda x: x.mid)
    spatial_stats["lon_center"] = spatial_stats["lon_bin"].apply(lambda x: x.mid)

    # Filter bins with sufficient data
    spatial_stats = spatial_stats[spatial_stats["mae_count"] >= 10]

    return spatial_stats


def prepare_uncertainty_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for uncertainty analysis.

    Args:
        df: DataFrame with uncertainty columns

    Returns:
        DataFrame prepared for uncertainty analysis
    """
    df = modify_df(df)

    # Check for uncertainty columns
    uncertainty_cols = [col for col in df.columns if "unc" in col.lower()]

    if not uncertainty_cols:
        print("Warning: No uncertainty columns found in data")
        return df

    # Ensure we have total uncertainty
    if "pred_total_unc" not in df.columns:
        if "pred_epistemic_unc" in df.columns and "pred_aleatoric_unc" in df.columns:
            df["pred_total_unc"] = np.sqrt(
                df["pred_epistemic_unc"] ** 2 + df["pred_aleatoric_unc"] ** 2
            )
        elif len(uncertainty_cols) == 1:
            df["pred_total_unc"] = df[uncertainty_cols[0]]

    return df


def calculate_performance_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        df: DataFrame with predictions and targets

    Returns:
        Dictionary of performance metrics
    """
    df = modify_df(df)

    # Basic metrics
    mae = df["mae"].mean()
    rmse = calc_rmse(df["residual"].values)
    bias = df["residual"].mean()

    # Correlation metrics
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score

    r2 = r2_score(df["target_stec"], df["pred_stec"])
    corr, p_val = pearsonr(df["target_stec"], df["pred_stec"])

    # Quantile metrics
    mae_q95 = df["mae"].quantile(0.95)
    mae_q99 = df["mae"].quantile(0.99)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "r2_score": r2,
        "correlation": corr,
        "correlation_pval": p_val,
        "mae_q95": mae_q95,
        "mae_q99": mae_q99,
        "n_samples": len(df),
    }

    # Uncertainty metrics if available
    if "pred_total_unc" in df.columns:
        unc_corr, unc_p = pearsonr(df["pred_total_unc"], df["abs_residual"])
        metrics["uncertainty_correlation"] = unc_corr
        metrics["uncertainty_correlation_pval"] = unc_p
        metrics["mean_uncertainty"] = df["pred_total_unc"].mean()

    return metrics
