"""
Base visualization utilities and configuration.

This module provides common plotting utilities, styling constants,
and shared functionality for all visualization modules.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


# Standardized matplotlib configuration for presentation-ready plots
PLOT_CONFIG = {
    "font.size": 16,  # Base font size
    "axes.titlesize": 22,  # Title font size
    "axes.labelsize": 20,  # Axis label font size
    "xtick.labelsize": 16,  # X-tick label size
    "ytick.labelsize": 16,  # Y-tick label size
    "legend.fontsize": 16,  # Legend font size
    "figure.titlesize": 24,  # Figure title size
    "axes.grid": True,  # Enable grid by default
    "grid.alpha": 0.3,  # Grid transparency
    "lines.linewidth": 2,  # Thicker lines
    "axes.linewidth": 1.2,  # Thicker axes
    "xtick.major.width": 1.2,  # Thicker tick marks
    "ytick.major.width": 1.2,
    "figure.dpi": 300,  # High resolution
    "savefig.dpi": 300,  # High DPI for saved figures
    "savefig.bbox": "tight",  # Tight bounding box
}

# Standardized figure sizes for consistent text scaling
FIGSIZE_SQUARE = (12, 12)           # Square plots: scatter, correlation, calibration
FIGSIZE_WIDE = (16, 10)             # Wide plots: spatial maps, single panel timelines
FIGSIZE_DOUBLE_WIDE = (24, 10)      # Double width: 2-panel side-by-side plots
FIGSIZE_QUAD = (20, 16)             # Quad: 4-panel 2x2 grids
FIGSIZE_HISTOGRAM = (16, 10)        # Histogram/distribution plots (same as WIDE)
FIGSIZE_HEATMAP = (16, 10)          # Heatmaps and spatial plots


def configure_plotting() -> None:
    """Apply standardized plotting configuration."""
    plt.rcParams.update(PLOT_CONFIG)


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_scientific_label(column_name: str) -> str:
    """Convert column names to scientific presentation labels."""
    label_mapping = {
        "target_stec": "True STEC [TECU]",
        "pred_stec": "Predicted STEC [TECU]",
        "residual": "Residual [TECU]",
        "mae": "Mean Absolute Error [TECU]",
        "doy": "Day of Year",
        "sod": "Seconds of Day [s]",
        "time": "Local Solar Time [h]",
        "satele": "Elevation Angle [°]",
        "satazi": "Azimuth Angle [°]",
        "lat_ipp": "IPP Latitude [°]",
        "lon_ipp": "IPP Longitude [°]",
        "sm_lat_ipp": "Solar Magnetic IPP Latitude [°]",
        "sm_lon_ipp": "Solar Magnetic IPP Longitude [°]",
        "kp": "Kp Index",
        "kp_binned": "Kp Index (binned)",
        "dst": "Dst Index [nT]",
        "f107": "F10.7 Solar Flux [sfu]",
        "sunspot": "Sunspot Number",
        "year": "Year",
        "pred_total_unc": "Total Uncertainty [TECU]",
        "pred_epistemic_unc": "Epistemic Uncertainty [TECU]",
        "pred_aleatoric_unc": "Aleatoric Uncertainty [TECU]",
    }
    return label_mapping.get(column_name, column_name.replace("_", " ").title())


def save_plot(
    fig: matplotlib.figure.Figure, filename: str, output_dir: str = "plots"
) -> None:
    """Save plot with standardized settings."""
    ensure_dir(output_dir)
    full_path = os.path.join(output_dir, filename)
    fig.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Initialize plotting configuration when module is imported
configure_plotting()


def create_temporal_metrics_summaries(
    df: pd.DataFrame, output_dir: str = "plots"
) -> None:
    """
    Create temporal split metrics summaries as txt files for the temporal splits and also total.

    Args:
        df: DataFrame with predictions, targets, and temporal columns
        output_dir: Directory to save summary files
    """
    temporal_dir = f"{output_dir}/temporal_analysis"
    ensure_dir(temporal_dir)

    def calculate_metrics(data):
        """Calculate comprehensive metrics for a data subset."""
        if len(data) == 0:
            return None

        y_true = data["target_stec"].values
        y_pred = data["pred_stec"].values

        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        corr, p_val = pearsonr(y_true, y_pred)
        bias = np.mean(y_pred - y_true)

        # Additional statistics
        residuals = y_pred - y_true
        abs_residuals = np.abs(residuals)

        metrics = {
            "count": len(data),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "correlation": corr,
            "p_value": p_val,
            "bias": bias,
            "std_residual": np.std(residuals),
            "mean_target": np.mean(y_true),
            "std_target": np.std(y_true),
            "mean_prediction": np.mean(y_pred),
            "std_prediction": np.std(y_pred),
            "median_abs_error": np.median(abs_residuals),
            "q75_abs_error": np.percentile(abs_residuals, 75),
            "q95_abs_error": np.percentile(abs_residuals, 95),
            "max_abs_error": np.max(abs_residuals),
        }

        # Add uncertainty metrics if available
        if "pred_total_unc" in data.columns:
            total_unc = data["pred_total_unc"].values
            metrics["mean_uncertainty"] = np.mean(total_unc)
            metrics["std_uncertainty"] = np.std(total_unc)

            # Coverage analysis
            within_1sigma = np.sum(abs_residuals <= total_unc) / len(data) * 100
            within_2sigma = np.sum(abs_residuals <= 2 * total_unc) / len(data) * 100
            within_3sigma = np.sum(abs_residuals <= 3 * total_unc) / len(data) * 100

            metrics["coverage_1sigma"] = within_1sigma
            metrics["coverage_2sigma"] = within_2sigma
            metrics["coverage_3sigma"] = within_3sigma

            # Calibration metric (correlation between uncertainty and absolute error)
            if len(total_unc) > 1:
                calib_corr, calib_p = pearsonr(total_unc, abs_residuals)
                metrics["uncertainty_calibration"] = calib_corr
                metrics["uncertainty_calibration_p"] = calib_p

        return metrics

    def write_metrics_file(filename, title, metrics_dict, description=""):
        """Write metrics to a formatted text file."""
        filepath = f"{temporal_dir}/{filename}"
        with open(filepath, "w") as f:
            f.write(f"{'='*80}\n")
            f.write(f"{title.center(80)}\n")
            f.write(f"{'='*80}\n\n")

            if description:
                f.write(f"{description}\n\n")

            if metrics_dict is None:
                f.write("No data available for this split.\n")
                return

            # Basic performance metrics
            f.write("📊 PREDICTION PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Sample Count:           {metrics_dict['count']:,}\n")
            f.write(f"RMSE:                  {metrics_dict['rmse']:.4f} TECU\n")
            f.write(f"MAE:                   {metrics_dict['mae']:.4f} TECU\n")
            f.write(f"R²:                    {metrics_dict['r2']:.4f}\n")
            f.write(
                f"Pearson Correlation:   {metrics_dict['correlation']:.4f} (p={metrics_dict['p_value']:.2e})\n"
            )
            f.write(f"Bias (Mean Error):     {metrics_dict['bias']:.4f} TECU\n")
            f.write(
                f"Residual Std Dev:      {metrics_dict['std_residual']:.4f} TECU\n\n"
            )

            # Target statistics
            f.write("🎯 TARGET STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Mean Target STEC:      {metrics_dict['mean_target']:.4f} TECU\n")
            f.write(f"Target Std Dev:        {metrics_dict['std_target']:.4f} TECU\n")
            f.write(
                f"Mean Prediction:       {metrics_dict['mean_prediction']:.4f} TECU\n"
            )
            f.write(
                f"Prediction Std Dev:    {metrics_dict['std_prediction']:.4f} TECU\n\n"
            )

            # Error distribution
            f.write("📈 ERROR DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            f.write(
                f"Median Abs Error:      {metrics_dict['median_abs_error']:.4f} TECU\n"
            )
            f.write(
                f"75th Percentile AE:    {metrics_dict['q75_abs_error']:.4f} TECU\n"
            )
            f.write(
                f"95th Percentile AE:    {metrics_dict['q95_abs_error']:.4f} TECU\n"
            )
            f.write(
                f"Maximum Abs Error:     {metrics_dict['max_abs_error']:.4f} TECU\n\n"
            )

            # Uncertainty metrics (if available)
            if "mean_uncertainty" in metrics_dict:
                f.write("🔍 UNCERTAINTY ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"Mean Uncertainty:      {metrics_dict['mean_uncertainty']:.4f} TECU\n"
                )
                f.write(
                    f"Uncertainty Std Dev:   {metrics_dict['std_uncertainty']:.4f} TECU\n\n"
                )

                f.write("📊 COVERAGE ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(
                    f"1σ Coverage:           {metrics_dict['coverage_1sigma']:.2f}% (Expected: 68.27%)\n"
                )
                f.write(
                    f"2σ Coverage:           {metrics_dict['coverage_2sigma']:.2f}% (Expected: 95.45%)\n"
                )
                f.write(
                    f"3σ Coverage:           {metrics_dict['coverage_3sigma']:.2f}% (Expected: 99.73%)\n\n"
                )

                if "uncertainty_calibration" in metrics_dict:
                    f.write("🎯 UNCERTAINTY CALIBRATION\n")
                    f.write("-" * 40 + "\n")
                    f.write(
                        f"Calibration Correlation: {metrics_dict['uncertainty_calibration']:.4f}\n"
                    )
                    f.write("(How well uncertainty correlates with actual error)\n")
                    f.write(
                        f"Calibration p-value:     {metrics_dict['uncertainty_calibration_p']:.2e}\n\n"
                    )

            # Performance assessment
            f.write("🏆 PERFORMANCE ASSESSMENT\n")
            f.write("-" * 40 + "\n")

            # R² assessment
            if metrics_dict["r2"] >= 0.9:
                r2_assessment = "Excellent"
            elif metrics_dict["r2"] >= 0.8:
                r2_assessment = "Very Good"
            elif metrics_dict["r2"] >= 0.7:
                r2_assessment = "Good"
            elif metrics_dict["r2"] >= 0.5:
                r2_assessment = "Fair"
            else:
                r2_assessment = "Poor"

            f.write(f"R² Assessment:         {r2_assessment}\n")

            # Bias assessment
            if abs(metrics_dict["bias"]) < 0.5:
                bias_assessment = "Low bias (Good)"
            elif abs(metrics_dict["bias"]) < 1.0:
                bias_assessment = "Moderate bias"
            else:
                bias_assessment = "High bias (Concerning)"

            f.write(f"Bias Assessment:       {bias_assessment}\n")

            # Uncertainty calibration assessment (if available)
            if "coverage_1sigma" in metrics_dict:
                sigma1_diff = abs(metrics_dict["coverage_1sigma"] - 68.27)
                if sigma1_diff < 5:
                    calib_assessment = "Well-calibrated"
                elif sigma1_diff < 10:
                    calib_assessment = "Moderately calibrated"
                else:
                    calib_assessment = "Poorly calibrated"

                f.write(f"Uncertainty Calibration: {calib_assessment}\n")

    # Calculate metrics for total dataset
    total_metrics = calculate_metrics(df)
    write_metrics_file(
        "total_metrics_summary.txt",
        "TOTAL DATASET METRICS SUMMARY",
        total_metrics,
        f"Complete dataset analysis with {len(df):,} samples",
    )

    # Calculate metrics by year if year column exists
    if "year" in df.columns:
        unique_years = sorted(df["year"].unique())
        for year in unique_years:
            year_data = df[df["year"] == year]
            year_metrics = calculate_metrics(year_data)
            write_metrics_file(
                f"year_{year}_metrics_summary.txt",
                f"YEAR {year} METRICS SUMMARY",
                year_metrics,
                f"Year {year} analysis with {len(year_data):,} samples",
            )

    # Calculate metrics by month if both year and doy exist
    if "year" in df.columns and "doy" in df.columns:
        # Create month from doy (approximate)
        df_temp = df.copy()
        df_temp["month"] = ((df_temp["doy"] - 1) // 30.44 + 1).astype(int)
        df_temp["month"] = df_temp["month"].clip(1, 12)

        unique_months = sorted(df_temp["month"].unique())
        for month in unique_months:
            month_data = df_temp[df_temp["month"] == month]
            month_metrics = calculate_metrics(month_data)
            month_names = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            month_name = month_names[month - 1]
            write_metrics_file(
                f"month_{month:02d}_{month_name}_metrics_summary.txt",
                f"MONTH {month_name.upper()} METRICS SUMMARY",
                month_metrics,
                f"Month {month_name} analysis with {len(month_data):,} samples",
            )

    # Calculate metrics by season if doy exists
    if "doy" in df.columns:

        def get_season(doy):
            if doy < 80 or doy >= 355:
                return "Winter"
            elif doy < 172:
                return "Spring"
            elif doy < 266:
                return "Summer"
            else:
                return "Fall"

        df_temp = df.copy()
        df_temp["season"] = df_temp["doy"].apply(get_season)

        for season in ["Winter", "Spring", "Summer", "Fall"]:
            season_data = df_temp[df_temp["season"] == season]
            if len(season_data) > 0:
                season_metrics = calculate_metrics(season_data)
                write_metrics_file(
                    f"season_{season.lower()}_metrics_summary.txt",
                    f"{season.upper()} METRICS SUMMARY",
                    season_metrics,
                    f"{season} analysis with {len(season_data):,} samples",
                )
