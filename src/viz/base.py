"""
Base visualization utilities and configuration.

This module provides common plotting utilities, styling constants,
and shared functionality for all visualization modules.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
FIGSIZE_SQUARE = (12, 12)  # Square plots: scatter, correlation, calibration
FIGSIZE_WIDE = (16, 10)  # Wide plots: spatial maps, multi-panel layouts
FIGSIZE_HISTOGRAM = (14, 8)  # Histogram/distribution plots
FIGSIZE_HEATMAP = (16, 10)  # Heatmaps and spatial plots


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
