"""
Distribution and statistical visualization functions.

This module handles boxplots, histograms, and other statistical distribution plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict
from datetime import datetime, timedelta

from .base import (
    FIGSIZE_HISTOGRAM,
    get_scientific_label,
    save_plot,
)


def plot_binned_boxplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bins: int = 20,
    output_dir: str = "plots",
    bin_range_dict: Optional[Dict[str, tuple]] = None,
) -> None:
    """
    Plot boxplot of y_col values grouped by binned x_col intervals with MAE/RMSE overlay.

    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        bins: Number of bins for x_col
        output_dir: Directory to save plot
        bin_range_dict: Optional dict with (min, max) ranges for features
    """
    df = df.copy()

    # Determine bin edges
    if bin_range_dict and x_col in bin_range_dict:
        min_val, max_val = bin_range_dict[x_col]
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        df["x_bin"] = pd.cut(df[x_col], bins=bin_edges, include_lowest=True)
    else:
        df["x_bin"] = pd.cut(df[x_col], bins=bins)

    # Group data by bins and calculate statistics
    grouped = df.groupby("x_bin", observed=True)
    box_data = [grouped.get_group(bin)[y_col].tolist() for bin in grouped.groups]
    x_labels = [f"{(b.left):.0f}–{b.right:.0f}" for b in grouped.groups]

    # Calculate MAE and RMSE if y_col is residual
    if (
        y_col == "residual"
        and "target_stec" in df.columns
        and "pred_stec" in df.columns
    ):
        bin_stats = []
        for bin_val in grouped.groups:
            bin_data = grouped.get_group(bin_val)
            mae = np.mean(np.abs(bin_data[y_col]))
            rmse = np.sqrt(np.mean(bin_data[y_col] ** 2))
            bin_stats.append({"mae": mae, "rmse": rmse})
    else:
        bin_stats = None

    # Create plot
    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)

    # Add zero reference line
    ax.axhline(y=0, color="red", linestyle="-", linewidth=2, zorder=1, alpha=0.8)

    # Create boxplot with styling
    bp = ax.boxplot(
        box_data,
        labels=x_labels,
        showfliers=False,
        zorder=2,
        patch_artist=True,
        notch=False,
    )

    # Style the boxplot
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_linewidth(2)

    # Add MAE and RMSE lines if available (on same y-axis as boxplots)
    if bin_stats:
        positions = range(1, len(bin_stats) + 1)  # boxplot positions start at 1
        mae_values = [stat["mae"] for stat in bin_stats]
        rmse_values = [stat["rmse"] for stat in bin_stats]

        # Plot MAE and RMSE lines on the same axis as the boxplots
        ax.plot(
            positions,
            mae_values,
            color="green",
            marker="o",
            linewidth=3,
            markersize=8,
            label="MAE",
            alpha=0.9,
            zorder=10,
        )
        ax.plot(
            positions,
            rmse_values,
            color="orange",
            marker="s",
            linewidth=3,
            markersize=8,
            label="RMSE",
            alpha=0.9,
            zorder=10,
        )

        # Add legend for MAE/RMSE
        ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

    ax.tick_params(axis="x", rotation=45, labelsize=18)

    # Set labels
    x_label = get_scientific_label(x_col)
    y_label = get_scientific_label(y_col)

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(f"{y_label} vs {x_label}", fontweight="bold", pad=20)

    plt.tight_layout()
    filename = f"{y_col}_vs_{x_col}_boxplot.png"
    save_plot(fig, filename, output_dir)


def plot_binned_boxplot_clipped(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    bins: int = 20,
    output_dir: str = "plots",
    bin_range_dict: Optional[Dict[str, tuple]] = None,
    x_limits: Optional[tuple] = None,
    y_limits: Optional[tuple] = None,
    suffix: str = "_clipped",
) -> None:
    """
    Plot boxplot with axis clipping to focus on specific data ranges.

    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        bins: Number of bins for x_col
        output_dir: Directory to save plot
        bin_range_dict: Optional dict with (min, max) ranges for features
        x_limits: Optional (min, max) limits for x-axis
        y_limits: Optional (min, max) limits for y-axis
        suffix: Suffix to add to filename
    """
    df = df.copy()

    # Determine bin edges
    if bin_range_dict and x_col in bin_range_dict:
        min_val, max_val = bin_range_dict[x_col]
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        df["x_bin"] = pd.cut(df[x_col], bins=bin_edges, include_lowest=True)
    else:
        df["x_bin"] = pd.cut(df[x_col], bins=bins)

    # Group data by bins
    grouped = df.groupby("x_bin", observed=True)[y_col].apply(list)
    box_data = [grouped[bin] for bin in grouped.index]
    x_labels = [f"{(b.left):.0f}–{b.right:.0f}" for b in grouped.index]

    # Create plot
    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)

    # Add zero reference line
    ax.axhline(y=0, color="red", linestyle="-", linewidth=2, zorder=1, alpha=0.8)

    # Create boxplot with styling
    bp = ax.boxplot(
        box_data,
        labels=x_labels,
        showfliers=False,
        zorder=2,
        patch_artist=True,
        notch=False,
    )

    # Style the boxplot
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_linewidth(2)

    ax.tick_params(axis="x", rotation=45, labelsize=18)

    # Apply axis limits if specified
    if x_limits:
        ax.set_xlim(x_limits)
    if y_limits:
        ax.set_ylim(y_limits)

    # Set labels
    x_label = get_scientific_label(x_col)
    y_label = get_scientific_label(y_col)

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(f"{y_label} vs {x_label}", fontweight="bold", pad=20)

    plt.tight_layout()
    filename = f"{y_col}_vs_{x_col}_boxplot{suffix}.png"
    save_plot(fig, filename, output_dir)


def plot_histogram_of_residuals(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """Plot histogram of model residuals."""
    residuals = df["target_stec"] - df["pred_stec"]

    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)

    # Create histogram
    n, bins, patches = ax.hist(
        residuals, bins=50, density=True, alpha=0.7, color="skyblue", edgecolor="black"
    )

    # Add statistics text
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax.axvline(
        mean_res,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_res:.3f}",
    )
    ax.axvline(0, color="black", linestyle="-", linewidth=2, alpha=0.8)

    # Labels and title
    ax.set_xlabel("Residuals [TECU]", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.set_title("Distribution of Residuals", fontweight="bold", pad=20)
    ax.legend()

    # Add statistics text box
    textstr = f"Mean: {mean_res:.3f}\nStd: {std_res:.3f}\nN: {len(residuals)}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    save_plot(fig, "residuals_histogram.png", output_dir)


def plot_mae_vs_doy(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """Plot Mean Absolute Error vs Day of Year."""
    df = df.copy()
    df["mae"] = np.abs(df["target_stec"] - df["pred_stec"])
    plot_binned_boxplot(df, "doy", "mae", bins=24, output_dir=output_dir)


def plot_residuals_vs_feature(
    df: pd.DataFrame,
    feature: str,
    num_bins: int = 24,
    output_dir: str = "plots",
    bin_range_dict: Optional[Dict[str, tuple]] = None,
) -> None:
    """Plot residuals vs any feature with proper scientific formatting."""
    df = df.copy()
    df["residual"] = df["target_stec"] - df["pred_stec"]
    plot_binned_boxplot(
        df,
        feature,
        "residual",
        bins=num_bins,
        output_dir=output_dir,
        bin_range_dict=bin_range_dict,
    )


def plot_residuals_vs_feature_clipped(
    df: pd.DataFrame,
    feature: str,
    num_bins: int = 24,
    output_dir: str = "plots",
    bin_range_dict: Optional[Dict[str, tuple]] = None,
    x_limits: Optional[tuple] = None,
    y_limits: Optional[tuple] = None,
) -> None:
    """Plot residuals vs feature with axis clipping."""
    df = df.copy()
    df["residual"] = df["target_stec"] - df["pred_stec"]
    plot_binned_boxplot_clipped(
        df,
        feature,
        "residual",
        bins=num_bins,
        output_dir=output_dir,
        bin_range_dict=bin_range_dict,
        x_limits=x_limits,
        y_limits=y_limits,
    )


def plot_box_by_date(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Creates a two-panel plot showing RMSE/MAE and residual boxplots over time.

    Args:
        df: DataFrame with test results containing year, doy, target_stec, pred_stec
        output_dir: Directory to save the plot
    """
    from analysis.metrics import calc_rmse

    df = df.copy()
    df["error"] = df["target_stec"] - df["pred_stec"]
    df["ae"] = np.abs(df["error"])

    # Create datetime from year and doy
    def create_date(row):
        try:
            year = int(row["year"])
            doy = int(row["doy"])
            date = datetime(year, 1, 1) + timedelta(days=doy - 1)
            return date
        except (ValueError, TypeError, OverflowError):
            return None

    df["date"] = df.apply(create_date, axis=1)
    df = df.dropna(subset=["date"])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    order = sorted(df["year_month"].unique())
    pos = np.arange(len(order))

    
    # Set shared style
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
    plt.rcParams['figure.dpi'] = 300
    colors = sns.color_palette("colorblind")

    # Calculate metrics, ensuring order is maintained
    rmse_lat = (
        df.groupby("year_month")["error"].apply(calc_rmse).reindex(order).reset_index()
    )
    mae_lat = df.groupby("year_month")["ae"].mean().reindex(order).reset_index()

    grouped = df.groupby("year_month")["error"].apply(list).reindex(order)
    box_data = [grouped[date] for date in order]

    fig, ax = plt.subplots(figsize=(16, 8))

    # 1. Boxplot of residuals (Background)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.5, zorder=1, alpha=0.8)
    bp = ax.boxplot(
        box_data,
        widths=0.5,
        positions=pos,
        showfliers=False,
        zorder=2,
        patch_artist=True,
        notch=False,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(colors[2])
        patch.set_alpha(0.5)
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_linewidth(1.5)
            if element == 'medians':
                item.set_color('black')

    # 2. RMSE and MAE Lines (Foreground)
    ax.plot(pos, rmse_lat["error"], marker="o", label="RMSE", color=colors[0], linewidth=3, markersize=8, zorder=3)
    ax.plot(pos, mae_lat["ae"], marker="s", label="MAE", color=colors[1], linewidth=3, markersize=8, zorder=3)

    # Styling
    ax.set_xticks(pos)
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_ylim([-30, 30])
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Residual / Error [TECU]")

    # Legend
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.title("Monthly Performance Metrics", fontweight="bold", y=1.02)
    plt.tight_layout()
    save_plot(fig, "year_month_summary.png", output_dir)
    plt.close(fig)


def plot_residuals_vs_local_time(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Plot residuals vs local solar time with combined boxplot and metrics overlay.

    Args:
        df: DataFrame with residuals and time columns
        output_dir: Directory to save plot
    """
    import logging

    logger = logging.getLogger(__name__)

    if "time" not in df.columns:
        if "sod" in df.columns:
            df = df.copy()
            df["time"] = df["sod"] / 3600.0  # Convert seconds to hours
        else:
            logger.warning(
                "Neither 'time' nor 'sod' column found. Skipping local time analysis."
            )
            return
    else:
        df = df.copy()
    df["residual"] = df["target_stec"] - df["pred_stec"]
    df["abs_residual"] = np.abs(df["residual"])

    # Create hourly bins (0-24 hours)
    df["hour_bin"] = pd.cut(df["time"], bins=24, labels=range(24))

    # Calculate statistics per hour
    hourly_stats = (
        df.groupby("hour_bin", observed=True)
        .agg({"residual": ["mean", "std", "count"], "abs_residual": "mean"})
        .reset_index()
    )

    # Flatten column names
    hourly_stats.columns = ["hour", "mean_residual", "std_residual", "count", "mae"]

    # Calculate RMSE for each hour
    rmse_list = []
    for hour in hourly_stats["hour"]:
        hour_data = df[df["hour_bin"] == hour]
        if len(hour_data) > 0:
            rmse = np.sqrt(
                np.mean((hour_data["target_stec"] - hour_data["pred_stec"]) ** 2)
            )
            rmse_list.append(rmse)
        else:
            rmse_list.append(np.nan)

    hourly_stats["rmse"] = rmse_list
    hourly_stats["hour_numeric"] = hourly_stats["hour"].astype(int)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set x-axis limits and ticks first to ensure full range is shown
    ax.set_xlim(-0.5, 23.5)
    
    # Use ticker to ensure only the desired ticks are shown
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(0, 24, 3)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(i) for i in range(0, 24, 3)]))

    # Prepare boxplot data - ensure all 24 hours are represented
    box_data = []
    box_positions = []
    mae_values = []
    rmse_values = []
    valid_hours = []
    
    for hour in range(24):
        hour_residuals = df[df["hour_bin"] == hour]["residual"].values
        hour_stats = hourly_stats[hourly_stats["hour"] == hour]
        
        if len(hour_residuals) >= 5:  # Only include hours with sufficient data for boxplot
            box_data.append(hour_residuals)
            box_positions.append(hour)
            valid_hours.append(hour)
        
        # Always include MAE/RMSE data if available
        if not hour_stats.empty:
            mae_values.append(hour_stats["mae"].iloc[0])
            rmse_values.append(hour_stats["rmse"].iloc[0])
        else:
            mae_values.append(np.nan)
            rmse_values.append(np.nan)

    # Create boxplot
    if box_data:
        ax.boxplot(
            box_data,
            positions=valid_hours,  # Use actual hour numbers, not sequential positions
            widths=0.6,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", edgecolor="black", alpha=0.7),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="red", linewidth=2),
        )

    # Plot MAE and RMSE lines for all 24 hours
    hours_range = list(range(24))
    ax.plot(
        hours_range,
        mae_values,
        color="green",
        marker="o",
        linewidth=3,
        markersize=8,
        label="MAE",
        alpha=0.9,
        zorder=10,
    )
    ax.plot(
        hours_range,
        rmse_values,
        color="orange",
        marker="s",
        linewidth=3,
        markersize=8,
        label="RMSE",
        alpha=0.9,
        zorder=10,
    )

    # Add legend
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

    # Styling
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, linewidth=2)
    ax.set_xlabel("Local Solar Time [hours]", fontweight="bold", fontsize=14)
    ax.set_ylabel("Residual [TECU]", fontweight="bold", fontsize=14)
    ax.set_title(
        "Residuals vs Local Solar Time", fontweight="bold", pad=20, fontsize=16
    )

    # Ensure x-axis shows full range (already set above, but reinforce)
    ax.set_xlim(-0.5, 23.5)
    
    # Set ticks after all plotting to ensure they take precedence
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([str(i) for i in range(0, 24, 3)])
    
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, "residuals_vs_local_time.png", output_dir)


def plot_residuals_vs_solar_indices(
    df: pd.DataFrame, output_dir: str = "plots"
) -> None:
    """
    Plot residuals vs solar indices (F10.7 and Dst) with combined boxplot and metrics overlay.

    Args:
        df: DataFrame with residuals and solar index columns
        output_dir: Directory to save plot
    """
    import logging

    logger = logging.getLogger(__name__)

    df = df.copy()
    df["residual"] = df["target_stec"] - df["pred_stec"]
    df["abs_residual"] = np.abs(df["residual"])

    # Apply OMNI2 scaling corrections for proper units in plotting
    if "Kp_index" in df.columns:
        df["Kp_index"] = df["Kp_index"] / 10.0  # Kp stored as Kp*10 in OMNI2

    # Check which solar indices are available
    solar_indices = []
    
    # F10.7 Solar Flux
    if "f107_index" in df.columns:
        solar_indices.append(("f107_index", "F10.7 Solar Flux [sfu]"))
    
    # Dst Index
    if "Dst-index,_nT" in df.columns:
        solar_indices.append(("Dst-index,_nT", "Dst Index [nT]"))
    
    # Kp Index
    if "Kp_index" in df.columns:
        solar_indices.append(("Kp_index", "Kp Index"))
    
    # Sunspot Number
    if "R_Sunspot_No" in df.columns:
        solar_indices.append(("R_Sunspot_No", "Sunspot Number"))
    
    # AE Index
    if "AE-index,_nT" in df.columns:
        solar_indices.append(("AE-index,_nT", "AE Index [nT]"))

    if not solar_indices:
        logger.warning(
            "No solar indices (f107_index, Dst-index, Kp_index, R_Sunspot_No, AE-index) found. Skipping solar index analysis."
        )
        return

    # Create subplots for each available solar index
    n_indices = len(solar_indices)
    fig, axes = plt.subplots(n_indices, 1, figsize=(16, 6 * n_indices))
    if n_indices == 1:
        axes = [axes]

    for i, (col, label) in enumerate(solar_indices):
        ax = axes[i]

        # Create appropriate bins for each solar index
        if col == "Kp_index":
            # Kp is reported as integers 0-9, round fractional values to nearest integer
            df[col] = np.round(df[col]).astype(int)
            # Create bins for all possible Kp values (0-9), even if some are empty
            bins = np.arange(0, 11, 1)  # [0,1), [1,2), ..., [9,10)
            df[f"{col}_bin"] = pd.cut(df[col], bins=bins, right=False, include_lowest=True)
            
            # Ensure all Kp bins (0-9) are represented, even if empty
            all_kp_bins = [pd.Interval(left=i, right=i+1, closed='left') for i in range(10)]
            existing_bins = df[f"{col}_bin"].cat.categories
            missing_bins = [b for b in all_kp_bins if b not in existing_bins]
            
            if missing_bins:
                # Add missing bins to categorical
                df[f"{col}_bin"] = df[f"{col}_bin"].cat.add_categories(missing_bins)
        elif col == "Dst-index,_nT":
            # Dst ranges ~ -500 to 100, use 50 nT bins
            bins = np.arange(-500, 101, 50)
            df[f"{col}_bin"] = pd.cut(df[col], bins=bins, right=False, include_lowest=True)
        elif col == "AE-index,_nT":
            # AE ranges 0-2500, use 200 nT bins
            bins = np.arange(0, 2501, 200)
            df[f"{col}_bin"] = pd.cut(df[col], bins=bins, right=False, include_lowest=True)
        elif col == "ap_index,_nT":
            # ap ranges 0-400, use 25 nT bins
            bins = np.arange(0, 401, 25)
            df[f"{col}_bin"] = pd.cut(df[col], bins=bins, right=False, include_lowest=True)
        elif col == "f107_index":
            # F10.7 ranges ~60-420, use 30 sfu bins
            bins = np.arange(60, 421, 30)
            df[f"{col}_bin"] = pd.cut(df[col], bins=bins, right=False, include_lowest=True)
        elif col == "R_Sunspot_No":
            # Sunspot number ranges 0-300, use 25 unit bins
            bins = np.arange(0, 301, 25)
            df[f"{col}_bin"] = pd.cut(df[col], bins=bins, right=False, include_lowest=True)
        else:
            # Fallback to quantile bins for unknown indices
            n_bins = 15
            df[f"{col}_bin"] = pd.qcut(df[col], q=n_bins, duplicates="drop")

        # Calculate statistics per bin
        if col == "Kp_index":
            # For Kp, create stats for all possible values (0-9), even if empty
            all_kp_intervals = [pd.Interval(left=i, right=i+1, closed='left') for i in range(10)]
            bin_stats = []
            
            for interval in all_kp_intervals:
                bin_data = df[df[f"{col}_bin"] == interval]
                if len(bin_data) > 0:
                    rmse = np.sqrt(np.mean((bin_data["target_stec"] - bin_data["pred_stec"]) ** 2))
                    stats = {
                        "bin": interval,
                        "bin_center": bin_data[col].mean(),
                        "mean_residual": bin_data["residual"].mean(),
                        "std_residual": bin_data["residual"].std() if len(bin_data) > 1 else 0,
                        "count": len(bin_data),
                        "mae": bin_data["abs_residual"].mean(),
                        "rmse": rmse
                    }
                else:
                    stats = {
                        "bin": interval,
                        "bin_center": interval.left + 0.5,  # Center of interval
                        "mean_residual": 0,
                        "std_residual": 0,
                        "count": 0,
                        "mae": 0,
                        "rmse": 0
                    }
                bin_stats.append(stats)
            
            bin_stats = pd.DataFrame(bin_stats)
        else:
            bin_stats = (
                df.groupby(f"{col}_bin", observed=True)
                .agg(
                    {
                        col: "mean",
                        "residual": ["mean", "std", "count"],
                        "abs_residual": "mean",
                    }
                )
                .reset_index()
            )

            # Flatten column names
            bin_stats.columns = [
                "bin",
                "bin_center",
                "mean_residual",
                "std_residual",
                "count",
                "mae",
            ]

            # Calculate RMSE for each bin
            rmse_list = []
            for bin_val in bin_stats["bin"]:
                bin_data = df[df[f"{col}_bin"] == bin_val]
                if len(bin_data) > 0:
                    rmse = np.sqrt(
                        np.mean((bin_data["target_stec"] - bin_data["pred_stec"]) ** 2)
                    )
                    rmse_list.append(rmse)
                else:
                    rmse_list.append(np.nan)

            bin_stats["rmse"] = rmse_list

        # Prepare boxplot data
        box_data = []
        box_positions = []
        for j, bin_val in enumerate(bin_stats["bin"]):
            bin_residuals = df[df[f"{col}_bin"] == bin_val]["residual"].values
            if len(bin_residuals) >= 5:  # Only include bins with sufficient data
                box_data.append(bin_residuals)
                box_positions.append(j)

        # Create boxplot
        if box_data:
            ax.boxplot(
                box_data,
                positions=box_positions,
                widths=0.6,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor="lightblue", edgecolor="black", alpha=0.7),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                medianprops=dict(color="red", linewidth=2),
            )

        # Plot MAE and RMSE lines on same axis as boxplots
        ax.plot(
            range(len(bin_stats)),
            bin_stats["mae"],
            color="green",
            marker="o",
            linewidth=3,
            markersize=8,
            label="MAE",
            alpha=0.9,
            zorder=10,
        )
        ax.plot(
            range(len(bin_stats)),
            bin_stats["rmse"],
            color="orange",
            marker="s",
            linewidth=3,
            markersize=8,
            label="RMSE",
            alpha=0.9,
            zorder=10,
        )

        # Styling
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, linewidth=2)
        ax.set_xlabel(label, fontweight="bold", fontsize=14)
        ax.set_ylabel("Residual [TECU]", fontweight="bold", fontsize=14)
        ax.set_title(f"Residuals vs {label}", fontweight="bold", pad=20, fontsize=16)

        # Add legend
        ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

        # Format x-axis with appropriate labeling for each solar index
        if col == "Kp_index":
            # For Kp, show all integer values (0-9) 
            tick_positions = list(range(10))  # All Kp values 0-9
            tick_labels = [f'{i}' for i in range(10)]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
        elif col in ["Dst-index,_nT", "AE-index,_nT", "ap_index,_nT", "f107_index", "R_Sunspot_No"]:
            # For other indices, use smart tick spacing to avoid overlap
            n_bins = len(bin_stats)
            if n_bins <= 8:
                # Show all bins
                tick_indices = list(range(n_bins))
                tick_labels = [
                    f'[{int(bin_stats.iloc[idx]["bin"].left)}, {int(bin_stats.iloc[idx]["bin"].right)})' 
                    for idx in tick_indices
                ]
            else:
                # Show every other bin or use smart spacing
                step = max(1, n_bins // 8)
                tick_indices = list(range(0, n_bins, step))
                tick_labels = [
                    f'[{int(bin_stats.iloc[idx]["bin"].left)}, {int(bin_stats.iloc[idx]["bin"].right)})' 
                    for idx in tick_indices
                ]
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_labels)
            # Rotate labels for better readability
            ax.tick_params(axis='x', rotation=45)
            # Set horizontal alignment for rotated labels
            for label in ax.get_xticklabels():
                label.set_ha('right')
        else:
            # Fallback for quantile bins
            n_ticks = min(8, len(bin_stats))
            tick_indices = np.linspace(0, len(bin_stats) - 1, n_ticks, dtype=int)
            tick_labels = [
                f'{bin_stats.iloc[idx]["bin_center"]:.1f}' for idx in tick_indices
            ]
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_labels)

        ax.set_xlim(-0.5, len(bin_stats) - 0.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # Add extra space at the bottom for rotated x-axis labels
    plt.subplots_adjust(bottom=0.15)
    save_plot(fig, "residuals_vs_solar_indices.png", output_dir)
