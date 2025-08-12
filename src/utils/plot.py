import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_binned_boxplot(df, x_col, y_col, bins=20, output_dir='plots', bin_range_dict=None):
    """
    Plots a boxplot of y_col values grouped by binned x_col intervals.
    Allows feature-specific min/max binning through bin_range_dict.
    """
    df = df.copy()

    # Determine min and max from bin_range_dict if provided
    if bin_range_dict and x_col in bin_range_dict:
        min_val, max_val = bin_range_dict[x_col]
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        df['x_bin'] = pd.cut(df[x_col], bins=bin_edges, include_lowest=True)
    else:
        df['x_bin'] = pd.cut(df[x_col], bins=bins)

    grouped = df.groupby('x_bin')[y_col].apply(list)
    box_data = [grouped[bin] for bin in grouped.index]
    x_labels = [f"{b.left:.0f}–{b.right:.0f}" for b in grouped.index]

    plt.figure(figsize=(12, 6))
    plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5, zorder=1)
    plt.boxplot(box_data, labels=x_labels, showfliers=False, zorder=2)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{y_col} vs {x_col} (Binned Boxplot)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{y_col}_vs_{x_col}_boxplot.png')
    plt.close()



def plot_mae_vs_doy(df, output_dir='plots'):
    df = df.copy()
    df['mae'] = np.abs(df['target_stec'] - df['pred_stec'])
    plot_binned_boxplot(df, 'doy', 'mae', bins=24, output_dir=output_dir)


def plot_residuals_vs_feature(df, feature, num_bins=24, output_dir='plots', bin_range_dict=None):
    df = df.copy()
    df['residual'] = df['target_stec'] - df['pred_stec']
    plot_binned_boxplot(df, feature, 'residual', bins=num_bins, output_dir=output_dir, bin_range_dict=bin_range_dict)


def plot_prediction_scatter(df, output_dir):
    plt.figure(figsize=(6, 6))
    plt.scatter(df['target_stec'], df['pred_stec'], alpha=0.3)
    plt.plot([df['target_stec'].min(), df['target_stec'].max()],
             [df['target_stec'].min(), df['target_stec'].max()], 'r--')
    plt.xlabel('True STEC')
    plt.ylabel('Predicted STEC')
    plt.title('Predicted vs. True STEC')
    plt.grid(True)
    plt.savefig(f'{output_dir}/prediction_scatter.png')
    plt.close()


def plot_spatial_error_map(df, output_dir):
    plt.figure(figsize=(12, 6))
    heatmap_data = df.copy()
    heatmap_data['lon_bin'] = pd.cut(df['lon_sta'], bins=72)
    heatmap_data['lat_bin'] = pd.cut(df['lat_sta'], bins=36)

    # Save full category ranges
    lon_cats = heatmap_data['lon_bin'].cat.categories
    lat_cats = heatmap_data['lat_bin'].cat.categories

    # Group and compute residuals
    grouped = heatmap_data.groupby(['lon_bin', 'lat_bin'])[['target_stec', 'pred_stec']].mean().reset_index()
    grouped['residual'] = np.abs(grouped['target_stec'] - grouped['pred_stec'])

    # Pivot and reindex to include all bins
    pivot = grouped.pivot(index='lat_bin', columns='lon_bin', values='residual')
    pivot = pivot.reindex(index=lat_cats, columns=lon_cats)

    # Plot
    ax = sns.heatmap(pivot, cmap='viridis', cbar_kws={'label': 'Absolute Residual'})

    # Tick label formatting: every 5th bin only
    xticks = ax.get_xticks()
    xlabels = []
    for tick in xticks:
        idx = int(round(tick))
        if 0 <= idx < len(pivot.columns):
            if idx % 1 == 0:
                xlabels.append(int(round(pivot.columns[idx].mid)))
            else:
                xlabels.append("")
    ax.set_xticklabels(xlabels, rotation=45)

    yticks = ax.get_yticks()
    ylabels = []
    for tick in yticks:
        idx = int(round(tick))
        if 0 <= idx < len(pivot.index):
            if idx % 1 == 0:
                ylabels.append(int(round(pivot.index[idx].mid)))
            else:
                ylabels.append("")
    ax.set_yticklabels(ylabels, rotation=0)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Spatial Distribution of Errors (Heatmap)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/spatial_error_map_heatmap.png')
    plt.close()

def plot_histogram_of_residuals(df, output_dir):
    plt.figure(figsize=(8, 5))
    residuals = df['target_stec'] - df['pred_stec']
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residual (STEC)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(f'{output_dir}/residual_histogram.png')
    plt.close()


def plot_uncertainty_calibration(df, output_dir):
    plt.figure(figsize=(8, 5))
    abs_residual = np.abs(df['target_stec'] - df['pred_stec'])
    plt.scatter(df['pred_total_unc'], abs_residual, alpha=0.3)
    max_val = max(df['pred_total_unc'].max(), abs_residual.max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel('Predicted Total Uncertainty')
    plt.ylabel('|Residual|')
    plt.title('Uncertainty Calibration')
    plt.grid(True)
    plt.savefig(f'{output_dir}/uncertainty_calibration.png')
    plt.close()


def plot_az_el_heatmap(df, output_dir, metric='residual'):
    """
    Plots a heatmap of residuals or MAE by azimuth and elevation.

    Parameters:
    - df: DataFrame containing 'satazi', 'satele', 'target_stec', and 'pred_stec' columns.
    - output_dir: Directory to save the plot.
    - metric: Metric to plot ('residual' or 'mae').
    """
    plt.figure(figsize=(12, 6))
    heatmap_data = df.copy()

    # Create bin columns with fixed categories
    az_min, az_max = 0, 360  # Define min and max for azimuth
    el_min, el_max = 5, 90   # Define min and max for elevation

    heatmap_data['az_bin'] = pd.cut(df['satazi'], bins=np.linspace(az_min, az_max, 181))
    heatmap_data['el_bin'] = pd.cut(df['satele'], bins=np.linspace(el_min, el_max, 87))

    # Save full category ranges
    az_cats = heatmap_data['az_bin'].cat.categories
    el_cats = heatmap_data['el_bin'].cat.categories

    # Group and compute the desired metric
    grouped = heatmap_data.groupby(['az_bin', 'el_bin'])[['target_stec', 'pred_stec']].mean().reset_index()
    if metric == 'residual':
        grouped['value'] = grouped['target_stec'] - grouped['pred_stec']
        cbar_label = 'Residual'
        title = 'Residuals by Azimuth and Elevation'
        filename = 'az_el_residuals_heatmap.png'
    elif metric == 'mae':
        grouped['value'] = np.abs(grouped['target_stec'] - grouped['pred_stec'])
        cbar_label = 'Mean Absolute Error'
        title = 'Mean Absolute Error by Azimuth and Elevation'
        filename = 'az_el_mae_heatmap.png'
    else:
        raise ValueError("Invalid metric. Choose 'residual' or 'mae'.")

    # Pivot and reindex to include all bins
    pivot = grouped.pivot(index='el_bin', columns='az_bin', values='value')
    pivot = pivot.reindex(index=el_cats, columns=az_cats)

    # Plot heatmap
    ax = sns.heatmap(pivot, cmap='RdBu_r', cbar_kws={'label': cbar_label})

    # Tick label formatting
    xticks = ax.get_xticks()
    xlabels = []
    for tick in xticks:
        idx = int(round(tick))
        if 0 <= idx < len(pivot.columns):
            if idx % 1 == 0 or idx == 0:
                xlabels.append(int(round(pivot.columns[idx].mid)))
            else:
                xlabels.append("")
    ax.set_xticklabels(xlabels, rotation=90)

    yticks = ax.get_yticks()
    ylabels = []
    for tick in yticks:
        idx = int(round(tick))
        if 0 <= idx < len(pivot.index):
            if idx % 1 == 0 or idx == 0:
                ylabels.append(int(round(pivot.index[idx].mid)))
            else:
                ylabels.append("")
    ax.set_yticklabels(ylabels, rotation=0)

    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}')
    plt.close()


def modify_df(df):
    """
    Modifies the DataFrame to ensure nice plotting using feature registry.
    """
    # Convert seconds of day to hours if SOD exists
    if 'sod' in df.columns:
        df['time'] = df['sod'] / 3600
    
    # Convert Kp index if it exists
    if 'kp' in df.columns:
        df['kp_binned'] = df['kp'].apply(lambda x: int(round(x / 10)))
    
    return df

def get_default_bin_ranges(feature_registry):
    """Get default bin ranges from feature registry statistics."""
    bin_ranges = {}
    
    # Get ranges from registry for enabled features
    for feature_name in feature_registry.get_all_enabled_features():
        feature_norm = feature_registry._features[feature_name]['normalization']
        if feature_norm is not None:
            bin_ranges[feature_name] = (feature_norm[0], feature_norm[1])

    # Add derived features
    bin_ranges['time'] = (0, 24)  # SOD converted to hours
    if 'kp' in bin_ranges:
        bin_ranges['kp_binned'] = (0, 9)  # Kp index binned
    
    return bin_ranges

def plot_residuals_vs_date(df, output_dir='plots'):
    """
    Plots residuals aggregated by month using year and day-of-year.
    Creates boxplots for each month present in the test data.
    """
    df = df.copy()
    df['residual'] = df['target_stec'] - df['pred_stec']
    
    # Create datetime from year and doy
    def create_date(row):
        try:
            year = int(row['year'])
            doy = int(row['doy'])
            # Create date from year and day of year
            date = datetime(year, 1, 1) + timedelta(days=doy - 1)
            return date
        except:
            return None
    
    df['date'] = df.apply(create_date, axis=1)
    df = df.dropna(subset=['date'])
    
    # Extract year-month for grouping
    df['year_month'] = df['date'].dt.to_period('M')
    
    # Sort by year_month to ensure chronological order
    unique_months = sorted(df['year_month'].unique())
    
    # Prepare data for boxplot
    box_data = []
    month_labels = []
    
    for month in unique_months:
        month_residuals = df[df['year_month'] == month]['residual'].values
        if len(month_residuals) > 0:
            box_data.append(month_residuals)
            month_labels.append(str(month))
    
    if not box_data:
        print("No valid data for residuals vs date plot")
        return
    
    plt.figure(figsize=(15, 6))
    plt.axhline(y=0, color='r', linestyle='-', linewidth=0.5, zorder=1)
    
    # Create boxplot
    bp = plt.boxplot(box_data, labels=month_labels, showfliers=False, zorder=2)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Month')
    plt.ylabel('Residual (STEC)')
    plt.title('Residuals by Month')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/residuals_vs_date_monthly.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_test_metrics(test_df, output_dir='plots', feature_registry=None):
    output_dir = os.path.join(output_dir, 'test_metrics')
    ensure_dir(output_dir)

    test_df = modify_df(test_df)

    # Get bin ranges from feature registry if available
    if feature_registry:
        bin_range_dict = get_default_bin_ranges(feature_registry)
    else:
        # Fallback to hardcoded ranges
        bin_range_dict = {
            'time': (0, 24),
            'doy': (1, 366),
            'satele': (5, 90),
            'kp_binned': (0, 9),
            'dst': (-400, 100),
            'f107': (50, 300),
            'sunspot': (0, 250)
        }

    # Plot metrics for available features only
    available_features = test_df.columns.tolist()
    
    if 'doy' in available_features:
        plot_mae_vs_doy(test_df, output_dir)
    
    # Plot residuals vs date if year and doy are available
    if 'year' in available_features and 'doy' in available_features:
        plot_residuals_vs_date(test_df, output_dir)
    
    # Define feature-specific plot configurations
    plot_configs = [
        ('time', 24),
        ('doy', 24), 
        ('satele', 17),
        ('satazi', 24),
        ('kp_binned', 9),
        ('dst', 20),
        ('f107', 10),
        ('sunspot', 10),
        ('target_stec', 20),
        ('pred_stec', 20)
    ]
    
    for feature, num_bins in plot_configs:
        if feature in available_features:
            plot_residuals_vs_feature(test_df, feature, num_bins=num_bins, 
                                    output_dir=output_dir, bin_range_dict=bin_range_dict)

    plot_prediction_scatter(test_df, output_dir)
    
    # Only plot spatial/azimuth plots if the required features exist
    required_spatial = ['lon_sta', 'lat_sta']
    if all(col in available_features for col in required_spatial):
        plot_spatial_error_map(test_df, output_dir)
    
    required_directional = ['satazi', 'satele']
    if all(col in available_features for col in required_directional):
        plot_az_el_heatmap(test_df, output_dir, metric='mae')
        plot_az_el_heatmap(test_df, output_dir, metric='residual')

    plot_histogram_of_residuals(test_df, output_dir)
    plot_uncertainty_calibration(test_df, output_dir)
