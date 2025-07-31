import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


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
    plot_binned_boxplot(df, 'DOY', 'mae', bins=24, output_dir=output_dir)


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
    heatmap_data['lon_bin'] = pd.cut(df['Lon_sta'], bins=72)
    heatmap_data['lat_bin'] = pd.cut(df['Lat_sta'], bins=36)

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


def plot_az_el_heatmap(df, output_dir):
    plt.figure(figsize=(12, 6))
    heatmap_data = df.copy()

    # Create bin columns with fixed categories
    heatmap_data['az_bin'] = pd.cut(df['Azimuth'], bins=72)
    heatmap_data['el_bin'] = pd.cut(df['Elevation'], bins=36)

    # Save full category ranges
    az_cats = heatmap_data['az_bin'].cat.categories
    el_cats = heatmap_data['el_bin'].cat.categories

    # Group and compute residuals
    grouped = heatmap_data.groupby(['az_bin', 'el_bin'])[['target_stec', 'pred_stec']].mean().reset_index()
    grouped['residual'] = grouped['target_stec'] - grouped['pred_stec']

    # Pivot and reindex to include all bins
    pivot = grouped.pivot(index='el_bin', columns='az_bin', values='residual')
    pivot = pivot.reindex(index=el_cats, columns=az_cats)

    # Plot heatmap
    ax = sns.heatmap(pivot, cmap='RdBu_r', center=0)

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

    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    plt.title('Residuals by Azimuth and Elevation')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/az_el_heatmap.png')
    plt.close()


def modify_df(df):
    """
    Modifies the DataFrame to ensure nice plotting.
    """
    df['time'] = df['SOD'] / 3600  # Convert seconds of day to hours
    df['Kp_index'] = df['Kp_index'].apply(lambda x: int(round(x / 10)))
    return df


def plot_test_metrics(test_df, output_dir='plots'):
    output_dir = os.path.join(output_dir, 'test_metrics')
    ensure_dir(output_dir)

    test_df = modify_df(test_df)

    # Define min/max bin ranges per feature
    bin_range_dict = {
        'time': (0, 24),
        'DOY': (1, 366),
        'Elevation': (5, 90),
        'Kp_index': (0, 9),
        'Dst_index': (-400, 100),
        'f107': (50, 300),
        'R_sunspot_number': (0, 250)
    }

    plot_mae_vs_doy(test_df, output_dir)
    plot_residuals_vs_feature(test_df, 'time', num_bins=24, output_dir=output_dir, bin_range_dict=bin_range_dict)
    plot_residuals_vs_feature(test_df, 'DOY', num_bins=24, output_dir=output_dir, bin_range_dict=bin_range_dict)
    plot_residuals_vs_feature(test_df, 'Elevation', num_bins=17, output_dir=output_dir, bin_range_dict=bin_range_dict)
    plot_residuals_vs_feature(test_df, 'Azimuth', num_bins=24, output_dir=output_dir, bin_range_dict=bin_range_dict)
    plot_residuals_vs_feature(test_df, 'Kp_index', num_bins=9, output_dir=output_dir, bin_range_dict=bin_range_dict)
    plot_residuals_vs_feature(test_df, 'Dst_index', num_bins=20, output_dir=output_dir, bin_range_dict=bin_range_dict)
    plot_residuals_vs_feature(test_df, 'f107', num_bins=10, output_dir=output_dir, bin_range_dict=bin_range_dict)
    plot_residuals_vs_feature(test_df, 'R_sunspot_number', num_bins=10, output_dir=output_dir, bin_range_dict=bin_range_dict)
    plot_residuals_vs_feature(test_df, 'target_stec', num_bins=20, output_dir=output_dir, bin_range_dict=bin_range_dict)
    plot_residuals_vs_feature(test_df, 'pred_stec', num_bins=20, output_dir=output_dir, bin_range_dict=bin_range_dict)

    plot_prediction_scatter(test_df, output_dir)
    plot_spatial_error_map(test_df, output_dir)
    plot_histogram_of_residuals(test_df, output_dir)
    plot_uncertainty_calibration(test_df, output_dir)
    plot_az_el_heatmap(test_df, output_dir)
