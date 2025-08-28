import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm

# Set presentation-ready matplotlib parameters
plt.rcParams.update({
    'font.size': 16,           # Base font size (increased from 14)
    'axes.titlesize': 22,      # Title font size (increased from 18)
    'axes.labelsize': 20,      # Axis label font size (increased from 16)
    'xtick.labelsize': 16,     # X-tick label size (increased from 14)
    'ytick.labelsize': 16,     # Y-tick label size (increased from 14)
    'legend.fontsize': 16,     # Legend font size (increased from 12)
    'figure.titlesize': 24,    # Figure title size (increased from 20)
    'axes.grid': True,         # Enable grid by default
    'grid.alpha': 0.3,         # Grid transparency
    'lines.linewidth': 2,      # Thicker lines
    'axes.linewidth': 1.2,     # Thicker axes
    'xtick.major.width': 1.2,  # Thicker tick marks
    'ytick.major.width': 1.2,
    'figure.dpi': 300,         # High resolution
    'savefig.dpi': 300,        # High DPI for saved figures
    'savefig.bbox': 'tight',   # Tight bounding box
})

# Standardized figure sizes for consistent text scaling
FIGSIZE_SQUARE = (12, 12)      # Square plots: scatter, correlation, calibration
FIGSIZE_WIDE = (16, 10)        # Wide plots: spatial maps, multi-panel layouts  
FIGSIZE_HISTOGRAM = (14, 8)    # Histogram/distribution plots
FIGSIZE_HEATMAP = (16, 10)     # Heatmaps and spatial plots


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
    x_labels = [f"{(b.left):.0f}–{b.right:.0f}" for b in grouped.index]

    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)
    
    # Add zero reference line
    ax.axhline(y=0, color='red', linestyle='-', linewidth=2, zorder=1, alpha=0.8)
    
    # Create boxplot with better styling
    bp = ax.boxplot(box_data, labels=x_labels, showfliers=False, zorder=2,
                   patch_artist=True, notch=False)
    
    # Style the boxplot
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_linewidth(2)
    
    ax.tick_params(axis='x', rotation=45, labelsize=18)
    
    # Improved axis labels based on column names
    x_label = get_scientific_label(x_col)
    y_label = get_scientific_label(y_col)
    
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title(f'{y_label} vs {x_label}', fontweight='bold', pad=20)
    
    plt.tight_layout()
    filename = f'{output_dir}/{y_col}_vs_{x_col}_boxplot.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def get_scientific_label(column_name):
    """Convert column names to scientific presentation labels"""
    label_mapping = {
        'target_stec': 'True STEC [TECU]',
        'pred_stec': 'Predicted STEC [TECU]',
        'residual': 'Residual [TECU]',
        'mae': 'Mean Absolute Error [TECU]',
        'doy': 'Day of Year',
        'sod': 'Seconds of Day [s]',
        'time': 'Local Solar Time [h]',
        'satele': 'Elevation Angle [°]',
        'satazi': 'Azimuth Angle [°]',
        'lat_ipp': 'IPP Latitude [°]',
        'lon_ipp': 'IPP Longitude [°]',
        'sm_lat_ipp': 'Solar Magnetic IPP Latitude [°]',
        'sm_lon_ipp': 'Solar Magnetic IPP Longitude [°]',
        'kp': 'Kp Index',
        'kp_binned': 'Kp Index (binned)',
        'dst': 'Dst Index [nT]',
        'f107': 'F10.7 Solar Flux [sfu]',
        'sunspot': 'Sunspot Number',
        'year': 'Year',
        'pred_total_unc': 'Total Uncertainty [TECU]',
        'pred_epistemic_unc': 'Epistemic Uncertainty [TECU]',
        'pred_aleatoric_unc': 'Aleatoric Uncertainty [TECU]',
    }
    return label_mapping.get(column_name, column_name.replace('_', ' ').title())


def plot_mae_vs_doy(df, output_dir='plots'):
    """Plot Mean Absolute Error vs Day of Year"""
    df = df.copy()
    df['mae'] = np.abs(df['target_stec'] - df['pred_stec'])
    plot_binned_boxplot(df, 'doy', 'mae', bins=24, output_dir=output_dir)


def plot_residuals_vs_feature(df, feature, num_bins=24, output_dir='plots', bin_range_dict=None):
    """Plot residuals vs any feature with proper scientific formatting"""
    df = df.copy()
    df['residual'] = df['target_stec'] - df['pred_stec']
    plot_binned_boxplot(df, feature, 'residual', bins=num_bins, output_dir=output_dir, bin_range_dict=bin_range_dict)


def plot_binned_boxplot_clipped(df, x_col, y_col, bins=20, output_dir='plots', bin_range_dict=None, 
                               x_limits=None, y_limits=None, suffix='_clipped'):
    """
    Plots a boxplot of y_col values grouped by binned x_col intervals with axis clipping.
    Allows feature-specific min/max binning through bin_range_dict.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis  
        bins: Number of bins
        output_dir: Output directory
        bin_range_dict: Dictionary with bin ranges
        x_limits: Tuple (x_min, x_max) for x-axis limits
        y_limits: Tuple (y_min, y_max) for y-axis limits
        suffix: Suffix to add to filename
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
    x_labels = [f"{(b.left):.0f}–{b.right:.0f}" for b in grouped.index]

    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)
    
    # Add zero reference line
    ax.axhline(y=0, color='red', linestyle='-', linewidth=2, zorder=1, alpha=0.8)
    
    # Create boxplot with better styling
    bp = ax.boxplot(box_data, labels=x_labels, showfliers=False, zorder=2,
                   patch_artist=True, notch=False)
    
    # Style the boxplot
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_linewidth(2)
    
    # Apply axis limits if specified
    if x_limits:
        ax.set_xlim(x_limits)
    if y_limits:
        ax.set_ylim(y_limits)
    
    ax.tick_params(axis='x', rotation=45, labelsize=18)
    
    # Improved axis labels based on column names
    x_label = get_scientific_label(x_col)
    y_label = get_scientific_label(y_col)
    
    ax.set_xlabel(x_label, fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title(f'{y_label} vs {x_label}', fontweight='bold', pad=20)
    
    plt.tight_layout()
    filename = f'{output_dir}/{y_col}_vs_{x_col}_boxplot{suffix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals_vs_feature_clipped(df, feature, num_bins=24, output_dir='plots', bin_range_dict=None,
                                     x_limits=None, y_limits=None):
    """Plot residuals vs any feature with axis clipping"""
    df = df.copy()
    df['residual'] = df['target_stec'] - df['pred_stec']
    plot_binned_boxplot_clipped(df, feature, 'residual', bins=num_bins, output_dir=output_dir, 
                               bin_range_dict=bin_range_dict, x_limits=x_limits, y_limits=y_limits)


def plot_prediction_scatter(df, output_dir):
    """Create comprehensive prediction scatter plots with improved presentation formatting"""
    
    # Create hexagonal density plot with logarithmic scaling
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    
    hb = ax.hexbin(df['target_stec'], df['pred_stec'], gridsize=100, cmap='BuGn', 
                   mincnt=1, norm=LogNorm(), alpha=0.8)
    
    # Add perfect prediction line
    min_val = min(df['target_stec'].min(), df['pred_stec'].min())
    max_val = max(df['target_stec'].max(), df['pred_stec'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=3, 
            label='Perfect Prediction', alpha=0.9)
    
    ax.set_xlabel('True STEC [TECU]', fontweight='bold')
    ax.set_ylabel('Predicted STEC [TECU]', fontweight='bold')
    ax.set_title('Predicted vs True STEC', fontweight='bold', pad=20)
    
    # colorbar
    cbar = plt.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Number of Points', fontweight='bold', rotation=270, labelpad=35)
    cbar.ax.tick_params(labelsize=16)
    
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_density.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2D histogram version with log scale
    min_v = float(np.nanmin([df['target_stec'].min(), df['pred_stec'].min()]))
    max_v = float(np.nanmax([df['target_stec'].max(), df['pred_stec'].max()]))

    fig, ax = plt.subplots(figsize=(12, 12))

    # Force the histogram to use the same square data range
    h = ax.hist2d(
        df['target_stec'], df['pred_stec'],
        bins=100, cmap='BuGn', norm=LogNorm(),
        range=[[min_v, max_v], [min_v, max_v]], alpha=0.8
    )

    # Perfect diagonal within that same range
    ax.plot([min_v, max_v], [min_v, max_v], 'r-', linewidth=3, label='Perfect Prediction', alpha=0.9)

    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('True STEC [TECU]', fontweight='bold')
    ax.set_ylabel('Predicted STEC [TECU]', fontweight='bold')
    ax.set_title('Predicted vs True STEC', fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)

    # keep colorbar from squeezing the square axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(h[3], cax=cax)
    cbar.set_label('Density', fontweight='bold', rotation=270, labelpad=35)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_hist2d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Standard scatter plot for comparison
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    
    ax.scatter(df['target_stec'], df['pred_stec'], alpha=0.3, s=1, c='blue')
    ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=3, 
            label='Perfect Prediction', alpha=0.9)
    
    ax.set_xlabel('True STEC [TECU]', fontweight='bold')
    ax.set_ylabel('Predicted STEC [TECU]', fontweight='bold')
    ax.set_title('Predicted vs True STEC', fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_spatial_error_map(df, output_dir):
    """Create geographic heatmap with coastlines using cartopy - presentation ready"""
    fig = plt.figure(figsize=FIGSIZE_HEATMAP)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    heatmap_data = df.copy()
    
    # Define longitude and latitude bin edges
    lon_edges = np.linspace(-180, 180, 145)
    lat_edges = np.linspace(-90, 90, 73)
    
    # Create bins
    heatmap_data['lon_bin'] = pd.cut(df['lon_ipp'], bins=lon_edges)
    heatmap_data['lat_bin'] = pd.cut(df['lat_ipp'], bins=lat_edges)

    # Group and compute residuals
    grouped = heatmap_data.groupby(['lon_bin', 'lat_bin'])[['target_stec', 'pred_stec']].mean().reset_index()
    grouped['residual'] = np.abs(grouped['target_stec'] - grouped['pred_stec'])
    
    # Get bin centers for plotting
    grouped['lon_center'] = grouped['lon_bin'].apply(lambda x: x.mid)
    grouped['lat_center'] = grouped['lat_bin'].apply(lambda x: x.mid)
    
    # Create 2D grid for heatmap
    lon_centers = np.array([interval.mid for interval in pd.cut([], bins=lon_edges).categories])
    lat_centers = np.array([interval.mid for interval in pd.cut([], bins=lat_edges).categories])
    
    # Initialize grid with NaN
    Z = np.full((len(lat_centers), len(lon_centers)), np.nan)
    
    # Fill grid with residual values
    for _, row in grouped.iterrows():
        lon_idx = np.argmin(np.abs(lon_centers - row['lon_center']))
        lat_idx = np.argmin(np.abs(lat_centers - row['lat_center']))
        Z[lat_idx, lon_idx] = row['residual']
    
    # Create meshgrid for plotting
    LON, LAT = np.meshgrid(lon_centers, lat_centers)
    
    # Plot heatmap using pcolormesh
    vmax = np.nanpercentile(Z, 95)  # clip top 5% of values
    im = ax.pcolormesh(LON, LAT, Z, cmap='coolwarm', shading='auto', 
                    transform=ccrs.PlateCarree(), alpha=0.8,
                    vmin=0, vmax=vmax)
    
    # Add coastlines and geographic features with better styling
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2, color='black', alpha=0.8)
    
    # Add gridlines with improved formatting
    gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                     linewidth=1, alpha=0.6, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 14, 'weight': 'bold'}
    gl.ylabel_style = {'size': 14, 'weight': 'bold'}
    
    # Set global extent
    ax.set_global()
    
    # Add improved colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05, extend='max')
    cbar.set_label('Absolute Residual [TECU]', rotation=270, labelpad=35, 
                   fontweight='bold', fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    
    ax.set_title('Spatial Distribution of STEC Prediction Errors', 
                fontweight='bold', fontsize=20, pad=25)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/spatial_error_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_spatial_error_map_by_local_time(df, output_dir):
    """
    Create spatial error heatmaps for each hour of local time.
    Requires the DataFrame to have 'time' column (local solar time in hours).
    """
    ensure_dir(os.path.join(output_dir, 'spatial_by_time'))
    
    # Ensure we have the time column (should be created by modify_df)
    if 'time' not in df.columns:
        print("Warning: 'time' column not found. Make sure modify_df() was called first.")
        return
    
    # Create hourly bins for local time
    df_time = df.copy()
    df_time['time_bin'] = pd.cut(df['time'], bins=np.arange(0, 25, 1), include_lowest=True, right=False)
    
    # Group by time bins
    time_groups = df_time.groupby('time_bin')
    
    # Define spatial bin edges
    lon_edges = np.linspace(-180, 180, 145)
    lat_edges = np.linspace(-90, 90, 73)
    
    for time_bin, group_df in time_groups:
        if len(group_df) < 10:  # Skip if too few observations
            continue
            
        # Create the spatial heatmap for this time bin
        fig = plt.figure(figsize=FIGSIZE_HEATMAP)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        heatmap_data = group_df.copy()
        
        # Create spatial bins
        heatmap_data['lon_bin'] = pd.cut(group_df['lon_ipp'], bins=lon_edges)
        heatmap_data['lat_bin'] = pd.cut(group_df['lat_ipp'], bins=lat_edges)

        # Group and compute residuals
        grouped = heatmap_data.groupby(['lon_bin', 'lat_bin'])[['target_stec', 'pred_stec']].mean().reset_index()
        grouped['residual'] = np.abs(grouped['target_stec'] - grouped['pred_stec'])
        
        # Get bin centers for plotting
        grouped['lon_center'] = grouped['lon_bin'].apply(lambda x: x.mid)
        grouped['lat_center'] = grouped['lat_bin'].apply(lambda x: x.mid)
        
        # Create 2D grid for heatmap
        lon_centers = np.array([interval.mid for interval in pd.cut([], bins=lon_edges).categories])
        lat_centers = np.array([interval.mid for interval in pd.cut([], bins=lat_edges).categories])
        
        # Initialize grid with NaN
        Z = np.full((len(lat_centers), len(lon_centers)), np.nan)
        
        # Fill grid with residual values
        for _, row in grouped.iterrows():
            lon_idx = np.argmin(np.abs(lon_centers - row['lon_center']))
            lat_idx = np.argmin(np.abs(lat_centers - row['lat_center']))
            Z[lat_idx, lon_idx] = row['residual']
        
        # Create meshgrid for plotting
        LON, LAT = np.meshgrid(lon_centers, lat_centers)
        
        # Plot heatmap using pcolormesh
        vmax = np.nanpercentile(Z, 95)  # clip top 5% of values
        im = ax.pcolormesh(LON, LAT, Z, cmap='coolwarm', shading='auto', 
                        transform=ccrs.PlateCarree(), alpha=0.8,
                        vmin=0, vmax=vmax)
        
        # Add coastlines and geographic features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False,
                         linewidth=0.5, alpha=0.5, color='gray')
        gl.top_labels = False
        gl.right_labels = False
        
        # Set global extent
        ax.set_global()
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05, extend='max')
        cbar.set_label('Absolute Residual [TECU]', rotation=270, labelpad=35,
                       fontweight='bold', fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        
        # Create title with time range
        time_start = int(time_bin.left)
        time_end = int(time_bin.right)
        plt.title(f'Spatial Distribution of Errors {time_start:02d}:00-{time_end:02d}:00 Local Solar Time',
                  fontweight='bold', fontsize=20, pad=20)
        plt.tight_layout()
        
        # Save with time in filename
        plt.savefig(f'{output_dir}/spatial_by_time/spatial_error_heatmap_{time_start:02d}h.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def plot_solar_magnetic_ipp_error_map(df, output_dir):
    """
    Spatial error heatmap for solar magnetic IPPs (no coastlines).
    Keeps equal data aspect and avoids colorbar squeezing.
    """

    # ---- Config ----
    lon_col = 'sm_lon_ipp'
    lat_col = 'sm_lat_ipp'
    coord_type = 'Solar Magnetic'
    filename = 'solar_magnetic_ipp_error_heatmap.png'

    # Edges (~2.5° bins)
    lon_edges = np.linspace(-180, 180, 145)
    lat_edges = np.linspace(-90, 90, 73)

    # Guard: keep only rows with required values
    cols_needed = [lon_col, lat_col, 'target_stec', 'pred_stec']
    data = df[cols_needed].dropna(subset=[lon_col, lat_col, 'target_stec', 'pred_stec']).copy()

    if data.empty:
        print("No data after dropping NaNs — nothing to plot.")
        return

    # Bin using pd.cut (right-closed, include_lowest to capture -180/-90)
    data['lon_bin'] = pd.cut(data[lon_col], bins=lon_edges, include_lowest=True, right=True)
    data['lat_bin'] = pd.cut(data[lat_col], bins=lat_edges, include_lowest=True, right=True)

    # Categories from the actual cuts (critical to avoid mismatch)
    lon_idx = data['lon_bin'].cat.categories
    lat_idx = data['lat_bin'].cat.categories

    # Group & residual per bin
    grouped = (
        data.groupby(['lat_bin', 'lon_bin'], observed=True)[['target_stec', 'pred_stec']]
            .mean()
    )
    res = (grouped['target_stec'] - grouped['pred_stec']).abs()

    # Build complete grid in the same interval space
    res_grid = res.unstack('lon_bin').reindex(index=lat_idx, columns=lon_idx)

    # Convert to Z; shape (ny, nx)
    Z = res_grid.to_numpy()

    # pcolormesh prefers **edge** arrays of length nx+1/ny+1; we already have edges
    vmax = np.nanpercentile(Z, 95) if np.isfinite(Z).any() else 1.0

    # Figure & axes
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    im = ax.pcolormesh(lon_edges, lat_edges, Z, cmap='coolwarm',
                       shading='auto', alpha=0.8, vmin=0, vmax=vmax)

    # Axes formatting
    ax.set_xlabel(f'{coord_type} Longitude [°]', fontweight='bold')
    ax.set_ylabel(f'{coord_type} Latitude [°]', fontweight='bold')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)

    # Equal data aspect on the main axes
    ax.set_aspect('equal', adjustable='box')

    # Colorbar beside without squeezing
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, extend='max')
    cbar.set_label('Absolute Residual [TECU]', rotation=270, labelpad=35,
                    fontsize=16, fontweight='bold')

    ax.set_title(f'{coord_type} IPP Spatial Distribution of Errors',
                 fontsize=20, pad=20, fontweight='bold')

    fig.tight_layout()
    fig.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)



def plot_histogram_of_residuals(df, output_dir):
    """Create presentation-ready histogram of residuals"""
    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)
    
    residuals = df['target_stec'] - df['pred_stec']
    
    # Create histogram with better styling
    n, bins, patches = ax.hist(residuals, bins=50, alpha=0.7, color='steelblue', 
                              edgecolor='black', linewidth=0.5)
    
    # Add statistics
    mean_res = residuals.mean()
    std_res = residuals.std()
    median_res = residuals.median()
    
    # Add vertical lines for statistics
    ax.axvline(mean_res, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_res:.3f} TECU', alpha=0.8)
    ax.axvline(median_res, color='orange', linestyle='--', linewidth=2,
              label=f'Median: {median_res:.3f} TECU', alpha=0.8)
    ax.axvline(0, color='black', linestyle='-', linewidth=2,
              label='Zero Bias', alpha=0.8)
    
    # Add text box with statistics
    textstr = f'σ = {std_res:.3f} TECU\nN = {len(residuals):,}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=18,
           verticalalignment='top', bbox=props)
    
    ax.set_xlabel('Residual [TECU]', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of STEC Prediction Residuals', fontweight='bold', pad=20)
    ax.legend(fontsize=18, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residual_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_uncertainty_calibration(df, output_dir):
    """Create presentation-ready uncertainty calibration plots"""
    # Ensure the uncertainty_analysis directory exists
    ensure_dir(f'{output_dir}/uncertainty_analysis')
    
    abs_residual = np.abs(df['target_stec'] - df['pred_stec'])
    
    # Hexagonal density plot with log scale
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    
    hb = ax.hexbin(df['pred_total_unc'], abs_residual, gridsize=100, cmap='BuGn', 
                   mincnt=1, norm=LogNorm(), alpha=0.8)
    
    # Add perfect calibration line
    max_val = max(df['pred_total_unc'].max(), abs_residual.max())
    ax.plot([0, max_val], [0, max_val], 'r-', linewidth=3, 
            label='Perfect Calibration (1σ)', alpha=0.9)
    
    ax.set_xlabel('Predicted Total Uncertainty [TECU]', fontweight='bold')
    ax.set_ylabel('|Residual| [TECU]', fontweight='bold')
    ax.set_title('Uncertainty Calibration', fontweight='bold', pad=20)
    
    cbar = plt.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Number of Points', fontweight='bold', rotation=270, labelpad=35)
    cbar.ax.tick_params(labelsize=16)
    
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uncertainty_analysis/uncertainty_calibration_density.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2D histogram version with log scale
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    
    h = ax.hist2d(df['pred_total_unc'], abs_residual, bins=100, cmap='BuGn', 
                  norm=LogNorm(), alpha=0.8)
    
    ax.plot([0, max_val], [0, max_val], 'r-', linewidth=3, 
            label='Perfect Calibration (1σ)', alpha=0.9)
    
    ax.set_xlabel('Predicted Total Uncertainty [TECU]', fontweight='bold')
    ax.set_ylabel('|Residual| [TECU]', fontweight='bold')
    ax.set_title('Uncertainty Calibration', fontweight='bold', pad=20)
    
    cbar = plt.colorbar(h[3], ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Density', fontweight='bold', rotation=270, labelpad=35)
    cbar.ax.tick_params(labelsize=16)
    
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uncertainty_analysis/uncertainty_calibration_hist2d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Standard scatter plot for comparison
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    
    ax.scatter(df['pred_total_unc'], abs_residual, alpha=0.3, s=1, c='blue')
    ax.plot([0, max_val], [0, max_val], 'r-', linewidth=3, 
            label='Perfect Calibration (1σ)', alpha=0.9)
    
    ax.set_xlabel('Predicted Total Uncertainty [TECU]', fontweight='bold')
    ax.set_ylabel('|Residual| [TECU]', fontweight='bold')
    ax.set_title('Uncertainty Calibration', fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uncertainty_analysis/uncertainty_calibration_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_az_el_heatmap(df, output_dir, metric='residual'):
    """ Plots a presentation-ready heatmap of residuals or MAE by azimuth and elevation.

    Parameters:
        - df: DataFrame containing 'satazi', 'satele', 'target_stec', and 'pred_stec' columns.
        - output_dir: Directory to save the plot.
        - metric: Metric to plot ('residual' or 'mae').
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    heatmap_data = df.copy()

    # Fixed bins
    az_min, az_max = 0, 360
    el_min, el_max = 5, 90

    heatmap_data['az_bin'] = pd.cut(df['satazi'], bins=np.linspace(az_min, az_max, 181))
    heatmap_data['el_bin'] = pd.cut(df['satele'], bins=np.linspace(el_min, el_max, 87))

    az_cats = heatmap_data['az_bin'].cat.categories
    el_cats = heatmap_data['el_bin'].cat.categories

    # Group and compute the desired metric
    grouped = heatmap_data.groupby(['az_bin', 'el_bin'])[['target_stec', 'pred_stec']].mean().reset_index()
    if metric == 'residual':
        grouped['value'] = grouped['target_stec'] - grouped['pred_stec']
        cbar_label = 'Residual [TECU]'
        title = 'STEC Prediction Residuals by Satellite Geometry'
        filename = 'az_el_residuals_heatmap.png'
        cmap_colors = 'RdBu_r'
        center = 0
    elif metric == 'mae':
        grouped['value'] = np.abs(grouped['target_stec'] - grouped['pred_stec'])
        cbar_label = 'Mean Absolute Error [TECU]'
        title = 'STEC Prediction MAE by Satellite Geometry'
        filename = 'az_el_mae_heatmap.png'
        cmap_colors = 'coolwarm'
        center = None
    else:
        raise ValueError("Invalid metric. Choose 'residual' or 'mae'.")

    # Pivot and reindex to include all bins
    pivot = grouped.pivot(index='el_bin', columns='az_bin', values='value')
    pivot = pivot.reindex(index=el_cats, columns=az_cats)
    # Reverse the order of elevation bins so high elevations are at top
    pivot = pivot.iloc[::-1]
    vals = pivot.to_numpy()

    # Compute 95% quantile limits (ignore NaNs)
    if metric == 'residual':
        vmax_q = np.nanpercentile(np.abs(vals), 95)
        vmin, vmax = -vmax_q, vmax_q
        data_min, data_max = np.nanmin(vals), np.nanmax(vals)
        extend = ('both' if (data_min < vmin) and (data_max > vmax)
                  else 'min' if data_min < vmin
                  else 'max' if data_max > vmax
                  else 'neither')
    else:  # mae
        vmin = 0.0
        vmax = np.nanpercentile(vals, 95)
        data_max = np.nanmax(vals)
        extend = 'max' if data_max > vmax else 'neither'

    # Create heatmap with improved styling
    heatmap = sns.heatmap(
        pivot, cmap=cmap_colors, vmin=vmin, vmax=vmax,
        cbar_kws={'label': cbar_label, 'extend': extend, 'shrink': 0.8},
        center=center, ax=ax, square=False
    )

    # Improve colorbar formatting
    cbar = heatmap.collections[0].colorbar
    cbar.set_label(cbar_label, fontweight='bold', fontsize=16, rotation=270, labelpad=35)
    cbar.ax.tick_params(labelsize=16)

    # Define nice tick values manually
    # X-axis (Azimuth: 0-360°) - every 60 degrees
    az_tick_values = [0, 60, 120, 180, 240, 300, 360]
    x_tick_positions = []
    x_tick_labels = []
    
    for az_val in az_tick_values:
        # Find the closest bin index for this azimuth value
        closest_idx = None
        min_distance = float('inf')
        for i, cat in enumerate(pivot.columns):
            bin_center = cat.mid
            distance = abs(bin_center - az_val)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        if closest_idx is not None:
            x_tick_positions.append(closest_idx)
            x_tick_labels.append(f'{az_val}°')
    
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=18)
    
    # Y-axis (Elevation: 5-90°) - every 15 degrees plus min/max
    el_tick_values = [5, 15, 30, 45, 60, 75, 90]
    y_tick_positions = []
    y_tick_labels = []
    
    for el_val in el_tick_values:
        # Find the closest bin index for this elevation value in the original el_cats
        closest_idx = None
        min_distance = float('inf')
        for i, cat in enumerate(el_cats):
            bin_center = cat.mid
            distance = abs(bin_center - el_val)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        if closest_idx is not None:
            # Since we reversed the pivot with iloc[::-1], we need to reverse the index
            reversed_idx = len(el_cats) - 1 - closest_idx
            y_tick_positions.append(reversed_idx)
            y_tick_labels.append(f'{el_val}°')
    
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, rotation=0, fontsize=18)

    ax.set_xlabel('Azimuth Angle [°]', fontweight='bold', fontsize=16)
    ax.set_ylabel('Elevation Angle [°]', fontweight='bold', fontsize=16)
    ax.set_title(title, fontweight='bold', fontsize=18, pad=25)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

def modify_df(df):
    """
    Modifies the DataFrame to ensure nice plotting using feature registry.
    """
    # Convert seconds of day to local solar time if SOD and longitude exist
    if 'sod' in df.columns:
        if 'lon_ipp' in df.columns:
            # Convert to local solar time using longitude
            # Local solar time = UTC + (longitude / 15) hours
            utc_hours = df['sod'] / 3600
            longitude_offset = df['lon_ipp'] / 15.0  # 15 degrees per hour
            df['time'] = (utc_hours + longitude_offset) % 24
        elif 'lon_sta' in df.columns:
            # Fallback to station longitude if IPP longitude not available
            utc_hours = df['sod'] / 3600
            longitude_offset = df['lon_sta'] / 15.0  # 15 degrees per hour
            df['time'] = (utc_hours + longitude_offset) % 24
        else:
            # Fallback to UTC if no longitude data available
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
    Creates presentation-ready boxplots for each month present in the test data.
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
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    # Add zero reference line
    ax.axhline(y=0, color='red', linestyle='-', linewidth=2, zorder=1, alpha=0.8)
    
    # Create boxplot with improved styling
    bp = ax.boxplot(box_data, labels=month_labels, showfliers=False, zorder=2,
                   patch_artist=True, notch=False)
    
    # Style the boxplot
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_linewidth(2)
    
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45, labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    
    ax.set_xlabel('Month', fontweight='bold', fontsize=20)
    ax.set_ylabel('Residual [TECU]', fontweight='bold', fontsize=20)
    ax.set_title('STEC Prediction Residuals by Month', fontweight='bold', fontsize=22, pad=25)
    ax.grid(axis='y', alpha=0.3)
    
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
            
            # Create clipped version for target_stec
            if feature == 'target_stec':
                plot_residuals_vs_feature_clipped(test_df, feature, num_bins=num_bins, 
                                                output_dir=output_dir, bin_range_dict=bin_range_dict,
                                                x_limits=None, y_limits=(-100, 100))

    plot_prediction_scatter(test_df, output_dir)
    
    # Only plot spatial/azimuth plots if the required features exist
    required_spatial = ['lon_ipp', 'lat_ipp']
    if all(col in available_features for col in required_spatial):
        plot_spatial_error_map(test_df, output_dir)
        
        # Plot spatial errors by local time (requires 'time' column from modify_df)
        if 'time' in available_features:
            plot_spatial_error_map_by_local_time(test_df, output_dir)
        
        # Plot solar magnetic IPP error map
        plot_solar_magnetic_ipp_error_map(test_df, output_dir)
    
    required_directional = ['satazi', 'satele']
    if all(col in available_features for col in required_directional):
        plot_az_el_heatmap(test_df, output_dir, metric='mae')
        plot_az_el_heatmap(test_df, output_dir, metric='residual')

    plot_histogram_of_residuals(test_df, output_dir)
    plot_uncertainty_calibration(test_df, output_dir)
    
    # New comprehensive uncertainty analysis
    if all(col in available_features for col in ['pred_epistemic_unc', 'pred_aleatoric_unc', 'pred_total_unc']):
        plot_comprehensive_uncertainty_analysis(test_df, output_dir)


def plot_comprehensive_uncertainty_analysis(df, output_dir):
    """
    Comprehensive uncertainty analysis including sigma interval coverage 
    for total, epistemic, and aleatoric uncertainties.
    
    INTERPRETATION GUIDE:
    - Epistemic Uncertainty: Model uncertainty (reducible with more training/data)
    - Aleatoric Uncertainty: Data noise uncertainty (irreducible measurement/physical noise)
    - Total Uncertainty: sqrt(epistemic² + aleatoric²)
    
    EXPECTED COVERAGE for well-calibrated uncertainties:
    - 1σ: 68.27% of observations should fall within uncertainty bounds
    - 2σ: 95.45% of observations should fall within uncertainty bounds  
    - 3σ: 99.73% of observations should fall within uncertainty bounds
    """
    # Create uncertainty subdirectory
    uncertainty_dir = os.path.join(output_dir, 'uncertainty_analysis')
    ensure_dir(uncertainty_dir)
    
    # Calculate absolute residuals
    abs_residuals = np.abs(df['target_stec'] - df['pred_stec'])
    
    # Extract uncertainties
    total_unc = df['pred_total_unc'].values
    epistemic_unc = df['pred_epistemic_unc'].values
    aleatoric_unc = df['pred_aleatoric_unc'].values
    
    # 1. SIGMA INTERVAL COVERAGE ANALYSIS
    def calculate_sigma_coverage(uncertainties, residuals, unc_type):
        """Calculate coverage for 1σ, 2σ, and 3σ intervals"""
        n_total = len(residuals)
        
        # Count observations within each sigma interval
        within_1sigma = np.sum(residuals <= uncertainties)
        within_2sigma = np.sum(residuals <= 2 * uncertainties)
        within_3sigma = np.sum(residuals <= 3 * uncertainties)
        
        # Calculate percentages
        pct_1sigma = (within_1sigma / n_total) * 100
        pct_2sigma = (within_2sigma / n_total) * 100
        pct_3sigma = (within_3sigma / n_total) * 100
        
        # Expected percentages for normal distribution
        expected_1sigma = 68.27
        expected_2sigma = 95.45
        expected_3sigma = 99.73
                
        return {
            '1sigma': {'observed': pct_1sigma, 'expected': expected_1sigma, 'count': within_1sigma},
            '2sigma': {'observed': pct_2sigma, 'expected': expected_2sigma, 'count': within_2sigma},
            '3sigma': {'observed': pct_3sigma, 'expected': expected_3sigma, 'count': within_3sigma}
        }
    
    # Calculate coverage for all uncertainty types
    total_coverage = calculate_sigma_coverage(total_unc, abs_residuals, "TOTAL")
    epistemic_coverage = calculate_sigma_coverage(epistemic_unc, abs_residuals, "EPISTEMIC")
    aleatoric_coverage = calculate_sigma_coverage(aleatoric_unc, abs_residuals, "ALEATORIC")
    
    # 2. SUMMARY STATISTICS
    
    # Correlation analysis
    from scipy.stats import pearsonr
    corr_total, p_total = pearsonr(total_unc, abs_residuals)
    corr_epistemic, p_epistemic = pearsonr(epistemic_unc, abs_residuals)
    corr_aleatoric, p_aleatoric = pearsonr(aleatoric_unc, abs_residuals)
    
    # 3. INDIVIDUAL VISUALIZATION PLOTS - Each as separate PNG
    # Define colors for consistency
    colors = {'total': 'navy', 'epistemic': 'darkred', 'aleatoric': 'darkgreen', 'expected': 'gray'}
    
    # 1. Coverage comparison with clear interpretation
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sigma_levels = ['1σ', '2σ', '3σ']
    expected_values = [68.27, 95.45, 99.73]
    total_observed = [total_coverage['1sigma']['observed'], 
                     total_coverage['2sigma']['observed'], 
                     total_coverage['3sigma']['observed']]
    epistemic_observed = [epistemic_coverage['1sigma']['observed'], 
                         epistemic_coverage['2sigma']['observed'], 
                         epistemic_coverage['3sigma']['observed']]
    aleatoric_observed = [aleatoric_coverage['1sigma']['observed'], 
                         aleatoric_coverage['2sigma']['observed'], 
                         aleatoric_coverage['3sigma']['observed']]
    
    x = np.arange(len(sigma_levels))
    width = 0.2
    
    # Plot with clear legend and interpretation
    bars1 = ax.bar(x + 0.0*width, expected_values, width, label='Expected (Perfect)', 
                   alpha=0.8, color=colors['expected'], edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + 1.0*width, total_observed, width, label='Total', 
                   alpha=0.8, color=colors['total'], edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + 2.0*width, epistemic_observed, width, label='Epistemic (Model)', 
                   alpha=0.8, color=colors['epistemic'], edgecolor='black', linewidth=1.5)
    bars4 = ax.bar(x + 3.0*width, aleatoric_observed, width, label='Aleatoric (Data Noise)', 
                   alpha=0.8, color=colors['aleatoric'], edgecolor='black', linewidth=1.5)
    
    # Add horizontal reference lines
    for i, exp_val in enumerate(expected_values):
        ax.axhline(y=exp_val, xmin=(i-0.4)/len(sigma_levels), xmax=(i+0.4)/len(sigma_levels), 
                   color='red', linestyle='--', alpha=0.7, linewidth=3)
    
    ax.set_xlabel('Sigma Levels', fontsize=16, fontweight='bold')
    ax.set_ylabel('Coverage [%]', fontsize=16, fontweight='bold')
    ax.set_title('Uncertainty Coverage Analysis\nCloser to Expected = Better Calibrated', 
                fontsize=20, fontweight='bold', pad=25)
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(sigma_levels, fontsize=18, fontweight='bold')
    ax.legend(fontsize=18, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    autolabel(bars1, expected_values)
    autolabel(bars2, total_observed)
    autolabel(bars3, epistemic_observed)
    autolabel(bars4, aleatoric_observed)
    
    plt.tight_layout()
    plt.savefig(f'{uncertainty_dir}/sigma_coverage_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Uncertainty distributions with clear interpretation
    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)
    
    # Create histogram with better styling
    n1, bins1, patches1 = ax.hist(total_unc, bins=50, alpha=0.7, label='Total', 
                                  color=colors['total'], edgecolor='black', linewidth=0.5)
    n2, bins2, patches2 = ax.hist(epistemic_unc, bins=50, alpha=0.7, label='Epistemic (Model)', 
                                  color=colors['epistemic'], edgecolor='black', linewidth=0.5)
    n3, bins3, patches3 = ax.hist(aleatoric_unc, bins=50, alpha=0.7, label='Aleatoric (Data Noise)', 
                                  color=colors['aleatoric'], edgecolor='black', linewidth=0.5)
    
    # Add mean lines with improved styling
    ax.axvline(total_unc.mean(), color=colors['total'], linestyle='--', linewidth=3, alpha=0.9, 
               label=f'Total mean: {total_unc.mean():.3f} TECU')
    ax.axvline(epistemic_unc.mean(), color=colors['epistemic'], linestyle='--', linewidth=3, alpha=0.9,
               label=f'Epistemic mean: {epistemic_unc.mean():.3f} TECU')
    ax.axvline(aleatoric_unc.mean(), color=colors['aleatoric'], linestyle='--', linewidth=3, alpha=0.9,
               label=f'Aleatoric mean: {aleatoric_unc.mean():.3f} TECU')
    
    ax.set_xlabel('Uncertainty [TECU]', fontsize=16, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=16, fontweight='bold')
    ax.set_title('Uncertainty Distributions', fontsize=20, fontweight='bold', pad=25)
    ax.legend(fontsize=18, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{uncertainty_dir}/uncertainty_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Calibration plot with sigma lines
    plt.figure(figsize=FIGSIZE_SQUARE)
    
    # Create scatter plots with different point sizes for clarity
    plt.scatter(total_unc, abs_residuals, alpha=0.4, s=4, label='Total', color=colors['total'])
    plt.scatter(epistemic_unc, abs_residuals, alpha=0.4, s=4, label='Epistemic', color=colors['epistemic'])
    plt.scatter(aleatoric_unc, abs_residuals, alpha=0.4, s=4, label='Aleatoric', color=colors['aleatoric'])
    
    # Add perfect calibration lines with labels
    max_unc = max(total_unc.max(), epistemic_unc.max(), aleatoric_unc.max())
    plt.plot([0, max_unc], [0, max_unc], 'k--', linewidth=3, alpha=0.8, label='Perfect 1σ line')
    plt.plot([0, max_unc/2], [0, max_unc], 'k:', linewidth=2, alpha=0.6, label='Perfect 2σ line')
    plt.plot([0, max_unc/3], [0, max_unc], 'k:', linewidth=1, alpha=0.4, label='Perfect 3σ line')
    
    plt.xlabel('Predicted Uncertainty [TECU]', fontsize=18, fontweight='bold')
    plt.ylabel('|Residual| [TECU]', fontsize=18, fontweight='bold')
    plt.title('Calibration Plot', fontsize=18, fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{uncertainty_dir}/calibration_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Epistemic vs Aleatoric relationship
    plt.figure(figsize=FIGSIZE_SQUARE)
    scatter = plt.scatter(epistemic_unc, aleatoric_unc, alpha=0.5, s=4, c=abs_residuals, 
                         cmap='BuGn', norm=LogNorm())
    plt.xlabel('Epistemic Uncertainty [TECU]', fontsize=18, fontweight='bold')
    plt.ylabel('Aleatoric Uncertainty [TECU]', fontsize=18, fontweight='bold')
    plt.title('Epistemic vs Aleatoric Uncertainty Relationship', fontsize=18, fontweight='bold')
    cbar = plt.colorbar(scatter)
    cbar.set_label('|Residual| [TECU]', fontsize=16)
    plt.grid(True, alpha=0.3)
    
    # Add diagonal line to show epistemic=aleatoric
    min_val = min(epistemic_unc.min(), aleatoric_unc.min())
    max_val = max(epistemic_unc.max(), aleatoric_unc.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2, 
             label='Equal uncertainties')
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{uncertainty_dir}/epistemic_vs_aleatoric.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Coverage vs uncertainty magnitude
    plt.figure(figsize=FIGSIZE_HISTOGRAM)
    
    # Calculate rolling coverage for visualization
    for unc_vals, unc_name, color in [(total_unc, 'Total', colors['total']), 
                                      (epistemic_unc, 'Epistemic', colors['epistemic']), 
                                      (aleatoric_unc, 'Aleatoric', colors['aleatoric'])]:
        sorted_indices = np.argsort(unc_vals)
        sorted_unc = unc_vals[sorted_indices]
        sorted_residuals = abs_residuals[sorted_indices]
        
        # Calculate rolling coverage
        window_size = max(len(unc_vals) // 20, 100)  # At least 100 points per window
        rolling_coverage_1sigma = []
        rolling_uncertainty = []
        
        for i in range(window_size, len(sorted_unc) - window_size, window_size//2):
            window_unc = sorted_unc[i-window_size:i+window_size]
            window_res = sorted_residuals[i-window_size:i+window_size]
            
            coverage_1 = np.mean(window_res <= window_unc) * 100
            rolling_coverage_1sigma.append(coverage_1)
            rolling_uncertainty.append(window_unc.mean())
        
        plt.plot(rolling_uncertainty, rolling_coverage_1sigma, 'o-', 
                label=f'{unc_name}', color=color, alpha=0.8, markersize=4)
    
    plt.axhline(68.27, color='red', linestyle='--', alpha=0.7, linewidth=2, 
               label='Expected 1σ (68.27%)')
    plt.xlabel('Uncertainty Magnitude [TECU]', fontsize=18, fontweight='bold')
    plt.ylabel('1σ Coverage (%)', fontsize=18, fontweight='bold')
    plt.title('Coverage vs Uncertainty Magnitude', fontsize=18, fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f'{uncertainty_dir}/coverage_vs_magnitude.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Summary interpretation as text file
    interpretation_file = f'{uncertainty_dir}/uncertainty_analysis_summary.txt'
    with open(interpretation_file, 'w') as f:
        f.write("🏆 UNCERTAINTY ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("📊 YOUR MODEL PERFORMANCE:\n")
        f.write(f"• Total Uncertainty: {'Over-confident' if total_coverage['1sigma']['observed'] > 75 else 'Well-calibrated' if total_coverage['1sigma']['observed'] > 60 else 'Under-confident'}\n")
        f.write(f"• Epistemic (Model): {'Under-confident' if epistemic_coverage['1sigma']['observed'] < 60 else 'Well-calibrated'}\n")
        f.write(f"• Aleatoric (Data): {'Excellent!' if abs(aleatoric_coverage['1sigma']['observed'] - 68.27) < 5 else 'Good' if abs(aleatoric_coverage['1sigma']['observed'] - 68.27) < 10 else 'Needs improvement'}\n\n")
        
        f.write("🎯 KEY INSIGHTS:\n")
        f.write(f"• Epistemic: {epistemic_unc.mean():.1f} ± {epistemic_unc.std():.1f} TECU\n")
        f.write(f"• Aleatoric: {aleatoric_unc.mean():.1f} ± {aleatoric_unc.std():.1f} TECU\n")
        f.write(f"• Typical error: {abs_residuals.mean():.1f} ± {abs_residuals.std():.1f} TECU\n\n")
        
        f.write("💡 RECOMMENDATIONS:\n")
        if epistemic_coverage['1sigma']['observed'] < 60:
            f.write("• Train longer to improve epistemic uncertainty\n")
        if abs(aleatoric_coverage['1sigma']['observed'] - 68.27) < 5:
            f.write("• Aleatoric uncertainty is excellently calibrated!\n")
        if epistemic_unc.mean() < aleatoric_unc.mean() * 0.5:
            f.write("• Consider ensemble methods for better epistemic uncertainty\n")
        f.write("\n")
        
        f.write("🔗 CORRELATIONS WITH ERRORS:\n")
        f.write(f"• Total: r = {corr_total:.3f} ({'Strong' if abs(corr_total) > 0.5 else 'Moderate' if abs(corr_total) > 0.3 else 'Weak'})\n")
        f.write(f"• Epistemic: r = {corr_epistemic:.3f} ({'Strong' if abs(corr_epistemic) > 0.5 else 'Moderate' if abs(corr_epistemic) > 0.3 else 'Weak'})\n")
        f.write(f"• Aleatoric: r = {corr_aleatoric:.3f} ({'Strong' if abs(corr_aleatoric) > 0.5 else 'Moderate' if abs(corr_aleatoric) > 0.3 else 'Weak'})\n")
        
    # 4. SEPARATE DETAILED PLOTS FOR EACH UNCERTAINTY TYPE
    
    # Detailed calibration plots for each uncertainty type
    uncertainty_types = [
        ('Total', total_unc, 'blue'),
        ('Epistemic', epistemic_unc, 'red'),
        ('Aleatoric', aleatoric_unc, 'green')
    ]
    
    for unc_name, unc_values, color in uncertainty_types:
        
        # Plot 1: Calibration plot with confidence intervals
        plt.figure(figsize=FIGSIZE_SQUARE)
        plt.hexbin(unc_values, abs_residuals, gridsize=50, cmap='BuGn', 
                   mincnt=1, norm=LogNorm())
        
        # Add sigma lines
        max_val = max(unc_values.max(), abs_residuals.max())
        plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='1σ line')
        plt.plot([0, max_val], [0, 2*max_val], 'k:', linewidth=2, label='2σ line')
        plt.plot([0, max_val], [0, 3*max_val], 'k:', linewidth=1, label='3σ line')
        
        plt.xlabel(f'{unc_name} Uncertainty [TECU]', fontsize=18, fontweight='bold')
        plt.ylabel('|Residual| [TECU]', fontsize=18, fontweight='bold')
        plt.title(f'{unc_name} Uncertainty Calibration Plot', fontsize=16, fontweight='bold')
        plt.legend(fontsize=16)
        plt.grid(True, alpha=0.3)
        cbar = plt.colorbar()
        cbar.set_label('Number of Points', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{uncertainty_dir}/{unc_name.lower()}_uncertainty_calibration_plot.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Distribution histogram
        plt.figure(figsize=FIGSIZE_HISTOGRAM)
        plt.hist(unc_values, bins=50, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
        plt.axvline(unc_values.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {unc_values.mean():.2f} TECU')
        plt.axvline(np.median(unc_values), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(unc_values):.2f} TECU')
        plt.xlabel(f'{unc_name} Uncertainty [TECU]', fontsize=16, fontweight='bold')
        plt.ylabel('Density', fontsize=16, fontweight='bold')
        plt.title(f'{unc_name} Uncertainty Distribution', fontsize=18, fontweight='bold')
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{uncertainty_dir}/{unc_name.lower()}_uncertainty_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Binned calibration plot
        plt.figure(figsize=FIGSIZE_SQUARE)
        n_bins = 20
        bin_edges = np.linspace(0, np.percentile(unc_values, 95), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        binned_mean_unc = []
        binned_mean_residual = []
        binned_std_residual = []
        
        for i in range(len(bin_centers)):
            mask = (unc_values >= bin_edges[i]) & (unc_values < bin_edges[i+1])
            if np.sum(mask) > 10:  # Only if sufficient samples
                binned_mean_unc.append(unc_values[mask].mean())
                binned_mean_residual.append(abs_residuals[mask].mean())
                binned_std_residual.append(abs_residuals[mask].std())
            else:
                binned_mean_unc.append(np.nan)
                binned_mean_residual.append(np.nan)
                binned_std_residual.append(np.nan)
        
        binned_mean_unc = np.array(binned_mean_unc)
        binned_mean_residual = np.array(binned_mean_residual)
        binned_std_residual = np.array(binned_std_residual)
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(binned_mean_unc)
        
        plt.errorbar(binned_mean_unc[valid_mask], binned_mean_residual[valid_mask], 
                    yerr=binned_std_residual[valid_mask], fmt='o-', color=color, 
                    capsize=5, alpha=0.8, linewidth=2, markersize=6)
        
        max_plot_val = max(binned_mean_unc[valid_mask].max() if valid_mask.any() else 0, 
                          binned_mean_residual[valid_mask].max() if valid_mask.any() else 0)
        plt.plot([0, max_plot_val], [0, max_plot_val], 'k--', linewidth=2, label='Perfect Calibration')
        plt.xlabel(f'Binned {unc_name} Uncertainty [TECU]', fontsize=18, fontweight='bold')
        plt.ylabel('Mean |Residual| ± Std [TECU]', fontsize=18, fontweight='bold')
        plt.title(f'Binned {unc_name} Uncertainty Calibration', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{uncertainty_dir}/{unc_name.lower()}_uncertainty_binned_calibration.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Coverage as a function of predicted uncertainty
        plt.figure(figsize=FIGSIZE_HISTOGRAM)
        sorted_indices = np.argsort(unc_values)
        sorted_unc = unc_values[sorted_indices]
        sorted_residuals = abs_residuals[sorted_indices]
        
        # Calculate rolling coverage
        window_size = len(unc_values) // 50  # 50 points
        rolling_coverage_1sigma = []
        rolling_coverage_2sigma = []
        rolling_coverage_3sigma = []
        rolling_uncertainty = []
        
        for i in range(window_size, len(sorted_unc) - window_size, window_size):
            window_unc = sorted_unc[i-window_size:i+window_size]
            window_res = sorted_residuals[i-window_size:i+window_size]
            
            coverage_1 = np.mean(window_res <= window_unc) * 100
            coverage_2 = np.mean(window_res <= 2 * window_unc) * 100
            coverage_3 = np.mean(window_res <= 3 * window_unc) * 100
            
            rolling_coverage_1sigma.append(coverage_1)
            rolling_coverage_2sigma.append(coverage_2)
            rolling_coverage_3sigma.append(coverage_3)
            rolling_uncertainty.append(window_unc.mean())
        
        plt.plot(rolling_uncertainty, rolling_coverage_1sigma, 'o-', label='1σ coverage', 
                alpha=0.8, linewidth=2, markersize=5)
        plt.plot(rolling_uncertainty, rolling_coverage_2sigma, 's-', label='2σ coverage', 
                alpha=0.8, linewidth=2, markersize=5)
        plt.plot(rolling_uncertainty, rolling_coverage_3sigma, '^-', label='3σ coverage', 
                alpha=0.8, linewidth=2, markersize=5)
        
        plt.axhline(68.27, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Expected 1σ (68.27%)')
        plt.axhline(95.45, color='gray', linestyle=':', alpha=0.7, linewidth=2, label='Expected 2σ (95.45%)')
        plt.axhline(99.73, color='gray', linestyle='-.', alpha=0.7, linewidth=2, label='Expected 3σ (99.73%)')
        
        plt.xlabel(f'{unc_name} Uncertainty [TECU]', fontsize=18, fontweight='bold')
        plt.ylabel('Coverage [%]', fontsize=18, fontweight='bold')
        plt.title(f'{unc_name} Uncertainty Coverage vs Magnitude', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(f'{uncertainty_dir}/{unc_name.lower()}_uncertainty_coverage_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. SAVE QUANTITATIVE RESULTS TO FILE
    results_file = f'{uncertainty_dir}/uncertainty_analysis_results.txt'
    with open(results_file, 'w') as f:
        f.write("COMPREHENSIVE UNCERTAINTY ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Dataset: {len(df):,} observations\n\n")
        
        f.write("SIGMA INTERVAL COVERAGE ANALYSIS\n")
        f.write("-"*40 + "\n")
        
        for unc_type, coverage in [("Total", total_coverage), 
                                  ("Epistemic", epistemic_coverage), 
                                  ("Aleatoric", aleatoric_coverage)]:
            f.write(f"\n{unc_type} Uncertainty:\n")
            for sigma in ['1sigma', '2sigma', '3sigma']:
                obs = coverage[sigma]['observed']
                exp = coverage[sigma]['expected']
                count = coverage[sigma]['count']
                f.write(f"  {sigma}: {obs:.2f}% ({count:,} obs) - Expected: {exp:.2f}% - Diff: {obs-exp:+.2f}%\n")
        
        f.write(f"\nUNCERTAINTY STATISTICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total uncertainty:     {total_unc.mean():.4f} ± {total_unc.std():.4f}\n")
        f.write(f"Epistemic uncertainty: {epistemic_unc.mean():.4f} ± {epistemic_unc.std():.4f}\n")
        f.write(f"Aleatoric uncertainty: {aleatoric_unc.mean():.4f} ± {aleatoric_unc.std():.4f}\n")
        f.write(f"Mean |residual|:       {abs_residuals.mean():.4f} ± {abs_residuals.std():.4f}\n")
        
        f.write(f"\nCORRELATIONS WITH |RESIDUALS|\n")
        f.write("-"*40 + "\n")
        f.write(f"Total uncertainty:     {corr_total:.4f} (p={p_total:.2e})\n")
        f.write(f"Epistemic uncertainty: {corr_epistemic:.4f} (p={p_epistemic:.2e})\n")
        f.write(f"Aleatoric uncertainty: {corr_aleatoric:.4f} (p={p_aleatoric:.2e})\n")
    
