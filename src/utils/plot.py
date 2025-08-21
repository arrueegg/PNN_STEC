import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm


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
    plt.figure(figsize=(8, 8))
    
    # Create hexagonal density plot with logarithmic scaling
    plt.hexbin(df['target_stec'], df['pred_stec'], gridsize=50, cmap='BuGn', 
               mincnt=1, norm=LogNorm())
    
    # Add perfect prediction line
    min_val = min(df['target_stec'].min(), df['pred_stec'].min())
    max_val = max(df['target_stec'].max(), df['pred_stec'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True STEC')
    plt.ylabel('Predicted STEC')
    plt.title('Predicted vs. True STEC (Density Plot - Log Scale)')
    plt.colorbar(label='Number of Points (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_density.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a 2D histogram version with log scale
    plt.figure(figsize=(8, 8))
    plt.hist2d(df['target_stec'], df['pred_stec'], bins=50, cmap='BuGn', 
               norm=LogNorm())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('True STEC')
    plt.ylabel('Predicted STEC')
    plt.title('Predicted vs. True STEC (2D Histogram - Log Scale)')
    plt.colorbar(label='Density (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_hist2d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # create a standard scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(df['target_stec'], df['pred_stec'], alpha=0.2)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('True STEC')
    plt.ylabel('Predicted STEC')
    plt.title('Predicted vs. True STEC (Scatter Plot)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_spatial_error_map(df, output_dir):
    # Create geographic heatmap with coastlines using cartopy
    fig = plt.figure(figsize=(15, 8))
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
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05, extend='max')
    cbar.set_label('Absolute Residual (STEC)', rotation=270, labelpad=15)
    
    plt.title('Spatial Distribution of Errors')
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
        fig = plt.figure(figsize=(15, 8))
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
        cbar.set_label('Absolute Residual (STEC)', rotation=270, labelpad=15)
        
        # Create title with time range
        time_start = int(time_bin.left)
        time_end = int(time_bin.right)
        plt.title(f'Spatial Distribution of Errors ({time_start:02d}:00-{time_end:02d}:00 Local Solar Time)')
        plt.tight_layout()
        
        # Save with time in filename
        plt.savefig(f'{output_dir}/spatial_by_time/spatial_error_heatmap_{time_start:02d}h.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def plot_solar_magnetic_ipp_error_map(df, output_dir):
    """
    Create spatial error heatmap for solar magnetic IPPs without coastlines.
    Uses magnetic coordinates if available, otherwise falls back to geographic.
    """
    # use magnetic coordinates 
    lon_col = 'sm_lon_ipp'
    lat_col = 'sm_lat_ipp'
    coord_type = 'Solar Magnetic'
    filename = 'solar_magnetic_ipp_error_heatmap.png'
    # Magnetic coordinates typically range -180 to 180 for longitude, -90 to 90 for latitude
    lon_edges = np.linspace(-180, 180, 145)
    lat_edges = np.linspace(-90, 90, 73)
    
    plt.figure(figsize=(15, 8))
    heatmap_data = df.copy()
    
    # Create bins
    heatmap_data['lon_bin'] = pd.cut(df[lon_col], bins=lon_edges)
    heatmap_data['lat_bin'] = pd.cut(df[lat_col], bins=lat_edges)

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
    
    # Plot heatmap using pcolormesh (without cartopy for magnetic coordinates)
    vmax = np.nanpercentile(Z, 95)  # clip top 5% of values
    im = plt.pcolormesh(LON, LAT, Z, cmap='coolwarm', shading='auto', 
                       alpha=0.8, vmin=0, vmax=vmax)
    
    # Add gridlines (simple matplotlib grid)
    plt.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    
    # Set labels and limits
    plt.xlabel(f'{coord_type} Longitude (degrees)')
    plt.ylabel(f'{coord_type} Latitude (degrees)')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8, pad=0.05, extend='max')
    cbar.set_label('Absolute Residual (STEC)', rotation=270, labelpad=15)
    
    plt.title(f'{coord_type} IPP Spatial Distribution of Errors')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
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
    plt.figure(figsize=(8, 8))
    abs_residual = np.abs(df['target_stec'] - df['pred_stec'])
    
    # Create hexagonal density plot with log scale
    plt.hexbin(df['pred_total_unc'], abs_residual, gridsize=50, cmap='BuGn', 
               mincnt=1, norm=LogNorm())
    
    # Add perfect calibration line
    max_val = max(df['pred_total_unc'].max(), abs_residual.max())
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Perfect Calibration')
    
    plt.xlabel('Predicted Total Uncertainty')
    plt.ylabel('|Residual|')
    plt.title('Uncertainty Calibration (Density Plot - Log Scale)')
    plt.colorbar(label='Number of Points (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uncertainty_calibration_density.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a 2D histogram version with log scale
    plt.figure(figsize=(8, 8))
    plt.hist2d(df['pred_total_unc'], abs_residual, bins=50, cmap='BuGn', 
               norm=LogNorm())
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Perfect Calibration')
    plt.xlabel('Predicted Total Uncertainty')
    plt.ylabel('|Residual|')
    plt.title('Uncertainty Calibration (2D Histogram - Log Scale)')
    plt.colorbar(label='Density (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uncertainty_calibration_hist2d.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plot standard scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(df['pred_total_unc'], abs_residual, alpha=0.2)
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Predicted Total Uncertainty')
    plt.ylabel('|Residual|')
    plt.title('Uncertainty Calibration (Scatter Plot)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uncertainty_calibration_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_az_el_heatmap(df, output_dir, metric='residual'):
    """ Plots a heatmap of residuals or MAE by azimuth and elevation with 95% quantiles on color scale.

    Parameters:
        - df: DataFrame containing 'satazi', 'satele', 'target_stec', and 'pred_stec' columns.
        - output_dir: Directory to save the plot.
        - metric: Metric to plot ('residual' or 'mae').
    """
    plt.figure(figsize=(12, 6))
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
        cbar_label = 'Residual'
        title = 'Residuals by Azimuth and Elevation'
        filename = 'az_el_residuals_heatmap.png'
        cmap_colors = 'RdBu_r'
        center = 0
    elif metric == 'mae':
        grouped['value'] = np.abs(grouped['target_stec'] - grouped['pred_stec'])
        cbar_label = 'Mean Absolute Error'
        title = 'Mean Absolute Error by Azimuth and Elevation'
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

    # Plot (seaborn honors vmin/vmax; center is only needed for diverging maps)
    ax = sns.heatmap(
        pivot, cmap=cmap_colors, vmin=vmin, vmax=vmax,
        cbar_kws={'label': cbar_label, 'extend': extend},
        center=center
    )

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
    ax.set_xticklabels(x_tick_labels, rotation=0)
    
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
    ax.set_yticklabels(y_tick_labels, rotation=0)

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
    ensure_dir(output_dir)
    
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
    plt.figure(figsize=(12, 8))
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
    plt.bar(x + 0.0*width, expected_values, width, label='Expected (Perfect)', 
            alpha=0.8, color=colors['expected'], edgecolor='black', linewidth=1)
    plt.bar(x + 1.0*width, total_observed, width, label='Total', 
            alpha=0.8, color=colors['total'], edgecolor='black', linewidth=1)
    plt.bar(x + 2.0*width, epistemic_observed, width, label='Epistemic (Model)', 
            alpha=0.8, color=colors['epistemic'], edgecolor='black', linewidth=1)
    plt.bar(x + 3.0*width, aleatoric_observed, width, label='Aleatoric (Data Noise)', 
            alpha=0.8, color=colors['aleatoric'], edgecolor='black', linewidth=1)
    
    # Add horizontal reference lines
    for i, exp_val in enumerate(expected_values):
        plt.axhline(y=exp_val, xmin=(i-0.4)/len(sigma_levels), xmax=(i+0.4)/len(sigma_levels), 
                   color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.xlabel('Sigma Levels', fontsize=14, fontweight='bold')
    plt.ylabel('Coverage (%)', fontsize=14, fontweight='bold')
    plt.title('Uncertainty Coverage Analysis\n(Closer to Expected = Better)', fontsize=16, fontweight='bold')
    plt.xticks(x, sigma_levels)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sigma_coverage_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Uncertainty distributions with clear interpretation
    plt.figure(figsize=(12, 8))
    plt.hist(total_unc, bins=50, alpha=0.7, label='Total', color=colors['total'])
    plt.hist(epistemic_unc, bins=50, alpha=0.7, label='Epistemic (Model)', color=colors['epistemic'])
    plt.hist(aleatoric_unc, bins=50, alpha=0.7, label='Aleatoric (Data Noise)', color=colors['aleatoric'])
    
    # Add mean lines
    plt.axvline(total_unc.mean(), color=colors['total'], linestyle='--', linewidth=2, alpha=0.8, 
                label=f'Total mean: {total_unc.mean():.3f}')
    plt.axvline(epistemic_unc.mean(), color=colors['epistemic'], linestyle='--', linewidth=2, alpha=0.8,
                label=f'Epistemic mean: {epistemic_unc.mean():.3f}')
    plt.axvline(aleatoric_unc.mean(), color=colors['aleatoric'], linestyle='--', linewidth=2, alpha=0.8,
                label=f'Aleatoric mean: {aleatoric_unc.mean():.3f}')
    
    plt.xlabel('Uncertainty (TECU)', fontsize=14, fontweight='bold')
    plt.ylabel('Density', fontsize=14, fontweight='bold')
    plt.title('📊 Uncertainty Distributions\n(Dashed lines = means)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uncertainty_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Calibration plot with sigma lines
    plt.figure(figsize=(12, 10))
    
    # Create scatter plots with different point sizes for clarity
    plt.scatter(total_unc, abs_residuals, alpha=0.4, s=4, label='Total', color=colors['total'])
    plt.scatter(epistemic_unc, abs_residuals, alpha=0.4, s=4, label='Epistemic', color=colors['epistemic'])
    plt.scatter(aleatoric_unc, abs_residuals, alpha=0.4, s=4, label='Aleatoric', color=colors['aleatoric'])
    
    # Add perfect calibration lines with labels
    max_unc = max(total_unc.max(), epistemic_unc.max(), aleatoric_unc.max())
    plt.plot([0, max_unc], [0, max_unc], 'k--', linewidth=3, alpha=0.8, label='Perfect 1σ line')
    plt.plot([0, max_unc/2], [0, max_unc], 'k:', linewidth=2, alpha=0.6, label='Perfect 2σ line')
    plt.plot([0, max_unc/3], [0, max_unc], 'k:', linewidth=1, alpha=0.4, label='Perfect 3σ line')
    
    plt.xlabel('Predicted Uncertainty (TECU)', fontsize=14, fontweight='bold')
    plt.ylabel('|Residual| (TECU)', fontsize=14, fontweight='bold')
    plt.title('Calibration Plot\n(Points on diagonal = perfect)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/calibration_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Epistemic vs Aleatoric relationship
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(epistemic_unc, aleatoric_unc, alpha=0.5, s=4, c=abs_residuals, 
                         cmap='viridis', norm=LogNorm())
    plt.xlabel('Epistemic Uncertainty (TECU)', fontsize=14, fontweight='bold')
    plt.ylabel('Aleatoric Uncertainty (TECU)', fontsize=14, fontweight='bold')
    plt.title('Epistemic vs Aleatoric Uncertainty Relationship\n(Color = |residual|)', fontsize=16, fontweight='bold')
    cbar = plt.colorbar(scatter)
    cbar.set_label('|Residual| (TECU)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add diagonal line to show epistemic=aleatoric
    min_val = min(epistemic_unc.min(), aleatoric_unc.min())
    max_val = max(epistemic_unc.max(), aleatoric_unc.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2, 
             label='Equal uncertainties')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/epistemic_vs_aleatoric.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Coverage vs uncertainty magnitude
    plt.figure(figsize=(12, 8))
    
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
    plt.xlabel('Uncertainty Magnitude (TECU)', fontsize=14, fontweight='bold')
    plt.ylabel('1σ Coverage (%)', fontsize=14, fontweight='bold')
    plt.title('📈 Coverage vs Uncertainty Magnitude\n(Should be flat at ~68%)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/coverage_vs_magnitude.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Summary interpretation as text file
    interpretation_file = f'{output_dir}/uncertainty_analysis_summary.txt'
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
        plt.figure(figsize=(10, 8))
        plt.hexbin(unc_values, abs_residuals, gridsize=50, cmap='BuGn', 
                   mincnt=1, norm=LogNorm())
        
        # Add sigma lines
        max_val = max(unc_values.max(), abs_residuals.max())
        plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='1σ line')
        plt.plot([0, max_val], [0, 2*max_val], 'k:', linewidth=2, label='2σ line')
        plt.plot([0, max_val], [0, 3*max_val], 'k:', linewidth=1, label='3σ line')
        
        plt.xlabel(f'{unc_name} Uncertainty (TECU)', fontsize=12, fontweight='bold')
        plt.ylabel('|Residual| (TECU)', fontsize=12, fontweight='bold')
        plt.title(f'{unc_name} Uncertainty Calibration Plot', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        cbar = plt.colorbar()
        cbar.set_label('Number of Points (log scale)', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{unc_name.lower()}_uncertainty_calibration_plot.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(unc_values, bins=50, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
        plt.axvline(unc_values.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {unc_values.mean():.2f} TECU')
        plt.axvline(np.median(unc_values), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(unc_values):.2f} TECU')
        plt.xlabel(f'{unc_name} Uncertainty (TECU)', fontsize=12, fontweight='bold')
        plt.ylabel('Density', fontsize=12, fontweight='bold')
        plt.title(f'{unc_name} Uncertainty Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{unc_name.lower()}_uncertainty_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Binned calibration plot
        plt.figure(figsize=(10, 8))
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
        plt.xlabel(f'Binned {unc_name} Uncertainty (TECU)', fontsize=12, fontweight='bold')
        plt.ylabel('Mean |Residual| ± Std (TECU)', fontsize=12, fontweight='bold')
        plt.title(f'Binned {unc_name} Uncertainty Calibration', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{unc_name.lower()}_uncertainty_binned_calibration.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Coverage as a function of predicted uncertainty
        plt.figure(figsize=(10, 8))
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
        
        plt.xlabel(f'{unc_name} Uncertainty (TECU)', fontsize=12, fontweight='bold')
        plt.ylabel('Coverage (%)', fontsize=12, fontweight='bold')
        plt.title(f'{unc_name} Uncertainty Coverage vs Magnitude', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{unc_name.lower()}_uncertainty_coverage_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. SAVE QUANTITATIVE RESULTS TO FILE
    results_file = f'{output_dir}/uncertainty_analysis_results.txt'
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
    
