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
    
    # Create hexagonal density plot
    plt.hexbin(df['target_stec'], df['pred_stec'], gridsize=50, cmap='BuGn', mincnt=1)
    
    # Add perfect prediction line
    min_val = min(df['target_stec'].min(), df['pred_stec'].min())
    max_val = max(df['target_stec'].max(), df['pred_stec'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('True STEC')
    plt.ylabel('Predicted STEC')
    plt.title('Predicted vs. True STEC (Density Plot)')
    plt.colorbar(label='Number of Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_density.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a 2D histogram version for comparison
    plt.figure(figsize=(8, 8))
    plt.hist2d(df['target_stec'], df['pred_stec'], bins=50, cmap='BuGn', density=True)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('True STEC')
    plt.ylabel('Predicted STEC')
    plt.title('Predicted vs. True STEC (2D Histogram)')
    plt.colorbar(label='Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_hist2d.png', dpi=300, bbox_inches='tight')
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
    
    # Create hexagonal density plot
    plt.hexbin(df['pred_total_unc'], abs_residual, gridsize=50, cmap='BuGn', mincnt=1)
    
    # Add perfect calibration line
    max_val = max(df['pred_total_unc'].max(), abs_residual.max())
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Perfect Calibration')
    
    plt.xlabel('Predicted Total Uncertainty')
    plt.ylabel('|Residual|')
    plt.title('Uncertainty Calibration (Density Plot)')
    plt.colorbar(label='Number of Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uncertainty_calibration_density.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a 2D histogram version
    plt.figure(figsize=(8, 8))
    plt.hist2d(df['pred_total_unc'], abs_residual, bins=50, cmap='BuGn', density=True)
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label='Perfect Calibration')
    plt.xlabel('Predicted Total Uncertainty')
    plt.ylabel('|Residual|')
    plt.title('Uncertainty Calibration (2D Histogram)')
    plt.colorbar(label='Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/uncertainty_calibration_hist2d.png', dpi=300, bbox_inches='tight')
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
