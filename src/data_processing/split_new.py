import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from itertools import chain

# Set global matplotlib parameters for scientific plots
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'figure.titlesize': 20,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3
})

# fix seeds
np.random.seed(42)

def download_igs_station_list(url, output_file):
    """Download IGS station list from a URL and save it to a file."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded station list to {output_file}.")
    else:
        raise RuntimeError(f"Failed to download. HTTP Status Code: {response.status_code}")

def load_stations(file_path):
    """Load station data from a CSV file."""
    df = pd.read_csv(file_path, usecols=['#StationName', 'Latitude', 'Longitude'])
    df.columns = ['name', 'lat', 'lon']
    df['name'] = df['name'].str[:4]
    return df

def create_grid(grid_width, grid_height):
    """Create a grid dividing the map into cells."""
    num_columns = 360 // grid_width
    num_rows = 180 // grid_height
    grid = [
        [
            (-180 + col * grid_width, -90 + row * grid_height, 
             -180 + (col + 1) * grid_width, -90 + (row + 1) * grid_height)
            for col in range(num_columns)
        ]
        for row in range(num_rows)
    ]
    return grid

def spatial_split(stations, train_frac=0.7, val_frac=0.15, seed=42):
    """Split stations spatially into training, validation, and test sets."""
    np.random.seed(seed)
    unique_grids = list(set(stations['grid']))
    np.random.shuffle(unique_grids)

    num_train = int(len(unique_grids) * train_frac)
    num_val = int(len(unique_grids) * val_frac)

    train_grids = unique_grids[:num_train]
    val_grids = unique_grids[num_train:num_train + num_val]
    test_grids = unique_grids[num_train + num_val:]

    train_data = stations[stations['grid'].isin(train_grids)]
    val_data = stations[stations['grid'].isin(val_grids)]
    test_data = stations[stations['grid'].isin(test_grids)]

    return train_data, val_data, test_data

def count_stations_in_grid(grid, stations):
    """Count the number of stations in each grid cell."""
    station_counts = [[0] * len(grid[0]) for _ in range(len(grid))]
    grid_num = np.zeros(len(stations))

    for i, (lat, lon) in enumerate(zip(stations['lat'], stations['lon'])):
        for row_idx, row in enumerate(grid):
            for col_idx, (x1, y1, x2, y2) in enumerate(row):
                if x1 <= lon <= x2 and y1 <= lat <= y2:
                    station_counts[row_idx][col_idx] += 1
                    grid_num[i] = row_idx * len(grid[0]) + col_idx
                    break
    stations['grid'] = grid_num
    return station_counts

def split_data_by_grid(data, station_counts, train_fraction=0.7, val_fraction=0.15, random_seed=72):
    """Split station data into training, validation, and testing sets based on grid."""
    train_stations, val_stations, test_stations = [], [], []

    for grid_id in np.unique(data['grid']):
        grid_data = data[data['grid'] == grid_id].sample(frac=1, random_state=random_seed)
        if len(grid_data) == 0:
            continue
        n_train = round(len(grid_data) * train_fraction)
        n_val = round(len(grid_data) * val_fraction)

        train_stations.extend(grid_data.iloc[:n_train].to_dict(orient='records'))
        val_stations.extend(grid_data.iloc[n_train:n_train + n_val].to_dict(orient='records'))
        test_stations.extend(grid_data.iloc[n_train + n_val:].to_dict(orient='records'))

    return pd.DataFrame(train_stations), pd.DataFrame(val_stations), pd.DataFrame(test_stations)

def save_to_files(train_stations, val_stations, test_stations, output_dir):
    """Save training, validation, and testing stations data and lists to files."""
    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(os.path.join(output_dir, "train_station.list"), train_stations['name'].values, fmt='%s')
    np.savetxt(os.path.join(output_dir, "val_station.list"), val_stations['name'].values, fmt='%s')
    np.savetxt(os.path.join(output_dir, "test_station.list"), test_stations['name'].values, fmt='%s')

def plot_station_distribution(train_stations, val_stations, test_stations, output_file):
    """Plot training, validation, and testing station distributions on a world map."""
    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add colorful stock image background
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor="#FFFFFF")
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff")
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')  # Add coastlines
    
    # Add gridlines with better formatting
    gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', 
                     alpha=0.6, linestyle='--')
    gl.top_labels, gl.right_labels = False, False
    gl.xlabel_style = {'size': 12, 'color': 'black', 'weight': 'bold'}
    gl.ylabel_style = {'size': 12, 'color': 'black', 'weight': 'bold'}
    
    # Plot stations with custom colors and better density handling
    scatter_size = 35
    alpha = 1.0  # Fully opaque
    
    # Apply minimal jitter to help with overlapping stations
    train_lon_j, train_lat_j = train_stations['lon'].values, train_stations['lat'].values
    val_lon_j, val_lat_j = val_stations['lon'].values, val_stations['lat'].values
    test_lon_j, test_lat_j = test_stations['lon'].values, test_stations['lat'].values

    # Use slightly darker colors for the perfect balance
    train_scatter = ax.scatter(train_lon_j, train_lat_j, 
                              s=scatter_size, c="#215ACC", label='Training',  # Slightly darker red
                              zorder=3, alpha=alpha)
    val_scatter = ax.scatter(val_lon_j, val_lat_j, 
                            s=scatter_size, c="#5ACC21", label='Validation',  # Slightly darker green
                            zorder=4, alpha=alpha)
    test_scatter = ax.scatter(test_lon_j, test_lat_j, 
                             s=scatter_size, c="#CC215A", label='Test',  # Slightly darker blue
                             zorder=5, alpha=alpha)
    
    # Add simplified title
    ax.set_title('IGS Station Distribution for STEC Database', 
                fontweight='bold', pad=20)
    
    # Create legend with integrated statistics
    total_stations = len(train_stations) + len(val_stations) + len(test_stations)
    
    # Custom legend labels with counts and percentages
    train_label = f'Training: {len(train_stations)} ({len(train_stations)/total_stations*100:.1f}%)'
    val_label = f'Validation: {len(val_stations)} ({len(val_stations)/total_stations*100:.1f}%)'
    test_label = f'Test: {len(test_stations)} ({len(test_stations)/total_stations*100:.1f}%)'
    
    # Update scatter plot labels
    train_scatter.set_label(train_label)
    val_scatter.set_label(val_label)
    test_scatter.set_label(test_label)
    
    # Create legend with title showing total
    legend = ax.legend(title=f'Total Stations: {total_stations}', 
                      loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3,
                      frameon=True, fancybox=True, shadow=False)
    legend.get_frame().set_linewidth(1.2)
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontsize(12)
    
    # Set global extent
    ax.set_global()

    # Save with high quality and transparent background
    plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    plt.close()
    
    print(f"Station distribution map saved to: {output_file}")
    print(f"Map shows {total_stations} IGS stations distributed across train/val/test sets")

def temporal_split():
    """Create temporal splits for training, validation, and testing datasets."""
    
    # 1) build the per-year, per-split month mapping
    #    months are numbered 1=Jan ... 12=Dec
    temp_split_map = {
        2010: {'train': list(range(3,13)),        'val': [2], 'test': [1]},
        2011: {'train': [m for m in range(1,13) if m not in (4,5)],  'val': [5], 'test': [4]},
        2012: {'train': [m for m in range(1,13) if m not in (7,8)],  'val': [8], 'test': [7]},
        2013: {'train': [m for m in range(1,13) if m not in (10,11)],'val': [11],'test': [10]},
        2014: {'train': [m for m in range(1,13) if m not in (6,7)],  'val': [7], 'test': [6]},
        2015: {'train': [m for m in range(1,13) if m not in (9,10)], 'val': [10],'test': [9]},
        2016: {'train': [m for m in range(1,13) if m not in (12,1)], 'val': [1], 'test': [12]},
        2017: {'train': [m for m in range(1,13) if m not in (3,4)],  'val': [4], 'test': [3]},
        2018: {'train': [m for m in range(1,13) if m not in (11,12)],'val': [12],'test': [11]},
        2019: {'train': [m for m in range(1,13) if m not in (2,3)],  'val': [3], 'test': [2]},
        2020: {'train': [m for m in range(1,13) if m not in (5,6)],  'val': [6], 'test': [5]},
        2021: {'train': [m for m in range(1,13) if m not in (8,9)],  'val': [9], 'test': [8]},
        2022: {'train': [m for m in range(1,13) if m not in (11,12)],'val': [11],'test': [12]},
        2023: {'train': [m for m in range(1,13) if m not in (7,8)],  'val': [7], 'test': [8]},
        2024: {'train': [1,2,3],                     'val': [4], 'test': list(range(5,13))},
    }

    # 2) build the list of all months in your full span
    all_months = pd.period_range("2010-01", "2024-12", freq="M")

    # 3) assign each month to train/val/test
    train_months = []
    val_months   = []
    test_months  = []

    for p in all_months:
        year, m = p.year, p.month
        cfg = temp_split_map.get(year)
        if cfg is None:
            continue
        if   m in cfg['train']:
            train_months.append(p)
        elif m in cfg['val']:
            val_months.append(p)
        elif m in cfg['test']:
            test_months.append(p)

    # optional: turn into strings “YYYY-MM”
    train_str = [str(p) for p in train_months]
    val_str   = [str(p) for p in val_months]
    test_str  = [str(p) for p in test_months]

    # 4) save to files
    outdir = "./src/data_processing"
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "train_dates.list"), "w") as f:
        f.write("\n".join(train_str))

    with open(os.path.join(outdir, "val_dates.list"), "w") as f:
        f.write("\n".join(val_str))

    with open(os.path.join(outdir, "test_dates.list"), "w") as f:
        f.write("\n".join(test_str))

    print(f"Temporal splits: train={len(train_str)} months, val={len(val_str)}, test={len(test_str)}")


if __name__ == "__main__":
    # Define constants
    csv_url = "https://files.igs.org/pub/station/general/IGSNetwork.csv"
    cwd = os.path.join(os.getcwd(), "src/data_processing")
    output_filename = "./IGSNetwork.csv"
    csv_file = os.path.join(cwd, output_filename)
    grid_width, grid_height = 60, 30

    # Define the list of stations for PVT testing
    PVT_test_station = [
        "REYK","ALGO","WROC","URUM","BIK0","JPLM","CPVG","LMMF",
        "POVE","CHPG","NKLG","RGDG","CAS1","FAA1","ULAB","WUH2",
        "SOLO","NNOR"
    ]

    # Download and load station data
    download_igs_station_list(csv_url, csv_file)
    stations = load_stations(csv_file)

    # pull the PVT stations out
    forced_test = stations[stations['name'].isin(PVT_test_station)].copy()
    # keep the rest for spatial splitting
    remaining = stations[~stations['name'].isin(PVT_test_station)].copy()

    use_grid = False  # Set to False if you want to skip grid-based splitting
    if use_grid:
        train_fraction = 0.725
        val_frac_eff = 0.16
        # Process station data by grid
        grid = create_grid(grid_width, grid_height)
        station_counts = count_stations_in_grid(grid, remaining)
        print(f"Number of stations in each grid cell: {list(chain(*station_counts))}")
        train_data, val_data, test_data = split_data_by_grid(remaining, station_counts,
                                                        train_fraction=train_fraction, val_fraction=val_frac_eff)
    else:
        # If not using grid, just split the remaining stations randomly
        train_fraction = 0.725
        val_frac_eff = 0.56
        train_data = remaining.sample(frac=train_fraction, random_state=42)
        remaining = remaining.drop(train_data.index)
        val_data = remaining.sample(frac=val_frac_eff, random_state=42)
        test_data = remaining.drop(val_data.index)

    # now stick the forced-test stations back onto the test set
    test_data = pd.concat([test_data, forced_test], ignore_index=True)

    # (just to be safe, drop any of those names from train/val if something slipped)
    train_data = train_data[~train_data['name'].isin(PVT_test_station)]
    val_data   = val_data[~val_data['name'].isin(PVT_test_station)]

    print(f"Train stations: {len(train_data)} ({round((len(train_data)/len(stations))*100, 1)}%), \
            Validation stations: {len(val_data)} ({round((len(val_data)/len(stations))*100, 1)}%), \
            Test stations: {len(test_data)} ({round((len(test_data)/len(stations))*100, 1)}%)")

    # Save and plot results
    save_to_files(train_data, val_data, test_data, cwd)
    plot_station_distribution(train_data, val_data, test_data, os.path.join(cwd, "stations.png"))

    temporal_split()
