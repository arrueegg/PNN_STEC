#!/usr/bin/env python3
"""
Positioning Metrics Module

Parses PPPx .pos output files and computes positioning accuracy metrics.
Supports both single-station analysis and daily aggregation across multiple stations.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def xyz2blh(xyz):
    """
    Convert ECEF to Geodetic position.
    
    Args:
        xyz: nx3 ECEF array [m]
    
    Returns:
        nx3 array [lat(deg), lon(deg), height(m)]
    """
    R2D = 180.0 / np.pi
    a = 6378137.0000    # WGS84 semi-major axis
    b = 6356752.3142    # WGS84 semi-minor axis

    x2 = xyz[:, 0] ** 2
    y2 = xyz[:, 1] ** 2
    z2 = xyz[:, 2] ** 2

    e = np.sqrt(1 - (b / a) ** 2)
    b2 = b * b
    e2 = e ** 2
    ep = e * (a / b)
    r = np.sqrt(x2 + y2)
    r2 = r * r
    E2 = a ** 2 - b ** 2
    F = 54 * b2 * z2
    G = r2 + (1 - e2) * z2 - e2 * E2
    c = ((e2 * e2) * F * r2) / (G ** 3)
    s = (1 + c + np.sqrt(c * c + 2 * c)) ** (1 / 3)
    P = F / (3 * (s + 1 / s + 1) ** 2 * G * G)
    Q = np.sqrt(1 + 2 * e2 * e2 * P)
    ro = -(P * e2 * r) / (1 + Q) + np.sqrt(
        (a * a / 2) * (1 + 1 / Q) - (P * (1 - e2) * z2) / (Q * (1 + Q)) - P * r2 / 2
    )
    tmp = (r - e2 * ro) ** 2
    U = np.sqrt(tmp + z2)
    V = np.sqrt(tmp + (1 - e2) * z2)
    zo = (b2 * xyz[:, 2]) / (a * V)

    h = U * (1 - b2 / (a * V))
    lat = np.arctan((xyz[:, 2] + ep * ep * zo) / r) * R2D

    lon = np.arctan(xyz[:, 1] / xyz[:, 0]) * R2D
    ind = np.logical_and(xyz[:, 0] < 0, xyz[:, 1] >= 0)
    lon[ind] = lon[ind] + 180
    ind = np.logical_and(xyz[:, 0] < 0, xyz[:, 1] < 0)
    lon[ind] = lon[ind] - 180

    return np.column_stack((lat, lon, h))


def load_sinex_coords(snx_file):
    """
    Parse IGS SINEX file to extract station coordinates.
    
    Args:
        snx_file: Path to .SNX file
    
    Returns:
        Dictionary: {STATION_NAME: [x, y, z]}
    """
    snx_path = Path(snx_file)
    if not snx_path.exists():
        print(f"Warning: SINEX file not found: {snx_file}")
        return {}
    
    coords = {}
    try:
        with open(snx_path, 'r', errors='ignore') as f:
            in_estimate = False
            for line in f:
                if line.startswith('+SOLUTION/ESTIMATE'):
                    in_estimate = True
                    continue
                if line.startswith('-SOLUTION/ESTIMATE'):
                    break
                
                if in_estimate:
                    # Example line:
                    #    1 STAX  ZIMM  A    1  05:159:43200 m    01  2104332.8845 0.0011
                    parts = line.split()
                    if len(parts) >= 9:
                        entry_type = parts[1]
                        station = parts[2].upper()
                        
                        # Skip if value is placeholder (e.g., '___ESTIMATED_VALUE___')
                        value_str = parts[8]
                        if '_' in value_str:
                            continue
                        
                        try:
                            value = float(value_str)
                        except ValueError:
                            # Skip non-numeric values
                            continue
                        
                        if entry_type in ['STAX', 'STAY', 'STAZ']:
                            if station not in coords:
                                coords[station] = [0.0, 0.0, 0.0]
                            
                            if entry_type == 'STAX': coords[station][0] = value
                            elif entry_type == 'STAY': coords[station][1] = value
                            elif entry_type == 'STAZ': coords[station][2] = value
        
        return coords
    except Exception as e:
        print(f"Error parsing SINEX file {snx_file}: {e}")
        return {}


def xyz2enu(xyz, orgxyz):
    """
    Convert ECEF to ENU (East-North-Up) coordinates.
    
    Args:
        xyz: nx3 ECEF array [m]
        orgxyz: 1x3 reference ECEF position [m]
    
    Returns:
        nx3 ENU array [m]
    """
    D2R = np.pi / 180.0
    n = xyz.shape[0]
    difxyz = xyz - np.tile(orgxyz, (n, 1))
    orgllh = xyz2blh(orgxyz.reshape(1, -1))
    phi = orgllh[0, 0] * D2R
    lam = orgllh[0, 1] * D2R
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    sinlam = np.sin(lam)
    coslam = np.cos(lam)
    R = np.array([
        [-sinlam, coslam, 0],
        [-sinphi * coslam, -sinphi * sinlam, cosphi],
        [cosphi * coslam, cosphi * sinlam, sinphi]
    ])
    enu = np.dot(R, difxyz.T).T
    return enu


def parse_pos_file(pos_file_path, ref_pos=None):
    """
    Parse PPPx .pos output file.
    
    Args:
        pos_file_path: Path to .pos file
        ref_pos: Optional 1x3 reference position [x, y, z] (m)
                 If None, uses mean position of the day.
    
    Returns:
        DataFrame with positioning results
    """
    try:
        df = pd.read_csv(
            pos_file_path,
            sep=r'\s+',
            usecols=(1, 2, 3, 4, 5, 9, 10, 11, 12),
            skiprows=1,
            names=['sod', 'nsat', 'x', 'y', 'z', 'rck', 'zhd', 'zwd', 'dzwd']
        )
        
        # Compute derived quantities
        df['hour'] = df['sod'] / 3600
        df['ztd'] = df['zhd'] + df['zwd'] + df['dzwd']
        
        # Determine reference position
        if ref_pos is not None:
            xyz_ref = np.array(ref_pos).reshape(1, -1)
            df['ref_source'] = 'ground_truth'
        else:
            xyz_ref = df[['x', 'y', 'z']].mean().values.reshape(1, -1)
            df['ref_source'] = 'mean'
            
        xyz_array = df[['x', 'y', 'z']].values
        enu = xyz2enu(xyz_array, xyz_ref)
        
        df['e'] = enu[:, 0]
        df['n'] = enu[:, 1]
        df['u'] = enu[:, 2]
        
        # Compute 2D and 3D errors
        df['error_2d'] = np.sqrt(df['e']**2 + df['n']**2)
        df['error_3d'] = np.sqrt(df['e']**2 + df['n']**2 + df['u']**2)
        
        return df
    
    except Exception as e:
        print(f"Error parsing {pos_file_path}: {e}")
        return None


def compute_metrics(df):
    """
    Compute positioning accuracy metrics from positioning results.
    
    Args:
        df: DataFrame from parse_pos_file()
    
    Returns:
        Dictionary with metrics
    """
    if df is None or len(df) == 0:
        return None
    
    metrics = {
        'n_epochs': len(df),
        'mean_nsat': df['nsat'].mean(),
        'ref_source': df['ref_source'].iloc[0] if 'ref_source' in df.columns else 'unknown',
        
        # East component
        'e_mean': df['e'].mean(),
        'e_std': df['e'].std(),
        'e_rms': np.sqrt((df['e']**2).mean()),
        
        # North component
        'n_mean': df['n'].mean(),
        'n_std': df['n'].std(),
        'n_rms': np.sqrt((df['n']**2).mean()),
        
        # Up component
        'u_mean': df['u'].mean(),
        'u_std': df['u'].std(),
        'u_rms': np.sqrt((df['u']**2).mean()),
        
        # 2D errors
        'error_2d_mean': df['error_2d'].mean(),
        'error_2d_std': df['error_2d'].std(),
        'error_2d_rms': np.sqrt((df['error_2d']**2).mean()),
        'error_2d_95th': df['error_2d'].quantile(0.95),
        
        # 3D errors
        'error_3d_mean': df['error_3d'].mean(),
        'error_3d_std': df['error_3d'].std(),
        'error_3d_rms': np.sqrt((df['error_3d']**2).mean()),
        'error_3d_95th': df['error_3d'].quantile(0.95),
    }
    
    return metrics


def aggregate_daily_metrics(results_dir, year, doy, method_name, stations=None, snx_file=None):
    """
    Aggregate metrics across all stations for a specific day and method.
    
    Args:
        results_dir: Directory containing .pos files
        year: Year (int)
        doy: Day of year (int)
        method_name: Name of method ("model" or "gim")
        stations: Optional list of station names to process
        snx_file: Optional path to SINEX file for ground truth coordinates
    
    Returns:
        DataFrame with per-station metrics
    """
    results_path = Path(results_dir)
    
    # Load ground truth coordinates if SINEX provided
    gt_coords = {}
    require_snx = False
    if snx_file:
        print(f"✓ Loading SINEX ground truth coordinates from: {snx_file}")
        gt_coords = load_sinex_coords(snx_file)
        print(f"  Loaded {len(gt_coords)} station coordinates from SINEX")
        require_snx = True  # Will exclude stations without SINEX coords
    else:
        print(f"⚠️  No SINEX file provided - will use day-mean as reference (WARNING: NOT true error!)")
    
    # Find all .pos files for this day/method (including hidden files in subdirectories)
    if stations:
        pos_files = []
        for station in stations:
            # Look for hidden .pos files in station subdirectory
            station_dir = results_path / station
            if station_dir.exists():
                hidden_pos = list(station_dir.glob(".*.pos"))
                regular_pos = list(station_dir.glob("*.pos"))
                pos_files.extend(hidden_pos + regular_pos)
    else:
        # Search recursively for all .pos files (hidden and regular)
        pos_files = list(results_path.glob("**/.*.pos")) + list(results_path.glob("**/*.pos"))
    
    if not pos_files:
        print(f"No .pos files found for {method_name} in {results_dir}")
        return None
    
    # Compute metrics for each station
    all_metrics = []
    
    for pos_file in pos_files:
        # Extract station name from parent directory or filename
        if pos_file.parent.name != results_path.name:
            # File is in subdirectory - use directory name as station
            station = pos_file.parent.name
        else:
            # Extract from filename (remove hidden file prefix and method suffix)
            station = pos_file.stem.lstrip('.').split('_')[0]
        
        # Get reference position for this station
        ref_pos = gt_coords.get(station.upper())
        
        # If SINEX was provided and this station is not in it, skip the station
        if require_snx and not ref_pos:
            print(f"SKIPPING: Station {station.upper()} not found in SINEX file (no ground-truth coordinates)")
            continue
        
        if snx_file and ref_pos:
            print(f"INFO: Using SINEX ground truth for {station.upper()}")
        
        # Parse and compute metrics
        df = parse_pos_file(pos_file, ref_pos=ref_pos)
        metrics = compute_metrics(df)
        
        if metrics:
            metrics['station'] = station
            metrics['method'] = method_name
            metrics['year'] = year
            metrics['doy'] = doy
            all_metrics.append(metrics)
    
    if not all_metrics:
        return None
    
    # Create DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Reorder columns
    cols = ['station', 'method', 'year', 'doy'] + [c for c in metrics_df.columns if c not in ['station', 'method', 'year', 'doy']]
    metrics_df = metrics_df[cols]
    
    return metrics_df


def save_daily_summary(metrics_model, metrics_gim, output_path):
    """
    Save daily summary comparing model and GIM results.
    
    Args:
        metrics_model: DataFrame with model metrics
        metrics_gim: DataFrame with GIM metrics
        output_path: Path to save summary CSV
    """
    # Combine both DataFrames
    combined = pd.concat([metrics_model, metrics_gim], ignore_index=True)
    
    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False, float_format='%.4f')
    
    # Print summary statistics
    print("\n" + "="*80)
    print(f"DAILY SUMMARY: {metrics_model['year'].iloc[0]}/{metrics_model['doy'].iloc[0]:03d}")
    print("="*80)
    
    unique_methods = combined['method'].unique()
    for method in unique_methods:
        method_data = combined[combined['method'] == method]
        if len(method_data) > 0:
            print(f"\n{method.upper()} ({len(method_data)} stations):")
            print(f"  2D RMS: {method_data['error_2d_rms'].mean():.4f} m (mean), {method_data['error_2d_rms'].std():.4f} m (std)")
            print(f"  3D RMS: {method_data['error_3d_rms'].mean():.4f} m (mean), {method_data['error_3d_rms'].std():.4f} m (std)")
            print(f"  2D 95th: {method_data['error_2d_95th'].mean():.4f} m")
            print(f"  3D 95th: {method_data['error_3d_95th'].mean():.4f} m")
    
    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute positioning metrics")
    parser.add_argument("--pos_file", type=str, help="Single .pos file to analyze")
    parser.add_argument("--results_dir", type=str, help="Directory with multiple .pos files")
    parser.add_argument("--year", type=int, help="Year (for aggregation)")
    parser.add_argument("--doy", type=int, help="Day of year (for aggregation)")
    parser.add_argument("--method", type=str, choices=['model', 'gim'], help="Method name")
    
    args = parser.parse_args()
    
    if args.pos_file:
        # Single file analysis
        df = parse_pos_file(args.pos_file)
        metrics = compute_metrics(df)
        
        print("\nPositioning Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    elif args.results_dir and args.year and args.doy and args.method:
        # Aggregate analysis
        metrics_df = aggregate_daily_metrics(args.results_dir, args.year, args.doy, args.method)
        
        if metrics_df is not None:
            print(f"\nAggregated {len(metrics_df)} stations:")
            print(metrics_df[['station', 'error_2d_rms', 'error_3d_rms', 'n_epochs']])
    
    else:
        parser.print_help()
