#!/usr/bin/env python3
"""
Compare predicted STEC values from a CSV file with ground truth from test.h5

This script compares model predictions (stored in CSV format) against the actual
ground truth values from the test dataset for a specific station and day.

Usage:
    python compare_predictions_with_ground_truth.py <csv_file> [--station STATION] [--year YEAR] [--doy DOY]
    
Example:
    python compare_predictions_with_ground_truth.py path/to/CHPG.csv --station CHPG --year 2024 --doy 122
"""

import argparse
import os
import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats


def load_ground_truth(h5_file: str, station: str, year: int, doy: int, 
                      use_casdcb: bool = False) -> pd.DataFrame:
    """
    Load ground truth data from HDF5 file for a specific station and date.
    
    Args:
        h5_file: Path to the HDF5 file containing ground truth data
        station: Station name (e.g., 'CHPG')
        year: Year (e.g., 2024)
        doy: Day of year (e.g., 122)
        use_casdcb: If True, load from CASDCB database format (hierarchical structure)
                    If False, load from test.h5 format (flat structure)
    
    Returns:
        DataFrame with ground truth data
    """
    with h5py.File(h5_file, 'r') as f:
        if use_casdcb:
            # CASDCB format: hierarchical structure [year][doy]['all_data']
            data = f[str(year)][str(doy)]['all_data'][:]
            # Filter by station only (year/doy already in path)
            station_bytes = station.encode('utf-8')
            filtered_data = data[data['station'] == station_bytes]
            
            # CASDCB has slightly different field names
            df = pd.DataFrame({
                'station': [s.decode('utf-8') for s in filtered_data['station']],
                'sat': [s.decode('utf-8') for s in filtered_data['sat']],
                'stec_true': filtered_data['stec'],
                'vtec': filtered_data['vtec'],
                'vtec_stddev': filtered_data['vtec_stddev'],
                'satele': filtered_data['satele'],
                'satazi': filtered_data['satazi'],
                'lon_ipp': filtered_data['lon_ipp'],
                'lat_ipp': filtered_data['lat_ipp'],
                'sod': filtered_data['sod'],
                'lat_sta': filtered_data['lat_sta'],
                'lon_sta': filtered_data['lon_sta'],
            })
        else:
            # test.h5 format: flat structure with year/doy fields
            data = f['data'][:]
            # Filter by station, year, and DOY
            station_bytes = station.encode('utf-8')
            mask = (data['station'] == station_bytes) & (data['year'] == year) & (data['doy'] == doy)
            filtered_data = data[mask]
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'station': [s.decode('utf-8') for s in filtered_data['station']],
                'sat': [s.decode('utf-8') for s in filtered_data['sat']],
                'year': filtered_data['year'],
                'doy': filtered_data['doy'],
                'stec_true': filtered_data['stec'],
                'vtec': filtered_data['vtec'],
                'satele': filtered_data['satele'],
                'satazi': filtered_data['satazi'],
                'lon_ipp': filtered_data['lon_ipp'],
                'lat_ipp': filtered_data['lat_ipp'],
                'sod': filtered_data['sod'],
                'lat_sta': filtered_data['lat_sta'],
                'lon_sta': filtered_data['lon_sta'],
            })
    
    return df


def load_predictions(csv_file: str) -> pd.DataFrame:
    """
    Load predicted STEC values from CSV file.
    
    Args:
        csv_file: Path to the CSV file containing predictions
    
    Returns:
        DataFrame with predictions
    """
    df = pd.read_csv(csv_file)
    
    # Rename columns to match naming convention
    df = df.rename(columns={
        'second_of_day': 'sod',
        'PRN': 'sat',
        'ipp_latitude': 'lat_ipp',
        'ipp_longitude': 'lon_ipp',
        'stec': 'stec_pred',
        'uncertainty': 'stec_uncertainty'
    })
    
    return df


def merge_predictions_and_truth(pred_df: pd.DataFrame, truth_df: pd.DataFrame, 
                                 tolerance: float = 1e-3) -> pd.DataFrame:
    """
    Merge predictions with ground truth by matching on second_of_day and satellite.
    
    Args:
        pred_df: DataFrame with predictions
        truth_df: DataFrame with ground truth
        tolerance: Tolerance for matching second_of_day and IPP coordinates
    
    Returns:
        Merged DataFrame
    """
    # Round second_of_day to avoid floating point precision issues
    pred_df['sod_rounded'] = pred_df['sod'].round(1)
    truth_df['sod_rounded'] = truth_df['sod'].round(1)
    
    # Merge on second_of_day and satellite
    merged = pd.merge(
        pred_df, 
        truth_df, 
        on=['sod_rounded', 'sat'],
        suffixes=('_pred', '_true')
    )
    
    # Verify IPP coordinates match (within tolerance)
    lat_diff = np.abs(merged['lat_ipp_pred'] - merged['lat_ipp_true'])
    lon_diff = np.abs(merged['lon_ipp_pred'] - merged['lon_ipp_true'])
    
    # Filter out mismatches
    valid_mask = (lat_diff < tolerance) & (lon_diff < tolerance)
    if not valid_mask.all():
        n_mismatches = (~valid_mask).sum()
        print(f"Warning: Removed {n_mismatches} records with mismatched IPP coordinates")
        merged = merged[valid_mask]
    
    return merged


def compute_metrics(merged_df: pd.DataFrame) -> dict:
    """
    Compute comparison metrics between predictions and ground truth.
    
    Args:
        merged_df: DataFrame with both predictions and ground truth
    
    Returns:
        Dictionary of metrics
    """
    errors = merged_df['stec_pred'] - merged_df['stec_true']
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2
    
    metrics = {
        'n_samples': len(merged_df),
        'mae': np.mean(abs_errors),
        'rmse': np.sqrt(np.mean(sq_errors)),
        'bias': np.mean(errors),
        'std': np.std(errors),
        'median_ae': np.median(abs_errors),
        'max_error': np.max(abs_errors),
        'min_error': np.min(abs_errors),
        'r2_score': 1 - (np.sum(sq_errors) / np.sum((merged_df['stec_true'] - merged_df['stec_true'].mean()) ** 2)),
        'pearson_corr': stats.pearsonr(merged_df['stec_pred'], merged_df['stec_true'])[0],
        'spearman_corr': stats.spearmanr(merged_df['stec_pred'], merged_df['stec_true'])[0],
    }
    
    # Mean uncertainty (if available)
    if 'stec_uncertainty' in merged_df.columns:
        metrics['mean_uncertainty'] = merged_df['stec_uncertainty'].mean()
        metrics['median_uncertainty'] = merged_df['stec_uncertainty'].median()
        
        # Calibration metric: percentage of errors within 1-sigma uncertainty
        within_1sigma = abs_errors <= merged_df['stec_uncertainty']
        metrics['calibration_1sigma'] = within_1sigma.mean() * 100  # percentage
        
        within_2sigma = abs_errors <= 2 * merged_df['stec_uncertainty']
        metrics['calibration_2sigma'] = within_2sigma.mean() * 100  # percentage
    
    return metrics


def plot_comparison(merged_df: pd.DataFrame, output_dir: str = None):
    """
    Create comparison plots.
    
    Args:
        merged_df: DataFrame with both predictions and ground truth
        output_dir: Directory to save plots (if None, displays instead)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Scatter plot: Predicted vs True
    ax = axes[0, 0]
    ax.scatter(merged_df['stec_true'], merged_df['stec_pred'], 
               alpha=0.5, s=20, edgecolors='none')
    
    # Perfect prediction line
    min_val = min(merged_df['stec_true'].min(), merged_df['stec_pred'].min())
    max_val = max(merged_df['stec_true'].max(), merged_df['stec_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('True STEC (TECU)', fontsize=12)
    ax.set_ylabel('Predicted STEC (TECU)', fontsize=12)
    ax.set_title('Predicted vs True STEC', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax = axes[0, 1]
    errors = merged_df['stec_pred'] - merged_df['stec_true']
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(errors.mean(), color='green', linestyle='--', linewidth=2, 
               label=f'Mean error: {errors.mean():.3f} TECU')
    ax.set_xlabel('Prediction Error (TECU)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Error vs Time
    ax = axes[1, 0]
    ax.scatter(merged_df['sod_true'], errors, alpha=0.5, s=20, edgecolors='none')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Second of Day', fontsize=12)
    ax.set_ylabel('Prediction Error (TECU)', fontsize=12)
    ax.set_title('Error vs Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Uncertainty calibration (if available)
    ax = axes[1, 1]
    if 'stec_uncertainty' in merged_df.columns:
        abs_errors = np.abs(errors)
        ax.scatter(merged_df['stec_uncertainty'], abs_errors, 
                   alpha=0.5, s=20, edgecolors='none')
        
        # 1:1 line (perfect calibration)
        max_unc = merged_df['stec_uncertainty'].max()
        ax.plot([0, max_unc], [0, max_unc], 'r--', linewidth=2, label='Perfect calibration')
        
        ax.set_xlabel('Predicted Uncertainty (TECU)', fontsize=12)
        ax.set_ylabel('Absolute Error (TECU)', fontsize=12)
        ax.set_title('Uncertainty Calibration', fontsize=14, fontweight='bold')
        ax.legend()
    else:
        # Error by satellite
        satellite_stats = merged_df.groupby('sat').agg({
            'stec_pred': 'count',
            'stec_true': lambda x: np.abs(merged_df.loc[x.index, 'stec_pred'] - x).mean()
        }).rename(columns={'stec_pred': 'count', 'stec_true': 'mae'})
        satellite_stats = satellite_stats.sort_values('mae', ascending=False).head(15)
        
        ax.barh(range(len(satellite_stats)), satellite_stats['mae'])
        ax.set_yticks(range(len(satellite_stats)))
        ax.set_yticklabels(satellite_stats.index)
        ax.set_xlabel('Mean Absolute Error (TECU)', fontsize=12)
        ax.set_ylabel('Satellite', fontsize=12)
        ax.set_title('MAE by Satellite (Top 15)', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'comparison_plots.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plots saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def print_metrics(metrics: dict, station: str, year: int, doy: int):
    """
    Print metrics in a formatted manner.
    
    Args:
        metrics: Dictionary of computed metrics
        station: Station name
        year: Year
        doy: Day of year
    """
    print("\n" + "="*70)
    print(f"COMPARISON RESULTS: {station}, Year {year}, DOY {doy}")
    print("="*70)
    print(f"\nNumber of matched samples: {metrics['n_samples']}")
    print("\n--- Error Metrics ---")
    print(f"MAE (Mean Absolute Error):     {metrics['mae']:.4f} TECU")
    print(f"RMSE (Root Mean Square Error): {metrics['rmse']:.4f} TECU")
    print(f"Median Absolute Error:         {metrics['median_ae']:.4f} TECU")
    print(f"Bias (Mean Error):             {metrics['bias']:.4f} TECU")
    print(f"Standard Deviation:            {metrics['std']:.4f} TECU")
    print(f"Max Absolute Error:            {metrics['max_error']:.4f} TECU")
    print(f"Min Absolute Error:            {metrics['min_error']:.4f} TECU")
    
    print("\n--- Correlation Metrics ---")
    print(f"R² Score:                      {metrics['r2_score']:.4f}")
    print(f"Pearson Correlation:           {metrics['pearson_corr']:.4f}")
    print(f"Spearman Correlation:          {metrics['spearman_corr']:.4f}")
    
    if 'mean_uncertainty' in metrics:
        print("\n--- Uncertainty Metrics ---")
        print(f"Mean Uncertainty:              {metrics['mean_uncertainty']:.4f} TECU")
        print(f"Median Uncertainty:            {metrics['median_uncertainty']:.4f} TECU")
        print(f"Calibration (within 1σ):       {metrics['calibration_1sigma']:.2f}% (expect ~68%)")
        print(f"Calibration (within 2σ):       {metrics['calibration_2sigma']:.2f}% (expect ~95%)")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare predicted STEC values with ground truth from test.h5'
    )
    parser.add_argument('csv_file', type=str, 
                        help='Path to CSV file with predictions')
    parser.add_argument('--station', type=str, default=None,
                        help='Station name (default: inferred from CSV filename)')
    parser.add_argument('--year', type=int, default=2024,
                        help='Year (default: 2024)')
    parser.add_argument('--doy', type=int, default=122,
                        help='Day of year (default: 122)')
    parser.add_argument('--test-file', type=str, default='data/test.h5',
                        help='Path to test.h5 file (default: data/test.h5)')
    parser.add_argument('--casdcb', action='store_true',
                        help='Use CASDCB database format instead of test.h5 format')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save plots (default: show plots)')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='Tolerance for matching IPP coordinates (default: 1e-3)')
    
    args = parser.parse_args()
    
    # Infer station from filename if not provided
    if args.station is None:
        csv_filename = Path(args.csv_file).stem
        args.station = csv_filename.upper()
        print(f"Inferred station name from filename: {args.station}")
    
    # Resolve test file path relative to script location
    script_dir = Path(__file__).parent.parent
    test_file = script_dir / args.test_file
    
    if not test_file.exists():
        print(f"Error: Test file not found at {test_file}")
        sys.exit(1)
    
    if not Path(args.csv_file).exists():
        print(f"Error: CSV file not found at {args.csv_file}")
        sys.exit(1)
    
    print(f"Loading predictions from: {args.csv_file}")
    pred_df = load_predictions(args.csv_file)
    print(f"\nLoading ground truth from: {test_file}")
    if args.casdcb:
        print("  Using CASDCB database format")
    truth_df = load_ground_truth(str(test_file), args.station, args.year, args.doy, 
                                 use_casdcb=args.casdcb)
    print(f"  Loaded {len(truth_df)} ground truth records")
    
    if len(truth_df) == 0:
        print(f"\nError: No ground truth data found for station {args.station}, "
              f"year {args.year}, DOY {args.doy}")
        sys.exit(1)
    
    print("\nMerging predictions with ground truth...")
    merged_df = merge_predictions_and_truth(pred_df, truth_df, args.tolerance)
    print(f"  Successfully matched {len(merged_df)} records")
    
    if len(merged_df) == 0:
        print("\nError: No matching records found between predictions and ground truth")
        sys.exit(1)
    
    print("\nComputing metrics...")
    metrics = compute_metrics(merged_df)
    
    print_metrics(metrics, args.station, args.year, args.doy)
    
    print("Generating comparison plots...")
    plot_comparison(merged_df, args.output_dir)
    
    # Save detailed results to CSV if output directory specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_file = os.path.join(args.output_dir, 'detailed_results.csv')
        merged_df.to_csv(results_file, index=False)
        print(f"Detailed results saved to: {results_file}")
        
        # Save metrics to text file
        metrics_file = os.path.join(args.output_dir, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Station: {args.station}\n")
            f.write(f"Year: {args.year}\n")
            f.write(f"DOY: {args.doy}\n")
            f.write(f"Number of samples: {metrics['n_samples']}\n\n")
            for key, value in metrics.items():
                if key != 'n_samples':
                    f.write(f"{key}: {value}\n")
        print(f"Metrics saved to: {metrics_file}")


if __name__ == '__main__':
    main()
