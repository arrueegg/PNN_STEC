#!/usr/bin/env python3
"""
Manual Plotting Script for Multi-Day Positioning Results

Loads the aggregated summary CSV, filters out specified outlier days/stations,
and regenerates the paper-ready plots.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def plot_trends(df, output_dir):
    """Generate paper-ready trend plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Common Style Settings
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'font.family': 'sans-serif'
    })
    
    # Define colors
    model_color = '#1f77b4'  # Blue
    gim_color = '#ff7f0e'    # Orange
    
    # Pre-process Data
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # Renaming known columns if they differ
    if 'error_3d_rms' in df.columns:
        df['3d_rms'] = df['error_3d_rms'] * 100 # Convert m to cm

    # Ensure standard column names
    if '3d_rms' in df.columns and 'method' in df.columns:
        
        # 1. High-Quality Time Series (Line Plot with Error Bands)
        # -------------------------------------------------------------------------
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Calculate daily stats
        daily_stats = df.groupby(['date', 'method'])['3d_rms'].agg(['mean', 'std', 'count']).reset_index()
        # Calculate standard error of the mean
        daily_stats['sem'] = daily_stats['std'] / (daily_stats['count'] ** 0.5)
        
        methods = daily_stats['method'].unique()
        
        for method in methods:
            subset = daily_stats[daily_stats['method'] == method]
            color = model_color if 'model' in str(method).lower() else gim_color
            label = "Model Correction" if 'model' in str(method).lower() else "IGS GIM"
            
            plt.plot(subset['date'], subset['mean'], marker='o', markersize=4, 
                     linewidth=2, label=label, color=color)
            
            plt.fill_between(subset['date'], 
                             subset['mean'] - subset['sem'], 
                             subset['mean'] + subset['sem'], 
                             color=color, alpha=0.2)
        
        plt.ylabel('3D RMS Error [cm]', fontweight='bold')
        plt.xlabel('Date', fontweight='bold')
        plt.title('Daily Positioning Performance Trend', fontweight='bold', pad=15)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(frameon=True, framealpha=0.9, loc='upper right')
        
        # Format X-axis dates
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "paper_trend_3d_rms_timeseries.png", dpi=300)
        plt.close()
        
        # 2. Comparative Boxplot Distribution (Aggregated)
        # -------------------------------------------------------------------------
        plt.figure(figsize=(8, 6), dpi=300)
        
        # Filter for relevant methods only if there are many
        plot_df = df[df['method'].isin(methods)] # Just safety
        
        # Create boxplot
        sns.boxplot(x='method', y='3d_rms', hue='method', data=plot_df, 
                    width=0.5, palette=[model_color, gim_color], 
                    showfliers=False, legend=False) # Hide outliers for cleaner look
                    
        # Add swarmplot for data density visibility (optional, good for papers with few points)
        if len(plot_df) < 500:
            sns.swarmplot(x='method', y='3d_rms', data=plot_df, 
                          color=".2", alpha=0.5, size=3)
            
        plt.ylabel('3D RMS Error [cm]', fontweight='bold')
        plt.xlabel('Correction Method', fontweight='bold')
        plt.title('Overall Positioning Accuracy Distribution', fontweight='bold', pad=15)
        
        # Rename x-tick labels for clarity
        current_labels = [l.get_text() for l in plt.gca().get_xticklabels()]
        new_labels = ["Model" if 'model' in l.lower() else "IGS GIM" for l in current_labels]
        ax = plt.gca()
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(new_labels)
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "paper_overall_distribution_boxplot.png", dpi=300)
        plt.close()

        # 3. CDF Plot (Cumulative Distribution Function) - Very common in GNSS papers
        # -------------------------------------------------------------------------
        plt.figure(figsize=(10, 6), dpi=300)
        
        for method in methods:
            subset = df[df['method'] == method].sort_values('3d_rms')
            data = subset['3d_rms'].values
            y = np.arange(1, len(data) + 1) / len(data) * 100 # Percentage
            
            color = model_color if 'model' in str(method).lower() else gim_color
            label = "Model Correction" if 'model' in str(method).lower() else "IGS GIM"
            
            plt.plot(data, y, linewidth=2.5, label=label, color=color)
            
            # Add 95th percentile marker
            p95 = np.percentile(data, 95)
            plt.plot([0, p95], [95, 95], linestyle=':', color=color, alpha=0.5)
            plt.plot([p95, p95], [0, 95], linestyle=':', color=color, alpha=0.5)
            
        plt.xlabel('3D RMS Error [cm]', fontweight='bold')
        plt.ylabel('Cumulative Probability [%]', fontweight='bold')
        plt.title('Error Cumulative Distribution Function (CDF)', fontweight='bold', pad=15)
        
        plt.xlim(0, df['3d_rms'].quantile(0.99) * 1.1)
        plt.ylim(0, 105)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_dir / "paper_cdf_3d_rms.png", dpi=300)
        plt.close()
        
        logger.info(f"Plots saved to: {output_dir}")
        
    else:
        logger.error(f"Required columns (3d_rms or error_3d_rms, method) not found. Columns: {df.columns.tolist()}")    


def main():
    parser = argparse.ArgumentParser(description="Manual Plotting of Positioning Results")
    parser.add_argument("--input", default="multiday_results/positioning/multiday_summary.csv", 
                        help="Path to input summary CSV")
    parser.add_argument("--exclude_dates", help="Comma-separated list of dates/DOYs to exclude (e.g. 2024-05-01,130,2024-12-25)")
    parser.add_argument("--output_dir", default="multiday_results/positioning/manual_plots", 
                        help="Directory to save new plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
        
    logger.info(f"Loading data from: {args.input}")
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return 1
        
    # Pre-process dates
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        
    initial_count = len(df)
    
    # Filter outliers if specified
    if args.exclude_dates:
        exclude_list = [d.strip() for d in args.exclude_dates.split(',')]
        logger.info(f"Excluding dates matching: {exclude_list}")
        
        # Create filter mask
        mask = pd.Series(False, index=df.index)
        
        for ex in exclude_list:
            # Check if it looks like a DOY (integer)
            if ex.isdigit():
                doy = int(ex)
                if 'doy' in df.columns:
                    mask |= (df['doy'] == doy)
            # Check if it looks like a date string
            else:
                try:
                    ts = pd.to_datetime(ex)
                    if 'date' in df.columns:
                        mask |= (df['date'] == ts)
                except:
                    logger.warning(f"Could not parse exclusion date: {ex}")
                    
        df = df[~mask]
        logger.info(f"Removed {initial_count - len(df)} rows. Remaining: {len(df)}")
        
    if len(df) == 0:
        logger.error("No data remaining after filtering!")
        return 1
        
    # Run plotting
    plot_trends(df, args.output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())