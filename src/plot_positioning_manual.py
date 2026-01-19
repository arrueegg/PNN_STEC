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

def prepare_data(df):
    """Normalize columns and units."""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # Renaming known columns if they differ
    if 'error_3d_rms' in df.columns:
        df['3d_rms'] = df['error_3d_rms'] * 100 # Convert m to cm
    if 'error_2d_rms' in df.columns:
        df['2d_rms'] = df['error_2d_rms'] * 100 # Convert m to cm
    if 'u_rms' in df.columns:
        df['up_rms'] = df['u_rms'] * 100 # Convert m to cm
        
    return df

def get_robust_limits(data, percentile=99.0):
    """Get robust axis limits excluding extreme outliers."""
    if len(data) == 0:
        return 0, 1
    return 0, np.percentile(data, percentile) * 1.05

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
        
        # Robust Y-axis limit
        all_means = daily_stats['mean']
        _, y_max = get_robust_limits(all_means, 99)
        plt.ylim(0, y_max)
        
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
        
        # Robust Y-axis
        # For boxplots, let seaborn handle it usually, but we can clamp
        # Since we use showfliers=False, it is already somewhat robust
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "paper_overall_distribution_boxplot.png", dpi=300)
        plt.close()

        # 3. CDF Plot (Cumulative Distribution Function) - Very common in GNSS papers
        # -------------------------------------------------------------------------
        plt.figure(figsize=(10, 6), dpi=300)
        
        robust_max = 0
        for method in methods:
            subset = df[df['method'] == method].sort_values('3d_rms')
            data = subset['3d_rms'].values
            
            # Robust max update
            _, local_max = get_robust_limits(data, 99)
            robust_max = max(robust_max, local_max)
            
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
        
        plt.xlim(0, robust_max * 1.1)
        plt.ylim(0, 105)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_dir / "paper_cdf_3d_rms.png", dpi=300)
        plt.close()
        
        logger.info(f"Main trend plots saved to: {output_dir}")
        
    else:
        logger.error(f"Required columns (3d_rms or error_3d_rms, method) not found. Columns: {df.columns.tolist()}")    


def plot_extended_analysis(df, output_dir):
    """Generate extended analysis plots for deeper insights."""
    output_dir = Path(output_dir)
    
    # Define colors
    model_color = '#1f77b4'  # Blue
    gim_color = '#ff7f0e'    # Orange
    
    methods = df['method'].unique() if 'method' in df.columns else []
    
    # 4. Vertical vs Horizontal Error Scatter
    # -------------------------------------------------------------------------
    if '2d_rms' in df.columns and 'up_rms' in df.columns:
        plt.figure(figsize=(10, 8), dpi=300)
        
        # Global robust limits for scatter
        _, x_max = get_robust_limits(df['2d_rms'], 99.5)
        _, y_max = get_robust_limits(df['up_rms'], 99.5)
        max_limit = max(x_max, y_max)
        
        for method in methods:
            subset = df[df['method'] == method]
            color = model_color if 'model' in str(method).lower() else gim_color
            label = "Model Correction" if 'model' in str(method).lower() else "IGS GIM"
            
            plt.scatter(subset['2d_rms'], subset['up_rms'], 
                       alpha=0.5, label=label, color=color, s=30)
            
        plt.xlabel('2D (Horizontal) RMS Error [cm]', fontweight='bold')
        plt.ylabel('Vertical (Up) RMS Error [cm]', fontweight='bold')
        plt.title('Vertical vs Horizontal Positioning Error', fontweight='bold', pad=15)
        
        # Add 1:1 line
        plt.plot([0, max_limit], [0, max_limit], 'k--', alpha=0.3, label='1:1 Ratio')
        
        plt.xlim(0, max_limit)
        plt.ylim(0, max_limit)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "analysis_vertical_vs_horizontal.png", dpi=300)
        plt.close()
    
    # 5. Error vs Satellite Count
    # -------------------------------------------------------------------------
    if 'mean_nsat' in df.columns and '3d_rms' in df.columns:
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Robust Y limit
        _, y_max = get_robust_limits(df['3d_rms'], 99.5)
        
        for method in methods:
            subset = df[df['method'] == method]
            color = model_color if 'model' in str(method).lower() else gim_color
            label = "Model Correction" if 'model' in str(method).lower() else "IGS GIM"
            
            plt.scatter(subset['mean_nsat'], subset['3d_rms'], 
                       alpha=0.5, label=label, color=color, s=30)

            # Fit trend line
            if len(subset) > 1:
                z = np.polyfit(subset['mean_nsat'], subset['3d_rms'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(subset['mean_nsat'].min(), subset['mean_nsat'].max(), 100)
                plt.plot(x_range, p(x_range), linestyle='--', color=color, alpha=0.8)
        
        plt.xlabel('Mean Number of Satellites', fontweight='bold')
        plt.ylabel('3D RMS Error [cm]', fontweight='bold')
        plt.title('Impact of Satellite Availability on Accuracy', fontweight='bold', pad=15)
        
        plt.ylim(0, y_max)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "analysis_error_vs_satellites.png", dpi=300)
        plt.close()

    # 6. Error Distribution Histogram
    # -------------------------------------------------------------------------
    if '3d_rms' in df.columns:
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Establish robust common bin range
        _, robust_max = get_robust_limits(df['3d_rms'], 99.5)
        bins = np.linspace(0, robust_max, 50)
        
        for method in methods:
            subset = df[df['method'] == method]
            color = model_color if 'model' in str(method).lower() else gim_color
            label = "Model Correction" if 'model' in str(method).lower() else "IGS GIM"
            
            sns.histplot(subset['3d_rms'], color=color, label=label, 
                        kde=True, alpha=0.3, element="step", bins=bins)
            
        plt.xlabel('3D RMS Error [cm]', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        plt.title('Distribution of 3D Positioning Errors', fontweight='bold', pad=15)
        
        plt.xlim(0, robust_max)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "analysis_error_histogram.png", dpi=300)
        plt.close()

    # 7. Error by Station (Top 20 Worst/Best)
    # -------------------------------------------------------------------------
    if 'station' in df.columns and '3d_rms' in df.columns:
        # Group by station and take mean error
        station_stats = df.groupby(['station', 'method'])['3d_rms'].mean().reset_index()
        
        # Focus on Model performance for sorting
        model_stats = station_stats[station_stats['method'].apply(lambda x: 'model' in str(x).lower())]
        
        if not model_stats.empty:
            # Sort by error
            sorted_stations = model_stats.sort_values('3d_rms', ascending=False)
            
            # Helper to plot bar chart
            def plot_bar_stations(stats, title_suffix, filename):
                plt.figure(figsize=(12, 6), dpi=300)
                sns.barplot(x='station', y='3d_rms', data=stats, palette='viridis')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Mean 3D RMS Error [cm]', fontweight='bold')
                plt.xlabel('Station', fontweight='bold')
                plt.title(f'Station Performance: {title_suffix}', fontweight='bold', pad=15)
                plt.grid(True, axis='y', linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(output_dir / filename, dpi=300)
                plt.close()

            # Plot top 20 worst stations
            if len(sorted_stations) > 0:
                plot_bar_stations(sorted_stations.head(20), "Highest Error Stations", "analysis_stations_worst.png")
                
            # Plot top 20 best stations
            if len(sorted_stations) > 20:
                plot_bar_stations(sorted_stations.tail(20), "Lowest Error Stations", "analysis_stations_best.png")
    
    logger.info(f"Extended analysis plots saved to: {output_dir}")


def plot_model_vs_gim_comparison(df, output_dir):
    """Generate direct comparison plots between Model and GIM."""
    output_dir = Path(output_dir)
    
    # Check for necessary components for pivoting
    if 'method' not in df.columns or '3d_rms' not in df.columns:
        logger.warning("Missing 'method' or '3d_rms' columns, skipping comparison plots")
        return

    pivot_cols = []
    if 'station' in df.columns:
        pivot_cols.append('station')
    
    if 'date' in df.columns:
        pivot_cols.append('date')
    elif 'year' in df.columns and 'doy' in df.columns:
        pivot_cols.append('year')
        pivot_cols.append('doy')
    
    if not pivot_cols:
        logger.warning("Could not identify grouping columns (date/station) for pivot.")
        return
        
    try:
        # Filter for only relevant methods
        df_filtered = df[df['method'].isin(['model', 'gim'])].copy()
        if df_filtered.empty:
            # Try fuzzy matching if exact 'model'/'gim' not found
            df['method_norm'] = df['method'].apply(lambda x: 'model' if 'model' in str(x).lower() else ('gim' if 'gim' in str(x).lower() else None))
            df_filtered = df[df['method_norm'].notna()].copy()
            df_filtered['method'] = df_filtered['method_norm']
        else:
            # Normalize method names (e.g. 'model' vs 'Model')
            df_filtered['method'] = df_filtered['method'].apply(lambda x: 'model' if 'model' in str(x).lower() else 'gim')
        
        # Pivot table to put Model and GIM side-by-side
        pivoted = df_filtered.pivot_table(
            index=pivot_cols, 
            columns='method', 
            values='3d_rms'
        ).dropna()
        
        if pivoted.empty:
            logger.warning("No paired Model/GIM data found for comparison plots (after pivoting).")
            return

        # Calculate differences
        pivoted['diff'] = pivoted['gim'] - pivoted['model'] # Positive means Model is better (lower error)
        pivoted['rel_benefit'] = (pivoted['gim'] - pivoted['model']) / pivoted['gim'] * 100
        
        # 1. Scatter Plot (Model vs GIM)
        # -------------------------------------------------------------------------
        plt.figure(figsize=(8, 8), dpi=300)
        
        # Robust Limits
        _, x_max = get_robust_limits(pivoted['gim'], 99.5)
        _, y_max = get_robust_limits(pivoted['model'], 99.5)
        max_val = max(x_max, y_max)
        
        plt.scatter(pivoted['gim'], pivoted['model'], alpha=0.4, s=20, color='purple', edgecolors='none')
        
        # 1:1 Line
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.8, linewidth=1.5, label='1:1 (Equal Performance)')
        
        # Region separation and labels
        plt.fill_between([0, max_val], [0, max_val], 0, color='green', alpha=0.05)
        plt.fill_between([0, max_val], [0, max_val], max_val, color='red', alpha=0.05)
        
        # Add text annotations for regions
        plt.text(max_val*0.7, max_val*0.3, "Model Better", color='green', fontweight='bold', ha='center', fontsize=12, alpha=0.6)
        plt.text(max_val*0.3, max_val*0.7, "GIM Better", color='red', fontweight='bold', ha='center', fontsize=12, alpha=0.6)
        
        plt.xlabel('IGS GIM 3D RMS [cm]', fontweight='bold')
        plt.ylabel('Model 3D RMS [cm]', fontweight='bold')
        plt.title('Direct Performance Comparison: Model vs GIM', fontweight='bold', pad=15)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_scatter_model_vs_gim.png", dpi=300)
        plt.close()

        # 2. Histogram of Improvement
        # -------------------------------------------------------------------------
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Robust bin range
        diff_data = pivoted['diff'].dropna()
        p01, p99 = np.percentile(diff_data, [0.5, 99.5])
        
        # Clip data for visualization only
        viz_data = np.clip(diff_data, p01, p99)
        bins = np.linspace(p01, p99, 50)
        
        sns.histplot(viz_data, kde=True, bins=bins, color='teal', alpha=0.6)
        plt.axvline(0, color='k', linestyle='--', linewidth=2)
        plt.xlabel('Improvement (GIM Error - Model Error) [cm]\nPositive values = Model is better', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        plt.title('Distribution of Model Improvement over GIM', fontweight='bold', pad=15)
        
        plt.xlim(p01, p99)
        
        # Stats annotation
        mean_imp = pivoted['diff'].mean()
        median_imp = pivoted['diff'].median()
        stats_text = f"Mean Imp.: {mean_imp:.2f} cm\nMedian Imp.: {median_imp:.2f} cm"
        plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                       verticalalignment='top', horizontalalignment='right', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_improvement_histogram.png", dpi=300)
        plt.close()

        # 3. Win Rate Pie Chart
        # -------------------------------------------------------------------------
        better_count = (pivoted['model'] < pivoted['gim']).sum()
        total_count = len(pivoted)
        win_rate = better_count / total_count * 100
        
        plt.figure(figsize=(7, 7), dpi=300)
        plt.pie([better_count, total_count - better_count], 
               labels=[f'Model Better\n({win_rate:.1f}%)', f'GIM Better\n({100-win_rate:.1f}%)'],
               colors=['#2ca02c', '#d62728'],
               autopct='%1.1f%%', startangle=90, 
               explode=(0.02, 0),
               textprops={'fontsize': 12, 'weight': 'bold'})
        plt.title('Global Performance: Win Rate', fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_win_rate_pie.png", dpi=300)
        plt.close()
        
        logger.info(f"Comparison plots saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating comparison plots: {e}")
        import traceback
        logger.error(traceback.format_exc())


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
    
    # Prepare data (normalize units and column names)
    df = prepare_data(df)
        
    # Run plotting
    plot_trends(df, args.output_dir)
    plot_extended_analysis(df, args.output_dir)
    plot_model_vs_gim_comparison(df, args.output_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())