#!/usr/bin/env python3
"""
Manual Plotting Script for Multi-Day Positioning Results

Loads the aggregated summary CSV, filters out specified outlier days/stations,
and regenerates the paper-ready plots with advanced visual styling.
Supports complex comparison between Direct STEC, VTEC, and GIM methods.
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

# Style Definitions
STEC_COLOR = '#1f77b4'  # Blue
VTEC_COLOR = '#ff7f0e'  # Orange
GIM_COLOR = '#2ca02c'   # Green
PRETRAINED_COLOR = '#9467bd'  # Purple

def get_style(method_name):
    """Return styling based on normalized method name."""
    m_lower = str(method_name).lower()
    if 'pretrained' in m_lower:
        return PRETRAINED_COLOR, "Pretrained Direct STEC", 'd'
    elif 'stec' in m_lower and 'direct' in m_lower:
        return STEC_COLOR, "Direct STEC", 'o'
    elif 'vtec' in m_lower:
        return VTEC_COLOR, "VTEC + Mapping", 's'
    elif 'gim' in m_lower:
        return GIM_COLOR, "IGS GIM + Mapping", '^'
    
    # Fallbacks for legacy/other names
    if 'model' in m_lower: return STEC_COLOR, "Direct STEC", 'o'
    
    return 'gray', method_name, 'x'

def prepare_data(df):
    """Normalize columns and units."""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
    
    # Renaming known columns if they differ
    if 'error_3d_rms' in df.columns:
        df['3d_rms'] = df['error_3d_rms'] # Keep in meters
    if 'error_2d_rms' in df.columns:
        df['2d_rms'] = df['error_2d_rms'] # Keep in meters
    if 'u_rms' in df.columns:
        df['up_rms'] = df['u_rms'] # Keep in meters

    # Clean Method Names and Extract Weighting
    if 'method' in df.columns:
        # Extract weighting if it exists (e.g. STEC_elev -> weighting=elev)
        def extract_weighting(m):
            m_str = str(m)
            if '_' in m_str:
                return m_str.split('_')[-1]
            return 'default'
        
        def extract_base_method(m):
            m_str = str(m)
            if '_' in m_str:
                base = '_'.join(m_str.split('_')[:-1])
                return base
            return m_str

        # Add weighting column
        df['weighting'] = df['method'].apply(extract_weighting)
        # Normalize base method name
        df['method'] = df['method'].apply(extract_base_method)
        
        df['method'] = df['method'].replace({
            'Model': 'Direct STEC', # Legacy
            'STEC': 'Direct STEC',
            'stec': 'Direct STEC',
            'Finetuned STEC': 'Direct STEC', # Support new pipeline naming
            'Pretrained_STEC': 'Pretrained Direct STEC',
            'Pretrained STEC': 'Pretrained Direct STEC',
            'IGS GIM': 'IGS GIM + Mapping',
            'igs gim': 'IGS GIM + Mapping',
            'GIM': 'IGS GIM + Mapping',
            'gim': 'IGS GIM + Mapping',
            'model': 'Direct STEC',
            'VTEC': 'VTEC + Mapping',
            'vtec': 'VTEC + Mapping'
        })
        
    return df

def get_robust_limits(data, percentile=99.0):
    """Get robust axis limits excluding extreme outliers."""
    if len(data) == 0:
        return 0, 1
    return 0, np.percentile(data, percentile) * 1.2

def plot_trends(df, output_dir, threshold_cm=None):
    """Generate paper-ready trend plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply filtering if threshold provided
    if threshold_cm is not None:
        logger.info(f"Filtering data with threshold: {threshold_cm} m")
        
        # Get filtered out records before filtering
        filtered_out = df[df['3d_rms'] > threshold_cm]
        
        if len(filtered_out) > 0:
            logger.info(f"Filtered out {len(filtered_out)} records:")
            # Print each day-station combination that's filtered
            for _, row in filtered_out.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                station = row.get('station', 'N/A')
                method = row.get('method', 'N/A')
                rms = row['3d_rms']
                logger.info(f"  {date_str} | {station} | {method}: {rms:.2f} m")
        
        df = df[df['3d_rms'] <= threshold_cm].copy()
        logger.info(f"Data after filtering: {len(df)} records remaining")
    
    # Common Style Settings
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'font.family': 'sans-serif',
        'lines.linewidth': 2
    })
    
    if '3d_rms' in df.columns and 'method' in df.columns:
        
        # 1. High-Quality Time Series (Line Plot with Error Bands)
        # -------------------------------------------------------------------------
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Calculate daily stats
        daily_stats = df.groupby(['date', 'method'])['3d_rms'].agg(['mean', 'std', 'count']).reset_index()
        daily_stats['sem'] = daily_stats['std'] / (daily_stats['count'] ** 0.5)
        
        unique_methods = daily_stats['method'].unique()
        # Sort methods: Pretrained STEC, Finetuned STEC, VTEC, GIM
        ordered_methods = sorted(unique_methods, key=lambda x: (
            0 if 'pretrained' in str(x).lower() else
            1 if 'stec' in str(x).lower() else 
            2 if 'vtec' in str(x).lower() else 
            3
        ))
        
        for method in ordered_methods:
            subset = daily_stats[daily_stats['method'] == method]
            color, label, marker = get_style(method)
            
            plt.plot(subset['date'], subset['mean'], marker=marker, markersize=5, 
                     linewidth=2, label=label, color=color, zorder=3-ordered_methods.index(method))
            
            plt.fill_between(subset['date'], 
                             subset['mean'] - subset['sem'], 
                             subset['mean'] + subset['sem'], 
                             color=color, alpha=0.2)
        
        plt.ylabel('3D RMS Error [m]', fontweight='bold')
        plt.xlabel('Date', fontweight='bold')
        # plt.title('Daily Positioning Performance Trend', fontweight='bold', pad=15)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(frameon=True, framealpha=0.9, loc='best')
        
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.xticks(rotation=45)
        
        plt.ylim(0, 3.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / "paper_trend_3d_rms_timeseries.png", dpi=300)
        plt.close()

        # 2. Daily Improvement vs GIM
        # -------------------------------------------------------------------------
        daily_pivot = daily_stats.pivot(index='date', columns='method', values='mean')
        
        gim_col = next((c for c in daily_pivot.columns if 'gim' in str(c).lower()), None)
        
        if gim_col:
            model_cols = [c for c in daily_pivot.columns if c != gim_col]
            
            if model_cols:
                plt.figure(figsize=(10, 6), dpi=300)
                
                for m_col in model_cols[::-1]:
                    color, label, marker = get_style(m_col)
                    
                    # Improvement: (GIM - Model) / GIM * 100
                    improvement = (daily_pivot[gim_col] - daily_pivot[m_col]) / daily_pivot[gim_col] * 100
                    
                    # Plot the line
                    plt.plot(improvement.index, improvement.values, marker=marker, markersize=4,
                             linewidth=1.5, label=f"Imp. by {label}", color=color)
                    
                    
                    # Add stats to legend or as text
                    mean_imp = improvement.mean()
                    logger.info(f"Mean improvement for {m_col}: {mean_imp:.1f}%")

                plt.axhline(0, color='black', linestyle='--', alpha=0.5)
                
                plt.ylabel('Improvement over GIM [%]', fontweight='bold')
                plt.xlabel('Date', fontweight='bold')
                # plt.title('Daily Relative Improvement in 3D Accuracy', fontweight='bold', pad=15)
                
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                ax = plt.gca()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig(output_dir / "paper_trend_improvement_timeseries.png", dpi=300)
                plt.close()

        logger.info(f"Main trend plots saved to: {output_dir}")

def plot_extended_analysis(df, output_dir, threshold_cm=None):
    """Generate extended analysis plots for deeper insights."""
    output_dir = Path(output_dir)
    
    methods = df['method'].unique() if 'method' in df.columns else []
    
    # 1. Overall Distribution (Boxplot with Styling) - Uses FULL data
    # -------------------------------------------------------------------------
    if '3d_rms' in df.columns and 'method' in df.columns:
        plt.figure(figsize=(8, 6), dpi=300)
        
        # Sort methods: Pretrained STEC, Finetuned STEC, VTEC, GIM
        ordered_methods = sorted(methods, key=lambda x: (
            0 if 'pretrained' in str(x).lower() else
            1 if 'stec' in str(x).lower() else 
            2 if 'vtec' in str(x).lower() else 
            3
        ))
        
        plot_df = df[df['method'].isin(ordered_methods)]
        palette = {m: get_style(m)[0] for m in ordered_methods}
        
        sns.boxplot(x='method', y='3d_rms', hue='method', data=plot_df, 
                    width=0.5, palette=palette, 
                    showfliers=False, order=ordered_methods)
        
        plt.ylabel('3D RMS Error [m]', fontweight='bold')
        plt.xlabel('Correction Method', fontweight='bold')
        # plt.title('Overall Positioning Accuracy Distribution', fontweight='bold', pad=15)
        
        new_labels = [get_style(m)[1] for m in ordered_methods]
        ax = plt.gca()
        ax.set_xticks(range(len(ordered_methods)))
        ax.set_xticklabels(new_labels)
        
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / "paper_overall_distribution_boxplot.png", dpi=300)
        plt.close()

        # 2. CDF Plot (Critical for Papers) - Uses FULL data
        # -------------------------------------------------------------------------
        plt.figure(figsize=(8, 6), dpi=300)
        
        robust_max = 0
        
        for method in ordered_methods:
            subset = df[df['method'] == method]['3d_rms'].dropna().sort_values()
            
            if len(subset) == 0: continue
            
            # Compute CDF
            y_vals = np.arange(1, len(subset) + 1) / len(subset) * 100
            
            color, label, _ = get_style(method)
            plt.plot(subset, y_vals, label=label, linewidth=2.5, color=color)
            
            # Update limits
            curr_max = np.percentile(subset, 98)
            robust_max = max(robust_max, curr_max)
            
            # Print stats
            rms_95 = np.percentile(subset, 95)
            logger.info(f"{method} 95%: {rms_95:.2f} m")

        plt.xlabel('3D RMS Error [m]', fontweight='bold')
        plt.ylabel('Cumulative Probability [%]', fontweight='bold')
        # plt.title('Error Cumulative Distribution Function (CDF)', fontweight='bold', pad=15)
        
        # Use threshold for x-axis limit if provided, otherwise use robust limits
        if threshold_cm:
            plt.xlim(0, threshold_cm)
        else:
            plt.xlim(0, robust_max * 1.2)
        plt.ylim(0, 102)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(output_dir / "paper_cdf_3d_rms.png", dpi=300)
        plt.close()
    
    # 3. Vertical vs Horizontal Error Scatter - Uses FULL data
    # -------------------------------------------------------------------------
    if '2d_rms' in df.columns and 'up_rms' in df.columns:
        plt.figure(figsize=(10, 8), dpi=300)
        
        # Global robust limits for scatter
        _, x_max = get_robust_limits(df['2d_rms'], 99.5)
        _, y_max = get_robust_limits(df['up_rms'], 99.5)
        max_limit = max(x_max, y_max)
        
        # Use threshold if provided
        if threshold_cm:
            max_limit = threshold_cm
        
        for method in methods:
            subset = df[df['method'] == method]
            color, label, _ = get_style(method)
            
            plt.scatter(subset['2d_rms'], subset['up_rms'], 
                       alpha=0.4, label=label, color=color, s=25)
            
        plt.xlabel('2D (Horizontal) RMS Error [m]', fontweight='bold')
        plt.ylabel('Vertical (Up) RMS Error [m]', fontweight='bold')
        # plt.title('Vertical vs Horizontal Positioning Error', fontweight='bold', pad=15)
        
        # Add 1:1 line
        plt.plot([0, max_limit], [0, max_limit], 'k--', alpha=0.3, label='1:1 Ratio')
        
        plt.xlim(0, max_limit)
        plt.ylim(0, max_limit)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "analysis_vertical_vs_horizontal.png", dpi=300)
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
            color, label, _ = get_style(method)
            
            sns.histplot(subset['3d_rms'], color=color, label=label, 
                        kde=True, alpha=0.3, element="step", bins=bins)
            
        plt.xlabel('3D RMS Error [m]', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        # plt.title('Distribution of 3D Positioning Errors', fontweight='bold', pad=15)
        
        plt.xlim(0, robust_max)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "analysis_error_histogram.png", dpi=300)
        plt.close()
        
    logger.info(f"Extended analysis plots saved to: {output_dir}")

def plot_model_vs_gim_comparison(df, output_dir, threshold_cm=None):
    """Generate direct comparison plots between Model and GIM."""
    output_dir = Path(output_dir)
    
    # Check for necessary components for pivoting
    if 'method' not in df.columns or '3d_rms' not in df.columns:
        logger.warning("Missing 'method' or '3d_rms' columns, skipping comparison plots")
        return

    # Identify columns to pivot on
    pivot_cols = []
    if 'station' in df.columns: pivot_cols.append('station')
    if 'date' in df.columns:
        pivot_cols.append('date')
    elif 'year' in df.columns and 'doy' in df.columns:
        pivot_cols.extend(['year', 'doy'])
    
    if not pivot_cols:
        logger.warning("Could not identify grouping columns (date/station) for pivot.")
        return
        
    try:
        # Create normalized method column for pivoting: 'stec', 'vtec', 'gim', 'pretrained'
        def normalize_method(m):
            m = str(m).lower()
            if 'pretrained' in m: return 'pretrained'
            if 'stec' in m or 'direct' in m: return 'stec'
            if 'vtec' in m: return 'vtec'
            if 'gim' in m: return 'gim'
            return None

        # Work on a copy
        df_comp = df.copy()
        df_comp['pivot_method'] = df_comp['method'].apply(normalize_method)
        df_filtered = df_comp[df_comp['pivot_method'].notna()].copy()
        
        # Pivot table
        pivoted = df_filtered.pivot_table(
            index=pivot_cols, 
            columns='pivot_method', 
            values='3d_rms'
        ).dropna() # Only keep rows where ALL selected methods exist
        
        if pivoted.empty:
            logger.warning("No paired data found for comparison plots.")
            return

        if 'gim' not in pivoted.columns:
            logger.warning("GIM data missing from pivot. Cannot compare.")
            return

        # Iterate over model types (stec, vtec) to compare against GIM
        model_types = [c for c in pivoted.columns if c != 'gim']
        
        # Define comparisons: Each model vs GIM
        comparisons = []
        for m in model_types:
            if m == 'stec': challenger_label = "Direct STEC"
            elif m == 'vtec': challenger_label = "VTEC + Mapping"
            elif m == 'pretrained': challenger_label = "Pretrained Direct STEC"
            else: challenger_label = str(m)
            
            comparisons.append({
                'challenger': m,
                'baseline': 'gim',
                'challenger_label': challenger_label,
                'baseline_label': "IGS GIM",
                'type': 'vs_gim'
            })
            
        # Add STEC vs VTEC comparison if both exist
        if 'stec' in pivoted.columns and 'vtec' in pivoted.columns:
            comparisons.append({
                'challenger': 'stec',
                'baseline': 'vtec',
                'challenger_label': "Direct STEC",
                'baseline_label': "VTEC + Mapping",
                'type': 'stec_vs_vtec'
            })
            
        # Add STEC vs Pretrained comparison if both exist
        if 'stec' in pivoted.columns and 'pretrained' in pivoted.columns:
            comparisons.append({
                'challenger': 'stec',
                'baseline': 'pretrained',
                'challenger_label': "Direct STEC",
                'baseline_label': "Pretrained Direct STEC",
                'type': 'stec_vs_pretrained'
            })
        
        for comp in comparisons:
            m_type = comp['challenger']
            baseline = comp['baseline']
            
            model_label = comp['challenger_label']
            baseline_label = comp['baseline_label']
            
            color, _, _ = get_style(model_label)
            safe_name = f"{m_type}_vs_{baseline}"
            
            # Differences: Baseline - Challenger (Positive = Challenger is better)
            diff_col = f'diff_{safe_name}'
            pivoted[diff_col] = pivoted[baseline] - pivoted[m_type]
            
            # 7. Scatter Plot
            # -------------------------------------------------------------------------
            plt.figure(figsize=(8, 8), dpi=300)
            
            # Robust Limits
            _, x_max = get_robust_limits(pivoted[baseline], 99.5)
            _, y_max = get_robust_limits(pivoted[m_type], 99.5)
            max_val = max(x_max, y_max)
            
            # Use threshold if provided
            if threshold_cm:
                max_val = threshold_cm
            
            plt.scatter(pivoted[baseline], pivoted[m_type], alpha=0.8, s=20, color=color, edgecolors='none')
            
            # 1:1 Line
            plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.8, linewidth=1.5, label='1:1 Line')
            
            # Shaded Regions
            # Green: Model Better (Under the 1:1 line) -> y < x
            plt.fill_between([0, max_val], [0, max_val], 0, color='green', alpha=0.05, label=f'{model_label} Better')
            # Red: Baseline Better (Above the 1:1 line) -> y > x
            plt.fill_between([0, max_val], [0, max_val], max_val, color='red', alpha=0.05, label=f'{baseline_label} Better')

            plt.xlabel(f'{baseline_label} 3D RMS [m]', fontweight='bold')
            plt.ylabel(f'{model_label} 3D RMS [m]', fontweight='bold')
            # plt.title(f'{model_label} vs {baseline_label} Performance', fontweight='bold', pad=15)
            
            # Text annotations for regions
            plt.text(max_val*0.75, max_val*0.25, f"{model_label}\nBetter", 
                    color='green', fontweight='bold', ha='center', fontsize=12, alpha=0.7)
            plt.text(max_val*0.25, max_val*0.75, f"{baseline_label}\nBetter", 
                    color='#d62728', fontweight='bold', ha='center', fontsize=12, alpha=0.7)
            
            plt.xlim(0, max_val)
            plt.ylim(0, max_val)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            
            plt.savefig(output_dir / f"comparison_scatter_{safe_name}.png", dpi=300)
            plt.close()

            # 8. Histogram of Improvement
            # -------------------------------------------------------------------------
            plt.figure(figsize=(10, 6), dpi=300)
            
            diff_data = pivoted[diff_col].dropna()
            if len(diff_data) > 0:
                p01, p99 = np.percentile(diff_data, [0.5, 99.5])
                
                # Clip data for visualization only
                viz_data = np.clip(diff_data, p01, p99)
                bins = np.linspace(p01, p99, 50)
                
                # Plot
                sns.histplot(viz_data, kde=True, bins=bins, color=color, alpha=0.6)
                
                # Color background areas
                y_min, y_max = plt.ylim()
                plt.axvspan(0, p99, color='green', alpha=0.05, label=f'{model_label} Better')
                plt.axvspan(p01, 0, color='red', alpha=0.05, label=f'{baseline_label} Better')
                plt.axvline(0, color='k', linestyle='--', linewidth=2)

                plt.xlabel(f'Improvement ({baseline_label} Error - {model_label} Error) [m]\nPositive values = {model_label} is better', fontweight='bold')
                plt.ylabel('Count', fontweight='bold')
                # plt.title(f'Distribution of {model_label} Improvement over {baseline_label}', fontweight='bold', pad=15)
                
                plt.xlim(p01, p99)
                
                # Stats annotation
                mean_imp = diff_data.mean()
                median_imp = diff_data.median()
                stats_text = f"Mean Imp.: {mean_imp:.2f} m\nMedian Imp.: {median_imp:.2f} m"
                plt.gca().text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
                            verticalalignment='top', horizontalalignment='right', 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
                
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / f"comparison_improvement_histogram_{safe_name}.png", dpi=300)
                plt.close()

            # 9. Win Rate Pie Chart
            # -------------------------------------------------------------------------
            better_count = (pivoted[m_type] < pivoted[baseline]).sum()
            total_count = len(pivoted)
            win_rate = better_count / total_count * 100
            
            # Use fixed axes to ensure consistent pie size regardless of title length
            fig = plt.figure(figsize=(8, 8), dpi=300)
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.7]) # Left, Bottom, Width, Height (Fixed Square)
            
            # Explicit shadow parameter and better colors
            wedges, texts, autotexts = ax.pie([better_count, total_count - better_count], 
                labels=[f'{model_label}\nBetter ({win_rate:.1f}%)', f'{baseline_label} Better\n({100-win_rate:.1f}%)'],
                colors=[color, '#d62728'], # Use model color for "better", red for "gim better"
                autopct='%1.1f%%', 
                startangle=90,
                explode=(0.05, 0), # Explode the winner slice slightly
                shadow=False, # Ensure shadow is off
                textprops={'fontsize': 12, 'weight': 'bold'})
                
            plt.setp(texts, size=12, weight="bold")
            plt.setp(autotexts, size=11, weight="bold", color="white")
            
            # Title attached to figure or axes, but axes size is fixed now
            # ax.set_title(f'Win Rate: {model_label} vs {baseline_label}\n(Based on 3D RMS)', pad=20, fontweight='bold', fontsize=14)
            
            # No tight_layout to preserve fixed axes dimensions
            plt.savefig(output_dir / f"comparison_win_rate_pie_{safe_name}.png", dpi=300)
            plt.close()
        
        logger.info(f"Comparison plots saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating comparison plots: {e}")
        import traceback
        logger.error(traceback.format_exc())

def generate_comparative_table(df, output_dir, threshold_m=None):
    """Generate a summary table comparing methods and weightings."""
    output_dir = Path(output_dir)
    
    # Filter by threshold if provided (to match plot logic)
    df_clean = df.copy()
    if threshold_m is not None:
        df_clean = df_clean[df_clean['3d_rms'] <= threshold_m]

    # Metrics to calculate
    metrics = {
        '3d_rms': ['mean', 'median', lambda x: np.percentile(x, 95)],
    }
    # Add 2D and Up if they exist
    if '2d_rms' in df_clean.columns: 
        metrics['2d_rms'] = ['mean']
    if 'up_rms' in df_clean.columns: 
        metrics['up_rms'] = ['mean']

    # Group and aggregate
    stats = df_clean.groupby(['method', 'weighting']).agg(metrics)
    
    # Flatten multi-index columns
    stats.columns = [
        f"{col}_{func if isinstance(func, str) else '95th'}" 
        for col, func in stats.columns
    ]
    stats = stats.reset_index()

    # Rename columns for clarity
    rename_map = {
        '3d_rms_mean': '3D Mean [m]',
        '3d_rms_median': '3D Med [m]',
        '3d_rms_95th': '3D 95% [m]',
        '2d_rms_mean': '2D Mean [m]',
        'up_rms_mean': 'Up Mean [m]'
    }
    stats = stats.rename(columns=rename_map)

    # Calculate Improvement vs GIM baseline within each weighting group
    stats['Imp. [%]'] = 0.0
    for w in stats['weighting'].unique():
        mask_w = stats['weighting'] == w
        gim_mask = stats['method'].str.contains('GIM', case=False) & mask_w
        if gim_mask.any():
            gim_val = stats.loc[gim_mask, '3D Mean [m]'].values[0]
            # Calculate for non-GIM models in this weighting
            model_mask = mask_w & ~gim_mask
            stats.loc[model_mask, 'Imp. [%]'] = (gim_val - stats.loc[model_mask, '3D Mean [m]']) / gim_val * 100

    # Sort by Weighting, then Method (Standard order: Pretrained STEC, STEC, VTEC, GIM)
    def method_priority(m):
        m = str(m).lower()
        if 'pretrained' in m: return 0
        if 'stec' in m or 'direct' in m: return 1
        if 'vtec' in m: return 2
        if 'gim' in m: return 3
        return 4
    
    stats['prio'] = stats['method'].apply(method_priority)
    stats = stats.sort_values(['weighting', 'prio']).drop('prio', axis=1)

    # Round numerical columns
    num_cols = [c for c in stats.columns if '[m]' in c or '[%]' in c]
    stats[num_cols] = stats[num_cols].round(4)

    # Save to CSV
    table_path = output_dir / "overall_metrics_comparison.csv"
    stats.to_csv(table_path, index=False)
    logger.info(f"Comparative table saved to: {table_path}")
    
    # Print a summary to the console
    print("\n" + "="*90)
    print("KEY PERFORMANCE METRICS (COMPARING WEIGHTINGS AND METHODS)")
    print("="*90)
    print(stats.to_string(index=False))
    print("="*90 + "\n")
    
    # NEW: Generate visual comparison of the table data
    plot_comparative_metrics(stats, output_dir)

def plot_comparative_metrics(stats_df, output_dir):
    """Generate grouped bar charts comparing metrics across methods and weightings."""
    output_dir = Path(output_dir)
    
    # 1. 3D Mean Error Comparison (Grouped Bar Chart)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Methods on X, Weighting as Hue
    # Use seaborn for easy grouped bars
    ax = sns.barplot(
        data=stats_df, 
        x='method', 
        y='3D Mean [m]', 
        hue='weighting',
        palette='viridis'
    )
    
    plt.ylabel('Mean 3D RMS Error [m]', fontweight='bold')
    plt.xlabel('Method', fontweight='bold')
    plt.title('Accuracy Comparison: Methods vs. Weighting Strategies', fontweight='bold', pad=15)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Weighting', frameon=True)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)
        
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_bar_3d_mean.png", dpi=300)
    plt.close()

    # 2. Improvement over GIM (Grouped Bar Chart)
    # -------------------------------------------------------------------------
    # Only plot for methods that aren't GIM
    imp_df = stats_df[~stats_df['method'].str.contains('GIM', case=False)].copy()
    
    if not imp_df.empty:
        plt.figure(figsize=(10, 6), dpi=300)
        
        ax = sns.barplot(
            data=imp_df, 
            x='method', 
            y='Imp. [%]', 
            hue='weighting',
            palette='magma'
        )
        
        plt.ylabel('Improvement over GIM [%]', fontweight='bold')
        plt.xlabel('Method', fontweight='bold')
        plt.title('Relative Improvement Score vs. IGS Baseline', fontweight='bold', pad=15)
        
        plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.legend(title='Weighting', frameon=True)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10)
            
        plt.tight_layout()
        plt.savefig(output_dir / "comparison_bar_improvement.png", dpi=300)
        plt.close()

    logger.info(f"Visual table comparisons saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Manual Plotting of Positioning Results")
    parser.add_argument("--input", default="multiday_results/positioning/multiday_summary.csv", 
                        help="Path to input summary CSV")
    parser.add_argument("--exclude_dates", help="Comma-separated list of dates/DOYs to exclude")
    parser.add_argument("--exclude_stations", help="Comma-separated list of station IDs to exclude")
    parser.add_argument("--exclude_threshold", type=float, help="RMS Error threshold (in meters) to exclude")
    parser.add_argument("--output_dir", default="multiday_results/positioning/manual_plots", 
                        help="Directory to save new plots")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
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
            if ex.isdigit():
                doy = int(ex)
                if 'doy' in df.columns:
                    mask |= (df['doy'] == doy)
            else:
                try:
                    ts = pd.to_datetime(ex)
                    if 'date' in df.columns:
                        mask |= (df['date'] == ts)
                except:
                    logger.warning(f"Could not parse exclusion date: {ex}")
                    
        df = df[~mask]
        
    if args.exclude_stations:
        stations = [s.strip() for s in args.exclude_stations.split(',')]
        logger.info(f"Excluding stations: {stations}")
        if 'station' in df.columns:
            df = df[~df['station'].isin(stations)]
            
    # Prepare Data
    threshold_m = None
    if args.exclude_threshold:
        logger.info(f"Filtering data with threshold: {args.exclude_threshold} m")
        threshold_m = args.exclude_threshold  # Keep in meters
    
    df = prepare_data(df)
    
    if len(df) < initial_count:
        logger.info(f"Filtered {initial_count - len(df)} rows. Remaining: {len(df)}")

    # Generate Comparative Table (New: compares all weightings in one place)
    generate_comparative_table(df, args.output_dir, threshold_m=threshold_m)

    # Generate Plots by Weighting Approach
    weightings = df['weighting'].unique()
    logger.info(f"Detected weightings: {weightings}")
    
    for weighting in weightings:
        logger.info(f"\n--- Generating plots for weighting: {weighting} ---")
        w_df = df[df['weighting'] == weighting].copy()
        
        # Create specialized output directory for this weighting
        if weighting != 'default':
            w_output_dir = Path(args.output_dir) / weighting
        else:
            w_output_dir = Path(args.output_dir)
            
        w_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate the standard suite of plots for this weighting group
        plot_trends(w_df, w_output_dir, threshold_cm=threshold_m)
        plot_extended_analysis(w_df, w_output_dir, threshold_cm=threshold_m)
        plot_model_vs_gim_comparison(w_df, w_output_dir, threshold_cm=threshold_m)
    
    logger.info(f"\nDone. All plots saved to: {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
