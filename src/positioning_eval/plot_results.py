#!/usr/bin/env python3
"""
Plot positioning evaluation results comparing model STEC vs IGS GIM.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def plot_positioning_results(summary_csv, output_dir=None):
    """
    Create visualization plots from positioning evaluation summary.
    
    Args:
        summary_csv: Path to summary CSV file
        output_dir: Directory to save plots (defaults to same as CSV)
    """
    # Read summary data
    df = pd.read_csv(summary_csv)
    
    if output_dir is None:
        output_dir = Path(summary_csv).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Horizontal RMS comparison
    fig, ax = plt.subplots(figsize=(14, 10))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, df['model_horizontal_rms'], width, 
                     label='Model STEC', color='#2E86AB', alpha=0.8)
    bars2 = ax.barh(x + width/2, df['gim_horizontal_rms'], width,
                     label='IGS GIM', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Horizontal RMS (m)', fontsize=12)
    ax.set_ylabel('Station', fontsize=12)
    ax.set_title('Horizontal Positioning Error: Model STEC vs IGS GIM', fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(df['station'], fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'horizontal_rms_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Vertical RMS comparison
    fig, ax = plt.subplots(figsize=(14, 10))
    
    bars1 = ax.barh(x - width/2, df['model_vertical_rms'], width,
                     label='Model STEC', color='#2E86AB', alpha=0.8)
    bars2 = ax.barh(x + width/2, df['gim_vertical_rms'], width,
                     label='IGS GIM', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Vertical RMS (m)', fontsize=12)
    ax.set_ylabel('Station', fontsize=12)
    ax.set_title('Vertical Positioning Error: Model STEC vs IGS GIM', fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(df['station'], fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'vertical_rms_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 3D RMS comparison
    fig, ax = plt.subplots(figsize=(14, 10))
    
    bars1 = ax.barh(x - width/2, df['model_3d_rms'], width,
                     label='Model STEC', color='#2E86AB', alpha=0.8)
    bars2 = ax.barh(x + width/2, df['gim_3d_rms'], width,
                     label='IGS GIM', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('3D RMS (m)', fontsize=12)
    ax.set_ylabel('Station', fontsize=12)
    ax.set_title('3D Positioning Error: Model STEC vs IGS GIM', fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(df['station'], fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3d_rms_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Scatter plot: Model vs GIM (Horizontal)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(df['gim_horizontal_rms'], df['model_horizontal_rms'], 
               s=100, alpha=0.6, c='#2E86AB', edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (equal performance)
    max_val = max(df['gim_horizontal_rms'].max(), df['model_horizontal_rms'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal performance')
    
    # Add labels for stations with significant differences
    for idx, row in df.iterrows():
        diff = abs(row['model_horizontal_rms'] - row['gim_horizontal_rms'])
        if diff > 0.5:  # Label if difference > 0.5m
            ax.annotate(row['station'], 
                       (row['gim_horizontal_rms'], row['model_horizontal_rms']),
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('IGS GIM Horizontal RMS (m)', fontsize=12)
    ax.set_ylabel('Model STEC Horizontal RMS (m)', fontsize=12)
    ax.set_title('Horizontal RMS: Model vs GIM Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'horizontal_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Scatter plot: Model vs GIM (3D)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(df['gim_3d_rms'], df['model_3d_rms'],
               s=100, alpha=0.6, c='#A23B72', edgecolors='black', linewidth=0.5)
    
    max_val = max(df['gim_3d_rms'].max(), df['model_3d_rms'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal performance')
    
    # Add labels for stations with significant differences
    for idx, row in df.iterrows():
        diff = abs(row['model_3d_rms'] - row['gim_3d_rms'])
        if diff > 1.0:  # Label if difference > 1.0m
            ax.annotate(row['station'],
                       (row['gim_3d_rms'], row['model_3d_rms']),
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('IGS GIM 3D RMS (m)', fontsize=12)
    ax.set_ylabel('Model STEC 3D RMS (m)', fontsize=12)
    ax.set_title('3D RMS: Model vs GIM Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3d_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Summary statistics box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Horizontal
    data_h = pd.DataFrame({
        'Model STEC': df['model_horizontal_rms'],
        'IGS GIM': df['gim_horizontal_rms']
    })
    data_h.boxplot(ax=axes[0])
    axes[0].set_ylabel('Horizontal RMS (m)', fontsize=11)
    axes[0].set_title('Horizontal Error Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Vertical
    data_v = pd.DataFrame({
        'Model STEC': df['model_vertical_rms'],
        'IGS GIM': df['gim_vertical_rms']
    })
    data_v.boxplot(ax=axes[1])
    axes[1].set_ylabel('Vertical RMS (m)', fontsize=11)
    axes[1].set_title('Vertical Error Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    
    # 3D
    data_3d = pd.DataFrame({
        'Model STEC': df['model_3d_rms'],
        'IGS GIM': df['gim_3d_rms']
    })
    data_3d.boxplot(ax=axes[2])
    axes[2].set_ylabel('3D RMS (m)', fontsize=11)
    axes[2].set_title('3D Error Distribution', fontsize=12, fontweight='bold')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Improvement analysis
    df['horizontal_improvement'] = df['gim_horizontal_rms'] - df['model_horizontal_rms']
    df['vertical_improvement'] = df['gim_vertical_rms'] - df['model_vertical_rms']
    df['3d_improvement'] = df['gim_3d_rms'] - df['model_3d_rms']
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Sort by improvement
    df_sorted = df.sort_values('horizontal_improvement')
    colors_h = ['green' if x > 0 else 'red' for x in df_sorted['horizontal_improvement']]
    axes[0].barh(range(len(df_sorted)), df_sorted['horizontal_improvement'], color=colors_h, alpha=0.7)
    axes[0].set_yticks(range(len(df_sorted)))
    axes[0].set_yticklabels(df_sorted['station'], fontsize=9)
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Improvement (m)', fontsize=11)
    axes[0].set_title('Horizontal RMS Improvement (Positive = Model Better)', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    df_sorted = df.sort_values('vertical_improvement')
    colors_v = ['green' if x > 0 else 'red' for x in df_sorted['vertical_improvement']]
    axes[1].barh(range(len(df_sorted)), df_sorted['vertical_improvement'], color=colors_v, alpha=0.7)
    axes[1].set_yticks(range(len(df_sorted)))
    axes[1].set_yticklabels(df_sorted['station'], fontsize=9)
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Improvement (m)', fontsize=11)
    axes[1].set_title('Vertical RMS Improvement (Positive = Model Better)', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    df_sorted = df.sort_values('3d_improvement')
    colors_3d = ['green' if x > 0 else 'red' for x in df_sorted['3d_improvement']]
    axes[2].barh(range(len(df_sorted)), df_sorted['3d_improvement'], color=colors_3d, alpha=0.7)
    axes[2].set_yticks(range(len(df_sorted)))
    axes[2].set_yticklabels(df_sorted['station'], fontsize=9)
    axes[2].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[2].set_xlabel('Improvement (m)', fontsize=11)
    axes[2].set_title('3D RMS Improvement (Positive = Model Better)', fontsize=12, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Summary statistics table as image
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    stats_data = []
    metrics = ['Horizontal RMS', 'Vertical RMS', '3D RMS']
    for metric_name, model_col, gim_col in [
        ('Horizontal RMS', 'model_horizontal_rms', 'gim_horizontal_rms'),
        ('Vertical RMS', 'model_vertical_rms', 'gim_vertical_rms'),
        ('3D RMS', 'model_3d_rms', 'gim_3d_rms')
    ]:
        model_mean = df[model_col].mean()
        model_std = df[model_col].std()
        gim_mean = df[gim_col].mean()
        gim_std = df[gim_col].std()
        improvement = gim_mean - model_mean
        improvement_pct = (improvement / gim_mean) * 100
        
        stats_data.append([
            metric_name,
            f'{model_mean:.3f} ± {model_std:.3f}',
            f'{gim_mean:.3f} ± {gim_std:.3f}',
            f'{improvement:.3f} ({improvement_pct:+.1f}%)'
        ])
    
    table = ax.table(cellText=stats_data,
                     colLabels=['Metric', 'Model STEC (m)', 'IGS GIM (m)', 'Improvement'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8E8E8')
    
    plt.title('Summary Statistics: Model STEC vs IGS GIM', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / 'summary_statistics_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Generated 8 plots in {output_dir}/")
    print(f"  - horizontal_rms_comparison.png")
    print(f"  - vertical_rms_comparison.png")
    print(f"  - 3d_rms_comparison.png")
    print(f"  - horizontal_scatter.png")
    print(f"  - 3d_scatter.png")
    print(f"  - error_distribution_boxplots.png")
    print(f"  - improvement_analysis.png")
    print(f"  - summary_statistics_table.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot positioning evaluation results")
    parser.add_argument("summary_csv", help="Path to summary CSV file")
    parser.add_argument("--output-dir", help="Output directory for plots (default: same as CSV)")
    
    args = parser.parse_args()
    
    plot_positioning_results(args.summary_csv, args.output_dir)
