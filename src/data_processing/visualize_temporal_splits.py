#!/usr/bin/env python3
"""
Visualize temporal splits as a timeline heatmap.

Creates a heatmap showing the temporal distribution of train/validation/test splits
across years (2010-2024) and months (Jan-Dec).

Color coding:
- Blue: Train
- Orange: Validation  
- Red: Test
- White: No data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import pandas as pd
import seaborn as sns
from pathlib import Path


def load_date_splits():
    """Load the train, validation, and test date splits from files."""
    base_path = Path(__file__).parent
    
    # Load date files
    train_dates = np.loadtxt(base_path / 'train_dates.list', dtype=str)
    val_dates = np.loadtxt(base_path / 'val_dates.list', dtype=str)
    test_dates = np.loadtxt(base_path / 'test_dates.list', dtype=str)
    
    return train_dates, val_dates, test_dates


def create_timeline_heatmap(train_dates, val_dates, test_dates, save_path=None):
    """
    Create a timeline heatmap showing temporal splits.
    
    Args:
        train_dates: Array of training dates in YYYY-MM format
        val_dates: Array of validation dates in YYYY-MM format  
        test_dates: Array of test dates in YYYY-MM format
        save_path: Optional path to save the figure
    """
    # Create a matrix for the heatmap (years x months)
    years = range(2010, 2025)  # 2010-2024
    months = range(1, 13)      # Jan-Dec
    
    # Initialize matrix with 0 (no data)
    # 0: No data, 1: Train, 2: Val, 3: Test
    data_matrix = np.zeros((len(years), len(months)))
    
    # Fill the matrix based on the date splits
    for date_str in train_dates:
        year, month = map(int, date_str.split('-'))
        if year in years:
            year_idx = year - 2010
            month_idx = month - 1
            data_matrix[year_idx, month_idx] = 1
    
    for date_str in val_dates:
        year, month = map(int, date_str.split('-'))
        if year in years:
            year_idx = year - 2010
            month_idx = month - 1
            data_matrix[year_idx, month_idx] = 2
    
    for date_str in test_dates:
        year, month = map(int, date_str.split('-'))
        if year in years:
            year_idx = year - 2010
            month_idx = month - 1
            data_matrix[year_idx, month_idx] = 3
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Define colors: white (no data), blue (train), orange (val), red (test)
    colors = ['white', '#215ACC', '#5ACC21', '#CC215A']  # white, blue, orange, red
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Create the heatmap
    im = ax.imshow(data_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=3)
    
    # Set ticks and labels
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)
    
    # Labels and title
    ax.set_xlabel("Month", fontweight='bold')
    ax.set_ylabel("Year", fontweight='bold')
    ax.set_title("Temporal Dataset Split", 
                fontweight='bold', pad=10)
    
    # Add grid for better visibility
    ax.set_xticks(np.arange(-0.5, len(months), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.8, alpha=0.8)
    
    # Remove minor ticks (keep only major ticks at year/month locations)
    ax.tick_params(which='minor', length=0)
    
    # Create custom legend with percentages
    total_months = len(train_dates) + len(val_dates) + len(test_dates)
    train_pct = len(train_dates) / total_months * 100
    val_pct = len(val_dates) / total_months * 100
    test_pct = len(test_dates) / total_months * 100
    
    legend_elements = [
        patches.Patch(color=colors[1], label=f'Training ({train_pct:.1f}%)'),
        patches.Patch(color=colors[2], label=f'Validation ({val_pct:.1f}%)'),
        patches.Patch(color=colors[3], label=f'Test ({test_pct:.1f}%)'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), 
             ncol=3, frameon=True, fancybox=True, shadow=False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timeline heatmap saved to: {save_path}")
    
    return fig, ax


def print_split_statistics(train_dates, val_dates, test_dates):
    """Print statistics about the temporal splits."""
    print("=" * 60)
    print("TEMPORAL SPLIT STATISTICS")
    print("=" * 60)
    
    print(f"Training months: {len(train_dates)}")
    print(f"Validation months: {len(val_dates)}")
    print(f"Test months: {len(test_dates)}")
    print(f"Total months with data: {len(train_dates) + len(val_dates) + len(test_dates)}")
    
    # Year coverage
    all_dates = list(train_dates) + list(val_dates) + list(test_dates)
    years_covered = sorted(set([int(date.split('-')[0]) for date in all_dates]))
    print(f"Years covered: {years_covered[0]}-{years_covered[-1]} ({len(years_covered)} years)")
    
    # Monthly distribution
    print("\nMonthly distribution:")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for split_name, dates in [('Train', train_dates), ('Val', val_dates), ('Test', test_dates)]:
        month_counts = {}
        for date in dates:
            month = int(date.split('-')[1])
            month_counts[month] = month_counts.get(month, 0) + 1
        
        print(f"  {split_name}:")
        for month in range(1, 13):
            count = month_counts.get(month, 0)
            print(f"    {month_names[month-1]}: {count:2d}", end="  ")
            if month % 4 == 0:
                print()
        print()
    
    # Year distribution
    print("\nYearly distribution:")
    for split_name, dates in [('Train', train_dates), ('Val', val_dates), ('Test', test_dates)]:
        year_counts = {}
        for date in dates:
            year = int(date.split('-')[0])
            year_counts[year] = year_counts.get(year, 0) + 1
        
        print(f"  {split_name}:")
        for year in years_covered:
            count = year_counts.get(year, 0)
            print(f"    {year}: {count:2d}", end="  ")
            if (year - years_covered[0] + 1) % 5 == 0:
                print()
        print()


def main():
    """Main function to create the temporal split visualization."""
    print("Loading temporal split data...")
    
    # Load the date splits
    train_dates, val_dates, test_dates = load_date_splits()
    
    # Print statistics
    print_split_statistics(train_dates, val_dates, test_dates)
    
    # Create the timeline heatmap
    print("\nCreating timeline heatmap...")
    save_path = Path(__file__).parent / 'temporal_splits_heatmap.png'
    fig, ax = create_timeline_heatmap(train_dates, val_dates, test_dates, save_path)
        
    print(f"\nVisualization complete!")
    print(f"Heatmap saved as: {save_path}")


if __name__ == "__main__":
    main()
