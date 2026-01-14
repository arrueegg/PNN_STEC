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
from pathlib import Path
import argparse


def load_date_splits():
    """Load the train, validation, and test date splits from files."""
    base_path = Path(__file__).parent

    # Load date files
    train_dates = np.loadtxt(base_path / "train_dates.list", dtype=str)
    val_dates = np.loadtxt(base_path / "val_dates.list", dtype=str)
    test_dates = np.loadtxt(base_path / "test_dates.list", dtype=str)

    return train_dates, val_dates, test_dates


def create_timeline_heatmap(
    train_dates,
    val_dates,
    test_dates,
    save_path=None,
    start_year=2014,
    end_year=2024,
):
    """
    Create a timeline heatmap showing temporal splits.

    Args:
        train_dates: Array of training dates in YYYY-MM format
        val_dates: Array of validation dates in YYYY-MM format
        test_dates: Array of test dates in YYYY-MM format
        save_path: Optional path to save the figure
        start_year: Start year for visualization
        end_year: End year for visualization (inclusive)
    """
    # Create a matrix for the heatmap (years x months)
    years = range(start_year, end_year + 1)
    months = range(1, 13)  # Jan-Dec

    # Initialize matrix with 0 (no data)
    # 0: No data, 1: Train, 2: Val, 3: Test
    data_matrix = np.zeros((len(years), len(months)))

    # Fill the matrix based on the date splits
    for date_str in train_dates:
        year, month = map(int, date_str.split("-"))
        if year in years:
            year_idx = year - start_year
            month_idx = month - 1
            data_matrix[year_idx, month_idx] = 1

    for date_str in val_dates:
        year, month = map(int, date_str.split("-"))
        if year in years:
            year_idx = year - start_year
            month_idx = month - 1
            data_matrix[year_idx, month_idx] = 2

    for date_str in test_dates:
        year, month = map(int, date_str.split("-"))
        if year in years:
            year_idx = year - start_year
            month_idx = month - 1
            data_matrix[year_idx, month_idx] = 3

    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Define colors: white (no data), blue (train), orange (val), red (test)
    colors = ["white", "#215ACC", "#5ACC21", "#CC215A"]  # white, blue, orange, red
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    # Create the heatmap
    ax.imshow(data_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=3)

    # Set ticks and labels
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )

    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)

    # Labels and title
    ax.set_xlabel("Month", fontweight="bold")
    ax.set_ylabel("Year", fontweight="bold")
    ax.set_title("Temporal Dataset Split", fontweight="bold", pad=10)

    # Add grid for better visibility
    ax.set_xticks(np.arange(-0.5, len(months), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.8, alpha=0.8)

    # Remove minor ticks (keep only major ticks at year/month locations)
    ax.tick_params(which="minor", length=0)

    # Create custom legend with percentages
    train_count = np.sum(data_matrix == 1)
    val_count = np.sum(data_matrix == 2)
    test_count = np.sum(data_matrix == 3)
    total_months = train_count + val_count + test_count

    if total_months > 0:
        train_pct = train_count / total_months * 100
        val_pct = val_count / total_months * 100
        test_pct = test_count / total_months * 100
    else:
        train_pct = 0.0
        val_pct = 0.0
        test_pct = 0.0

    legend_elements = [
        patches.Patch(color=colors[1], label=f"Training ({train_pct:.1f}%)"),
        patches.Patch(color=colors[2], label=f"Validation ({val_pct:.1f}%)"),
        patches.Patch(color=colors[3], label=f"Test ({test_pct:.1f}%)"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
        fancybox=True,
        shadow=False,
    )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Timeline heatmap saved to: {save_path}")

    return fig, ax


def print_split_statistics(train_dates, val_dates, test_dates, start_year=None, end_year=None):
    """Print statistics about the temporal splits."""
    
    # Filter dates if range provided
    if start_year is not None and end_year is not None:
        def filter_dates(dates):
            res = []
            for d in dates:
                y = int(d.split('-')[0])
                if start_year <= y <= end_year:
                    res.append(d)
            return res
            
        train_dates = filter_dates(train_dates)
        val_dates = filter_dates(val_dates)
        test_dates = filter_dates(test_dates)

    print("=" * 60)
    print("TEMPORAL SPLIT STATISTICS")
    if start_year and end_year:
        print(f"Range: {start_year} - {end_year}")
    print("=" * 60)

    print(f"Training months: {len(train_dates)}")
    print(f"Validation months: {len(val_dates)}")
    print(f"Test months: {len(test_dates)}")
    print(
        f"Total months with data: {len(train_dates) + len(val_dates) + len(test_dates)}"
    )

    # Year coverage
    if len(train_dates) + len(val_dates) + len(test_dates) > 0:
        all_dates = list(train_dates) + list(val_dates) + list(test_dates)
        years_covered = sorted(set([int(date.split("-")[0]) for date in all_dates]))
        print(
            f"Years covered: {years_covered[0]}-{years_covered[-1]} ({len(years_covered)} years)"
        )
    else:
        print("No data in selected range.")
        years_covered = []

    # Monthly distribution
    print("\nMonthly distribution:")
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    for split_name, dates in [
        ("Train", train_dates),
        ("Val", val_dates),
        ("Test", test_dates),
    ]:
        month_counts = {}
        for date in dates:
            month = int(date.split("-")[1])
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
    for split_name, dates in [
        ("Train", train_dates),
        ("Val", val_dates),
        ("Test", test_dates),
    ]:
        year_counts = {}
        for date in dates:
            year = int(date.split("-")[0])
            year_counts[year] = year_counts.get(year, 0) + 1

        print(f"  {split_name}:")
        for year in years_covered:
            count = year_counts.get(year, 0)
            if count > 0:
                print(f"    {year}: {count:2d}")


def main():
    """Main function to create the temporal split visualization."""
    parser = argparse.ArgumentParser(description="Visualize temporal splits.")
    parser.add_argument(
        "--start_year", type=int, default=2014, help="Start year for visualization"
    )
    parser.add_argument(
        "--end_year", type=int, default=2024, help="End year for visualization"
    )
    args = parser.parse_args()

    print("Loading temporal split data...")

    # Load the date splits
    train_dates, val_dates, test_dates = load_date_splits()

    # Print statistics
    print_split_statistics(
        train_dates, val_dates, test_dates, start_year=args.start_year, end_year=args.end_year
    )

    # Create the timeline heatmap
    print(
        f"\nCreating timeline heatmap from {args.start_year} to {args.end_year}..."
    )
    save_path = Path(__file__).parent / "temporal_splits_heatmap.png"
    fig, ax = create_timeline_heatmap(
        train_dates,
        val_dates,
        test_dates,
        save_path,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    print("\nVisualization complete!")
    print(f"Heatmap saved as: {save_path}")


if __name__ == "__main__":
    main()
