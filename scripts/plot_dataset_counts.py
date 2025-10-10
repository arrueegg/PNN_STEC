import os
import h5py
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import matplotlib.dates as mdates
import numpy as np

# Set global matplotlib parameters for scientific plots
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
    }
)


def count_observations_in_h5(file_path, year, doy, dataset_name):
    """
    Count total observations in an HDF5 file.
    Assumes structure: /year/doy/all_data
    """
    try:
        with h5py.File(file_path, "r") as f:
            if dataset_name in f[f"{year}/{doy}"]:
                return len(f[f"{year}/{doy}/{dataset_name}"])
        return 0
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0


def scan_database(data_root, start_year=2010, end_year=2024):
    """
    Scan the database and return a list of dates and observation counts.
    Assumes structure: data_root/year/doy/*.h5
    """
    dates, counts = [], []
    missing_files = []
    zero_observations = []

    for year in range(start_year, end_year + 1):
        year_path = os.path.join(data_root, str(year))
        if not os.path.exists(year_path):
            # Add all days for this year as missing
            days_in_year = (
                366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
            )
            for doy in range(1, days_in_year + 1):
                date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1)
                missing_files.append(date)
            continue

        # Get all DOY directories that exist
        existing_doys = set()
        for doy in tqdm(sorted(os.listdir(year_path)), desc=f"Scanning {year}"):
            doy_path = os.path.join(year_path, doy)
            if not os.path.isdir(doy_path):
                continue

            try:
                doy_int = int(doy)
                existing_doys.add(doy_int)
            except ValueError:
                continue

            file_found = False
            for file in os.listdir(doy_path):
                if file.endswith(".h5"):
                    file_path = os.path.join(doy_path, file)
                    date = datetime.datetime(year, 1, 1) + datetime.timedelta(
                        days=doy_int - 1
                    )

                    obs_count = count_observations_in_h5(
                        file_path, year, doy, "all_data"
                    )
                    dates.append(date)
                    counts.append(obs_count)

                    # Track zero observations
                    if obs_count == 0:
                        zero_observations.append(date)

                    file_found = True
                    break  # Only one file per day

            # If directory exists but no .h5 file found
            if not file_found:
                date = datetime.datetime(year, 1, 1) + datetime.timedelta(
                    days=doy_int - 1
                )
                missing_files.append(date)

        # Check for missing DOY directories
        days_in_year = (
            366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
        )
        for doy in range(1, days_in_year + 1):
            if doy not in existing_doys:
                date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy - 1)
                missing_files.append(date)

    # Print missing files and zero observations
    if missing_files:
        print(f"\n{'='*50}")
        print(f"MISSING FILES: {len(missing_files)} days")
        print(f"{'='*50}")
        for date in sorted(missing_files):
            print(
                f"Missing: {date.strftime('%Y-%m-%d')} (DOY {date.timetuple().tm_yday})"
            )

    if zero_observations:
        print(f"\n{'='*50}")
        print(f"ZERO OBSERVATIONS: {len(zero_observations)} days")
        print(f"{'='*50}")
        for date in sorted(zero_observations):
            print(
                f"Zero obs: {date.strftime('%Y-%m-%d')} (DOY {date.timetuple().tm_yday})"
            )

    print(f"\n{'='*50}")
    print("SCAN SUMMARY")
    print(f"{'='*50}")
    print(f"Days with data: {len(dates)}")
    print(f"Days missing files: {len(missing_files)}")
    print(f"Days with zero observations: {len(zero_observations)}")
    total_expected = sum(
        366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
        for year in range(start_year, end_year + 1)
    )
    print(f"Total expected days ({start_year}-{end_year}): {total_expected}")
    print(f"Data completeness: {len(dates)/total_expected*100:.1f}%")

    return dates, counts


def format_count_labels(counts):
    """Format count labels with appropriate units (K, M, B)"""
    labels = []
    for count in counts:
        if count >= 1_000_000_000:
            labels.append(f"{count/1_000_000_000:.1f}B")
        elif count >= 1_000_000:
            labels.append(f"{count/1_000_000:.1f}M")
        elif count >= 1_000:
            labels.append(f"{count/1_000:.0f}K")
        else:
            labels.append(f"{count}")
    return labels


def plot_observations(dates, counts):
    """
    Plot the number of observations per day.
    """
    if not dates:
        print("No data to plot.")
        return

    folder = "plots"
    os.makedirs(folder, exist_ok=True)

    dates, counts = zip(*sorted(zip(dates, counts)))

    ##################################
    # Yearly observations - Scientific color scheme
    yearly_counts = defaultdict(int)
    for date, count in zip(dates, counts):
        yearly_counts[date.year] += count

    years = sorted(yearly_counts.keys())
    total_counts = [yearly_counts[year] for year in years]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(
        years,
        total_counts,
        width=0.8,
        align="center",
        color="steelblue",
        alpha=0.8,
        edgecolor="navy",
        linewidth=0.8,
    )

    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel("Total STEC Observations", fontweight="bold")
    ax.set_title(
        "Annual STEC Database Coverage\n(2010-2024)", fontweight="bold", pad=20
    )

    # Format y-axis with scientific notation
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(12)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    # Add value labels on bars for key years
    max_val = max(total_counts)
    for i, (year, count) in enumerate(zip(years, total_counts)):
        ax.text(
            year,
            count + max_val * 0.01,
            format_count_labels([count])[0],
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.xticks(years[::2], rotation=45)  # Show every other year
    plt.tight_layout()
    plt.savefig("plots/yearly_observations.png", dpi=300, bbox_inches="tight")
    plt.close()

    ##################################
    # Monthly observations with better time series visualization
    monthly_counts = defaultdict(int)
    monthly_dates = []
    for date, count in zip(dates, counts):
        month_date = datetime.date(date.year, date.month, 1)
        monthly_counts[month_date] += count
        if month_date not in monthly_dates:
            monthly_dates.append(month_date)

    monthly_dates = sorted(set(monthly_dates))
    total_monthly_counts = [monthly_counts[month] for month in monthly_dates]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(
        monthly_dates,
        total_monthly_counts,
        linewidth=2,
        color="darkgreen",
        marker="o",
        markersize=4,
        alpha=0.8,
    )

    ax.set_xlabel("Time Period", fontweight="bold")
    ax.set_ylabel("Monthly STEC Observations", fontweight="bold")
    ax.set_title(
        "STEC Database Monthly Coverage Time Series\n(2010-2024)",
        fontweight="bold",
        pad=20,
    )

    # Format x-axis with better date formatting
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator([1, 7]))  # Jan and July
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Format y-axis
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(12)

    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/monthly_observations.png", dpi=300, bbox_inches="tight")
    plt.close()

    ##################################
    # Daily observations heatmap with better color scheme
    heatmap_data = np.full((len(set([d.year for d in dates])), 366), np.nan)
    years = sorted(set([d.year for d in dates]))

    year_to_idx = {year: i for i, year in enumerate(years)}
    for date, count in zip(dates, counts):
        doy = date.timetuple().tm_yday - 1  # 0-based DOY
        heatmap_data[year_to_idx[date.year], doy] = count

    fig, ax = plt.subplots(figsize=(12, 8))

    # Use a better colormap for scientific data
    cmap = plt.cm.viridis
    cmap.set_bad("lightgray", 1.0)  # Set color for missing data

    im = ax.imshow(heatmap_data, cmap=cmap, aspect="auto", interpolation="nearest")

    # Add colorbar with proper formatting
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Daily STEC Observations", fontweight="bold", fontsize=14)
    cbar.ax.ticklabel_format(style="scientific", scilimits=(0, 0))

    # Set axis labels and ticks
    ax.set_xlabel("Day of Year", fontweight="bold")
    ax.set_ylabel("Year", fontweight="bold")
    ax.set_title("STEC Dataset Daily Coverage", fontweight="bold", pad=10)

    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
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

    # Better tick formatting
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_starts, fontsize=11)
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years)

    # Add month labels on top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(month_starts)
    ax2.set_xticklabels(month_names, fontsize=11)
    ax2.set_xlabel("Month", fontweight="bold")

    plt.tight_layout()
    plt.savefig("plots/daily_observations_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    ###################################
    # Statistics summary with better formatting
    total_observations = sum(counts)
    mean_daily = np.mean(counts)
    std_daily = np.std(counts)

    print("\n" + "=" * 60)
    print("STEC DATABASE STATISTICS SUMMARY")
    print("=" * 60)
    print(f"Data Period: {min(dates).year} - {max(dates).year}")
    print(f"Total Days: {len(dates):,}")
    print(f"Total Observations: {total_observations:,}")

    if total_observations >= 1_000_000_000:
        print(f"Total Observations: {total_observations / 1_000_000_000:.2f} billion")
    elif total_observations >= 1_000_000:
        print(f"Total Observations: {total_observations / 1_000_000:.1f} million")

    print(f"Mean Daily Observations: {mean_daily:,.0f} ± {std_daily:,.0f}")
    print(f"Min Daily Observations: {min(counts):,}")
    print(f"Max Daily Observations: {max(counts):,}")
    print("=" * 60)


def main():
    DATA_ROOT = "/home/space/data/IONO/STEC_DB_CASDCB/"
    dates, counts = scan_database(DATA_ROOT)
    plot_observations(dates, counts)


if __name__ == "__main__":
    main()
