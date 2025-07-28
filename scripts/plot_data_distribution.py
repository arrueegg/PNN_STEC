import os
import h5py
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

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


def scan_database(data_root, start_year=2010, end_year=2025):
    """
    Scan the database and return a list of dates and observation counts.
    Assumes structure: data_root/year/doy/*.h5
    """
    dates, counts = [], []

    for year in range(start_year, end_year + 1):
        year_path = os.path.join(data_root, str(year))
        if not os.path.exists(year_path):
            continue

        for doy in sorted(os.listdir(year_path)):
            doy_path = os.path.join(year_path, doy)
            if not os.path.isdir(doy_path):
                continue

            for file in os.listdir(doy_path):
                if file.endswith(".h5"):
                    file_path = os.path.join(doy_path, file)
                    try:
                        doy_int = int(doy)
                        date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy_int - 1)
                    except ValueError:
                        continue

                    obs_count = count_observations_in_h5(file_path, year, doy, 'all_data')
                    dates.append(date)
                    counts.append(obs_count)
                    break  # Only one file per day
    return dates, counts


def plot_observations(dates, counts, title="Observations per Day (2010–2025)"):
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
    # Group counts by year
    yearly_counts = defaultdict(int)
    for date, count in zip(dates, counts):
        yearly_counts[date.year] += count

    years = sorted(yearly_counts.keys())
    total_counts = [yearly_counts[year] for year in years]

    plt.figure(figsize=(12, 6))
    plt.bar(years, total_counts, width=0.8, align='center')
    plt.xlabel("Year")
    plt.ylabel("Total Number of Observations")
    plt.title(title)
    plt.xticks(years, rotation=45)
    plt.tight_layout()
    plt.savefig("plots/yearly_observations.png", dpi=300)

    ##################################
    # Group counts by month
    monthly_counts = defaultdict(int)
    for date, count in zip(dates, counts):
        month_key = date.strftime("%b %Y")
        monthly_counts[month_key] += count

    months = sorted(monthly_counts.keys(), key=lambda x: datetime.datetime.strptime(x, "%b %Y"))
    total_monthly_counts = [monthly_counts[month] for month in months]
    plt.figure(figsize=(36, 6))
    plt.bar(months, total_monthly_counts, width=0.8, align='center')
    plt.xlabel("Month")
    plt.ylabel("Total Number of Observations")
    plt.title(title)
    plt.xticks(months, rotation=90)
    plt.tight_layout()
    plt.savefig("plots/monthly_observations.png", dpi=300)

    ##################################
    # Daily observations heatmap
    heatmap_data = np.full((len(set([d.year for d in dates])), 366), np.nan)
    years = sorted(set([d.year for d in dates]))

    year_to_idx = {year: i for i, year in enumerate(years)}
    for date, count in zip(dates, counts):
        doy = date.timetuple().tm_yday - 1  # 0-based DOY
        heatmap_data[year_to_idx[date.year], doy] = count

    plt.figure(figsize=(15, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu", cbar=True, xticklabels=30, yticklabels=years)
    plt.xlabel("Day of Year")
    plt.ylabel("Year")
    plt.title("Daily Observation Counts")
    plt.tight_layout()
    plt.savefig("plots/daily_observations_heatmap.png", dpi=300)


    ###################################
    # Total observations
    total_observations = sum(counts)
    print(f"Total number of observations: {total_observations:,}")
    if total_observations >= 1_000_000_000:
        print(f"Total number of observations: {total_observations / 1_000_000_000:.2f} billion")
    elif total_observations >= 1_000_000:
        print(f"Total number of observations: {total_observations / 1_000_000:.2f} million")    

def main():
    DATA_ROOT = "/home/space/data/IONO/STEC_DB_CASDCB/"
    dates, counts = scan_database(DATA_ROOT)
    plot_observations(dates, counts)


if __name__ == "__main__":
    main()
