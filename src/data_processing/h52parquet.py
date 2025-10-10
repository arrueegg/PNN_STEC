import pandas as pd
import h5py
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_year_doy_from_path(path):
    """Extract year and doy from folder structure like /.../2014/001/..."""
    parts = Path(path).parts
    for i in range(len(parts) - 1):
        if parts[i].isdigit() and len(parts[i]) == 4 and parts[i + 1].isdigit():
            return parts[i], parts[i + 1]
    return "unknown", "000"


def h5_to_parquet_per_station(h5_file, parquet_root):
    """
    Converts a single HDF5 dataset (with structured fields incl. 'station')
    into separate Parquet files for each station.

    Expected HDF5 structure:
      /<year>/<doy>/all_data -> Dataset with compound dtype (structured array)
    """

    with h5py.File(h5_file, "r") as h5f:
        # Navigate down to the all_data dataset
        year_group = next((k for k in h5f if k.isdigit()), None)
        doy_group = next((k for k in h5f[year_group] if k.isdigit()), None)
        dataset = h5f[year_group][doy_group]["all_data"]

        # Convert structured array to DataFrame
        df = pd.DataFrame.from_records(dataset[:])

    stations = df["station"].unique()
    print(f"Processing {h5_file} with {len(stations)} stations...")

    for station in tqdm(stations):
        try:
            station_df = df[df["station"] == station]
        except Exception as e:
            print(f"  ⚠️ Skipping station {station}: {e}")
            continue

        station = station.decode("utf-8") if isinstance(station, bytes) else station
        output_dir = Path(parquet_root) / year_group / doy_group / station
        output_dir.mkdir(parents=True, exist_ok=True)

        parquet_file = output_dir / f"{year_group}{doy_group}{station}.parquet"
        station_df.to_parquet(parquet_file, index=False)


def find_h5_files(base_dir):
    """Recursively find all *all.h5 files"""
    return [str(p) for p in Path(base_dir).rglob("*all.h5")]


def main():
    parser = argparse.ArgumentParser(
        description="Convert nested HDF5 GNSS STEC files to station-split Parquet."
    )
    parser.add_argument(
        "--h5_path",
        default="/scratch2/arrueegg/WP4/PNN_STEC/test_data",
        type=str,
        help="Path to the directory containing H5 files.",
    )
    parser.add_argument(
        "--parquet_path",
        default="/scratch2/arrueegg/WP4/PNN_STEC/parquet_data",
        type=str,
        help="Path to the directory for output Parquet files.",
    )
    args = parser.parse_args()

    h5_files = find_h5_files(args.h5_path)
    print(f"Found {len(h5_files)} HDF5 files.")

    for h5_file in h5_files:
        h5_to_parquet_per_station(h5_file, args.parquet_path)


if __name__ == "__main__":
    main()
