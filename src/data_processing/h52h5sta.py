import pandas as pd
import h5py
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def h5_to_h5_per_station(input_h5, output_root):
    """
    Splits a single HDF5 file containing a compound dataset '/<year>/<doy>/all_data'
    into separate HDF5 files per station, preserving year and doy in the path.

    Each output file is at:
      output_root/<year>/<doy>/<station>/<year><doy>_<station>.h5
    and contains a group named '<station>' with a dataset 'all_data'.
    """
    # Derive year and doy from input path (expects ..._YYYYDDD_...)
    fname = Path(input_h5).stem
    parts = fname.split("_")
    if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) == 7:
        year, doy = parts[1][:4], parts[1][4:]
    else:
        # fallback: scan path parts
        segs = Path(input_h5).parts
        year = next((p for p in segs if p.isdigit() and len(p) == 4), "unknown")
        doy = next((p for p in segs if p.isdigit() and len(p) == 3), "000")

    # Read input data
    with h5py.File(input_h5, "r") as h5f:
        try:
            ds = h5f[year][doy]["all_data"]
        except KeyError:
            print(
                f"⚠️  Skipping {input_h5}: expected path /{year}/{doy}/all_data not found."
            )
            return
        recs = ds[:]  # structured numpy array

    # Convert to DataFrame for easy filtering
    df = pd.DataFrame.from_records(recs)
    stations = df["station"].unique()
    print(f"Processing {input_h5}: found {len(stations)} stations...")

    # Prepare dtypes for HDF5: convert object/string columns to variable-length UTF-8
    col_dtypes = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype is object or dtype.kind in ("U", "S"):
            # variable-length UTF-8
            h5_dtype = h5py.string_dtype(encoding="utf-8")
        else:
            h5_dtype = dtype
        col_dtypes.append((col, h5_dtype))

    for station in tqdm(stations, desc=f"Stations in {Path(input_h5).name}"):
        # Filter rows
        station_df = df[df["station"] == station]
        station_key = (
            station.decode("utf-8") if isinstance(station, bytes) else str(station)
        )

        # Build structured array with correct dtypes
        n = len(station_df)
        struct_arr = np.empty(n, dtype=col_dtypes)
        for col in station_df.columns:
            struct_arr[col] = station_df[col].values

        # Prepare output path
        out_dir = Path(output_root) / year / doy / station_key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{year}{doy}_{station_key}.h5"

        # Write station-specific HDF5
        with h5py.File(out_file, "w") as out_f:
            grp = out_f.create_group(station_key)
            grp.create_dataset("all_data", data=struct_arr, compression="gzip")


def find_h5_files(base_dir):
    """Recursively find all files ending with 'all.h5'"""
    return [str(p) for p in Path(base_dir).rglob("*.h5")]


def main():
    parser = argparse.ArgumentParser(
        description="Split HDF5 GNSS STEC files into per-station HDF5 archives."
    )
    parser.add_argument(
        "--h5_path",
        default="/scratch2/arrueegg/WP4/PNN_STEC/test_data",
        type=str,
        help="Directory containing source HDF5 files.",
    )
    parser.add_argument(
        "--output_path",
        default="/scratch2/arrueegg/WP4/PNN_STEC/h5_data",
        type=str,
        help="Root directory for output station HDF5 files.",
    )
    args = parser.parse_args()

    h5_files = find_h5_files(args.h5_path)
    print(f"Found {len(h5_files)} HDF5 files under {args.h5_path}")

    for h5f in h5_files:
        h5_to_h5_per_station(h5f, args.output_path)


if __name__ == "__main__":
    main()
