import subprocess
import os
import pandas as pd
import numpy as np
import h5py


class OmniDownloader:
    def __init__(self, base_folder, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.base_folder = base_folder
        self.txt_file = os.path.join(
            base_folder, f"omni_hourly_{start_date[:4]}-{end_date[:4]}.txt"
        )
        self.h5_file = os.path.join(
            base_folder, f"omni_hourly_{start_date[:4]}-{end_date[:4]}.h5"
        )
        self.base_url = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
        self.vars_list = np.arange(1, 57).tolist()
        self.post_data = self._build_post_data()

    def _build_post_data(self):
        return (
            "activity=retrieve"
            f"&res=hour"
            f"&spacecraft=omni2"
            f"&start_date={self.start_date}"
            f"&end_date={self.end_date}"
            + "".join([f"&vars={v}" for v in self.vars_list])
            + "&scale=Linear"
            "&view=0"
            "&table=0"
        )

    def download_txt(self):
        """Download the data file if not already present."""
        if not os.path.exists(self.txt_file):
            print(f"Downloading data to {self.txt_file}...")
            cmd = [
                "wget",
                "--post-data",
                self.post_data,
                self.base_url,
                "-O",
                self.txt_file,
            ]
            subprocess.run(cmd, check=True)
        else:
            print(f"File {self.txt_file} already exists. Skipping download.")

    def _parse_headers(self):
        """Parse column names and descriptions from the TXT file."""
        with open(self.txt_file, "r") as f:
            lines = f.readlines()

        header_line_idx = None
        header_line = None
        for i, line in enumerate(lines):
            if line.startswith("YEAR DOY HR"):
                header_line_idx = i
                header_line = line.strip()
                break

        if header_line_idx is None:
            raise RuntimeError("Could not find header line with column names.")

        col_names = header_line.split()[:3]  # ['YEAR', 'DOY', 'HR']
        col_descriptions = {}

        # Parse "Selected parameters" section
        in_params_section = False
        for line in lines:
            line = line.strip()
            if "Selected parameters:" in line:
                in_params_section = True
                continue
            if in_params_section:
                if line == "" or line.startswith("YEAR DOY HR"):
                    break
                parts = line.split()
                try:
                    idx = int(parts[0])
                    raw_name = (
                        "_".join(parts[1:])
                        .replace("(", "")
                        .replace(")", "")
                        .replace(".", "")
                    )
                    name = raw_name
                    # Deduplicate name if it already exists
                    count = 1
                    while name in col_names:
                        count += 1
                        name = f"{raw_name}_{count}"
                    col_descriptions[idx] = name
                    col_names.append(name)
                except ValueError:
                    continue

        return header_line_idx, col_names

    def load_data(self):
        """Load and clean data from the TXT file."""
        header_line_idx, col_names = self._parse_headers()

        df = pd.read_csv(
            self.txt_file,
            sep=r"\s+",
            skiprows=header_line_idx + 1,
            names=col_names,
            comment="<",  # skip HTML-like tags
            dtype=str,  # read as string first
            skipfooter=15,
            engine="python",
        )

        # Convert all columns to numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        # Remove unwanted columns
        columns_to_remove = df.filter(
            regex="_2|Hour|sigma|ID|RMS|#|Proton|Alpha|flow|Flux|Mach|Quasy|Lat\. Angle|Long\. Angle|BX|BY|Temperature|Beta"
        ).columns
        df = df.drop(columns=columns_to_remove)

        # Replace flag values with NaN
        flag_values = np.array(
            [
                9.9999900e-01,
                9.9990000e00,
                9.9999000e00,
                9.0000000e01,
                9.9900000e01,
                9.9990000e01,
                9.9900000e02,
                9.9990000e02,
                9.9999000e02,
                9.9990000e03,
                9.9999000e04,
                9.9999990e04,
                9.9999999e05,
                9.9999990e06,
            ]
        )
        for col in [c for c in df.columns if c not in ["YEAR", "DOY"]]:
            df[col] = df[col].mask(np.isin(df[col], flag_values))

        # Forward fill NaN values
        df = df.fillna(method="ffill")

        # Print min/max
        for col in df.columns:
            if col not in ["YEAR", "DOY", "HR"]:
                print(f"{col}: min={df[col].min()}, max={df[col].max()}")

        return df

    def save_hdf5(self, df):
        """Split data into daily datasets and save as HDF5."""
        with h5py.File(self.h5_file, "w") as h5f:
            for (year, doy), day_data in df.groupby(["YEAR", "DOY"]):
                grp = h5f.require_group(str(year))
                dset = grp.create_dataset(
                    str(int(doy)).zfill(3), data=day_data.to_numpy()
                )
                dset.attrs["columns"] = np.array(day_data.columns, dtype="S")

        print(f"\n✅ Finished splitting into daily HDF5: {self.h5_file}")

    def run(self):
        """Run the entire pipeline."""
        self.download_txt()
        df = self.load_data()
        self.save_hdf5(df)


# === Example usage ===
if __name__ == "__main__":
    downloader = OmniDownloader(
        base_folder="/home/space/data/IONO/SWI/",
        start_date="20100101",
        end_date="20250625",
    )
    downloader.run()
