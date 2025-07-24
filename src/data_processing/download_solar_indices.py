import subprocess
import os
import pandas as pd
import numpy as np
import h5py


# === Configuration ===
start_date = "20100101"
end_date = "20250625"

txt_file = os.path.join(f"omni_hourly_{start_date[:4]}-{end_date[:4]}.txt")
h5_file = os.path.join(f"omni_hourly_{start_date[:4]}-{end_date[:4]}.h5")
base_url = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

# Download parameters
vars_list = np.arange(1, 57).tolist()  # all vars

post_data = (
    "activity=retrieve"
    f"&res=hour"
    f"&spacecraft=omni2"
    f"&start_date={start_date}"
    f"&end_date={end_date}"
    + ''.join([f"&vars={v}" for v in vars_list]) +
    "&scale=Linear"
    "&view=0"
    "&table=0"
)

# === STEP 1: Download TXT if not exists ===
if not os.path.exists(txt_file):
    print(f"Downloading data to {txt_file}...")
    cmd = ["wget", "--post-data", post_data, base_url, "-O", txt_file]
    subprocess.run(cmd, check=True)
else:
    print(f"File {txt_file} already exists. Skipping download.")

# === STEP 2: Detect header line for column names and descriptions ===
header_line_idx = None
header_line = None
col_descriptions = {}

with open(txt_file, "r") as f:
    lines = f.readlines()

# Find the line with "YEAR DOY HR"
for i, line in enumerate(lines):
    if line.startswith("YEAR DOY HR"):
        header_line_idx = i
        header_line = line.strip()
        break

if header_line_idx is None:
    raise RuntimeError("Could not find header line with column names.")

# Start with fixed columns
col_names = header_line.split()[:3]  # ['YEAR', 'DOY', 'HR']

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
            raw_name = "_".join(parts[1:]).replace("(", "").replace(")", "").replace(".", "")
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

# === STEP 3: Load data into DataFrame with cleaning ===
df = pd.read_csv(
    txt_file,
    sep=r"\s+",
    skiprows=header_line_idx + 1,
    names=col_names,
    comment='<',           # skip HTML-like tags
    dtype=str,             # read all as string first to avoid errors
    skipfooter=15,          # skip the last line of the file
    engine='python'        # use Python engine for regex sep
)

# Convert all columns to numeric values where possible
df = df.apply(pd.to_numeric, errors='coerce')

# Remove columns that contain 'sigma', 'ID', or 'RMS' in their names
columns_to_remove = df.filter(regex='_2|Hour|sigma|ID|RMS|#|Proton|Alpha|flow|Flux|Mach|Quasy|Lat\. Angle|Long\. Angle|BX|BY|Temperature|Beta').columns
df = df.drop(columns=columns_to_remove)

cols = df.columns.tolist()

flag_values = np.array([9.9999900e-01, 9.9990000e+00, 9.9999000e+00, 9.0000000e+01,
    9.9900000e+01, 9.9990000e+01, 4.0000000e+02, 9.9900000e+02, 9.9990000e+02,
    9.9999000e+02, 9.9990000e+03, 9.9999000e+04, 9.9999990e+04, 9.9999999e+05,
    9.9999990e+06
])

# Use vectorized replacement
for col in cols:
    df[col] = df[col].mask(np.isin(df[col], flag_values))

# Fill NaN values with the last known value
df = df.fillna(method='ffill')

# Calculate and print min and max values for each column
for col in cols:
    if col not in ['YEAR', 'DOY', 'HR']:  # Skip non-numeric columns
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"{col}: min={min_val}, max={max_val}")

# === STEP 4: Split into daily datasets and save HDF5 ===
with h5py.File(h5_file, "w") as h5f:
    for (year, doy), day_data in df.groupby(['YEAR', 'DOY']):
        grp = h5f.require_group(str(year))
        dset = grp.create_dataset(str(int(doy)).zfill(3), data=day_data.to_numpy())
        # Attach column names as dataset attribute
        dset.attrs['columns'] = np.array(day_data.columns, dtype='S')

print(f"\n✅ Finished splitting into daily HDF5: {h5_file}")

