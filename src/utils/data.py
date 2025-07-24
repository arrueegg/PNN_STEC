import os
import h5py
import pandas as pd
import torch
import numpy as np
import spacepy.toolbox
from spacepy.coordinates import Coords
from spacepy.time import Ticktock
from torch.utils.data import RandomSampler, Dataset, DataLoader, random_split
from utils.locationencoder.pe import SphericalHarmonics
import tables

import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

torch.multiprocessing.set_start_method('fork', force=True)
#spacepy.toolbox.update(leapsecs=True)


class PyTablesDatasetSplit(Dataset):
    def __init__(self, config, h5_file_path, split):
        self.config = config
        self.h5_file_path = h5_file_path
        self.doy = self.h5_file_path.split('/')[-1].split('_')[1][4:]
        self.year = self.h5_file_path.split('/')[-1].split('_')[1][:4]
        self.split = split
        self.SWI_data = self.load_SWI()
        self.SWI_cols = self.SWI_data.columns if self.SWI_data is not None else []
        self.file = tables.open_file(h5_file_path, mode='r')
        self.data = self.file.get_node(f'/{self.year}/{self.doy}/{self.split}_data')
        self.length = 100_000  # len(self.data)
    
    def load_SWI(self):
        if self.config['data']['use_SWI']:
            filename = os.path.join(self.config['data']['SWI_data_path'], f'omni_hourly_2010-2025.h5')
            if not os.path.exists(filename):
                raise FileNotFoundError(f"SWI data file not found: {filename}")
            with h5py.File(filename, "r") as f:
                data = f[str(self.year)][str(self.doy).zfill(3)][:]
                columns = [c.decode() for c in f[str(self.year)][str(self.doy).zfill(3)].attrs['columns']]
            return pd.DataFrame(data, columns=columns)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.data[idx]

        # Encode strings and combine with numeric features
        features = torch.tensor([
            row['sm_lat_sta'],
            row['sm_lon_sta'],
            row['satazi'],
            row['satele'],
            row['sod'],
            row['lat_ipp_450'],
            row['lon_ipp_450'],],
            dtype=torch.float32)
        
        if self.config['data']['use_SWI']:
            # Add SWI features if available
            swi_idx = (row['sod'] / 3600).astype(int)  # Convert seconds to hours
            swi_row = self.SWI_data.iloc[swi_idx]
            swi_features = torch.tensor(swi_row[self.SWI_cols].values, dtype=torch.float32)
            features = torch.cat((features, swi_features), dim=0)
        
        # Return features and label separately
        label = torch.tensor(row['stec'], dtype=torch.float32)

        # Check for NaN or similar in features or label
        if torch.isnan(features).any() or torch.isnan(label):
            raise ValueError(f"NaN detected in features or label at index {idx}")

        return features, label

    def __del__(self):
        self.file.close()  # Ensure the file is closed properly

class CollateWithSH:
    def __init__(self, config):
        # SH degree and flag
        self.sh_degree  = config["data"].get("SH_degree", 0) or 0
        self.sh_enabled = self.sh_degree > 0
        if self.sh_enabled:
            self.sh_encoder = SphericalHarmonics(legendre_polys=self.sh_degree)

        self.num_core_features = 7

    def transform(self, features):
        """
        Normalize and transform the batch of features (one IPP point).
        """
        # raw feature indices
        LAT_STA, LON_STA, AZI, ELE, SOD, IPP_LAT, IPP_LON = range(self.num_core_features)

        # --- time of day ---
        t      = features[:, SOD] / 86400 * (2 * torch.pi)
        sin_t  = torch.sin(t).unsqueeze(1)
        cos_t  = torch.cos(t).unsqueeze(1)
        norm_t = (2 * features[:, SOD] / 86400 - 1).unsqueeze(1)

        # --- azimuth / elevation ---
        a      = features[:, AZI] / 180 * torch.pi
        sin_a  = torch.sin(a).unsqueeze(1)
        cos_a  = torch.cos(a).unsqueeze(1)
        norm_e = (2 * features[:, ELE] / 90 - 1).unsqueeze(1)

        # --- station coords normalized to [-1,1] ---
        sm_lat_norm = ((features[:, LAT_STA] + 90) / 180 * 2 - 1).unsqueeze(1)
        sm_lon_norm = ((features[:, LON_STA] + 180) / 360 * 2 - 1).unsqueeze(1)

        # --- single IPP point normalized to [-1,1] ---
        ipp_lat_norm = ((features[:, IPP_LAT] + 90) / 180 * 2 - 1).unsqueeze(1)
        ipp_lon_norm = ((features[:, IPP_LON] + 180) / 360 * 2 - 1).unsqueeze(1)

        # concatenate core normalized features
        x_out = torch.cat([
            sm_lat_norm, sm_lon_norm,
            sin_a, cos_a, norm_e,
            sin_t, cos_t, norm_t,
            ipp_lat_norm, ipp_lon_norm
        ], dim=1)

        return x_out
    
    def norm_SWI(self, swi_features):
        """
        Normalize SWI features to [0, 1] range for each column by specified min/max values.
        """
        swi_features = swi_features.float()
        # Define min and max values for normalization
        min_max_values = {
            "Bartels_rotation_number": (2407, 3000),
            "Scalar_B,_nT": (0, 70),
            "Vector_B_Magnitude,nT": (0.0, 70),
            "Lat_Angle_of_B_GSE": (-90, 90),
            "Long_Angle_of_B_GSE": (0.0, 360.0),
            "BZ,_nT_GSE": (-50, 35),
            "BZ,_nT_GSM": (-50, 35),
            "SW_Plasma_Speed,_km/s": (240.0, 1100.0),
            "Flow_pressure": (0, 60),
            "E_elecrtric_field": (-20, 30),
            "Alfen_mach_number": (0, 120),
            "Kp_index": (0.0, 100.0),
            "R_Sunspot_No": (0.0, 300.0),
            "Dst-index,_nT": (-450, 100),
            "AE-index,_nT": (0.0, 2500.0),
            "ap_index,_nT": (0.0, 300.0),
            "f107_index": (62, 420),
            "pc-index": (-6, 16),
            "AL-index,_nT": (-2000.0, 20.0),
            "AU-index,_nT": (-200.0, 1200.0),
            "Magnetosonic_Much_num": (0, 15),
            "Lyman_alpha": (0, 0.015),
        }

        # Normalize each column using min-max scaling
        normalized_features = []
        for i in range(swi_features.shape[1]):
            column_name = list(min_max_values.keys())[i]
            min_val, max_val = min_max_values[column_name]
            normalized_column = (swi_features[:, i] - min_val) / (max_val - min_val)
            normalized_features.append(normalized_column.unsqueeze(1))

        # Concatenate normalized columns
        return torch.cat(normalized_features, dim=1)

    def SH_transform(self, raw, norm):
        """
        Append spherical-harmonic embeddings if enabled.
        `raw` is original features; `norm` is output from transform().
        """
        if not self.sh_enabled:
            return norm

        # SH embeddings for station (LON_STA, LAT_STA = cols 1,0)
        sta_lonlat = torch.stack([raw[:, 1], raw[:, 0]], dim=1)
        sh_sta = self.sh_encoder(sta_lonlat)

        # SH embeddings for IPP (IPP_LON, IPP_LAT = cols 6,5)
        ipp_lonlat = torch.stack([raw[:, 6], raw[:, 5]], dim=1)
        sh_ipp = self.sh_encoder(ipp_lonlat)

        # combine normalized + SH features
        return torch.cat([norm, sh_sta, sh_ipp], dim=1)
    
    def __call__(self, batch):
        """
        Process and collate a batch of (raw_features, labels).
        """
        feats, labels = zip(*batch)
        raw = torch.stack(feats, dim=0)
        y   = torch.stack(labels, dim=0)

        # Split core features and SWI features
        core = raw[:, :self.num_core_features]
        swi  = raw[:, self.num_core_features:]

        # Normalize features
        norm_core  = self.transform(core)
        x_out = self.SH_transform(core, norm_core)
        norm_swi = self.norm_SWI(swi) if swi.shape[1] > 0 else None

        # Append SWI features (raw or normalized if needed)
        if norm_swi:
            x_out = torch.cat([x_out, norm_swi], dim=1)

        return x_out, y
    
def generate_dates(months):
        dates = []
        for month in months:
            year, month = map(int, month.split('-'))
            start_date = datetime(year, month, 1)
            end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            current_date = start_date
            while current_date <= end_date:
                dates.append(current_date)
                current_date += timedelta(days=1)
        return dates

def get_split_file_lists(config, year, doy):
    gnss_path = config['data']['GNSS_data_path']   
    sampling = config['data']['sampling']

    train_months = sorted(set(np.loadtxt('./src/data_processing/train_dates.list', dtype=str)))
    val_months = sorted(set(np.loadtxt('./src/data_processing/val_dates.list', dtype=str)))
    test_months = sorted(set(np.loadtxt('./src/data_processing/test_dates.list', dtype=str)))

    # Filter dates to be only after 2019 for debuging
    #train_months = [m for m in train_months if int(m.split('-')[0]) >= 2019]
    #val_months = [m for m in val_months if int(m.split('-')[0]) >= 2019]
    #test_months = [m for m in test_months if int(m.split('-')[0]) >= 2019]

    train_dates = generate_dates(train_months)
    val_dates = generate_dates(val_months)
    test_dates = generate_dates(test_months)

    # Debugging: only take a subset of dates for testing
    train_dates = train_dates[::400]
    val_dates = val_dates[::400]
    test_dates = test_dates[::400]

    def get_file_paths(dates):
        file_paths = []
        for date in dates:
            file_path = os.path.join(gnss_path, f'Split_{date.year}{date.timetuple().tm_yday:03d}_30_5_subsampled_{sampling}.h5')
            if os.path.exists(file_path):
                file_paths.append(file_path)
        return file_paths

    return {
        'train': get_file_paths(train_dates),
        'val': get_file_paths(val_dates),
        'test': get_file_paths(test_dates)
    }

def get_data_loaders(config):

    collate_fn = CollateWithSH(config)

    file_splits = get_split_file_lists(config, config['year'], config['doy'])

    loaders = {}
    for split, file_paths in file_splits.items():
        datasets = [PyTablesDatasetSplit(config, file_path, split) for file_path in file_paths]
        combined_ds = torch.utils.data.ConcatDataset(datasets)
        print(f"Combined dataset into one. Total length: {len(combined_ds)}")
        
        #sampler = RandomSampler(combined_ds, num_samples=int(len(combined_ds)/30))
        
        loaders[split] = DataLoader(
            combined_ds,
            batch_size=config['finetune']['batch_size'],
            num_workers=config['finetune']['num_workers'],
            prefetch_factor=2,
            collate_fn=collate_fn,
            shuffle=True if split == 'train' else False,
            #sampler=sampler if split == 'train' else None
        )
    return loaders['train'], loaders['val'], loaders['test']