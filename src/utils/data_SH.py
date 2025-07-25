import os
import h5py
import pandas as pd
import torch
import numpy as np
from spacepy.coordinates import Coords
from spacepy.time import Ticktock
from torch.utils.data import RandomSampler, Dataset, DataLoader, random_split
from utils.locationencoder.pe import SphericalHarmonics
import tables

import warnings
warnings.filterwarnings("ignore")

torch.multiprocessing.set_start_method('fork', force=True)

class SingleGNSSDataset(Dataset):
    def __init__(self, config, split='random'):
        # Load and preprocess your data here
        self.year = config['year']
        self.doy = config['doy']
        self.elev = config['training']['elevation']  # Elevation cutoff

        data_file = os.path.join(config['data']['GNSS_data_path'], str(self.year), str(self.doy), f'ccl_{self.year}{self.doy}_30_5.h5')
        self.split = split
        self.data = self.load_data(data_file)
    
    def __len__(self):
        return len(self.data)

    def coord_transform(self, coords, epochs, inp_type, out_type):
        coord_inp = Coords(coords, inp_type, 'sph')
        coord_inp.ticks = Ticktock(epochs, 'UTC')
        return coord_inp.convert(out_type, 'sph')

    def load_data(self, data_file):

        if self.split != 'random':
            sta_list = np.loadtxt(f'./src/data_processing/{self.split}.list', dtype=str)
            
        data = {}

        with h5py.File(data_file, 'r') as h5_file:

            # If no specific columns are provided, load all columns
            self.columns_to_load = list(h5_file.keys())  # Load all column names
            
            if self.split != 'random':
                # Load the 'station' column to filter rows first
                station_column = h5_file['station'][:]
                station_column = np.array([x.decode('utf-8').upper() if isinstance(x, bytes) else x for x in station_column])
        
                # Filter indices for the wanted stations
                wanted_indices = np.isin(station_column, sta_list)

                # Load only the filtered rows for each column
                for column in self.columns_to_load:
                    data[column] = h5_file[column][wanted_indices]

            else:
                # Load the data for the specified columns
                for column in self.columns_to_load:
                    data[column] = h5_file[column][:]

        # Create a DataFrame from the data dictionary to facilitate merging
        data_df = pd.DataFrame(data)
        data_df['station'] = np.array([x.decode('utf-8').upper() if isinstance(x, bytes) else x for x in data_df['station']])

        df = self.filter_df(df)

        return df
    
    def filter_df(self, df):
        
        df['station'] = df['station'].apply(lambda x: x.decode('utf-8').upper() if isinstance(x, bytes) else x)

        if self.split != 'random':
            sta_list = np.loadtxt(f'./src/data_processing/{self.split}.list', dtype=str)
            df = df[df['station'].isin(sta_list)]

        # Filter data
        mask = (abs(df['dcbs']) > 1e-3) & (abs(df['dcbr']) > 1e-3) & (df['vtec'] > 2.0) & \
            (df['vtec'] <= 200) & (df['satele'] >= self.elev)
        df = df[mask]
        
        # Handle empty dataframe case
        if df.empty:
            raise ValueError("DataFrame is empty after filtering, check your filtering conditions or data.")

        # Normalize time features
        df.loc[:, 'sin_sod'] = np.sin(df['sod'] / 86400 * 2 * np.pi)
        df.loc[:, 'cos_sod'] = np.cos(df['sod'] / 86400 * 2 * np.pi)
        df.loc[:, 'sod_normalize'] = 2 * df['sod'] / 86400 - 1

        # Normalize spatial features
        #df.loc[:, 'sm_lon'] = (df['sm_lon'] + 180) % 360 - 180
        df.loc[:, 'sm_lon_sta'] = (df['sm_lon_sta'] + 180) % 360 - 180

        # Normalize azimuth and elevation
        df.loc[:, 'sin_azi'] = np.sin(df['satazi'] / 180 * np.pi)
        df.loc[:, 'cos_azi'] = np.cos(df['satazi'] / 180 * np.pi)
        df.loc[:, 'ele_normalize'] = 2 * df['satele'] / 90 - 1

        # Keep only the necessary columns
        columns_to_keep = ['stec', 'sm_lat_sta', 'sm_lon_sta', 'sin_azi', 'cos_azi', 'ele_normalize', 'sin_sod', 'cos_sod', 'sod_normalize']
        return torch.tensor(df[columns_to_keep].values, dtype=torch.float32)

    def __getitem__(self, idx):

        x = self.data[idx, 1:]  # All columns except the first one as features
        y = self.data[idx, 0]   # First column as the label

        return x, y

class PyTablesDataset(Dataset):
    def __init__(self, config, h5_file_path, split):
        print(f"Initialize dataset from {h5_file_path}")
        self.config = config
        self.h5_file_path = h5_file_path
        self.doy = self.h5_file_path.split('/')[-1].split('_')[1][4:]
        self.year = self.h5_file_path.split('/')[-1].split('_')[1][:4]
        self.split = split
        self.file = tables.open_file(h5_file_path, mode='r', driver='H5FD_SEC2')
        
        # Access the dataset (table) inside the group
        self.data = self.file.get_node(f'/{self.year}/{self.doy}/all_data')

        self.filtered_indices = self.get_indices()
        self.length = len(self.filtered_indices)

        self.file.close()  # Close the file after filtering
        del self.file  # Delete the file object
        del self.data  # Delete the data object

    def get_indices(self):
        stations = np.loadtxt(f'./src/data_processing/{self.split}.list', dtype=str)
        filtered_indices = []
    
        for row in self.data.iterrows():
            if row['station'].decode('utf-8') in stations and \
            row['satele'] >= self.config['data']['min_elevation'] and \
            2 < row['vtec'] <= 200 and \
            abs(row['dcbs']) > 1e-3 and abs(row['dcbr']) > 1e-3:
                filtered_indices.append(row.nrow)  # Store index

        return np.array(filtered_indices, dtype=np.int32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get the actual row index after filtering
        actual_idx = self.filtered_indices[idx]
        with tables.open_file(self.h5_file_path, mode='r') as file:
            data = file.get_node(f'/{self.year}/{self.doy}/all_data')
            row = data[actual_idx] # Efficient single row retrieval
        #row = self.data[actual_idx]

        # Encode strings and combine with numeric features
        features = torch.tensor([
            row['sm_lat_sta'],
            row['sm_lon_sta'],
            row['satazi'],
            row['satele'],
            row['sod'],
        ], dtype=torch.float32)

        # Return features and label separately
        label = torch.tensor(row['stec'], dtype=torch.float32)

        return features, label

    def __del__(self):
        self.file.close()  # Ensure the file is closed properly

class PyTablesDatasetSplit(Dataset):
    def __init__(self, config, h5_file_path, split):
        self.config = config
        self.h5_file_path = h5_file_path
        self.doy = self.h5_file_path.split('/')[-1].split('_')[1][4:]
        self.year = self.h5_file_path.split('/')[-1].split('_')[1][:4]
        self.split = split
        self.file = tables.open_file(h5_file_path, mode='r')
        self.data = self.file.get_node(f'/{self.year}/{self.doy}/{self.split}_data')
        self.length = len(self.data)

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
            row['sod']],
            dtype=torch.float32)

        # Return features and label separately
        label = torch.tensor(row['stec'], dtype=torch.float32)

        return features, label

    def __del__(self):
        self.file.close()  # Ensure the file is closed properly

class CollateWithSH:
    def __init__(self, config):
        self.sh_degree = config["data"]["SH_degree"]
        if self.sh_degree:
            self.sh_encoder = SphericalHarmonics(legendre_polys=self.sh_degree)

    def transform(self, features):
        """
        Normalize and transform the batch of features.
        """
        azi_ind = 2  # Index of 'satazi' in the features tensor
        ele_ind = 3  # Index of 'satele' in the features tensor
        sod_ind = 4  # Index of 'sod' in the features tensor
        sm_lon_sta_ind = 1  # Index of 'sm_lon_sta' in the features tensor
        sm_lat_sta_ind = 0  # Index of 'sm_lat_sta' in the features tensor

        ipp_lat_ind = [i for i in range(5, 5 + self.num_layers)]
        ipp_lon_ind = [i for i in range(5 + self.num_layers, 5 + 2 * self.num_layers)]
        
        # Normalize time features
        sin_sod = torch.sin(features[:, sod_ind] / 86400 * 2 * torch.pi)  # sin(sod)
        cos_sod = torch.cos(features[:, sod_ind] / 86400 * 2 * torch.pi)  # cos(sod)
        sod_normalize = 2 * features[:, sod_ind] / 86400 - 1              # Normalize sod

        # Normalize azimuth and elevation
        sin_azi = torch.sin(features[:, azi_ind] / 180 * torch.pi)  # sin(satazi)
        cos_azi = torch.cos(features[:, azi_ind] / 180 * torch.pi)  # cos(satazi)
        ele_normalize = 2 * features[:, ele_ind] / 90 - 1           # Normalize satele

        # Normalize station coordinates features
        sm_lon = (features[:, sm_lon_sta_ind] + 180) % 360 - 180  # Normalize 'sm_lon'
        sm_lat = (features[:, sm_lat_sta_ind] + 90) % 180 - 90  # Normalize 'sm_lat'

        # Normalize all IPP coordinates dynamically
        sm_lat_ipp_normalized = [(features[:, idx] + 90) % 180 - 90 for idx in ipp_lat_ind]
        sm_lon_ipp_normalized = [(features[:, idx] + 180) % 360 - 180 for idx in ipp_lon_ind]

        # Stack the IPP values dynamically
        sm_lat_ipp_tensor = torch.stack(sm_lat_ipp_normalized, dim=1)
        sm_lon_ipp_tensor = torch.stack(sm_lon_ipp_normalized, dim=1)

        # Replace transformed columns into the feature tensor
        features = torch.cat((
            sm_lat.unsqueeze(1),
            sm_lon.unsqueeze(1),
            sin_azi.unsqueeze(1),
            cos_azi.unsqueeze(1),
            ele_normalize.unsqueeze(1),
            sin_sod.unsqueeze(1),
            cos_sod.unsqueeze(1),
            sod_normalize.unsqueeze(1),
            sm_lat_ipp_tensor,
            sm_lon_ipp_tensor,
        ), dim=1)

        return features
    
    def SH_transform(self, features):
        """
        Compute spherical harmonic embeddings for features dynamically.

        Args:
            features (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: Transformed feature tensor with SH embeddings.
        """

        sm_lon_sta_ind = 1  # Index of 'sm_lon_sta' in the features tensor
        sm_lat_sta_ind = 0  # Index of 'sm_lat_sta' in the features tensor

        ipp_lat_ind = [i for i in range(8, 8 + self.num_layers)]
        ipp_lon_ind = [i for i in range(8 + self.num_layers, 8 + 2 * self.num_layers)]

        if self.sh_encoding:
            # Extract station lat/lon (in degrees, NOT normalized)
            sm_lat = features[:, sm_lat_sta_ind]
            sm_lon = features[:, sm_lon_sta_ind]
            lonlat = torch.stack((sm_lon, sm_lat), dim=-1)

            # Normalize lat/lon of stations for final output
            sm_lat_norm = (sm_lat + 90) / 180 * 2 - 1
            sm_lon_norm = (sm_lon + 180) / 360 * 2 - 1

            # Compute SH embeddings for station location
            embeddings = self.sh_encoder(lonlat)

            # Extract IPP lat/lon **in degrees**
            sm_lat_ipp = [features[:, idx] for idx in ipp_lat_ind]  # List of (N,)
            sm_lon_ipp = [features[:, idx] for idx in ipp_lon_ind]  # List of (N,)

            # Compute SH embeddings for each IPP layer separately
            embeddings_ipp = [self.sh_encoder(torch.stack((sm_lon_ipp[i], sm_lat_ipp[i]), dim=-1))  # (N, 2) → (N, SH_dim)
                            for i in range(self.num_layers)]

            # Convert lists to tensors
            sm_lat_ipp_tensor = torch.stack(sm_lat_ipp, dim=1)  # Shape: (N, num_layers)
            sm_lon_ipp_tensor = torch.stack(sm_lon_ipp, dim=1)  # Shape: (N, num_layers)
            embeddings_ipp_tensor = torch.stack(embeddings_ipp, dim=1)

            # Normalize lat/lon of IPPs for final output
            sm_lat_ipp_norm = (sm_lat_ipp_tensor + 90) / 180 * 2 - 1
            sm_lon_ipp_norm = (sm_lon_ipp_tensor + 180) / 360 * 2 - 1

            # Extract other features (excluding lat/lon columns)
            lat_lon_indices = [0, 1] + ipp_lat_ind + ipp_lon_ind
            other_feature_indices = [i for i in range(features.shape[1]) if i not in lat_lon_indices]
            other_features = features[:, other_feature_indices]

            # Interleave IPP lat, lon, and SH embeddings per layer
            ipp_features = torch.cat([torch.cat((sm_lat_ipp_norm[:, i].unsqueeze(1),
                                                sm_lon_ipp_norm[:, i].unsqueeze(1),
                                                embeddings_ipp_tensor[:, i, :]), dim=1)
                                    for i in range(self.num_layers)], dim=1)

            # Concatenate everything
            features = torch.cat([sm_lat_norm.unsqueeze(1), sm_lon_norm.unsqueeze(1), embeddings, ipp_features, other_features], dim=1)

        else:
            # If SH_encoding is False, just normalize and return the features
            sm_lat_norm = (features[:, sm_lat_sta_ind] + 90) / 180 * 2 - 1
            sm_lon_norm = (features[:, sm_lon_sta_ind] + 180) / 360 * 2 - 1

            sm_lat_ipp_normalized = [(features[:, idx] + 90) / 180 * 2 - 1 for idx in ipp_lat_ind]
            sm_lon_ipp_normalized = [(features[:, idx] + 180) / 360 * 2 - 1 for idx in ipp_lon_ind]

            sm_lat_ipp_tensor = torch.stack(sm_lat_ipp_normalized, dim=1)
            sm_lon_ipp_tensor = torch.stack(sm_lon_ipp_normalized, dim=1)

            # Extract other features
            lat_lon_indices = [0, 1] + ipp_lat_ind + ipp_lon_ind
            other_feature_indices = [i for i in range(features.shape[1]) if i not in lat_lon_indices]
            other_features = features[:, other_feature_indices]

            # Interleave IPP latitudes and longitudes
            ipp_features = torch.cat([torch.stack((sm_lat_ipp_tensor[:, i], sm_lon_ipp_tensor[:, i]), dim=1)
                                    for i in range(self.num_layers)], dim=1)

            features = torch.cat([
                sm_lat_norm.unsqueeze(1),
                sm_lon_norm.unsqueeze(1),
                ipp_features,
                other_features
            ], dim=1)

        return features
    
    def __call__(self, batch):
        """
        Process and collate a batch of (features, labels).
        """
        # Unzip features and labels from the batch
        features, labels = zip(*batch)

        # Stack features and labels into tensors
        features = torch.stack(features)
        labels = torch.stack(labels)

        # Normalize and transform features
        features = self.transform(features)

        # Apply SH encoding if enabled
        features = self.SH_transform(features)

        return features, labels

def get_split_file_lists(config, year, doy):
    gnss_path = config['data']['GNSS_data_path']   
    sampling = config['data']['sampling']

    start_date = pd.to_datetime(config['pretrain']['pretrain_start_date'])
    end_date = pd.to_datetime(config['pretrain']['pretrain_end_date'])

    all_dates = pd.date_range(start_date, end_date)
    test_dates = all_dates[:1]
    train_dates, val_dates = np.split(all_dates[1:], [int(0.8 * (len(all_dates)-len(test_dates)))])

    def get_file_paths(dates):
        file_paths = []
        for date in dates:
            file_path = os.path.join(gnss_path, f'Split_{date.year}{date.dayofyear:03d}_30_5_subsampled_{sampling}.h5')
            if os.path.exists(file_path):
                file_paths.append(file_path)
        return file_paths

    return {
        'train': get_file_paths(train_dates),
        'val': get_file_paths(val_dates),
        'test': get_file_paths(test_dates)
    }

def get_multi_data_loaders(config):

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
            batch_size=config['pretrain']['batch_size'],
            num_workers=config['pretrain']['num_workers'],
            prefetch_factor=2,
            collate_fn=collate_fn,
            shuffle=True if split == 'train' else False,
            #sampler=sampler if split == 'train' else None
        )
    return loaders['train'], loaders['val'], loaders['test']

def get_single_data_loaders(config):
    year, doy = config['year'], f'{int(config["doy"]):03d}'
    collate_fn = CollateWithSH(config)

    num_IPP_layers = config['model']['num_layers']
    sampling = config['training']['sampling']
    file = os.path.join(config['data']['GNSS_data_path'], f'Split_{year}{doy}_30_5_IL{num_IPP_layers}_subsampled_{sampling}.h5')

    loaders = {}
    splits = ['train', 'val', 'test']
    for split in splits:
        dataset = PyTablesDatasetSplit(config, file, split)

        loaders[split] = DataLoader(
            dataset,
            batch_size=config['pretrain']['batch_size'],
            num_workers=config['pretrain']['num_workers'],
            prefetch_factor=4,
            collate_fn=collate_fn,
            shuffle=True if split == 'train' else False
        )
    return loaders['train'], loaders['val'], loaders['test']
