#!/usr/bin/env python

import os
import sys
import time
import numpy as np
import pandas as pd

def mslm(elevation):
    # degree2rad
    elev = elevation*np.pi/180
    rb = 6371000
    hion = 450000
    rp = rb/(rb+hion)*np.sin(0.9782*(np.pi/2-elev))
    fs = 1/np.sqrt(1-rp*rp)
    return fs


def calculate_rmse(group):
    squared_error = (group['vtec_pred_error'])**2
    rmse = np.sqrt(squared_error.mean())
    mae = np.mean(np.abs(group['vtec_pred_error']))
    count = len(group['vtec_pred_error'])
    return pd.Series({'count': count, 'rmse': rmse, 'mae': mae})

def remove_outliers(group):
    Q1 = group.quantile(0.25)
    Q3 = group.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    return group[(group <= upper_bound)]

def dstec_assessment(group):
    nsample = len(group)
    # if < ~ 10min
    # if < ~ 5min

    dstec_rms = np.nan
    rmss = [np.nan, np.nan, np.nan]
    maes = [np.nan, np.nan, np.nan]
    res = [np.nan, np.nan, np.nan]

    group_id = group['id'].iloc[0]
    if nsample >= 10:
        idx = group[['elev']].idxmax()

        # to assume that the ref measurements and current measurements are independent 
        # when the elevation differences are larger that 20 degrees
        mask = (group['elev'] - group.loc[idx, 'elev'].values < -20)
        group['mask'] = mask
        # print('maximum elevation difference: %10.3f %10.3f' %(np.max(np.abs(group['elev'].values - group.loc[idx, 'elev'].values)), 
        #        group.loc[idx, 'elev'].values))

        dstec = np.array(group['gfphase'].values - group.loc[idx, 'gfphase'].values)
        dstec_ccl = np.array(group['vtec'].values*group['facion'].values - 
                              group.loc[idx, 'vtec'].values*group.loc[idx, 'facion'].values)
        # dstec_gim = np.array(group['gimstec'].values - group.loc[idx, 'gimstec'].values)
        dstec_gim = np.array(group['ac_vtec'].values*group['facion'].values - 
                             group.loc[idx, 'ac_vtec'].values*group.loc[idx, 'facion'].values)
        dstec_gim_var = np.array(group['ac_vtec_variance'].values*np.square(group['facion'].values) +
                                  group.loc[idx, 'ac_vtec_variance'].values*np.square(group.loc[idx, 'facion'].values))

        dstec_pred = np.array(group['vtec_pred'].values*group['facion'].values - 
                              group.loc[idx, 'vtec_pred'].values*group.loc[idx, 'facion'].values)
        dstec_pred_var = np.array(group['vtec_pred_variance'].values*np.square(group['facion'].values) +
                                  group.loc[idx, 'vtec_pred_variance'].values*np.square(group.loc[idx, 'facion'].values))
        epochs = group['epoch'].values
        masks = group['mask'].values
        ids = group['id'].values
        elevs = group['elev'].values
        dstec_pred_error = dstec_pred - dstec
        dstec_gim_error = dstec_gim - dstec
        df_dstec = pd.DataFrame({'id': ids, 'epoch': epochs, 'elev': elevs, 
                                 'dstec': dstec, 'dstec_ccl': dstec_ccl, 
                                 'dstec_gim': dstec_gim, 'dstec_gim_error': dstec_gim_error, 'dstec_gim_var': dstec_gim_var,
                                 'dstec_pred': dstec_pred, 'dstec_pred_error': dstec_pred_error, 
                                 'dstec_pred_var': dstec_pred_var, 'mask': masks})

        df_dstec['elev_max'] = group.loc[idx, 'elev'].values[0]
        df_dstec.to_csv(f'dstec_{group_id}_df.csv', index=None, float_format="%10.4f")

        # lists = [dstec, dstec_ccl, dstec_gim, dstec_pred, dstec_pred_var]
        # length = len(dstec)
        # if not all(len(l) == length for l in lists):
        #     print('not all lists have same length!')
        # else:
        #     # labels = ['GIML', 'GIM', 'NN']
        #     labels = ['CCL', 'GIM', 'NN']
        #     # mask = (dstec != 0)
        #     mask = masks
        #     dstec = dstec[mask]
        #     dstec_ccl = dstec_ccl[mask]
        #     dstec_gim = dstec_gim[mask]
        #     dstec_pred = dstec_pred[mask]
        #     maes = []
        #     rmss = []
        #     res = []

        #     dstec_rms = np.sqrt(np.nanmean(np.square(dstec)))

        #     for values in [dstec_ccl, dstec_gim, dstec_pred]:
        #         diff = values - dstec
        #         mae = np.nanmean(np.abs(diff))
        #         rmse = np.sqrt(np.nanmean(np.square(diff)))
        #         re = rmse/dstec_rms

        #         rmss.append(rmse)
        #         maes.append(mae)
        #         res.append(re)

    return pd.Series({'count': nsample, 'dstec_rms': dstec_rms,
           'rmse_ccl': rmss[0], 'mae_ccl': maes[0], 're_ccl': res[0],
           'rmse_gim': rmss[1], 'mae_gim': maes[1], 're_gim': res[1],
           'rmse_nn' : rmss[2], 'mae_nn' : maes[2], 're_nn' : res[2]})

# input data 
path = sys.argv[1]
file_gimvtec = sys.argv[2]

# time information
year = int(sys.argv[3])
doy = "%03d" %(int(sys.argv[4]))
ac = sys.argv[5]

print('Process: ', path, file_gimvtec, year, doy)

data = pd.read_csv(path )
data['epoch'] = pd.to_datetime(data['epoch'], format='%Y %m %d %H %M %S')
df_gim = pd.read_csv(file_gimvtec, sep='\s+')

print(data.head())

print('datashape: ', len(data), len(df_gim))
data['ac_vtec'] = df_gim['vtec'].values
data['ac_vtec_variance'] = df_gim['vtecvar'].values
data['mslm'] = data['elev'].apply(mslm)
print(data[['mslm', 'facion']])

data['id'] = data['station']+data['satellite']+data['slipc'].astype(int).astype(str)
id_rmse = data.groupby(['id']).apply(dstec_assessment).reset_index()
# id_rmse.to_csv(f'{ac}_per_id_{year}{doy}.csv', index=None, float_format="%10.4f")
