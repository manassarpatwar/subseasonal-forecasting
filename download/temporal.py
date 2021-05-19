# Author: Manas Saraptwar
# Date: 19/05/2021

import pandas as pd
import numpy as np
import os
import requests
from io import StringIO
import config


def get_colspecs(start, width, offset, size=12):
    # Column specifications as a list of tuples. Start and end indexes for each column per row as a half closed
    # tuple [start, end)
    # 1st tuple usually is different -> (0, 4)
    colspecs = [start]
    # Create list by adding width and offset.
    # range creates starting indices of columns
    return colspecs+[(i, i+width) for i in range(start[1]+width, start[1]+width+(width+offset)*size, width+offset)]

def mjo():
    # Daily values
    data = StringIO(requests.get('http://bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt').text)
    df = pd.read_fwf(data, header=None, colspecs=[(8, 12), (22,24), (34, 36), (37, 52), (53, 69), (77, 80), (83, 96), (97, 130)])
    df = df.iloc[2:, :-1]
    df.columns = ['year', 'month', 'day', 'RMM1', 'RMM2', 'phase', 'amplitude']

    df['start_date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df[['start_date', 'RMM1', 'RMM2', 'phase', 'amplitude']]
    df = df.set_index('start_date')
    df = df.astype({'RMM1': np.float32, 'RMM2': np.float32, 'phase': int, 'amplitude': np.float32})
    df['phase_cos'] = np.cos((2*np.pi*df['phase'])/8)
    df['phase_sin'] = np.sin((2*np.pi*df['phase'])/8)
    df.to_hdf(os.path.join(config.DATAFRAME_DIR, 'mjo.h5'), key='data')
    return df

def mei():
    # Bi monthly values
    df = pd.read_fwf('https://psl.noaa.gov/enso/mei/data/meiv2.data', colspecs=get_colspecs(start=(0,4), width=5, offset=4), names=["year","1", "2","3","4","5","6","7","8","9","10","11","12"])
    df = df.iloc[1:-5]

    df = pd.melt(df, id_vars='year', var_name='month', value_name='mei')
    df['start_date'] = pd.to_datetime(df['year']+'-'+df['month'])
    df = df.drop(['year', 'month'], 1)
    df = df.set_index('start_date')
    df = df.sort_values('start_date')
    # df['mei-2'] = df['mei-1'].shift(-1)
    dates = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(dates, method='nearest')
    df.index = df.index.rename('start_date')
    df = df.astype(np.float32)

    df.to_hdf(os.path.join(config.DATAFRAME_DIR, 'mei.h5'), key='data')

def iod():
    # Bi monthly values
    df = pd.read_fwf('https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data', colspecs=get_colspecs(start=(0,4), width=6, offset=4), names=["year","1", "2","3","4","5","6","7","8","9","10","11","12"])
    df = df.iloc[1:-7]

    df = pd.melt(df, id_vars='year', var_name='month', value_name='iod')
    df['start_date'] = pd.to_datetime(df['year']+'-'+df['month'])
    df = df.drop(['year', 'month'], 1)
    df = df.set_index('start_date')
    df = df.sort_values('start_date')
    dates = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(dates, method='nearest')
    df.index = df.index.rename('start_date')
    # df['iod-2'] = df['iod-1'].shift(-1)
    df = df.astype(np.float32)

    df.to_hdf(os.path.join(config.DATAFRAME_DIR, 'iod.h5'), key='data')

if __name__ == '__main__':
    mjo()
    mei()
    iod()