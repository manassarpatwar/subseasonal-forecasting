# Author: Manas Saraptwar
# Date: 19/05/2021

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from preprocess import get_merged_imdaa, get_merged_rodeo
import argparse, sys
import config



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--feature', help='Feature to visualise, eg: precip')
    parser.add_argument('-d', '--date', help='Date to visualise at, eg: 1979-06-01')
    parser.add_argument('-p', '--path',  help='Path of file to visualise')
    parser.add_argument('--lat', help='Latitude to visualise')
    parser.add_argument('--lon',  help='Longitude to visualise')
    parser.add_argument('-i', '--imdaa', help='Visualise imdaa feature', action='store_true')
    parser.add_argument('-r', '--rodeo', help='Visualise rodeo feature', action='store_true')

    args = parser.parse_args()

    feature = args.feature
    lat = float(args.lat) if args.lat else None
    lon = float(args.lon) if args.lon else None
    date = args.date
    path = args.path

    if path:
        df = pd.read_hdf(path)
        if 'start_date' in df.columns:
            df = df.set_index('start_date')
    else:
        if args.rodeo:
            df = get_merged_rodeo()
        else:
            df = get_merged_imdaa()
    
    latlon = f"-{lat}-{lon}" if lat and lon else ''

    if lat and lon:
        df = df.loc[(df['lat'] == lat) & (df['lon'] == lon)]
        df = df[feature]
        plt.plot(df)
    else:
        df = df.loc[date or '2017-06-30']
        lat, lon = df['lat'].nunique(), df['lon'].nunique()
        df = df[feature]
        plt.imshow(np.flipud(df.to_numpy().reshape(lat, lon)))
    
    plt.savefig(os.path.join(config.VISUALISATION_DIR, f"{feature}{latlon}.png"))
