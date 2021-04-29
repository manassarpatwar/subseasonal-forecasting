import matplotlib.pyplot as plt
import pandas as pd
import os
from preprocess import get_spatial_feature
import argparse, sys



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--feature', help='Feature to visualise, eg: rf')
    parser.add_argument('-p', '--path',  help='Path of file to visualise')
    parser.add_argument('--lat', help='Latitude to visualise')
    parser.add_argument('--lon',  help='Longitude to visualise')
    parser.add_argument('-a', '--average', help='Visualise averaged feature', action='store_true')

    args = parser.parse_args()

    feature = args.feature
    lat = float(args.lat)
    lon = float(args.lon)
    average = bool(args.average)

    path = args.path

    if path:
        df = pd.read_hdf(path)
        df = df.set_index('start_date')
    else:
        df = get_spatial_feature(feature, average)
    
    if lat and lon:
        df = df.loc[(df['lat'] == lat) & (df['lon'] == lon)]

    df = df[feature]
    plt.figure(figsize=(200,10))
    plt.plot(df)
    
    visualisation_dir = os.path.join('visualisations', feature)
    if not os.path.isdir(visualisation_dir):
        os.makedirs(visualisation_dir)
    
    plt.savefig(os.path.join(visualisation_dir, f"{feature}-{lat}-{lon}{'-avg' if average else ''}.png"))
