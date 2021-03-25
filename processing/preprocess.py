import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
import os
from sklearn.preprocessing import MinMaxScaler

features = {"rf": {"path": "imd_rf-1x1.nc", "time": "time", "name": "rf"}, 
            "temp": {"path": "imd_temp-1x1.nc", "time": "time", "name": "t"},
            "pres": {"path": "imdaa_reanl_PRES-sfc-1x1.nc", "time": "time", "name": "PRES_sfc"}, 
            "slp": {"path": "imdaa_reanl_PRMSL-msl-1x1.nc", "time": "time", "name": "PRMSL_msl"}, 
            "rhum": {"path": "imdaa_reanl_RH-2m-1x1.nc", "time": "time", "name": "RH_2m"}}

interpolated_data_dir = "data/interpolated"
processed_data_dir = "data/processed"
dataframe_dir = "data/dataframes"

if not os.path.isdir(interpolated_data_dir):
    os.makedirs(interpolated_data_dir)

if not os.path.isdir(processed_data_dir):
    os.makedirs(processed_data_dir)

if not os.path.isdir(dataframe_dir):
    os.makedirs(dataframe_dir)

def get_dataframe(feature, data, i):
    saved_file_name = f"{dataframe_dir}/{feature}.h5"
    if os.path.isfile(saved_file_name):
        print(f"--1.{i+1}--Using saved converted {feature}.h5")
        df = pd.read_hdf(saved_file_name, key="data")
    else:
        print(f"--1.{i+1}--Converting {feature}")
        d = xr.open_dataset(f"{interpolated_data_dir}/{data['path']}")
        # Delete time_bnds index and column
        if 'time_bnds' in d.variables:
            d = d.drop('time_bnds')
        df = d.to_dataframe()
        del d
        df.reset_index(inplace=True)
        print(df.head())

        # Rename variable
        df.rename(columns={data['time']: 'start_date', data['name']: feature}, inplace=True)

        # Format time to start_date    
        df['start_date'] = pd.to_datetime(df['start_date']).dt.strftime('%Y-%m-%d')
        print(df.shape)
        df.drop_duplicates(inplace=True)
        print(df.shape)
        print(df.head())
        df.to_hdf(saved_file_name, key="data")
    return df

def get_merged_dataframe(merge=False):
    saved_merged_file_name = f"{processed_data_dir}/processed_dataframe.h5"
    if not merge and os.path.isfile(saved_merged_file_name):
        print("--1--Loading saved merged dataframe")
        merged_df = pd.read_hdf(saved_merged_file_name, key="data")
    else:
        print("--1--Converting to dataframes and merging")
        merged_df = None
        for i, (feature, data) in enumerate(features.items()):
            df = get_dataframe(feature=feature, data=data, i=i)
            if merged_df is not None:
                print(f"     --Combining {feature} with {', '.join(list(merged_df.columns)[3:])}")
                merged_df = merged_df.merge(df, left_on=['start_date', 'lon', 'lat'], right_on=['start_date', 'lon', 'lat'], how='inner')
            else:
                merged_df = df.copy()
            # delete dataframe
            del df
            print(merged_df.shape)
     
        print("\n--2--Cleaning NaN values in merged dataframe")
        # Setting nan values to mean
        for i, feature in enumerate(features):
            nan = merged_df[feature].isnull()
            if nan.values.any():
                print(f"--2.{i+1}--Found NaN values in column {feature}")
                mean = merged_df[merged_df[feature].notnull()][feature].mean()
                merged_df.loc[nan, feature] = mean

        print("\n--3--Saving merged dataframe")
        print(merged_df.head())
        merged_df.to_hdf(saved_merged_file_name, key="data")

    return merged_df

def get_feature(feature, data):
    d = nc.Dataset(f"{interpolated_data_dir}/{data['path']}")
    print(d.variables[data['name']])
    return d[data['name']][:, :, :]

def get_normalized_dataframe(normalize=False):
    saved_normalized_file_name = f"{processed_data_dir}/processed_normalized_dataframe.h5"
    if not normalize and os.path.isfile(saved_normalized_file_name):
        print("--1--Loading saved normalized dataframe")
        df = pd.read_hdf(saved_normalized_file_name, key="data")
    else:
        print("--1--Creating normalized dataframe")
        df = get_merged_dataframe()
        scaler = MinMaxScaler()
        df[list(features)] = scaler.fit_transform(df[features])
        print("\n--2--Saving normalized dataframe")
        df.to_hdf(saved_normalized_file_name, key="data")
    return df

def get_region_tensor(target_months=[6,7,8,9], horizon=56, lookback=26):
    df = get_normalized_dataframe()
    dates = df["start_date"].unique()
    tensors = []
    
    for i, (feature, data) in enumerate(features.items()):
        t, mask = get_feature(feature=feature, data=data)
        tensors.append(t)
        masks.append(mask)
    return np.stack(tensors, axis=-1)

if __name__ == '__main__':
    get_merged_dataframe()




