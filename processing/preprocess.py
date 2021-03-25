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

def get_dataframe(feature, data):
    saved_file_name = f"{dataframe_dir}/{feature}.h5"
    if os.path.isfile(saved_file_name):
        print(f"--Using saved converted {feature}.h5")
        df = pd.read_hdf(saved_file_name, key="data")
    else:
        print(f"--Converting {feature}")
        d = xr.open_dataset(f"{interpolated_data_dir}/{data['path']}")
        # Delete time_bnds index and column
        if 'time_bnds' in d.variables:
            d = d.drop('time_bnds')
        df = d.to_dataframe()
        del d
        df = df.reset_index()
        print(df.head())

        # Rename variable
        df = df.rename(columns={data['time']: 'start_date', data['name']: feature})
        print(df.shape)
        df = df.drop_duplicates()
        print(df.shape)

        # Format start_date to datetime column  
        df['start_date'] = pd.to_datetime(df['start_date']).dt.date.astype('datetime64')

        df = df.set_index('start_date')
       
        print(df.head())
        df.to_hdf(saved_file_name, key="data")
    return df

def get_merged_dataframe(merge=False, clean=True, save=True):
    saved_merged_file_name = f"{processed_data_dir}/processed_dataframe.h5"
    if not merge and os.path.isfile(saved_merged_file_name):
        print("--Loading saved merged dataframe")
        merged_df = pd.read_hdf(saved_merged_file_name, key="data")
    else:
        print("--Converting to dataframes and merging")
        merged_df = None
        for feature, data in features.items():
            df = get_dataframe(feature=feature, data=data)
            if merged_df is not None:
                print(f"    --Combining {feature} with {', '.join(list(merged_df.columns)[3:])}")
                merged_df = merged_df.merge(df, left_on=['start_date', 'lon', 'lat'], right_on=['start_date', 'lon', 'lat'], how='inner')
            else:
                merged_df = df.copy()
            # delete dataframe
            del df
            print(merged_df.shape)
     
        if clean:
            print("\n--Cleaning NaN values in merged dataframe")
            # Setting nan values to mean
            for i, feature in enumerate(features):
                nan = merged_df[feature].isnull()
                if nan.values.any():
                    print(f"--Found NaN values in column {feature}")
                    mean = merged_df[merged_df[feature].notnull()][feature].mean()
                    merged_df.loc[nan, feature] = mean

        if save:
            print("\n--Saving merged dataframe")
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
        print("--Loading saved normalized dataframe")
        df = pd.read_hdf(saved_normalized_file_name, key="data")
    else:
        print("--Creating normalized dataframe")
        df = get_merged_dataframe()
        scaler = MinMaxScaler()
        df[list(features)] = scaler.fit_transform(df[features])
        print("\n--Saving normalized dataframe")
        df.to_hdf(saved_normalized_file_name, key="data")
    return df

def get_train_data(target_months=[6,7,8,9], horizon=56, lookback=26, features=['rf', 'temp', 'pres', 'slp', 'rhum'], target_feature='rf'):
    train = get_normalized_dataframe()
    # dates = df["start_date"].unique()
    # train = train[train.index.shift(periods=(horizon+lookback), freq='D').month.isin(target_months)]
    train = train.sort_values(by=['start_date', 'lat', 'lon'])
    train_grid = (train['lat'].nunique(), train['lon'].nunique())
    train = train[features]

    saved_file_name = f"{dataframe_dir}/{target_feature}.h5"
    print(f"--Using saved converted {target_feature}.h5")
    target = pd.read_hdf(saved_file_name, key="data")

    target = target[target.index.month.isin(target_months)]
    target = target.sort_values(by=['start_date', 'lat', 'lon'])

    # Drop na values
    target = target.dropna()
    saved_target_points_file_name = f"data/target_points_{target_feature}.csv"
    if not os.path.isfile(saved_target_points_file_name):
        target[['lat', 'lon']].drop_duplicates().to_csv(saved_target_points_file_name, index=None)
        
    target_shape = target.groupby(['lat', 'lon']).ngroups
    
    scaler = MinMaxScaler()
    target[target_feature] = scaler.fit_transform(target[[target_feature]])

    target = target[target_feature]

    return train, target, train_grid, target_shape
    
if __name__ == '__main__':
    get_merged_dataframe()




