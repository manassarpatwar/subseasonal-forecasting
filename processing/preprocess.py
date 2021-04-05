import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
import os
from sklearn.preprocessing import StandardScaler as Scaler

spatial_features = {"rf": {"file_name": "imd_rf-1x1.nc", "time": "time", "name": "rf"}, 
            "temp": {"file_name": "imd_temp-1x1.nc", "time": "time", "name": "t"},
            "pres": {"file_name": "imdaa_reanl_PRES-sfc-1x1.nc", "time": "time", "name": "PRES_sfc"}, 
            "slp": {"file_name": "imdaa_reanl_PRMSL-msl-1x1.nc", "time": "time", "name": "PRMSL_msl"}, 
            "rhum": {"file_name": "imdaa_reanl_RH-2m-1x1.nc", "time": "time", "name": "RH_2m"}}

temporal_features = {"mjo": {"file_name": "mjo.h5"}, 
                    "mei": {"file_name": "mei.h5"},
                    "iod": {"file_name": "iod.h5"}}

data_dir = "data"
interpolated_data_dir = os.path.join(data_dir, "interpolated")
processed_data_dir = os.path.join(data_dir, "processed")
dataframe_dir = os.path.join(data_dir, "dataframes")

if not os.path.isdir(interpolated_data_dir):
    os.makedirs(interpolated_data_dir)

if not os.path.isdir(processed_data_dir):
    os.makedirs(processed_data_dir)

if not os.path.isdir(dataframe_dir):
    os.makedirs(dataframe_dir)

def get_spatial_feature(feature, data, average=False):
    period = 14 #14 days/2 weeks
    feature_file_name = os.path.join(dataframe_dir, f"{feature}{'-2-week-avg' if average else ''}.h5")
    if os.path.isfile(feature_file_name):
        print(f"--Using saved converted {feature}.h5")
        df = pd.read_hdf(feature_file_name, key="data")
    else:
        print(f"--Converting {feature}")
        d = xr.open_dataset(os.path.join(interpolated_data_dir, data['file_name']))
        # Delete time_bnds index and column
        if 'time_bnds' in d.variables:
            d = d.drop('time_bnds')
            
        if average:
            print(f"--Calculating 2 week average for {feature}")
            d = d.rolling(time=period).mean()

        df = d.to_dataframe()
        del d
        df = df.reset_index()

        # Rename variable
        df = df.rename(columns={data['time']: 'start_date', data['name']: feature})
        df = df.drop_duplicates()

        # Format start_date to datetime column  
        df['start_date'] = pd.to_datetime(df['start_date']).dt.date.astype('datetime64')

        df = df.set_index('start_date')

        if average:
            df = df.iloc[period:]
       
        df.to_hdf(feature_file_name, key="data")
    return df

def get_spatial_dataframe(merge=False, clean=True, save=True, average=True):
    spatial_file_name = os.path.join(processed_data_dir, f"spatial{'-2-week-avg' if average else ''}.h5")
    if not merge and os.path.isfile(spatial_file_name):
        print("--Loading saved spatial dataframe")
        spatial_df = pd.read_hdf(spatial_file_name, key="data")
    else:
        print("--Converting .nc to dataframes and merging")
        spatial_df = None
        for feature, data in spatial_features.items():
            df = get_spatial_feature(feature=feature, data=data, average=average)
            if spatial_df is not None:
                print(f"--Combining {feature} with {', '.join(list(spatial_df.columns)[2:])}")
                spatial_df = spatial_df.merge(df, left_on=['start_date', 'lat', 'lon'], right_on=['start_date', 'lat', 'lon'], how='inner')
            else:
                spatial_df = df.copy()
            # delete dataframe
            del df
            print(spatial_df.shape)

        spatial_df = spatial_df.sort_values(by=['start_date', 'lat', 'lon'])
        
        if clean:
            print("\n--Cleaning NaN values in spatial dataframe")
            # Setting nan values to mean
            for i, feature in enumerate(spatial_features):
                nan = spatial_df[feature].isnull()
                if nan.values.any():
                    print(f"--Found NaN values in column {feature}")
                    # mean = spatial_df[spatial_df[feature].notnull()][feature].mean()
                    dates = spatial_df[feature].groupby(spatial_df.index)
                    mean = dates.mean()
                    spatial_df.loc[nan, feature] = mean


        if save:
            print("\n--Saving spatial dataframe")
            print(spatial_df.head())
            spatial_df.to_hdf(spatial_file_name, key="data")

    return spatial_df

def get_temporal_dataframe(merge=False, save=True):
    temporal_file_name = os.path.join(processed_data_dir, "temporal.h5")
    if not merge and os.path.isfile(temporal_file_name):
        print("--Loading saved temporal dataframe")
        temporal_df = pd.read_hdf(temporal_file_name, key="data")
    else:
        print("--Merging dataframes")
        temporal_df = None
        for feature, data in temporal_features.items():
            df = pd.read_hdf(os.path.join(dataframe_dir, data['file_name']), key='data')
            if temporal_df is not None:
                print(f"--Combining {', '.join(list(df.columns))} with {', '.join(list(temporal_df.columns))}")
                temporal_df = temporal_df.merge(df, left_on='start_date', right_on='start_date', how='inner')
            else:
                temporal_df = df.copy()
            # delete dataframe
            del df
            print(temporal_df.shape)

        temporal_df = temporal_df.sort_values(by='start_date')

        if save:
            print("\n--Saving temporal dataframe")
            print(temporal_df.head())
            temporal_df.to_hdf(temporal_file_name, key="data")

    return temporal_df


def get_train_data(target_months=[6,7,8,9], horizon=56, lookback=26, spatial_features=['rf', 'temp', 'pres', 'slp', 'rhum'], temporal_features=['phase_cos', 'phase_sin', 'mei', 'iod', 'amplitude'], target_feature='rf', years=(1979, 2010, 2015, 2020), average=False):
    spatial_df = get_spatial_dataframe(average=average)
    lat_lon_grid_shape = (spatial_df['lat'].nunique(), spatial_df['lon'].nunique())
    spatial_df = spatial_df[spatial_features]
    spatial_df = spatial_df.astype(np.float32)
  
    temporal_df = get_temporal_dataframe()
    temporal_df = temporal_df[temporal_features]
    scaler = Scaler()

    target_file_name =  os.path.join(dataframe_dir, f"{target_feature}.h5")

    target = pd.read_hdf(target_file_name, key='data')
    target = target.dropna()
    target = target[target.index.month.isin(target_months)]
    target_shape = target.groupby(['lat', 'lon']).ngroups

    target = target[target_feature]
    target = target.astype(np.float32)

    years = {'train': list(range(years[0], years[1])),
            'validation': list(range(years[1], years[2])),
            'test': list(range(years[2], years[3]))}

    train = {'spatial': spatial_df[spatial_df.index.year.isin(years['train'])], 
             'temporal': temporal_df[temporal_df.index.year.isin(years['train'])],
             'y': target[target.index.year.isin(years['train'])]}

    validation = {'spatial': spatial_df[spatial_df.index.year.isin(years['validation'])],
                  'temporal': temporal_df[temporal_df.index.year.isin(years['validation'])], 
                  'y': target[target.index.year.isin(years['validation'])]}

    test = {'spatial': spatial_df[spatial_df.index.year.isin(years['test'])], 
            'temporal': temporal_df[temporal_df.index.year.isin(years['test'])],
            'y': target[target.index.year.isin(years['test'])]}

    # Normalize
    train['spatial'] = pd.DataFrame(scaler.fit_transform(train['spatial']), index=train['spatial'].index, columns=train['spatial'].columns)
    validation['spatial'] = pd.DataFrame(scaler.transform(validation['spatial']), index=validation['spatial'].index, columns=validation['spatial'].columns)
    test['spatial'] = pd.DataFrame(scaler.transform(test['spatial']), index=test['spatial'].index, columns=test['spatial'].columns)

    return  train, validation, test, lat_lon_grid_shape, target_shape
    
if __name__ == '__main__':
    get_spatial_dataframe()




