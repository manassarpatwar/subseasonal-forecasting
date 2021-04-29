import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
import os
from sklearn.preprocessing import MinMaxScaler
from utils import Lookback
import config


def get_spatial_feature(feature, average=False, period=14):
    
    feature_file_name = os.path.join(config.DATAFRAME_DIR, f"{feature}{'-2-week-avg' if average else ''}.h5")
    if os.path.isfile(feature_file_name):
        print(f"--Using saved converted {feature}.h5")
        df = pd.read_hdf(feature_file_name, key="data")
    else:
        data = config.SPATIAL_FEATURES[feature]
        print(f"--Converting {feature}")
        d = xr.open_dataset(os.path.join(config.INTERPOLATED_DATA_DIR, data['file_name']))
        # Delete time_bnds index and column
        if 'time_bnds' in d.variables:
            d = d.drop('time_bnds')
            
        if average:
            print(f"--Calculating 2 week average for {feature}")
            d = d.rolling(time=period, center=True).mean()

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
            offset = int(period/2)
            dates = df.index.unique()
            df = df.drop(dates[:offset])
            df = df.drop(dates[-offset:])

        df = df.sort_values(['start_date',  'lat', 'lon'])
        df.to_hdf(feature_file_name, key="data")
    return df
 
def get_spatial_dataframe(merge=False, save=True, average=True):
    spatial_file_name = os.path.join(config.PROCESSED_DATA_DIR, f"spatial{'-2-week-avg' if average else ''}.h5")
    if not merge and os.path.isfile(spatial_file_name):
        print("--Loading saved spatial dataframe")
        spatial = pd.read_hdf(spatial_file_name, key="data")
    else:
        print("--Converting .nc to dataframes and merging")
        spatial = None
        for feature in config.SPATIAL_FEATURES:
            df = get_spatial_feature(feature=feature, average=average)
            if spatial is not None:
                print(f"--Combining {feature} with {', '.join(list(spatial.columns)[2:])}")
                spatial = spatial.merge(df, left_on=['start_date', 'lat', 'lon'], right_on=['start_date', 'lat', 'lon'], how='inner')
            else:
                spatial = df.copy()
            # delete dataframe
            del df
            print(spatial.shape)

        spatial = spatial.sort_values(by=['start_date', 'lat', 'lon'])            

        # if clean:
        #     print("\n--Cleaning NaN values in dataframe")
        #     spatial = spatial.fillna(0.0)
            # Setting nan values to mean
            # for i, feature in enumerate(spatial_features):
            #     nan = spatial[feature].isnull()
            #     if nan.values.any():
            #         print(f"--Found NaN values in column {feature}")
            #         mean = spatial[spatial[feature].notnull()][feature].mean()
            #         # dates = spatial[feature].groupby(spatial.index)
            #         # mean = dates.mean()
            #         spatial.loc[nan, feature] = mean

        if save:
            print("\n--Saving spatial dataframe")
            print(spatial.head())
            spatial.to_hdf(spatial_file_name, key="data")

    return spatial   

def get_temporal_dataframe(merge=False, save=True):
    temporal_file_name = os.path.join(config.PROCESSED_DATA_DIR, "temporal.h5")
    if not merge and os.path.isfile(temporal_file_name):
        print("--Loading saved temporal dataframe")
        temporal = pd.read_hdf(temporal_file_name, key="data")
    else:
        print("--Merging dataframes")
        temporal = None
        for feature, data in config.TEMPORAL_FEATURES.items():
            df = pd.read_hdf(os.path.join(config.DATAFRAME_DIR, data['file_name']), key='data')
            if temporal is not None:
                print(f"--Combining {', '.join(list(df.columns))} with {', '.join(list(temporal.columns))}")
                temporal = temporal.merge(df, left_on='start_date', right_on='start_date', how='inner')
            else:
                temporal = df.copy()
            # delete dataframe
            del df
            print(temporal.shape)

        temporal = temporal.sort_values(by='start_date')

        if save:
            print("\n--Saving temporal dataframe")
            print(temporal.head())
            temporal.to_hdf(temporal_file_name, key="data")

    return temporal

def normalize(df):
    return (df-df.min())/(df.max()-df.min())

def get_train_data(target_months=[1,2,3,4,5,6,7,8,9,10,11,12], 
                    horizon=21, 
                    lookback=Lookback(past=28), 
                    spatial_features=['rf', 'temp', 'pres', 'slp', 'rhum'], 
                    temporal_features=['phase_cos', 'phase_sin', 'mei', 'iod', 'amplitude'], 
                    target_feature='rf', 
                    split=(1979, 2010, 2015, 2020), 
                    average=False,
                    lat_lon=(20.0, 75.0)):

    spatial = get_spatial_dataframe(average=average)

    spatial = spatial.astype(np.float32)
  
    temporal = get_temporal_dataframe()
    temporal = temporal.astype(np.float32)

    temporal = temporal[temporal_features]

    train_years, validation_years, test_years = lookback.split_years(split=split)
    
    spatial_train = spatial[train_years][spatial_features]
    #normalize
    # spatial_mean = spatial_train.mean(axis=0)
    # spatial_std = spatial_train.std(axis=0)

    ground_truth = spatial[['lat', 'lon', target_feature]]

    spatial_dates = spatial.groupby(spatial.index)
    spatial_mean = spatial_dates.mean()
    spatial_std = spatial_dates.std()
    
    spatial = spatial.fillna(spatial_mean)
    spatial[spatial_features] = (spatial[spatial_features] - spatial_mean[spatial_features]) / spatial_std[spatial_features]

    # # Specific lat lon location
    if lat_lon:
        spatial = spatial.loc[(spatial['lat'] == lat_lon[0]) & (spatial['lon'] == lat_lon[1])]

    spatial_grid_shape = (spatial['lat'].nunique(), spatial['lon'].nunique())

    common_lat_lon = ground_truth.dropna().groupby(['lat', 'lon']).count().index
    target_mask = ground_truth.groupby(['lat', 'lon']).count().index.isin(common_lat_lon)
    
    ground_truth = ground_truth[pd.MultiIndex.from_frame(ground_truth[['lat', 'lon']]).isin(common_lat_lon)]
    ground_truth = ground_truth.fillna(spatial_mean)

    if lat_lon:
        ground_truth = ground_truth.loc[(ground_truth['lat'] == lat_lon[0]) & (ground_truth['lon'] == lat_lon[1])]
  
    #normalize
    ground_truth[target_feature] = (ground_truth[target_feature] - spatial_mean[target_feature])/spatial_std[target_feature]
    
    
    target_shape = ground_truth.groupby(['lat','lon']).ngroups
    min_target_date = lookback.min_target_date(ground_truth.index.min(), train_years.start, target_months[0], horizon)
    target_train_years = slice(min_target_date, train_years.stop)

    ground_truth = ground_truth[ground_truth.index.month.isin(target_months)]
    ground_truth = ground_truth[target_feature]

    target = {}
    target['train'] = ground_truth[target_train_years]
    target['validation'] = ground_truth[validation_years]
    target['test'] =  ground_truth[test_years]
    target['mask'] = target_mask
    target['shape'] = target_shape
    target['locations'] = common_lat_lon
    target['spatial_mean'] = spatial_mean
    target['spatial_std'] = spatial_std

    spatial = spatial[spatial_features]

    return spatial, temporal, target, spatial_grid_shape
    

def get_forecast_rodeo_data():
    spatial_file_name = "lat_lon_date_data-contest_precip_34w.h5"
    temporal_file_name = "date_data-contest_precip_34w.h5"

    spatial = get_spatial_dataframe(average=True)
    temporal = get_temporal_dataframe()

    spatial = spatial[spatial['rf'].notna()]

    spatial_da = spatial.set_index([spatial.index, 'lon', 'lat']).to_xarray()

    spatial_features = [col for col in spatial.columns if col not in ['lat', 'lon']]
    spatial_shift29 = spatial_da.shift(start_date=29).to_dataframe().reset_index(['lat', 'lon'])
    spatial_shift29 = spatial_shift29.rename(columns=dict([(col, f"{col}_shift29") for col in spatial_features]))

    spatial_shift58 = spatial_da.shift(start_date=58).to_dataframe().reset_index(['lat', 'lon'])
    spatial_shift58 = spatial_shift58.rename(columns=dict([(col, f"{col}_shift58") for col in spatial_features]))

    del spatial_da
    spatial = spatial.merge(spatial_shift29, left_on=['start_date', 'lat', 'lon'], right_on=['start_date', 'lat', 'lon'], how='inner')
    del spatial_shift29
    spatial = spatial.merge(spatial_shift58, left_on=['start_date', 'lat', 'lon'], right_on=['start_date', 'lat', 'lon'], how='inner')
    del spatial_shift58
    spatial = spatial.drop(spatial.index.unique()[:58])

    temporal = temporal.loc[spatial.index.unique()]

    spatial = spatial.reset_index()
    temporal = temporal.reset_index()
    
    spatial.to_hdf(os.path.join(config.FORECAST_RODEO_DIR, spatial_file_name), key="data")

    temporal.to_hdf(os.path.join(config.FORECAST_RODEO_DIR, temporal_file_name), key="data")

    return spatial, temporal

if __name__ == '__main__':
    get_spatial_dataframe()




