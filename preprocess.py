import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
import os
from sklearn.preprocessing import MinMaxScaler
from utils import Lookback
import config


def get_imdaa_spatial_feature(feature, average=True, period=14):
    
    feature_file_name = os.path.join(config.DATAFRAME_DIR, "{feature}{avg}.h5".format(feature=feature, avg=f"-{period}d-avg" if average else ''))
    if os.path.isfile(feature_file_name):
        print(f"-Using saved converted {feature}.h5")
        df = pd.read_hdf(feature_file_name, key="data")
    else:
        data = config.IMDAA['spatial-features'][feature]
        print(f"-Converting {feature}")
        d = xr.open_dataset(os.path.join(config.INTERPOLATED_DATA_DIR, data['file_name']))
        # Delete time_bnds index and column
        if 'time_bnds' in d.variables:
            d = d.drop('time_bnds')
            
        if average:
            print(f"-Calculating 2 week average for {feature}")
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
            dates = df.index.unique()
            df = df.drop(dates[:period-1])

        df = df.sort_values(['start_date',  'lat', 'lon'])
        df.to_hdf(feature_file_name, key="data")
    return df
 
def get_imdaa_spatial_dataframe(merge=False, save=True, average=True, period=14):
    spatial_file_name = os.path.join(config.PROCESSED_DATA_DIR, "spatial{avg}.h5".format(avg=f"-{period}d-avg" if average else ''))
    if not merge and os.path.isfile(spatial_file_name):
        print("-Loading saved spatial dataframe")
        spatial = pd.read_hdf(spatial_file_name, key="data")
    else:
        print("-Converting .nc to dataframes and merging")
        spatial = None
        for feature in config.IMDAA['spatial-features']:
            df = get_imdaa_spatial_feature(feature=feature, average=average, period=period)
            if spatial is not None:
                print(f"-Combining {feature} with {', '.join(list(spatial.columns)[2:])}")
                spatial = spatial.merge(df, left_on=['start_date', 'lat', 'lon'], right_on=['start_date', 'lat', 'lon'], how='inner')
            else:
                spatial = df.copy()
            # delete dataframe
            del df
            print(spatial.shape)

        spatial = spatial.sort_values(by=['start_date', 'lat', 'lon'])            

        # if clean:
        #     print("\n-Cleaning NaN values in dataframe")
        #     spatial = spatial.fillna(0.0)
            # Setting nan values to mean
            # for i, feature in enumerate(spatial_features):
            #     nan = spatial[feature].isnull()
            #     if nan.values.any():
            #         print(f"-Found NaN values in column {feature}")
            #         mean = spatial[spatial[feature].notnull()][feature].mean()
            #         # dates = spatial[feature].groupby(spatial.index)
            #         # mean = dates.mean()
            #         spatial.loc[nan, feature] = mean

        if save:
            print("\n-Saving spatial dataframe")
            print(spatial.head())
            spatial.to_hdf(spatial_file_name, key="data")

    return spatial   

def get_temporal_dataframe(merge=False, save=True):
    temporal_file_name = os.path.join(config.PROCESSED_DATA_DIR, "temporal.h5")
    if not merge and os.path.isfile(temporal_file_name):
        print("-Loading saved temporal dataframe")
        temporal = pd.read_hdf(temporal_file_name, key="data")
    else:
        print("-Merging dataframes")
        temporal = None
        for feature, data in config.IMDAA['temporal-features'].items():
            df = pd.read_hdf(os.path.join(config.DATAFRAME_DIR, data['file_name']), key='data')
            if temporal is not None:
                print(f"-Combining {', '.join(list(df.columns))} with {', '.join(list(temporal.columns))}")
                temporal = temporal.merge(df, left_on='start_date', right_on='start_date', how='inner')
            else:
                temporal = df.copy()
            # delete dataframe
            del df
            print(temporal.shape)

        temporal = temporal.sort_values(by='start_date')

        if save:
            print("\n-Saving temporal dataframe")
            print(temporal.head())
            temporal.to_hdf(temporal_file_name, key="data")

    return temporal

def get_rodeo_spatial_dataframe(merge=False, save=True, pad=True, res=(1.0,1.0), sg=4):
    # Make sure spatial granularity is an integer
    assert isinstance(sg, int)

    rodeo_spatial_file_name = os.path.join(config.PROCESSED_DATA_DIR, "rodeo-spatial.h5")
    if not merge and os.path.isfile(rodeo_spatial_file_name):
        print("-Loading saved rodeo spatial dataframe")
        spatial = pd.read_hdf(rodeo_spatial_file_name, key="data")
    else:
        print("-Merging rodeo dataframes")
        spatial = None
        for feature, data in config.RODEO['spatial-features'].items():
            df = pd.read_hdf(os.path.join(config.RODEO_DIR, data['file_name']))
            # df = df.rename(columns={data['name']: feature})
            df = df[['start_date', 'lat', 'lon', data['name']]]
            df = df.set_index('start_date')
            if pad:
                print(f"-Padding {feature}")
                # Get all latitude values from minimum to maximum 
                lat = np.arange(df['lat'].min()-float(sg*res[0]), df['lat'].max()+float(sg*res[0]), float(res[0]))
                # Get all longitude values from minimum to maximum 
                lon = np.arange(df['lon'].min()-float(sg*res[1]), df['lon'].max()+float(sg*res[1]), float(res[1]))
                # Unique dates
                dates = df.index.unique()
                # Get their combinations as a multiindex
                mi = pd.MultiIndex.from_product([dates, lat, lon], names=['start_date', 'lat', 'lon'])
                df = df.set_index(['lat', 'lon'], append=True).reindex(mi)
                df = df.reset_index().set_index('start_date')
                del mi

            if spatial is not None:
                print(f"-Combining {feature} with {', '.join(list(spatial.columns)[2:])}")
                spatial = spatial.merge(df, left_on=['start_date', 'lat', 'lon'], right_on=['start_date', 'lat', 'lon'], how='inner')
            else:
                spatial = df.copy()
            # delete dataframe
            del df
            print(spatial.shape)

        spatial = spatial.sort_values(by=['start_date', 'lat', 'lon'])            

        if save:
            print("\n-Saving spatial dataframe")
            print(spatial.head())
            spatial.to_hdf(rodeo_spatial_file_name, key="data")

    return spatial   

def get_train_data(target_months, 
                    horizon, 
                    lookback, 
                    spatial_features, 
                    temporal_features, 
                    target_feature, 
                    split, 
                    dataset):

    if dataset == 'IMDAA':
        spatial = get_imdaa_spatial_dataframe()
    elif dataset == 'RODEO':
        spatial = get_rodeo_spatial_dataframe()

    temporal = get_temporal_dataframe()
    temporal = temporal[temporal_features]

    # Validating the split
    split[0] = max(spatial.index.min().year, split[0])
    split[-1] = min(spatial.index.max().year, split[-1])
    train_years, validation_years, test_years = lookback.split_years(split=split)
    
    ground_truth = spatial[['lat', 'lon', target_feature]]

    spatial_train = spatial[train_years][spatial_features]
    #normalize
    spatial_dates = spatial.index.unique()
    spatial_days = spatial_train.groupby(spatial_train.index.day)
    spatial_mean = spatial_days.mean().reindex(spatial_dates.day, method='nearest').set_index(spatial_dates)
    spatial_std = spatial_days.std().reindex(spatial_dates.day, method='nearest').set_index(spatial_dates)
    
    spatial = spatial.fillna(spatial_mean)
    spatial[spatial_features] = (spatial[spatial_features] - spatial_mean[spatial_features]) / spatial_std[spatial_features]

    # Some inputs are completely nan making std=0.0, fill with mean of whole dataset
    spatial = spatial.fillna(spatial_mean.mean())

    spatial_grid_shape = (spatial['lat'].nunique(), spatial['lon'].nunique())

    common_lat_lon = ground_truth.dropna().groupby(['lat', 'lon']).count().index
    target_mask = ground_truth.groupby(['lat', 'lon']).count().index.isin(common_lat_lon)
    
    ground_truth = ground_truth[pd.MultiIndex.from_frame(ground_truth[['lat', 'lon']]).isin(common_lat_lon)]
    ground_truth = ground_truth.fillna(spatial_mean)

    #normalize
    ground_truth[target_feature] = (ground_truth[target_feature] - spatial_mean[target_feature])/spatial_std[target_feature]
    
    # Some inputs are completely nan making std=0.0, fill with mean of whole dataset
    ground_truth = ground_truth.fillna(spatial_mean.mean())
    
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
    target['mean'] = spatial_mean
    target['std'] = spatial_std

    spatial = spatial[spatial_features]

    return spatial, temporal, target, spatial_grid_shape
    

def get_forecast_rodeo_data():
    spatial_file_name = "lat_lon_date_data-contest_precip_34w.h5"
    temporal_file_name = "date_data-contest_precip_34w.h5"

    spatial = get_imdaa_spatial_dataframe(average=True)
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
    
    spatial.to_hdf(os.path.join(config.RODEO_DIR, spatial_file_name), key="data")

    temporal.to_hdf(os.path.join(config.RODEO_DIR, temporal_file_name), key="data")

    return spatial, temporal

if __name__ == '__main__':
    get_spatial_dataframe()




