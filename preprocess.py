import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
import os
from sklearn.preprocessing import MinMaxScaler
from utils import Lookback, dataset_data
import config
from config import IMDAA, RODEO
import argparse


def get_imdaa_feature(feature, avg=True):
    data = IMDAA['spatial_features'][feature]
    feature_file_name = os.path.join(config.DATAFRAME_DIR, f"{feature}{'-14d' if avg else ''}.h5")
    if os.path.isfile(feature_file_name):
        print(f"-Using saved converted {feature_file_name}")
        df = pd.read_hdf(feature_file_name, key="data")
    else:
        print(f"-Converting {feature}")
        d = xr.open_dataset(os.path.join(config.INTERPOLATED_DATA_DIR, f"{data['file_name']}{'-14d' if avg else ''}.nc"))
        # Delete time_bnds index and column
        if 'time_bnds' in d.variables:
            d = d.drop('time_bnds')
            
        df = d.to_dataframe()
        del d
        df = df.reset_index()

        # Rename variable
        df = df.rename(columns={data['time']: 'start_date', data['name']: feature})
        df = df.drop_duplicates()

        # Format start_date to datetime column  
        df['start_date'] = pd.to_datetime(df['start_date']).dt.date.astype('datetime64')

        df = df.set_index('start_date')
        df = df.sort_values(['start_date',  'lat', 'lon'])
        
        df.to_hdf(feature_file_name, key="data")

    if 'climatology' in data:
        climatology_file_name = os.path.join(config.CLIMATOLOGY_DIR, data['climatology'])
        if not os.path.isfile(f"{climatology_file_name}.h5"):
            print(f"-Converting climatology for {feature}")
            c = xr.open_dataset(f"{climatology_file_name}.nc")
            # climatology df
            cdf = c.to_dataframe()
            cdf = cdf.dropna().reset_index()
            cdf = cdf.rename(columns={data['time']: 'start_date', data['name']: feature})

            cdf.to_hdf(f"{climatology_file_name}.h5", key="data")

    return df

def get_merged_imdaa(merge=False, save=True, avg=True):
    imdaa_spatial_file_name = os.path.join(config.MERGED_DATA_DIR, f"imdaa-spatial{'-14d' if avg else ''}.h5")
    if not merge and os.path.isfile(imdaa_spatial_file_name):
        print("-Loading saved merged spatial dataframe")
        spatial = pd.read_hdf(imdaa_spatial_file_name, key="data")
    else:
        print("-Converting .nc to dataframes and merging")
        spatial = None
        for feature in IMDAA['spatial_features']:
            df = get_imdaa_feature(feature=feature, avg=avg)
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
            print(f"-Saving merged imdaa spatial dataframe to {imdaa_spatial_file_name}")
            print(spatial.head())
            spatial.to_hdf(imdaa_spatial_file_name, key="data")
    return spatial

def get_imdaa(process=False, save=True, avg=True, merge=False, calculate=False):
    imdaa_spatial_file_name = os.path.join(config.PROCESSED_DATA_DIR, f"imdaa-spatial{'-14d' if avg else ''}.h5")

    if not process and os.path.isfile(imdaa_spatial_file_name):
        print("-Loading saved spatial dataframe")
        spatial = pd.read_hdf(imdaa_spatial_file_name, key="data")
    else:
        spatial = get_merged_imdaa(avg=avg, merge=merge)         

        split = IMDAA['split']
        train = spatial.loc[str(split[0]):str(split[1])]

        daymean = spatial.groupby(spatial.index).mean().reindex(spatial.index, method='nearest')
        mean = train.groupby(train.index.dayofyear).mean().reindex(spatial.index.dayofyear, method='nearest').set_index(spatial.index)
        std = train.groupby(train.index.dayofyear).std().reindex(spatial.index.dayofyear, method='nearest').set_index(spatial.index)

        print("-Preprocessing target features")
        target_features = IMDAA['target_features']
        process_target_features(dataset='IMDAA', spatial=spatial, target_features=target_features, mean=daymean, calculate=calculate)
        
        for feature in IMDAA['spatial_features']:
            spatial[feature] = (spatial[feature]-mean[feature])/std[feature]
            spatial[feature] = spatial[feature].fillna(0.0)

        if save:
            print("-Saving processed imdaa spatial dataframe")
            print(spatial.head())
            spatial.to_hdf(imdaa_spatial_file_name, key="data")

    return spatial

def get_merged_rodeo(merge=False, save=True, pad=True, res=(1.0,1.0), sg=0):
    # Make sure spatial granularity is an integer
    assert isinstance(sg, int)

    rodeo_spatial_file_name = os.path.join(config.MERGED_DATA_DIR, "rodeo-spatial.h5")
    if not merge and os.path.isfile(rodeo_spatial_file_name):
        print("-Loading saved merged rodeo spatial dataframe")
        spatial = pd.read_hdf(rodeo_spatial_file_name, key="data")
    else:
        print("-Merging rodeo dataframes")
        spatial = None
        for feature, data in RODEO['spatial_features'].items():
            df = pd.read_hdf(os.path.join(config.RODEO_DIR, data['file_name']))
            df = df.reset_index()
            df = df[['start_date', 'lat', 'lon', data['name']]]
            df = df.rename(columns={data['name']: feature})
            df = df.set_index('start_date')

            if spatial is not None:
                print(f"-Combining {feature} with {', '.join(list(spatial.columns)[2:])}")
                spatial = spatial.merge(df, left_on=['start_date', 'lat', 'lon'], right_on=['start_date', 'lat', 'lon'], how='inner')
            else:
                spatial = df.copy()
            # delete dataframe
            del df
            print(spatial.shape)

        spatial = spatial.sort_values(by=['start_date', 'lat', 'lon'])            

        if pad:
            print(f"-Padding dataframe")
            # Get all latitude values from minimum to maximum 
            lat = np.arange(spatial['lat'].min()-float(sg*res[0]), spatial['lat'].max()+float((sg+1)*res[0]), float(res[0]))
            # Get all longitude values from minimum to maximum 
            lon = np.arange(spatial['lon'].min()-float(sg*res[1]), spatial['lon'].max()+float((sg+1)*res[1]), float(res[1]))
            # Unique dates
            dates = spatial.index.unique()
            # Get their combinations as a multiindex
            mi = pd.MultiIndex.from_product([dates, lat, lon], names=['start_date', 'lat', 'lon'])
            del dates
            spatial = spatial.set_index(['lat', 'lon'], append=True).reindex(mi)
            spatial = spatial.reset_index().set_index('start_date')
            del mi
        
        if save:
            print(f"-Saving merged rodeo spatial dataframe to {rodeo_spatial_file_name}")
            print(spatial.head())
            spatial.to_hdf(rodeo_spatial_file_name, key="data")

    return spatial    
        
def get_rodeo(process=False, save=True, merge=False, calculate=False):
    rodeo_spatial_file_name = os.path.join(config.PROCESSED_DATA_DIR, "rodeo-spatial.h5")
    if not process and os.path.isfile(rodeo_spatial_file_name):
        print("-Loading saved rodeo spatial dataframe")
        spatial = pd.read_hdf(rodeo_spatial_file_name, key="data")
    else:
        spatial = get_merged_rodeo(merge=merge)
        split = RODEO['split']
        train = spatial.loc[str(split[0]):str(split[1])]
            
        daymean = spatial.groupby(spatial.index).mean().reindex(spatial.index, method='nearest')

        mean = train.groupby(train.index.dayofyear).mean().reindex(spatial.index.dayofyear, method='nearest').set_index(spatial.index)
        std = train.groupby(train.index.dayofyear).std().reindex(spatial.index.dayofyear, method='nearest').set_index(spatial.index)

        print("-Preprocessing target features")
        target_features = RODEO['target_features']
        process_target_features(dataset='RODEO', spatial=spatial, target_features=target_features, mean=daymean, calculate=calculate)

        for feature in RODEO['spatial_features']:
            spatial[feature] = (spatial[feature]-mean[feature])/std[feature]
            spatial[feature] = spatial[feature].fillna(0.0)  

        if save:
            print("-Saving processed rodeo spatial dataframe")
            print(spatial.head())
            spatial.to_hdf(rodeo_spatial_file_name, key="data")

    return spatial  

def get_climatology(dataset, target_feature, calculate=False, save=True):
    climatology_years = config.CLIMATOLOGY_YEARS
    processed_climatology_file_name = os.path.join(config.PROCESSED_DATA_DIR, 'climatology', f"{dataset.lower()}-{target_feature}-{climatology_years[0]}-{climatology_years[1]}.h5")
    if not calculate and os.path.isfile(processed_climatology_file_name):
        print(f"-Loading saved climatology for {dataset} {target_feature} from {processed_climatology_file_name}")
        df = pd.read_hdf(processed_climatology_file_name, key="data")
    else:
        df = get_merged_spatial_dataframe(dataset=dataset)
        features = df.columns
        df = df.reset_index()
        df['d'] = df['start_date'].dt.day
        df['m'] = df['start_date'].dt.month
        df = df[['start_date', 'd', 'm', 'lat', 'lon']]
        
        climatology_file_name = f"{dataset_data(dataset)['spatial_features'][target_feature]['climatology']}"
        climatology_path = os.path.join(config.CLIMATOLOGY_DIR, climatology_file_name)
        climatology = pd.read_hdf(f"{climatology_path}.h5")
        climatology['d'] = climatology['start_date'].dt.day
        climatology['m'] = climatology['start_date'].dt.month
        climatology = climatology.drop('start_date', axis='columns')

        df = df.merge(climatology, left_on=['d', 'm', 'lat', 'lon'], right_on=['d', 'm', 'lat','lon'], how='left')
        df = df.drop(['d', 'm'], axis='columns')
        df = df.set_index('start_date')

        if save:
            print(f"-Saving climatology to {processed_climatology_file_name}")
            df.to_hdf(processed_climatology_file_name, key="data")

    return df

def process_target_features(dataset, spatial, target_features, mean, calculate=False):
    for feature in target_features:
        climatology = get_climatology(dataset=dataset, target_feature=feature, calculate=calculate)
        spatial[f"{feature}_target"] = np.nan
        spatial[f"{feature}_clim"] = np.nan
        spatial[f"{feature}_anom"] = np.nan

        common_lat_lon = spatial[['lat', 'lon', feature]].dropna().groupby(['lat', 'lon']).count().index
        target_locations = pd.MultiIndex.from_frame(spatial[['lat', 'lon']]).isin(common_lat_lon)
        spatial.loc[target_locations, f"{feature}_target"] = 1.0
        spatial.loc[target_locations, f"{feature}_clim"] = climatology.loc[target_locations, feature]
        spatial.loc[target_locations, feature] = spatial.loc[target_locations, feature].fillna(mean.loc[target_locations, feature])
        spatial.loc[target_locations, f"{feature}_anom"] = spatial.loc[target_locations, feature] - spatial.loc[target_locations, f"{feature}_clim"]

def get_merged_spatial_dataframe(dataset, avg=True, merge=False):
    if isinstance(dataset, str):
        dataset = dataset.upper()

    if dataset == 'IMDAA':
        spatial = get_merged_imdaa(avg=avg, merge=merge)
    elif dataset == 'RODEO':
        spatial = get_merged_rodeo(merge=merge)
    
    return spatial

def get_spatial_dataframe(dataset, avg=True, process=False, merge=False):
    if isinstance(dataset, str):
        dataset = dataset.upper()

    if dataset == 'IMDAA':
        spatial = get_imdaa(avg=avg, process=process, merge=merge)
    elif dataset == 'RODEO':
        spatial = get_rodeo(process=process)
    
    return spatial

def get_temporal_dataframe(merge=False, save=True):
    temporal_file_name = os.path.join(config.PROCESSED_DATA_DIR, "temporal.h5")
    if not merge and os.path.isfile(temporal_file_name):
        print("-Loading saved temporal dataframe")
        temporal = pd.read_hdf(temporal_file_name, key="data")
    else:
        print("-Merging dataframes")
        temporal = None
        for feature, data in IMDAA['temporal_features'].items():
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
            print("-Saving temporal dataframe")
            print(temporal.head())
            temporal.to_hdf(temporal_file_name, key="data")

    return temporal

def get_train_data(target_months, 
                    horizon, 
                    lookback, 
                    spatial_features, 
                    temporal_features, 
                    target_feature, 
                    dataset):
    spatial = get_spatial_dataframe(dataset=dataset)
    temporal = get_temporal_dataframe()
    temporal = temporal[temporal_features]

    # Validating the split
    split = dataset_data(dataset)['split']
    years = split.copy()
    years[0] = max(spatial.index.min(), pd.Timestamp(str(years[0]))).date()
    years[-1] = min(spatial.index.max(), pd.Timestamp(str(years[-1]))).date()
    train_years, validation_years, test_years = lookback.split_years(split=years, horizon=horizon)
    test_years = test_years[test_years.month.isin(target_months)]

    spatial_grid_shape = (spatial['lat'].nunique(), spatial['lon'].nunique())
    ground_truth = spatial[['lat', 'lon', f"{target_feature}_anom", f"{target_feature}_target"]]
    anomalies = ground_truth.dropna().drop(f"{target_feature}_target", axis='columns')

    print("-Getting target locations")
    common_lat_lon = anomalies.groupby(['lat', 'lon']).count().index
    target_mask = ground_truth.groupby(['lat', 'lon']).count().index.isin(common_lat_lon)
    
    target_shape = anomalies.groupby(['lat','lon']).ngroups
    anomalies = anomalies[anomalies.index.month.isin(target_months)]
    anomalies = anomalies[f"{target_feature}_anom"]

    min_target_date = lookback.min_target_date(anomalies.index.min(), train_years.start, target_months[0], horizon)
    target_train_years = slice(min_target_date, train_years.stop)

    target = {}
    target['train'] = anomalies.loc[target_train_years]
    target['validation'] = anomalies.loc[validation_years]
    target['test'] =  anomalies.loc[test_years]
    target['mask'] = target_mask
    target['shape'] = target_shape
    target['locations'] = common_lat_lon

    return spatial, temporal, target, spatial_grid_shape
    

def imdaa_to_rodeo(target_feature):
    imdaa_spatial_file_name = f"lat_lon_date_data-contest_{target_feature}_34w.h5"
    temporal_file_name = f"date_data-contest_{target_feature}_34w.h5"

    spatial = get_spatial_dataframe('IMDAA')
    temporal = get_temporal_dataframe()
    climatology = get_climatology(dataset='IMDAA', target_feature=target_feature, spatial=spatial)
    spatial['clim'] = climatology[target_feature]
    spatial['anom'] = spatial[target_feature] - spatial['clim']

    gt = spatial[['lat', 'lon', target_feature]]
    common_lat_lon = gt.dropna().groupby(['lat', 'lon']).count().index
    spatial = spatial[pd.MultiIndex.from_frame(gt[['lat', 'lon']]).isin(common_lat_lon)]

    spatial = spatial.fillna(spatial.groupby(spatial.index).mean())
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imdaa', help='Preprocess IMDAA', action='store_true')
    parser.add_argument('-r', '--rodeo',  help='Preprocess RODEO', action='store_true')
    parser.add_argument('-m', '--merge', help='Force merge', action='store_true')
    parser.add_argument('-c', '--climatology', help='Force calculate climatology', action='store_true')
    parser.add_argument('-p', '--process', help='Force process both datasets', action='store_true')
    args = parser.parse_args()

    merge = bool(args.merge) or bool(args.process)
    calculate = bool(args.climatology) or bool(args.process)
    
    if args.imdaa:
        get_imdaa(process=True, merge=merge, calculate=calculate)
    
    if args.rodeo:
        get_rodeo(process=True, merge=merge, calculate=calculate)