import pandas as pd

def load_column_names(filepath='data/processed/processed_features_1980-2019.h5'):
    df = pd.read_hdf(filepath)
    columns = df.columns.tolist()
    del df
    return columns

def load_locations(filepath='data/target_points.csv'):
    locations_df = pd.read_csv(filepath)
    locations_df['region_id'] = list(zip(locations_df['lat'], locations_df['lon']))
    return locations_df

def load_standardizers(rootdir='data/standardization/'):
    feature_means = np.array(np.load(rootdir + 'feature_means.npy'))
    feature_stds = np.array(np.load(rootdir + 'feature_stds.npy'))
    feature_scaler = joblib.load(rootdir + 'all_feature_scaler.pkl', "r")
    prec_scaler = joblib.load(rootdir + 'prec_scaler.pkl', "r")
    tmp_scaler = joblib.load(rootdir + 'temp_scaler.pkl', "r")
    return feature_means, feature_stds, feature_scaler, prec_scaler, tmp_scaler