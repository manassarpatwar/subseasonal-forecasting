import joblib
from glob import glob
import numpy as np
import pandas as pd
import sklearn
import os
# import tensorflow as tf

# Processing feature dataset functions

# TODO sensibly name and place these files
# Loading functions
from utils.load_functions import load_locations
from utils.standardize import PreprocessTemporalSpatialData

# Prediction functions
from processing.interpolate_feature_data import run_interpolation_pipeline
# from predict.prediction_processing import process_tar_data, get_prediction_data, prepare_single_spatial_temporal_region, \
#     generate_all_region_input, PreprocessTemporalSpatialDataPrediction

# Get params #TODO make some of these user input OR based on update run/full run/check to see if they are already updated
# TODO put inside a constants file
params = {'process_feature_data': True}

if __name__ == "__main__":

    # -1.1- Process & Interpolate feature data
    if params['process_feature_data']:
        run_interpolation_pipeline()

    locations = load_locations()
    df = pd.read_hdf('data/processed/processed_features_1980-2019.h5') 
    y = PreprocessTemporalSpatialData(df, locations, max_sg=5)
    y.preprocess_pipeline()