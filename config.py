import os

DATA_DIR = "data"
INTERPOLATED_DATA_DIR = os.path.join(DATA_DIR, "interpolated")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MERGED_DATA_DIR = os.path.join(DATA_DIR, "merged")
DATAFRAME_DIR = os.path.join(DATA_DIR, "dataframes")
CLIMATOLOGY_DIR = os.path.join(DATA_DIR, "climatology")
VISUALISATION_DIR = "visualisations"
RODEO_DIR = os.path.join(DATA_DIR, "rodeo")
RESULTS_DIR = "results"
RESULTS_TABLE_DIR = os.path.join(RESULTS_DIR, "tables")

CLIMATOLOGY_YEARS = ['1981', '2010']

IMDAA = {
    "spatial_features": {"precip": {"file_name": "imd_rf-1x1", "time": "time", "name": "rf", "climatology": "imdaa-precip-1981-2010"}, 
                         "tmp2m": {"file_name": "imd_temp-1x1", "time": "time", "name": "t", "climatology": "imdaa-tmp2m-1981-2010"},
                         "pres": {"file_name": "imdaa_reanl_PRES-sfc-1x1", "time": "time", "name": "PRES_sfc"}, 
                         "slp": {"file_name": "imdaa_reanl_PRMSL-msl-1x1", "time": "time", "name": "PRMSL_msl"}, 
                         "rhum": {"file_name": "imdaa_reanl_RH-2m-1x1", "time": "time", "name": "RH_2m"}
    },
    "temporal_features": {"mjo": {"file_name": "mjo.h5"}, 
                         "mei": {"file_name": "mei.h5"},
                         "iod": {"file_name": "iod.h5"}
    },
    'split': [1979, 2011, 2017, 2020],
    'target_features': ['tmp2m', 'precip'],
    # Location source https://www.latlong.net/category/states-102-14.html
    'visualize_locations': {'Tamil Nadu': (11.0, 79.0),
                            'Telangana': (18.0, 78.0),
                            'Madhya Pradesh': (26.0, 78.0),
                            'Haryana': (29.0, 76.0),
                            'Chhattisgarh': (21.0, 82.0),
                            'Maharashtra': (20.0, 75.0),
                            'Tripura': (24.0, 92.0),
                            'Karnataka': (15.0, 76.0),
                            'Kerala': (11.0, 76.0),
                            'Uttar Pradesh': (28.0, 80.0),
                            'Assam': (26.0, 93.0),
                            'West Bengal': (23.0, 88.0),
                            'Gujarat': (22.0, 72.0),
                            'Odisha': (21.0, 85.0),
                            'Rajasthan': (27.0, 73.0),
                            'Himachal Pradesh': (32.0, 78.0)}
}

RODEO = {
    "spatial_features": {"tmp2m": {"file_name": "gt-contest_tmp2m-14d-1979-2018.h5", "name": "tmp2m", "climatology": "official_climatology-contest_tmp2m-1981-2010"},
                         "precip": {"file_name": "gt-contest_precip-14d-1948-2018.h5", "name": "precip", "climatology": "official_climatology-contest_precip-1981-2010"},
                         "pres": {"file_name": "gt-contest_pres.sfc.gauss-14d-1948-2018.h5", "name": "pres"}, 
                         "slp": {"file_name": "gt-contest_slp-14d-1948-2018.h5", "name": "slp"}, 
                         "rhum": {"file_name": "gt-contest_rhum.sig995-14d-1948-2018.h5", "name": "rhum"}},
    'split': [1979, 2011, 2017, 2019],
    'target_features': ['tmp2m', 'precip'],
    'visualize_locations': {'Texas': (31.0, 259.0),
                            'South Dakota': (44.0, 259.0),
                            'Oregon': (44.0, 238.0),
                            'Nebraska': (42.0, 259.0),
                            'Kansas': (38.0, 261.0),
                            'Nevada': (40.0, 242.0),
                            'North Dakota': (48.0, 259.0),
                            'Oklahoma': (36.0, 262.0),
                            'Montana': (47.0, 249.0),
                            'Washington State': (48.0, 238.0),
                            'Utah': (39.0, 247.0),
                            'Colorado': (39.0, 254.0),
                            'New Mexico': (34.0, 253.0),
                            'Arizona': (34.0, 248.0),
                            'California': (37.0, 240.0),
                            'Idaho': (44.0, 244.0),
                            'Wyoming': (43.0, 252.0)}
}