
import os

IMDAA = {
    "spatial-features": {"rf": {"file_name": "imd_rf-1x1.nc", "time": "time", "name": "rf"}, 
                         "temp": {"file_name": "imd_temp-1x1.nc", "time": "time", "name": "t"},
                         "pres": {"file_name": "imdaa_reanl_PRES-sfc-1x1.nc", "time": "time", "name": "PRES_sfc"}, 
                         "slp": {"file_name": "imdaa_reanl_PRMSL-msl-1x1.nc", "time": "time", "name": "PRMSL_msl"}, 
                         "rhum": {"file_name": "imdaa_reanl_RH-2m-1x1.nc", "time": "time", "name": "RH_2m"}
    },
    "temporal-features": {"mjo": {"file_name": "mjo.h5"}, 
                         "mei": {"file_name": "mei.h5"},
                         "iod": {"file_name": "iod.h5"}
    }
}

RODEO = {
    "spatial-features": {"temp": {"file_name": "gt-contest_tmp2m-14d-1979-2018.h5", "name": "tmp2m"}}
}



DATA_DIR = "data"
INTERPOLATED_DATA_DIR = os.path.join(DATA_DIR, "interpolated")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
DATAFRAME_DIR = os.path.join(DATA_DIR, "dataframes")
RODEO_DIR = os.path.join(DATA_DIR, "rodeo")