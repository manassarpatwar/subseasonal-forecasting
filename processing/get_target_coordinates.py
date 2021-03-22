import netCDF4 as nc
import numpy as np
import pandas as pd

data_dir = 'data'
data = nc.Dataset(f"{data_dir}/interpolated/imd_rf-1x1.nc")
mask = np.logical_not(data.variables['RAINFALL'][0].mask)
target = np.argwhere(mask)
lat = data.variables['lat'][:].data.astype(int)
lon = data.variables['lon'][:].data.astype(int)
print(f"Min lat: {np.min(lat)} Max lat: {np.max(lat)} Min lon: {np.min(lon)} Max lon: {np.max(lon)}")
target_coordinates = np.column_stack((lat[target[:, 0]], lon[target[:, 1]]))
df = pd.DataFrame(target_coordinates)
df.columns = ['lat', 'lon']
df.to_csv(f"{data_dir}/target_points.csv", index=None)
