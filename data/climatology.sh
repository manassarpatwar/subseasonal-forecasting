#!/bin/bash
# Author: Manas Sarpatwar
# Date: 19/05/2021

cdo selyear,1981/2010 interpolated/imd_rf-1x1-14d.nc climatology/imdaa-precip.nc
cdo ydaymean climatology/imdaa-precip.nc climatology/imdaa-precip-1981-2010.nc
rm climatology/imdaa-precip.nc

cdo selyear,1981/2010 interpolated/imd_temp-1x1-14d.nc climatology/imdaa-tmp2m.nc
cdo ydaymean climatology/imdaa-tmp2m.nc climatology/imdaa-tmp2m-1981-2010.nc
rm climatology/imdaa-tmp2m.nc