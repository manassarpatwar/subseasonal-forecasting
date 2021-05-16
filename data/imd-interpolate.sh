
cdo remapbil,targetgrid imd/imd_temp.nc interpolated/imd_temp-1x1.nc
# Calculate rolling 14 day mean
RUNSTAT_DATE='last' cdo runmean,14 interpolated/imd_temp-1x1.nc interpolated/imd_temp-1x1-14d.nc

cdo remapbil,targetgrid imd/imd_rf.nc interpolated/imd_rf-1x1.nc
# Calculate rolling 14 day sum
RUNSTAT_DATE='last' cdo runsum,14 interpolated/imd_rf-1x1.nc interpolated/imd_rf-1x1-14d.nc