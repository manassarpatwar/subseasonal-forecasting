#!/bin/bash
# Author: Manas Sarpatwar
# Date: 19/05/2021

start=1979
end=2019

for ((i = $start; i <= $end; i++))
do
	cdo remapbil,targetgrid ncmrwf/ncum_imdaa_reanl_DY_PRES-sfc_"$i"0101-"$i"1231.nc interpolated/imdaa_reanl_PRES-sfc_$i-1x1.nc
done

cdo mergetime interpolated/imdaa_reanl_PRES-sfc_*-1x1.nc interpolated/imdaa_reanl_PRES-sfc-1x1.nc
rm interpolated/imdaa_reanl_PRES-sfc_*-1x1.nc
# Calculate rolling 14 day mean
RUNSTAT_DATE='last' cdo runmean,14 interpolated/imdaa_reanl_PRES-sfc-1x1.nc interpolated/imdaa_reanl_PRES-sfc-1x1-14d.nc


for ((i = $start; i <= $end; i++))
do
	cdo remapbil,targetgrid ncmrwf/ncum_imdaa_reanl_DY_PRMSL-msl_"$i"0101-"$i"1231.nc interpolated/imdaa_reanl_PRMSL-msl_$i-1x1.nc
done

cdo mergetime interpolated/imdaa_reanl_PRMSL-msl_*-1x1.nc interpolated/imdaa_reanl_PRMSL-msl-1x1.nc
rm interpolated/imdaa_reanl_PRMSL-msl_*-1x1.nc
# Calculate rolling 14 day mean
RUNSTAT_DATE='last' cdo runmean,14 interpolated/imdaa_reanl_PRMSL-msl-1x1.nc interpolated/imdaa_reanl_PRMSL-msl-1x1-14d.nc


for ((i = $start; i <= $end; i++))
do
	cdo remapbil,targetgrid ncmrwf/ncum_imdaa_reanl_DY_RH-2m_"$i"0101-"$i"1231.nc interpolated/imdaa_reanl_RH-2m_$i-1x1.nc
done

cdo mergetime interpolated/imdaa_reanl_RH-2m_*-1x1.nc interpolated/imdaa_reanl_RH-2m-1x1.nc
rm interpolated/imdaa_reanl_RH-2m_*-1x1.nc
# Calculate rolling 14 day mean
RUNSTAT_DATE='last' cdo runmean,14 interpolated/imdaa_reanl_RH-2m-1x1.nc interpolated/imdaa_reanl_RH-2m-1x1-14d.nc