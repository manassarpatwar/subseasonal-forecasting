#!/bin/bash
# Author: Manas Sarpatwar
# Date: 19/05/2021

start=1979
end=2019

rf(){
    curl -sX POST -F "rain=$1" https://www.imdpune.gov.in/Clim_Pred_LRF_New/rainfall.php -o data/imd/imd_rainfall_"$1"0101-"$1"1231.GRD
    python download/create_ctl.py rf $1 imd_rainfall_"$1"0101-"$1"1231
    cdo -f nc import_binary data/imd/imd_rainfall_"$1"0101-"$1"1231.ctl data/imd/imd_rainfall_"$1"0101-"$1"1231.nc
    rm data/imd/imd_rainfall_"$1"0101-"$1"1231.ctl
    rm data/imd/imd_rainfall_"$1"0101-"$1"1231.GRD
}

for ((i = $start; i <= $end; i++))
do
    rf $i &
done
wait

cdo mergetime data/imd/imd_rainfall_*.nc data/imd/imd_rf.nc
rm data/imd/imd_rainfall_*.nc

maxtemp(){
    curl -sX POST -F "maxtemp=$1" https://www.imdpune.gov.in/Clim_Pred_LRF_New/maxtemp.php -o data/imd/imd_maxtemp_"$1"0101-"$1"1231.GRD
    python download/create_ctl.py temp $1 imd_maxtemp_"$1"0101-"$1"1231
    cdo -f nc import_binary data/imd/imd_maxtemp_"$1"0101-"$1"1231.ctl data/imd/imd_maxtemp_"$1"0101-"$1"1231.nc
    rm data/imd/imd_maxtemp_"$1"0101-"$1"1231.ctl
    rm data/imd/imd_maxtemp_"$1"0101-"$1"1231.GRD
}

for ((i = $start; i <= $end; i++))
do
	maxtemp $i &
done
wait

cdo mergetime data/imd/imd_maxtemp_*.nc data/imd/imd_maxtemp.nc
rm data/imd/imd_maxtemp_*.nc

mintemp(){
    curl -sX POST -F "mintemp=$1" https://www.imdpune.gov.in/Clim_Pred_LRF_New/mintemp.php -o data/imd/imd_mintemp_"$1"0101-"$1"1231.GRD
    python download/create_ctl.py temp $1 imd_mintemp_"$1"0101-"$1"1231
    cdo -f nc import_binary data/imd/imd_mintemp_"$1"0101-"$1"1231.ctl data/imd/imd_mintemp_"$1"0101-"$1"1231.nc
    rm data/imd/imd_mintemp_"$1"0101-"$1"1231.ctl
    rm data/imd/imd_mintemp_"$1"0101-"$1"1231.GRD
}

for ((i = $start; i <= $end; i++))
do
	mintemp $i &
done
wait

cdo mergetime data/imd/imd_mintemp_*.nc data/imd/imd_mintemp.nc
rm data/imd/imd_mintemp_*.nc

cdo ensmean data/imd/imd_maxtemp.nc data/imd/imd_mintemp.nc data/imd/imd_temp.nc
