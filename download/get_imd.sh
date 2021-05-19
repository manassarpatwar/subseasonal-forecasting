#!/bin/bash
# Author: Manas Sarpatwar
# Date: 19/05/2021

feature="rf" #rf/temp
variable="rainfall" #rainfall/maxtemp/mintemp
key="rain" #rain/maxtemp/mintemp
start=1979
end=2019
for ((i = $start; i <= $end; i++))
do
	curl -X POST -F "$key=$i" https://www.imdpune.gov.in/Clim_Pred_LRF_New/"$variable".php -o data/imd/imd_"$variable"_"$i"0101-"$i"1231.GRD
	python download/create_ctl.py $feature $i imd_"$variable"_"$i"0101-"$i"1231
	cdo -f nc import_binary data/imd/imd_"$variable"_"$i"0101-"$i"1231.ctl data/imd/imd_"$variable"_"$i"0101-"$i"1231.nc
	rm data/imd/imd_"$variable"_"$i"0101-"$i"1231.ctl
done

cdo mergetime data/imd/imd_"$variable"_*.nc data/imd/imd_"$variable".nc

