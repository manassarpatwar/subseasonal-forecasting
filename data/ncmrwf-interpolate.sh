start=1979
end=2019
for ((i = $start; i <= $end; i++))
do
	cdo remapbil,targetgrid ncmrwf/ncum_imdaa_reanl_DY_$1_"$i"0101-"$i"1231.nc interpolated/imdaa_reanl_$1_$i-1x1.nc
done

cdo mergetime interpolated/imdaa_reanl_$1_*-1x1.nc interpolated/imdaa_reanl_$1-1x1.nc
rm interpolated/imdaa_reanl_$1_*-1x1.nc
# Calculate rolling 14 day mean
RUNSTAT_DATE='last' cdo runmean,14 interpolated/imdaa_reanl_$1-1x1.nc interpolated/imdaa_reanl_$1-1x1-14d.nc