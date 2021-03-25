#!/bin/bash
# Request 1 gigabytes of real memory (RAM)
#SBATCH --mem=1G
# Request 1 core
#SBATCH -c 1
# Email notifications to me@somedomain.com
#SBATCH --mail-user=mmsarpatwar1@sheffield.ac.uk
# Email notifications if the job fails
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name=imd-daily-download
#SBATCH -o imd-daily.out

module load Anaconda3/2019.07
source activate dissertation

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

