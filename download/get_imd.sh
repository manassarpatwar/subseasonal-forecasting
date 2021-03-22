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

start=1980
end=2019
for ((i = $start; i <= $end; i++))
do
	curl -X POST -F "RF25=$i" https://www.imdpune.gov.in/Clim_Pred_LRF_New/RF25.php -o data/imd/imd_rf_"$i"0101-"$i"1231.nc
done

