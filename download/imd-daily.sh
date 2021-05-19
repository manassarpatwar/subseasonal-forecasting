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

source download/get_imd.sh

