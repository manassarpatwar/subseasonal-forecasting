#!/bin/bash
# Request 2 gigabytes of real memory (RAM)
#SBATCH --mem=2G
# Request 1 core
#SBATCH -c 1
# Email notifications to me@somedomain.com
#SBATCH --mail-user=mmsarpatwar1@sheffield.ac.uk
# Email notifications if the job fails
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name=ncmrwf-daily-download
#SBATCH -o ncmrwf-daily.out
# BATCH --account=dcs-acad3
# BATCH --partition=dcs-acad


cd data/ncmrwf

source download.sh
