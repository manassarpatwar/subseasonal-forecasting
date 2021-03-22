#!/bin/bash
# Request 32 gigabytes of real memory (RAM)
#SBATCH --mem=32G
# Request 8 core
#SBATCH -c 8
# Maximum runtime of 1 days
#SBATCH -t 1-00:00:00
# Email notifications to me@somedomain.com
#SBATCH --mail-user=mmsarpatwar1@sheffield.ac.uk
# Email notifications if the job fails
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name=data-preprocessing
#SBATCH -o data-preprocessing.out
#SBATCH --account=dcs-acad3
#SBATCH --partition=dcs-acad

module load Anaconda3/2019.07

source activate dissertation

python predict.py
