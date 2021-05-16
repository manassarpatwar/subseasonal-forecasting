#!/bin/bash
# Request 1 core
#SBATCH -c 1
# Request 8 gigabytes of real memory (RAM)
#SBATCH --mem=8G
# Maximum runtime of 3 days
#SBATCH -t 3-00:00:00
# Email notifications to mmsarpatwar1@sheffield.ac.uk
#SBATCH --mail-user=mmsarpatwar1@sheffield.ac.uk
# Email notifications if the job fails
#SBATCH --mail-type=FAIL
#SBATCH --job-name=imdaa
# SBATCH --account=dcs-acad3
# SBATCH --partition=dcs-acad

module load Anaconda3/2019.07
source activate dissertation

wandb agent aca18mms/dissertation/zfre9gaw