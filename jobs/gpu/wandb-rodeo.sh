#!/bin/bash
# Request 1 core
#SBATCH -c 1
# Request 32 gigabytes of real memory (RAM)
#SBATCH --mem=16G
# Request 1 node
#SBATCH --nodes=1
# Request 1 gpu
#SBATCH --gpus-per-node=1
# Maximum runtime of 3 days
#SBATCH -t 3-00:00:00
# Email notifications to mmsarpatwar1@sheffield.ac.uk
#SBATCH --mail-user=mmsarpatwar1@sheffield.ac.uk
# Email notifications if the job fails
#SBATCH --mail-type=FAIL
#SBATCH --job-name=rodeo-gpu
# Run on acad gpu
# SBATCH --account=dcs-acad3
# SBATCH --partition=dcs-acad-pre
# Run on UoS gpu
#SBATCH --partition=gpu
#SBATCH --qos=gpu

module load cuDNN/7.6.4.38-gcccuda-2019b
module load Anaconda3/2019.07
source activate dissertation

wandb agent aca18mms/dissertation/xyzk1qj9