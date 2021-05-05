#!/bin/bash
# Request 8 cores
#SBATCH -c 8
# Request 32 gigabytes of real memory (RAM)
#SBATCH --mem=32G
# Request 1 node
#SBATCH --nodes=1 
# Request 1 gpu
#SBATCH --gpus-per-node=1
# Maximum runtime of 2 days
#SBATCH -t 2-00:00:00
# Email notifications to me@somedomain.com
#SBATCH --mail-user=mmsarpatwar1@sheffield.ac.uk
# Email notifications if the job fails
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --job-name=wandb-sweep-temp
#SBATCH -o wandb-sweep-temp.out
#SBATCH --account=dcs-acad3
#SBATCH --partition=dcs-acad

module load cuDNN/7.6.4.38-gcccuda-2019b;
module load Anaconda3/2019.07
source activate dissertation

wandb agent aca18mms/dissertation/iyvq8nv3