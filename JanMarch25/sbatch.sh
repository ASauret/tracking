#!/bin/sh
#SBATCH --job-name pyAT
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 2G
#SBATCH --time 24:00:00
#SBATCH --array 1-225
#SBATCH --partition asd
/machfs/swhite/pyenvs/python38_rnice/bin/python $(sed -n ${SLURM_ARRAY_TASK_ID}p fparam)
