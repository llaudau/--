#!/bin/bash
#SBATCH --partition=thcp1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=hmc_n1
#SBATCH --output=/home/khw/Documents/Git_repository/hmc/log/hmc_n1_%j.log

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PATH=/usr/local/ompi/bin:$PATH

cd /home/khw/Documents/Git_repository/hmc

srun ./build/hmc_run \
    --nodes 1 --Lz 8 --Lt 32 \
    --omp 8
