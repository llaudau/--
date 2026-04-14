#!/bin/bash
#SBATCH --partition=thcp1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=hmc_lzl4
#SBATCH --output=/home/khw/Documents/Git_repository/hmc/log/hmc_lzl4_%j.log

export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PATH=/usr/local/ompi/bin:$PATH

cd /home/khw/Documents/Git_repository/hmc

srun ./build/hmc_run \
    --lz_local 4 \
    --omp 16
