#!/bin/bash
#SBATCH --partition=thcp1
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=hmc_Lt32
#SBATCH --output=/thfs1/home/fengxu/wkh/hmc/log/hmc_Lt32_%j.log

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PATH=/usr/local/ompi/bin:$PATH

cd /thfs1/home/fengxu/wkh/hmc

srun ./build/hmc_run \
    --Lt 32 \
    --omp 8
