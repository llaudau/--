#!/bin/bash
#SBATCH --partition=thcp1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=hmc_n2
#SBATCH --output=/thfs1/home/fengxu/wkh/hmc/log/hmc_n2_%j.log

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PATH=/usr/local/ompi/bin:$PATH

cd /thfs1/home/fengxu/wkh/hmc

srun ./build/hmc_run \
    --nodes 2 \
    --omp 8
