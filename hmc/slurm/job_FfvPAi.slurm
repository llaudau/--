#!/bin/bash
#SBATCH --partition=thcp1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=hmc_Lt8
#SBATCH --output=/thfs1/home/fengxu/wkh/hmc/log/hmc_Lt8_%j.log

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PATH=/usr/local/ompi/bin:$PATH

cd /thfs1/home/fengxu/wkh/hmc

srun ./build/hmc_run \
    --Lx 4 --Ly 4 --Lz 4 --Lt 8 --beta 6.0 --thermal 5 --samples 3 --flow_t_max 1.0 \
    --omp 8
