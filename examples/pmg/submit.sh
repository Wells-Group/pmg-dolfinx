#!/bin/bash -l
#SBATCH --exclusive
#SBATCH --job-name=examplejob           # Job name
##SBATCH -o %x-j%j.out
#SBATCH --nodes=1                      # Total number of nodes
#SBATCH --ntasks-per-node=8
## SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:8
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00                  # Run time (d-hh:mm:ss)


export MPICH_GPU_SUPPORT_ENABLED=1

echo "Starting job $SLURM_JOB_ID at `date`"

ulimit -c unlimited
ulimit -s unlimited

gpu_bind=../select_gpu.sh
cpu_bind="--cpu-bind=map_cpu:49,57,17,23,1,9,33,41"



export AMD_LOG_LEVEL=0
export MPICH_GPU_SUPPORT_ENABLED=1
cd build
# make -j8
srun -N ${SLURM_NNODES} -n ${SLURM_NTASKS} ${cpu_bind} ${gpu_bind} ./pmg --ndofs 500000
