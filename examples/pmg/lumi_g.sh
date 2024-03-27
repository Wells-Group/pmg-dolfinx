# LUMI-G environment

export SLURM_ACCOUNT=project_465000633
export SLURM_PARTITION=ju-standard-g

export SBATCH_ACCOUNT=${SLURM_ACCOUNT}
export SBATCH_PARTITION=${SLURM_PARTITION}

export SALLOC_ACCOUNT=${SLURM_ACCOUNT}
export SALLOC_PARTITION=${SLURM_PARTITION}

module load LUMI/23.09
module load partition/G
module load gcc
module load rocm


export SPACK_DIR=/scratch/project_465000633/crichard/spack
source $SPACK_DIR/share/spack/setup-env.sh

export MPICH_OFI_NIC_POLICY=NUMA
