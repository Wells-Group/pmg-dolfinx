# LUMI-G environment

export SLURM_ACCOUNT=project_465000356
export SLURM_PARTITION=standard-g

export SBATCH_ACCOUNT=${SLURM_ACCOUNT}
export SBATCH_PARTITION=${SLURM_PARTITION}

export SALLOC_ACCOUNT=${SLURM_ACCOUNT}
export SALLOC_PARTITION=${SLURM_PARTITION}

module load PrgEnv-gnu
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load rocm


export SPACK_DIR=/scratch/project_465000356/adrianj/spack
source $SPACK_DIR/share/spack/setup-env.sh
spack env activate fenicsx-gpu-env
spack load fenics-dolfinx


export MPICH_OFI_NIC_POLICY=NUMA

export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"
export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)

export CXX=hipcc