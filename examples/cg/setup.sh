module load PrgEnv-gnu
module load craype-x86-trento
module load craype-accel-amd-gfx90a
module load rocm

export SPACK_DIR=/scratch/project_465000356/adrianj/spack
source $SPACK_DIR/share/spack/setup-env.sh
spack env activate fenicsx-gpu-env
spack load fenics-dolfinx


export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"
export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)

export CXX=hipcc

