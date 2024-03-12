. /scratch/project_465000633/spack/share/spack/setup-env.sh
spack env activate fenics-main-gpu

module load PrgEnv-aocc
module load rocm

export HIPCC_COMPILE_FLAGS_APPEND="--offload-arch=gfx90a $(CC --cray-print-opts=cflags)"
export HIPCC_LINK_FLAGS_APPEND=$(CC --cray-print-opts=libs)

export CXX=hipcc

