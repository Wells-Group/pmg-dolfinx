
## Running on LUMI AMD GPU

Install the spack environment in `hypre-rocm-lumi.yaml`

```
spack env create hypre-rocm hypre-rocm-lumi.yaml
spack env activate hypre-rocm
```

N.B. to install fenics-basix, you might need to patch spack slightly:
`spack edit fenics-basix` and add the following:

```
    def cmake_args(self):
        options = []
        options.append("-DBLAS_LIBRARIES=" + self.spec["blas"].libs.joined())
        options.append("-DLAPACK_LIBRARIES=" + self.spec["blas"].libs.joined())
        lapack_libs = self.spec['lapack'].libs.joined(';')
        blas_libs = self.spec['blas'].libs.joined(';')
        return options
```

Once `spack install` has completed successfully, you can build the examples in the `examples` folder using cmake.

```
# Setup required for "build" on login node
module load LUMI/23.09
module load partition/G
module load rocm
module load gcc
source spack/share/spack/setup-env.sh
spack env activate hypre-rocm

# Build an example
cd pmg-dolfinx/examples/pmg
mkdir build
cd build
CXX=hipcc cmake ..
make
```

Get a GPU node:

```
salloc --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --time=01:0:00 --partition=dev-g --account=ACCOUNT
export MPICH_GPU_SUPPORT_ENABLED=1
srun --ntasks=1 ./pmg --ndofs=50000
```

There are some instructions on running on GPU on the LUMI documentation, especially about [selecting GPU/CPU affinity](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/).