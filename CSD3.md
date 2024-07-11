## Running on CSD3 A100 GPU

Obtain a GPU A100 node to build the spack environment using `salloc`

Install the spack environment in `hypre-cuda.yaml`

```
spack env create hypre-cuda hypre-cuda.yaml
spack env activate hypre-cuda
```

Once `spack install` has completed successfully, you can build the examples in the `examples` folder using cmake.

Here is a suitable setup script
```
source spack/share/spack/setup-env.sh
spack env activate hypre-cuda
module use /usr/local/software/spack/spack-modules/a100-20221118/linux-rocky8-zen3
module load gcc/11.3.0/gcc/i4xnp7h5
module load openmpi/4.1.4/gcc/6kaxusn4
spack load cuda
spack load cmake
```

# Build an example
```
cd pmg-dolfinx/examples/pmg
mkdir build
cd build
cmake ..
make
```

# Run an example using 4 GPUs
```
mpirun -n 4 ./pmg --ndofs=50000
```
