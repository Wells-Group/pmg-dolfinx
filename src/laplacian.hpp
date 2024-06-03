// Copyright (C) 2023 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "hip/hip_runtime.h"

/// Compute y = A * x where A is the stiffness operator
/// for a set of entities (cells or facets) in a mesh.
///
/// The stiffness operator is defined as:
///
///     A = ∑_i ∫_Ω C ∇ϕ_i ∇ϕ_j dx
///
/// where C is a constant, ϕ_i and ϕ_j are the basis functions of the
/// finite element space, and ∇ϕ_i is the gradient of the basis function.
/// The integral is computed over the domain Ω of the entity using sum 
/// factorization. The basis functions are defined on the reference element 
/// and are transformed to the physical element using the geometry operator G. 
/// G is a 3x3 matrix per quadrature point per entity.
///
/// @tparam P Polynomial degree of the basis functions
/// @tparam T Data type of the input and output arrays
/// @param x Input vector of size (ndofs,)
/// @param entity_constants Array of size (n_entities,) with the constant C for each entity
/// @param y Output vector of size (ndofs,)
/// @param G_entity Array of size (n_entities, nq, 6) with the geometry operator G for each entity
/// @param entity_dofmap Array of size (n_entities, ndofs) with the dofmap for each entity
/// @param dphi Array of size (nq, ndofs) with the basis function gradients in 1D.
/// @param n_entities Number of entities (cells or facets) to compute the stiffness operator for

/// @note The kernel is launched with a 3D grid of 1D blocks, where each block
/// is responsible for computing the stiffness operator for a single entity.
/// The block size is (P+1, P+1, P+1) and the shared memory 4 * (P+1)^3 * sizeof(T).
template <int P, typename T>
__global__ void stiffness_operator(
    const T *x,
    const T *entity_constants,
    T *y,
    const T *G_entity,
    const int *entity_dofmap,
    const T *dphi,
    int n_entities)
{
    const int nd = P + 1; // Number of dofs per direction in 1D
    // const int nq = P + 2; // Currently unused, same as nd
    const int cube_nd = nd * nd * nd;
    const int square_nd = nd * nd;

    extern __shared__ T shared_mem[];
    
    T *scratch = shared_mem;
    T *scratchx = shared_mem + cube_nd;
    T *scratchy = scratchx + cube_nd;
    T *scratchz = scratchy + cube_nd;


    int tx = threadIdx.x; // 1d dofs x direction
    int ty = threadIdx.y; // 1d dofs y direction
    int tz = threadIdx.z; // 1d dofs z direction

    // thread_id = tx * nd * nd + ty * nd + tz represents the dof index in 3D
    int thread_id = tx * blockDim.y * blockDim.z + ty * blockDim.z + tz;
    int block_id = blockIdx.x; // Entity index

    // Check if the block_id is valid (i.e. within the number of entities)
    if (block_id >= n_entities)
        return;

    // Get dof index that this thread is responsible for

    int dof = entity_dofmap[block_id * Ndofs + thread_id];

    // Gather x value required by this thread
    scratch[tx * square_nd + ty * nd + tz] = x[dof];
    __syncthreads();

    // Apply contraction in the x-direction
    T val_x = 0.0;
    for (int ix = 0; ix < nd; ++ix)
    {
        val_x += dphi[tx * nd + ix] * scratch[ix * square_nd + ty * nd + tz];
    }

    // Apply contraction in the y-direction
    T val_y = 0.0;
    for (int iy = 0; iy < nd; ++iy)
    {
        val_y += dphi[ty * nd + iy] * scratch[tx * square_nd + iy * nd + tz];
    }

    // Apply contraction in the z-direction
    T val_z = 0.0;
    for (int iz = 0; iz < nd; ++iz)
    {
        val_z += dphi[tz * nd + iz] * scratch[tx * square_nd + ty * nd + iz];
    }

    // Apply transform
    int offset = block_id * cube_nd + thread_id * 6;
    T G0 = G_entity[offset + 0];
    T G1 = G_entity[offset + 1];
    T G2 = G_entity[offset + 2];
    T G3 = G_entity[offset + 3];
    T G4 = G_entity[offset + 4];
    T G5 = G_entity[offset + 5];

    // DG-0 Coefficient
    T coeff = entity_constants[block_id];

    // Apply geometry
    T fw0 = coeff * (G0 * val_x + G1 * val_y + G2 * val_z);
    T fw1 = coeff * (G1 * val_x + G3 * val_y + G4 * val_z);
    T fw2 = coeff * (G2 * val_x + G4 * val_y + G5 * val_z);

    scratchx[tx * square_nd + ty * nd + tz] = fw0;
    scratchy[tx * square_nd + ty * nd + tz] = fw1;
    scratchz[tx * square_nd + ty * nd + tz] = fw2;

    __syncthreads();

    // Apply contraction in the x-direction
    val_x = 0.0;
    for (int ix = 0; ix < nd; ++ix)
    {
        val_x += dphi[ix * nd + tx] * scratchx[ix * square_nd + ty * nd + tz];
    }

    // Apply contraction in the y-direction
    val_y = 0.0;
    for (int iy = 0; iy < nd; ++iy)
    {
        val_y += dphi[iy * nd + ty] * scratchy[tx * square_nd + iy * nd + tz];
    }

    // Apply contraction in the z-direction
    val_z = 0.0;
    for (int iz = 0; iz < nd; ++iz)
    {
        val_z += dphi[iz * nd + tz] * scratchz[tx * square_nd + ty * nd + iz];
    }

    // Add contributions
    T val = val_x + val_y + val_z;

    // Atomically add the computed value to the output array `y`
    atomicAdd(&y[dof], val);
}
