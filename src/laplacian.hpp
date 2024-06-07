// Copyright (C) 2023 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "hip/hip_runtime.h"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <thrust/device_vector.h>

#pragma once

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
template <typename T, int P>
__global__ void stiffness_operator(const T* x, const T* entity_constants, T* y, const T* G_entity,
                                   const std::int32_t* entity_dofmap, const T* dphi, int n_entities)
{
  constexpr int nd = P + 1; // Number of dofs per direction in 1D
  constexpr int nq = nd;    // Number of quadrature points in 1D (must be the same as nd)

  assert(blockDim.x == nd);
  assert(blockDim.y == nd);
  assert(blockDim.z == nd);

  constexpr int cube_nd = nd * nd * nd;
  constexpr int cube_nq = nq * nq * nq;
  constexpr int square_nd = nd * nd;
  constexpr int square_nq = nq * nq;

  extern __shared__ T shared_mem[];

  T* scratch = shared_mem;            // size nd^3
  T* scratchx = shared_mem + cube_nd; // size nq^3
  T* scratchy = scratchx + cube_nq;   // size nq^3
  T* scratchz = scratchy + cube_nq;   // size nq^3

  int tx = threadIdx.x; // 1d dofs x direction
  int ty = threadIdx.y; // 1d dofs y direction
  int tz = threadIdx.z; // 1d dofs z direction

  // thread_id represents the dof index in 3D
  int thread_id = tx * square_nd + ty * nd + tz;
  // block_id is the cell (or facet) index
  int block_id = blockIdx.x;

  // Check if the block_id is valid (i.e. within the number of entities)
  if (block_id >= n_entities)
    return;

  // Get dof index that this thread is responsible for
  int dof = entity_dofmap[block_id * cube_nd + thread_id];

  // Gather x values required in this cell
  // scratch has dimensions (nd, nd, nd)
  scratch[thread_id] = x[dof];
  __syncthreads();

  // Compute val_x, val_y, val_z at quadrature point of this thread
  // Apply contraction in the x-direction
  // tx is quadrature point index, ty, tz dof indices
  T val_x = 0.0;
  for (int ix = 0; ix < nd; ++ix)
  {
    val_x += dphi[tx * nd + ix] * scratch[ix * square_nd + ty * nd + tz];
  }
  // Because phi(nq, nd) is the identity for this choice of quadrature points,
  // we do not need to apply it in the y and z direction, and val_x is already the value at the
  // quadrature point of thread (tx, ty, tz). Similarly, below, for val_y and val_z.

  // Apply contraction in the y-direction
  // ty is quadrature point index, tx, tz dof indices
  T val_y = 0.0;
  for (int iy = 0; iy < nd; ++iy)
  {
    val_y += dphi[ty * nd + iy] * scratch[tx * square_nd + iy * nd + tz];
  }

  // Apply contraction in the z-direction
  // tz is quadrature point index, tx, ty dof indices
  T val_z = 0.0;
  for (int iz = 0; iz < nd; ++iz)
  {
    val_z += dphi[tz * nd + iz] * scratch[tx * square_nd + ty * nd + iz];
  }

  // Apply transform at each quadrature point (thread)
  int offset = (block_id * nq * nq * nq + thread_id) * 6;
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

  // Store values at quadrature points
  // scratchx, scratchy, scratchz all have dimensions (nq, nq, nq)
  scratchx[tx * square_nq + ty * nq + tz] = fw0;
  scratchy[tx * square_nq + ty * nq + tz] = fw1;
  scratchz[tx * square_nq + ty * nq + tz] = fw2;

  __syncthreads();

  // Apply contraction in the x-direction
  val_x = 0.0;
  // tx is dof index, ty, tz quadrature point indices
  for (int ix = 0; ix < nq; ++ix)
  {
    val_x += dphi[ix * nd + tx] * scratchx[ix * square_nq + ty * nd + tz];
  }

  // Apply contraction in the y-direction and add y contribution
  // ty is dof index, tx, tz quadrature point indices
  val_y = 0.0;
  for (int iy = 0; iy < nq; ++iy)
  {
    val_y += dphi[iy * nd + ty] * scratchy[tx * square_nq + iy * nd + tz];
  }

  // Apply contraction in the z-direction and add z contribution
  // tz is dof index, tx, ty quadrature point indices
  val_z = 0.0;
  for (int iz = 0; iz < nq; ++iz)
  {
    val_z += dphi[iz * nd + tz] * scratchz[tx * square_nq + ty * nd + iz];
  }

  // Sum contributions
  T val = val_x + val_y + val_z;

  // Atomically add the computed value to the output array `y`
  atomicAdd(&y[dof], val);
}

namespace dolfinx::acc
{

template <int P, typename T>
class MatFreeLaplacian
{
public:
  MatFreeLaplacian(int num_cells, std::span<const T> coefficients,
                   std::span<const std::int32_t> dofmap, std::span<const T> G)
      : num_cells(num_cells), cell_constants(coefficients), cell_dofmap(dofmap), G_entity(G)
  {
    std::map<int, int> Qdegree = {{2, 3}, {3, 4}, {4, 6}, {5, 8}};

    // Create 1D element
    auto element1D = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::interval, P,
        basix::element::lagrange_variant::gll_warped, basix::element::dpc_variant::unset, false);

    // Create quadrature
    auto [points, weights] = basix::quadrature::make_quadrature<T>(
        basix::quadrature::type::gll, basix::cell::type::interval, basix::polyset::type::standard,
        Qdegree[P]);

    // Tabulate 1D
    auto [table, shape] = element1D.tabulate(1, points, {weights.size(), 1});

    spdlog::debug("1D table = {}, G = {}", weights.size(), G.size());

    int nq = weights.size();
    // 6 geometry values per quadrature point on each cell
    assert(nq * nq * nq * num_cells * 6 == G.size());

    spdlog::debug("Create device vector for phi");
    // Basis value gradient evualation table
    dphi_d.resize(table.size() / 2);
    thrust::copy(std::next(table.begin(), table.size() / 2), table.end(), dphi_d.begin());
  }

  template <typename Vector>
  void operator()(Vector& in, Vector& out)
  {
    dim3 block_size(P + 1, P + 1, P + 1);
    int p1cubed = (P + 1) * (P + 1) * (P + 1);
    dim3 grid_size(num_cells);
    std::size_t shm_size = 4 * p1cubed * sizeof(T);

    T* x = in.mutable_array().data();
    T* y = out.mutable_array().data();
    std::span<const T> dphi(thrust::raw_pointer_cast(dphi_d.data()), dphi_d.size());
    hipLaunchKernelGGL(HIP_KERNEL_NAME(stiffness_operator<T, P>), grid_size, block_size, shm_size,
                       0, x, cell_constants.data(), y, G_entity.data(), cell_dofmap.data(),
                       dphi.data(), num_cells);

    err_check(hipGetLastError());
  }

private:
  int num_cells;

  // Reference to on-device storage for constants, dofmap etc.
  std::span<const T> cell_constants;
  std::span<const std::int32_t> cell_dofmap;
  std::span<const T> G_entity;

  // On device storage for dphi
  thrust::device_vector<T> dphi_d;
};

} // namespace dolfinx::acc
