// Copyright (C) 2023 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <thrust/device_vector.h>

#include "util.hpp"

#pragma once

/// @brief Computes weighted geometry tensor G from the coordinates and quadrature weights
/// @param [in] xgeom Geometry points [*, 3]
/// @param [out] G_entity geometry data [n_entities, nq, 6]
/// @param [in] geometry_dofmap Location of coordinates for each cell in xgeom [*, ncdofs]
/// @param [in] _dphi Basis derivative tabulation for cell at quadrature points [3, nq, ncdofs]
/// @param [in] weights Quadrature weights [nq]
/// @param [in] entities list of cells to compute for [n_entities]
/// @param [in] n_entities total number of cells to compute for
/// @tparam T scalar type
/// @tparam P degree of kernel to compute geometry for
template <typename T, int P>
__global__ void geometry_computation(const T* xgeom, T* G_entity,
                                     const std::int32_t* geometry_dofmap, const T* _dphi,
                                     const T* weights, const int* entities, int n_entities)
{
  // One block per cell
  int c = blockIdx.x;

  // Limit to cells in list
  if (c >= n_entities)
    return;

  // Cell index
  int cell = entities[c];

  // Number of quadrature points (must match arrays in weights and dphi)
  constexpr int nq = (P + 1) * (P + 1) * (P + 1);
  // Number of coordinate dofs
  constexpr int ncdofs = 8;
  // Geometric dimension
  constexpr int gdim = 3;

  extern __shared__ T shared_mem[];

  // coord_dofs has shape [ncdofs, gdim]
  T* _coord_dofs = shared_mem;

  // First collect geometry into shared memory
  int iq = threadIdx.x;
  if constexpr (P == 1)
  {
    // Only 8 threads when P == 1
    assert(iq < 8);
    for (int j = 0; j < 3; ++j)
      _coord_dofs[iq * 3 + j] = xgeom[3 * geometry_dofmap[cell * ncdofs + iq] + j];
  }
  else
  {
    int i = iq / gdim;
    int j = iq % gdim;
    if (i < ncdofs)
      _coord_dofs[iq] = xgeom[3 * geometry_dofmap[cell * ncdofs + i] + j];
  }

  __syncthreads();
  // One quadrature point per thread

  if (iq >= nq)
    return;

  // Jacobian
  T J[3][3];
  auto coord_dofs = [&_coord_dofs](int i, int j) -> T& { return _coord_dofs[i * gdim + j]; };

  // For each quadrature point / thread
  {
    // dphi has shape [gdim, ncdofs]
    auto dphi = [&_dphi, iq](int i, int j) -> const T { return _dphi[(i * nq + iq) * ncdofs + j]; };

    for (std::size_t i = 0; i < gdim; i++)
      for (std::size_t j = 0; j < gdim; j++)
      {
        J[i][j] = 0.0;
        for (std::size_t k = 0; k < ncdofs; k++)
          J[i][j] += coord_dofs(k, i) * dphi(j, k);
      }

    // Components of K = J^-1 (detJ)
    T K[3][3] = {{J[1][1] * J[2][2] - J[1][2] * J[2][1], -J[0][1] * J[2][2] + J[0][2] * J[2][1],
                  J[0][1] * J[1][2] - J[0][2] * J[1][1]},
                 {-J[1][0] * J[2][2] + J[1][2] * J[2][0], J[0][0] * J[2][2] - J[0][2] * J[2][0],
                  -J[0][0] * J[1][2] + J[0][2] * J[1][0]},
                 {J[1][0] * J[2][1] - J[1][1] * J[2][0], -J[0][0] * J[2][1] + J[0][1] * J[2][0],
                  J[0][0] * J[1][1] - J[0][1] * J[1][0]}};

    T detJ = J[0][0] * K[0][0] - J[1][0] * K[0][1] + J[0][2] * K[2][0];

    int offset = (c * nq + iq) * 6;
    G_entity[offset]
        = (K[0][0] * K[0][0] + K[0][1] * K[0][1] + K[0][2] * K[0][2]) * weights[iq] / detJ;
    G_entity[offset + 1]
        = (K[1][0] * K[0][0] + K[1][1] * K[0][1] + K[1][2] * K[0][2]) * weights[iq] / detJ;
    G_entity[offset + 2]
        = (K[2][0] * K[0][0] + K[2][1] * K[0][1] + K[2][2] * K[0][2]) * weights[iq] / detJ;
    G_entity[offset + 3]
        = (K[1][0] * K[1][0] + K[1][1] * K[1][1] + K[1][2] * K[1][2]) * weights[iq] / detJ;
    G_entity[offset + 4]
        = (K[2][0] * K[1][0] + K[2][1] * K[1][1] + K[2][2] * K[1][2]) * weights[iq] / detJ;
    G_entity[offset + 5]
        = (K[2][0] * K[2][0] + K[2][1] * K[2][1] + K[2][2] * K[2][2]) * weights[iq] / detJ;
  }
}

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
/// @param entities List of entities to compute on
/// @param n_entities Number of entries in `entities`

/// @note The kernel is launched with a 3D grid of 1D blocks, where each block
/// is responsible for computing the stiffness operator for a single entity.
/// The block size is (P+1, P+1, P+1) and the shared memory 4 * (P+1)^3 * sizeof(T).
template <typename T, int P>
__global__ void stiffness_operator(const T* x, const T* entity_constants, T* y, const T* G_entity,
                                   const std::int32_t* entity_dofmap, const T* dphi,
                                   const int* entities, int n_entities,
                                   const std::int8_t* bc_marker)
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
  int dof = entity_dofmap[entities[block_id] * cube_nd + thread_id];

  // Gather x values required in this cell
  // scratch has dimensions (nd, nd, nd)
  if (bc_marker[dof])
    scratch[thread_id] = 0.0;
  else
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

  // FIXME Set correct BC val for y outside kernel (multiple cell may share the dof)
  if (bc_marker[dof])
    y[dof] = x[dof];
  else
    // Atomically add the computed value to the output array `y`
    atomicAdd(&y[dof], val);
}

template <typename T, int P>
__global__ void mat_diagonal(const T* entity_constants, T* y, const T* G_entity,
                             const std::int32_t* entity_dofmap, const T* dphi, const int* entities,
                             int n_entities, const std::int8_t* bc_marker)
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

  T* scratchx = shared_mem;         // size nq^3
  T* scratchy = scratchx + cube_nq; // size nq^3
  T* scratchz = scratchy + cube_nq; // size nq^3

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
  T fw0 = 0;
  T fw1 = 0;
  T fw2 = 0;

  for (int ix = 0; ix < nq; ++ix)
    for (int iy = 0; iy < nq; ++iy)
      for (int iz = 0; iz < nq; ++iz)
      {
        T val_x = dphi[ix + tx * nd];
        T val_y = dphi[iy + ty * nd];
        T val_z = dphi[iz + tz * nd];
        fw0 += (G0 * val_x + G1 * val_y + G2 * val_z);
        fw1 += (G1 * val_x + G3 * val_y + G4 * val_z);
        fw2 += (G2 * val_x + G4 * val_y + G5 * val_z);
      }

  // Store values at quadrature points
  // scratchx, scratchy, scratchz all have dimensions (nq, nq, nq)
  scratchx[tx * square_nq + ty * nq + tz] = coeff * fw0;
  scratchy[tx * square_nq + ty * nq + tz] = coeff * fw1;
  scratchz[tx * square_nq + ty * nq + tz] = coeff * fw2;

  __syncthreads();

  T val = 0.0;
  for (int ix = 0; ix < nq; ++ix)
    for (int iy = 0; iy < nq; ++iy)
      for (int iz = 0; iz < nq; ++iz)
      {
        // tx is dof index, ty, tz quadrature point indices
        val_x = dphi[ix * nd + tx] * scratchx[ix * square_nq + ty * nd + tz];

        // Apply contraction in the y-direction and add y contribution
        // ty is dof index, tx, tz quadrature point indices
        val_y = dphi[iy * nd + ty] * scratchy[tx * square_nq + iy * nd + tz];

        // Apply contraction in the z-direction and add z contribution
        // tz is dof index, tx, ty quadrature point indices
        val_z = dphi[iz * nd + tz] * scratchz[tx * square_nq + ty * nd + iz];

        // Sum contributions
        val += val_x + val_y + val_z;
      }

  int dof = entity_dofmap[entities[block_id] * cube_nd + thread_id];
  if (bc_marker[dof])
    y[dof] = T(1.0);
  else
    y[dof] = val;
}

namespace dolfinx::acc
{

template <typename T>
class MatFreeLaplacian
{
public:
  using value_type = T;

  MatFreeLaplacian(int degree, std::span<const T> coefficients,
                   std::span<const std::int32_t> dofmap, std::span<const T> xgeom,
                   std::span<const std::int32_t> geometry_dofmap, std::span<const T> dphi_geometry,
                   std::span<const T> G_weights, const std::vector<int>& lcells,
                   const std::vector<int>& bcells, std::span<const std::int8_t> bc_marker,
                   std::size_t batch_size = 0)
      : degree(degree), cell_constants(coefficients), cell_dofmap(dofmap), xgeom(xgeom),
        geometry_dofmap(geometry_dofmap), dphi_geometry(dphi_geometry), G_weights(G_weights),
        bc_marker(bc_marker), batch_size(batch_size)
  {
    std::map<int, int> Qdegree = {{2, 3}, {3, 4}, {4, 6}, {5, 8}};

    // Create 1D element
    auto element1D = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::interval, degree,
        basix::element::lagrange_variant::gll_warped, basix::element::dpc_variant::unset, false);

    // Create quadrature
    auto [points, weights] = basix::quadrature::make_quadrature<T>(
        basix::quadrature::type::gll, basix::cell::type::interval, basix::polyset::type::standard,
        Qdegree[degree]);

    // Tabulate 1D
    auto [table, shape] = element1D.tabulate(1, points, {weights.size(), 1});

    spdlog::debug("Create device vector for phi");
    // Basis value gradient evualation table
    dphi_d.resize(table.size() / 2);
    thrust::copy(std::next(table.begin(), table.size() / 2), table.end(), dphi_d.begin());

    lcells_device.resize(lcells.size());
    thrust::copy(lcells.begin(), lcells.end(), lcells_device.begin());
    bcells_device.resize(bcells.size());
    thrust::copy(bcells.begin(), bcells.end(), bcells_device.begin());

    // If we're not batching the geomery, precompute it
    if (batch_size == 0)
    {
      // FIXME Store cells and local/ghost offsets instead to avoid this?
      spdlog::info("Precomputing geometry");
      thrust::device_vector<std::int32_t> cells_d(lcells_device.size() + bcells_device.size());
      thrust::copy(lcells_device.begin(), lcells_device.end(), cells_d.begin());
      thrust::copy(bcells_device.begin(), bcells_device.end(), cells_d.begin() + lcells_device.size());
      std::span<std::int32_t> cell_list_d(thrust::raw_pointer_cast(cells_d.data()), cells_d.size());

      // FIXME Tidy
      if (degree == 1)
        compute_geometry<1>(cell_list_d);
      else if (degree == 2)
        compute_geometry<2>(cell_list_d);
      else if (degree == 3)
        compute_geometry<3>(cell_list_d);
      else if (degree == 4)
        compute_geometry<4>(cell_list_d);
      else if (degree == 5)
        compute_geometry<5>(cell_list_d);
      else
        throw std::runtime_error("Unsupported degree");
      device_synchronize();
    }
  }

  // Compute weighted geometry data on GPU
  template <int P>
  void compute_geometry(std::span<int> cell_list_d)
  {
    G_entity.resize(G_weights.size() * cell_list_d.size() * 6);
    dim3 block_size(G_weights.size());
    dim3 grid_size(cell_list_d.size());

    spdlog::debug("xgeom size {}", xgeom.size());
    spdlog::debug("G_entity size {}", G_entity.size());
    spdlog::debug("geometry_dofmap size {}", geometry_dofmap.size());
    spdlog::debug("dphi_geometry size {}", dphi_geometry.size());
    spdlog::debug("G_weights size {}", G_weights.size());
    spdlog::debug("cell_list_d size {}", cell_list_d.size());
    spdlog::debug("Calling geometry_computation [{}]", P);

    std::size_t shm_size = 24 * sizeof(T); // coordinate size (8x3)
    geometry_computation<T, P><<<grid_size, block_size, shm_size, 0>>>(
        xgeom.data(), thrust::raw_pointer_cast(G_entity.data()), geometry_dofmap.data(),
        dphi_geometry.data(), G_weights.data(), cell_list_d.data(), cell_list_d.size());
  }

  template <int P, typename Vector>
  void impl_operator(Vector& in, Vector& out)
  {
    spdlog::debug("impl_operator operator start");

    in.scatter_fwd_begin();

    if (!lcells_device.empty())
    {
      std::size_t i = 0;
      std::size_t i_batch_size = (batch_size == 0) ? lcells_device.size() : batch_size;
      while (i < lcells_device.size())
      {
        std::size_t i_next = std::min(lcells_device.size(), i + i_batch_size);
        std::span<int> cell_list_d(thrust::raw_pointer_cast(lcells_device.data()) + i,
                                   (i_next - i));
        i = i_next;

        if (batch_size > 0)
        {
          spdlog::debug("Calling compute_geometry on local cells [{}]", cell_list_d.size());
          compute_geometry<P>(cell_list_d);
          device_synchronize();
        }

        dim3 block_size(P + 1, P + 1, P + 1);
        int p1cubed = (P + 1) * (P + 1) * (P + 1);
        dim3 grid_size(cell_list_d.size());
        std::size_t shm_size = 4 * p1cubed * sizeof(T);

        spdlog::debug("Calling stiffness_operator on local cells [{}]", cell_list_d.size());
        T* x = in.mutable_array().data();
        T* y = out.mutable_array().data();
        stiffness_operator<T, P><<<grid_size, block_size, shm_size, 0>>>(
            x, cell_constants.data(), y, thrust::raw_pointer_cast(G_entity.data()),
            cell_dofmap.data(), thrust::raw_pointer_cast(dphi_d.data()), cell_list_d.data(),
            cell_list_d.size(), bc_marker.data());

        check_device_last_error();
      }
    }

    spdlog::debug("impl_operator done lcells");

    spdlog::debug("cell_constants size {}", cell_constants.size());
    spdlog::debug("in size {}", in.array().size());
    spdlog::debug("out size {}", out.array().size());
    spdlog::debug("G_entity size {}", G_entity.size());
    spdlog::debug("cell_dofmap size {}", cell_dofmap.size());
    spdlog::debug("dphi_d size {}", dphi_d.size());
    spdlog::debug("bc_marker size {}", bc_marker.size());

    in.scatter_fwd_end();

    spdlog::debug("impl_operator after scatter");

    if (!bcells_device.empty())
    {
      spdlog::debug("impl_operator doing bcells. bcells size = {}", bcells_device.size());
      std::span<int> cell_list_d(thrust::raw_pointer_cast(bcells_device.data()),
                                 bcells_device.size());

      if (batch_size > 0)
      {
        compute_geometry<P>(cell_list_d);
        device_synchronize();
      }

      dim3 block_size(P + 1, P + 1, P + 1);
      int p1cubed = (P + 1) * (P + 1) * (P + 1);
      dim3 grid_size(cell_list_d.size());
      std::size_t shm_size = 4 * p1cubed * sizeof(T);

      T* x = in.mutable_array().data();
      T* y = out.mutable_array().data();

      stiffness_operator<T, P><<<grid_size, block_size, shm_size, 0>>>(
          x, cell_constants.data(), y, thrust::raw_pointer_cast(G_entity.data()),
          cell_dofmap.data(), thrust::raw_pointer_cast(dphi_d.data()), cell_list_d.data(),
          cell_list_d.size(), bc_marker.data());

      check_device_last_error();
    }

    device_synchronize();

    spdlog::debug("impl_operator done bcells");
  }

  template <typename Vector>
  void operator()(Vector& in, Vector& out)
  {
    spdlog::debug("Mat free operator start");
    out.set(T{0.0});

    if (degree == 1)
      impl_operator<1>(in, out);
    else if (degree == 2)
      impl_operator<2>(in, out);
    else if (degree == 3)
      impl_operator<3>(in, out);
    else if (degree == 4)
      impl_operator<4>(in, out);
    else if (degree == 5)
      impl_operator<5>(in, out);
    else
      throw std::runtime_error("Unsupported degree");

    spdlog::debug("Mat free operator end");
  }

  template <typename Vector>
  void get_diag_inverse(Vector& diag_inv)
  {
    thrust::copy(_diag_inv.begin(), _diag_inv.end(), diag_inv.mutable_array().begin());
  }

  template <typename Vector>
  void set_diag_inverse(const Vector& diag_inv)
  {
    _diag_inv.resize(diag_inv.array().size(), 0);
    thrust::copy(diag_inv.array().begin(), diag_inv.array().end(), _diag_inv.begin());
  }

private:
  int degree;

  // Reference to on-device storage for constants, dofmap etc.
  std::span<const T> cell_constants;
  std::span<const std::int32_t> cell_dofmap;

  // Reference to on-device storage of geometry data
  std::span<const T> xgeom;
  std::span<const std::int32_t> geometry_dofmap;
  std::span<const T> dphi_geometry;
  std::span<const T> G_weights;
  std::span<const std::int8_t> bc_marker;

  // On device storage for geometry data (computed for each batch of cells)
  thrust::device_vector<T> G_entity;

  // On device storage for dphi
  thrust::device_vector<T> dphi_d;

  // Lists of cells which are local (lcells) and boundary (bcells)
  thrust::device_vector<int> lcells_device, bcells_device;

  // On device storage for the inverse diagonal, needed for Jacobi
  // preconditioner (to remove in future)
  thrust::device_vector<T> _diag_inv;

  // Batch size for geometry computation (set to 0 for no batching)
  std::size_t batch_size;
};

} // namespace dolfinx::acc
