#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <cstdint>
#include <thrust/device_vector.h>

namespace
{

// Interpolate cells from Q1 to Q2 (prolongation) coarse->fine operator
// Input: N - number of cells
//        Q1dofs - dofmap Q1: list of shape N x Q1_dofs_per_cell
//        Q1_dofs_per_cell: int
//        Q2dofs - dofmap Q2: list of shape N x Q2_dofs_per_cell
//        Q2_dofs_per_cell: int
//        valuesQ1: vector of values for Q1
//        valuesQ2: vector of values for Q2
//        Mptr: CSR matrix row offsets for local interpolation matrix,
//          number of entries = Q2_dofs_per_cell + 1
//        Mcols: CSR matrix columns for local interpolation matrix
//        Mvals: CSR matrix values for local interpolation matrix
// Output: vector valuesQ2
template <typename T>
__global__ void interpolate_Q1Q2(int N, const std::int32_t* cell_list, const std::int32_t* Q1dofmap,
                                 int Q1_dofs_per_cell, const std::int32_t* Q2dofmap,
                                 int Q2_dofs_per_cell, const T* valuesQ1, T* valuesQ2,
                                 const std::int32_t* Mptr, const std::int32_t* Mcols,
                                 const T* Mvals)
{
  // Calculate the cell index for this thread.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the cell index is out of bounds.
  if (i < N)
  {
    const std::int32_t cell = cell_list[i];
    const std::int32_t* dofsQ1 = Q1dofmap + cell * Q1_dofs_per_cell;
    const std::int32_t* dofsQ2 = Q2dofmap + cell * Q2_dofs_per_cell;

    for (std::int32_t j = 0; j < Q2_dofs_per_cell; j++)
    {
      T vj = 0;
      for (std::int32_t k = Mptr[j]; k < Mptr[j + 1]; k++)
        vj += Mvals[k] * valuesQ1[dofsQ1[Mcols[k]]];
      valuesQ2[dofsQ2[j]] = vj;
    }
  }
}

// Reverse interpolation (restriction) fine->coarse operator
template <typename T>
__global__ void interpolate_Q2Q1(int N, const std::int32_t* cell_list, const std::int32_t* Q1dofmap,
                                 int Q1_dofs_per_cell, const std::int32_t* Q2dofmap,
                                 int Q2_dofs_per_cell, T* valuesQ1, const T* valuesQ2,
                                 const std::int32_t* Mptr, const std::int32_t* Mcols,
                                 const T* Mvals, const T* Q2mult)
{
  // Calculate the cell index for this thread.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the cell index is out of bounds.
  if (i < N)
  {
    const std::int32_t cell = cell_list[i];
    const std::int32_t* dofsQ1 = Q1dofmap + cell * Q1_dofs_per_cell;
    const std::int32_t* dofsQ2 = Q2dofmap + cell * Q2_dofs_per_cell;

    for (std::int32_t j = 0; j < Q1_dofs_per_cell; j++)
    {
      T vj = 0;
      for (std::int32_t k = Mptr[j]; k < Mptr[j + 1]; k++)
      {
        std::int32_t dofQ2 = dofsQ2[Mcols[k]];
        vj += Mvals[k] * valuesQ2[dofQ2] / Q2mult[dofQ2];
      }
      atomicAdd(&valuesQ1[dofsQ1[j]], vj);
    }
  }
}

} // namespace

template <typename T>
class Interpolator
{
public:
  // Set up interpolation from Q1 to Q2
  // Q1_element - element of input space (Q1)
  // Q2_element - element of output space (Q2)
  // Q1_dofmap - dofmap of input space (on device)
  // Q2_dofmap - dofmap of output space (on device)
  // l_cells - local cells, to interpolate immediately
  // b_cells - boundary cells, to interpolate after vector update
  Interpolator(const basix::FiniteElement<T>& Q1_element, const basix::FiniteElement<T>& Q2_element,
               std::span<const std::int32_t> Q1_dofmap, std::span<const std::int32_t> Q2_dofmap,
               std::span<const std::int32_t> l_cells, std::span<const std::int32_t> b_cells)
      : Q1_dofmap(Q1_dofmap), Q2_dofmap(Q2_dofmap), local_cells(l_cells), boundary_cells(b_cells)
  {
    num_cell_dofs_Q1 = Q1_element.dim();
    num_cell_dofs_Q2 = Q2_element.dim();

    // Checks on dofmap shapes and sizes
    assert(Q1_dofmap.size() % num_cell_dofs_Q1 == 0);
    assert(Q2_dofmap.size() % num_cell_dofs_Q2 == 0);
    assert(Q2_dofmap.size() / num_cell_dofs_Q2 == Q1_dofmap.size() / num_cell_dofs_Q1);

    // Get local interpolation matrix and compress to CSR format
    auto [mat, shape] = basix::compute_interpolation_operator(Q1_element, Q2_element);
    T tol = 1e-12;
    std::vector<std::int32_t> Mptr = {0};
    std::vector<std::int32_t> Mcolumns;
    std::vector<T> Mvalues;
    for (std::size_t row = 0; row < shape[0]; ++row)
    {
      for (std::size_t col = 0; col < shape[1]; ++col)
      {
        T val = mat[row * shape[1] + col];
        if (std::abs(val) > tol)
        {
          Mcolumns.push_back(col);
          Mvalues.push_back(val);
        }
      }
      Mptr.push_back(Mcolumns.size());
    }
    assert((num_cell_dofs_Q2 + 1) == Mptr.size());

    // Copy CSR to device
    Mptr_device.resize(Mptr.size());
    thrust::copy(Mptr.begin(), Mptr.end(), Mptr_device.begin());
    Mcol_device.resize(Mcolumns.size());
    thrust::copy(Mcolumns.begin(), Mcolumns.end(), Mcol_device.begin());
    Mval_device.resize(Mvalues.size());
    thrust::copy(Mvalues.begin(), Mvalues.end(), Mval_device.begin());

    // Create transpose matrix
    Mptr.clear();
    Mptr.push_back(0);
    Mcolumns.clear();
    Mvalues.clear();
    for (std::size_t row = 0; row < shape[1]; ++row)
    {
      for (std::size_t col = 0; col < shape[0]; ++col)
      {
        T val = mat[col * shape[1] + row];
        if (std::abs(val) > tol)
        {
          Mcolumns.push_back(col);
          Mvalues.push_back(val);
        }
      }
      Mptr.push_back(Mcolumns.size());
    }
    // Copy CSR to device (transpose)
    MptrT_device.resize(Mptr.size());
    thrust::copy(Mptr.begin(), Mptr.end(), MptrT_device.begin());
    McolT_device.resize(Mcolumns.size());
    thrust::copy(Mcolumns.begin(), Mcolumns.end(), McolT_device.begin());
    MvalT_device.resize(Mvalues.size());
    thrust::copy(Mvalues.begin(), Mvalues.end(), MvalT_device.begin());

    // Compute dofmap multiplicity
    std::int32_t max_dof = *std::max_element(Q2_dofmap.begin(), Q2_dofmap.end());
    std::vector<T> Q2count(max_dof + 1, 0);
    for (std::int32_t dof : Q2_dofmap)
      Q2count[dof] += 1.0;
    Q2mult.resize(Q2count.size());
    thrust::copy(Q2count.begin(), Q2count.end(), Q2mult.begin());
  }

  // Interpolate from input_values to output_values (both on device)
  // Use this only for prolongation, i.e. coarse->fine interpolation
  template <typename Vector>
  void interpolate(Vector& Q1_vector, Vector& Q2_vector)
  {
    dolfinx::common::Timer tt("% Interpolate Kernel");

    // Input (Q1) vector is also changed by MPI vector update
    T* Q1_values = Q1_vector.mutable_array().data();
    T* Q2_values = Q2_vector.mutable_array().data();

    int ncells = local_cells.size();
    thrust::device_vector<std::int32_t> cell_list_d(local_cells.begin(), local_cells.end());
    assert(ncells <= Q2_dofmap.size() / num_cell_dofs_Q2);

    dim3 block_size(256);
    dim3 grid_size((ncells + block_size.x - 1) / block_size.x);

    // Start vector update of Q1_vector
    Q1_vector.scatter_fwd_begin();

    spdlog::info("From {} to {} on {} cells", num_cell_dofs_Q1, num_cell_dofs_Q2, ncells);
    spdlog::info("Input dofmap size = {}, output dofmap size = {}", Q1_dofmap.size(),
                 Q2_dofmap.size());

    interpolate_Q1Q2<T><<<grid_size, block_size, 0, 0>>>(
        ncells, thrust::raw_pointer_cast(cell_list_d.data()), Q1_dofmap.data(), num_cell_dofs_Q1,
        Q2_dofmap.data(), num_cell_dofs_Q2, Q1_values, Q2_values,
        thrust::raw_pointer_cast(Mptr_device.data()), thrust::raw_pointer_cast(Mcol_device.data()),
        thrust::raw_pointer_cast(Mval_device.data()));

    check_device_last_error();

    // Wait for vector update of input_vector to complete
    Q1_vector.scatter_fwd_end();

    cell_list_d.resize(boundary_cells.size());
    thrust::copy(boundary_cells.begin(), boundary_cells.end(), cell_list_d.begin());
    ncells = boundary_cells.size();
    if (ncells > 0)
    {
      spdlog::info("From {} dofs/cell to {} on {} (boundary) cells", num_cell_dofs_Q1,
                   num_cell_dofs_Q2, ncells);

      interpolate_Q1Q2<T><<<grid_size, block_size, 0, 0>>>(
          ncells, thrust::raw_pointer_cast(cell_list_d.data()), Q1_dofmap.data(), num_cell_dofs_Q1,
          Q2_dofmap.data(), num_cell_dofs_Q2, Q1_values, Q2_values,
          thrust::raw_pointer_cast(Mptr_device.data()),
          thrust::raw_pointer_cast(Mcol_device.data()),
          thrust::raw_pointer_cast(Mval_device.data()));
    }

    device_synchronize();
    check_device_last_error();

    spdlog::debug("Done mat-free interpolation");
  }

  template <typename Vector>
  void reverse_interpolate(Vector& Q2_vector, Vector& Q1_vector)
  {
    spdlog::info("Reverse interpolate");

    dolfinx::common::Timer tt("% Reverse interpolate Kernel");

    // Input vector is also changed by MPI vector update
    T* Q1_values = Q1_vector.mutable_array().data();
    T* Q2_values = Q2_vector.mutable_array().data();

    int ncells = local_cells.size();
    thrust::device_vector<std::int32_t> cell_list_d(local_cells.begin(), local_cells.end());
    assert(ncells <= output_dofmap.size() / num_cell_dofs_Q2);

    dim3 block_size(256);
    dim3 grid_size((ncells + block_size.x - 1) / block_size.x);

    // Start vector update of input_vector
    Q2_vector.scatter_fwd_begin();

    spdlog::info("From {} to {} on {} cells", num_cell_dofs_Q2, num_cell_dofs_Q1, ncells);
    spdlog::info("Input dofmap size = {}, output dofmap size = {}", Q2_dofmap.size(),
                 Q1_dofmap.size());

    Q1_vector.set(0.0);
    interpolate_Q2Q1<T><<<grid_size, block_size, 0, 0>>>(
        ncells, thrust::raw_pointer_cast(cell_list_d.data()), Q1_dofmap.data(), num_cell_dofs_Q1,
        Q2_dofmap.data(), num_cell_dofs_Q2, Q1_values, Q2_values,
        thrust::raw_pointer_cast(MptrT_device.data()),
        thrust::raw_pointer_cast(McolT_device.data()),
        thrust::raw_pointer_cast(MvalT_device.data()), thrust::raw_pointer_cast(Q2mult.data()));

    check_device_last_error();

    // Wait for vector update of input_vector to complete
    Q2_vector.scatter_fwd_end();

    cell_list_d.resize(boundary_cells.size());
    thrust::copy(boundary_cells.begin(), boundary_cells.end(), cell_list_d.begin());
    ncells = boundary_cells.size();
    if (ncells > 0)
    {
      spdlog::info("From {} dofs/cell to {} on {} (boundary) cells", num_cell_dofs_Q1,
                   num_cell_dofs_Q2, ncells);

      interpolate_Q2Q1<T><<<grid_size, block_size, 0, 0>>>(
          ncells, thrust::raw_pointer_cast(cell_list_d.data()), Q1_dofmap.data(), num_cell_dofs_Q1,
          Q2_dofmap.data(), num_cell_dofs_Q2, Q1_values, Q2_values,
          thrust::raw_pointer_cast(MptrT_device.data()),
          thrust::raw_pointer_cast(McolT_device.data()),
          thrust::raw_pointer_cast(MvalT_device.data()), thrust::raw_pointer_cast(Q2mult.data()));
    }

    device_synchronize();
    check_device_last_error();

    spdlog::debug("Done mat-free interpolation");
  }

private:
  // Dofmap widths
  int num_cell_dofs_Q1;
  int num_cell_dofs_Q2;

  // Per-cell CSR interpolation matrix and transpose (on device)
  thrust::device_vector<std::int32_t> Mcol_device, Mptr_device;
  thrust::device_vector<T> Mval_device;
  thrust::device_vector<std::int32_t> McolT_device, MptrT_device;
  thrust::device_vector<T> MvalT_device;

  // Multiplicity of dofs in Q2
  thrust::device_vector<T> Q2mult;

  // Dofmaps (on device)
  std::span<const std::int32_t> Q1_dofmap;
  std::span<const std::int32_t> Q2_dofmap;

  // List of local cells, which can be updated before a Vector update
  std::span<const std::int32_t> local_cells;

  // List of cells which are in the "boundary region" which need to wait for a Vector update
  // before interpolation (on device)
  std::span<const std::int32_t> boundary_cells;
};
