#include "small-csr.hpp"
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>

namespace
{

// Interpolate cells from Q1 to Q2
// Input: N - number of cells
//        Q1dofs - dofmap Q1: list of shape N x Q1_dofs_per_cell
//        Q1_dofs_per_cell: int
//        Q2dofs - dofmap Q2: list of shape N x Q2_dofs_per_cell
//        Q2_dofs_per_cell: int
//        valuesQ1: vector of values for Q1
//        valuesQ2: vector of values for Q2
//        mat_row_offset: CSR matrix row offsets for local interpolation matrix, number of entries =
//        Q2_dofs_per_cell + 1 mat_column: CSR matrix columns for local interpolation matrix
//        mat_value: CSR matrix values for local interpolation matrix
// Output: vector valuesQ2
template <typename T>
__global__ void interpolate_Q1Q2(int N, const std::int32_t* cell_list, const std::int32_t* Q1dofmap,
                                 int Q1_dofs_per_cell, const std::int32_t* Q2dofmap,
                                 int Q2_dofs_per_cell, const T* valuesQ1, T* valuesQ2,
                                 const SmallCSRDevice<T>* mat)
{
  // Calculate the cell index for this thread.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the row index is out of bounds.
  if (i < N)
  {
    const std::int32_t cell = cell_list[i];
    const std::int32_t* cellQ1 = Q1dofmap + cell * Q1_dofs_per_cell;
    const std::int32_t* cellQ2 = Q2dofmap + cell * Q2_dofs_per_cell;
    mat->apply_indirect(cellQ1, cellQ2, valuesQ1, valuesQ2);
  }
}

} // namespace

template <typename T>
class Interpolator
{
public:
  // Set up interpolation from Q1 to Q2
  // inp_element - element of input space
  // out_element - element of output space
  // inp_dofmap - dofmap of input space (on device)
  // out_dofmap - dofmap of output space (on device)
  // b_cells - boundary cells, to interpolate after vector update
  // l_cells - local cells, to interpolate immediately
  Interpolator(const basix::FiniteElement<T>& inp_element,
               const basix::FiniteElement<T>& out_element, std::span<const std::int32_t> inp_dofmap,
               std::span<const std::int32_t> out_dofmap, std::span<const std::int32_t> b_cells,
               std::span<const std::int32_t> l_cells)
      : input_dofmap(inp_dofmap), output_dofmap(out_dofmap), boundary_cells(b_cells),
        local_cells(l_cells)
  {
    // Local CSR data to be copied to device
    std::vector<std::int32_t> _cols;
    std::vector<std::int32_t> _row_offset;
    std::vector<T> _vals;

    num_cell_dofs_Q1 = inp_element.dim();
    num_cell_dofs_Q2 = out_element.dim();

    // Checks on dofmap shapes and sizes
    assert(input_dofmap.size() % num_cell_dofs_Q1 == 0);
    assert(output_dofmap.size() % num_cell_dofs_Q2 == 0);
    assert(output_dofmap.size() / num_cell_dofs_Q2 == input_dofmap.size() / num_cell_dofs_Q1);
    assert(_row_offset.size() == num_cell_dofs_Q2 + 1);

    auto [mat, shape] = basix::compute_interpolation_operator(inp_element, out_element);
    _mat_csr = std::make_shared<SmallCSR<T>>(mat, shape);
  }

  // Interpolate from input_values to output_values (both on device)
  template <typename Vector>
  void interpolate(Vector& input_vector, Vector& output_vector)
  {
    dolfinx::common::Timer tt("% Interpolate Kernel");

    // Input vector is also changed by MPI vector update
    T* input_values = input_vector.mutable_array().data();
    T* output_values = output_vector.mutable_array().data();

    int ncells = local_cells.size();
    const std::int32_t* cell_list = local_cells.data();
    assert(ncells <= output_dofmap.size() / num_cell_dofs_Q2);

    dim3 block_size(256);
    dim3 grid_size((ncells + block_size.x - 1) / block_size.x);

    // Start vector update of input_vector
    input_vector.scatter_fwd_begin();

    LOG(INFO) << "From " << num_cell_dofs_Q1 << " dofs/cell to " << num_cell_dofs_Q2 << " on "
              << ncells << " cells";

    hipLaunchKernelGGL(interpolate_Q1Q2<T>, grid_size, block_size, 0, 0, ncells, cell_list,
                       input_dofmap.data(), num_cell_dofs_Q1, output_dofmap.data(),
                       num_cell_dofs_Q2, input_values, output_values, _mat_csr->device_matrix());

    err_check(hipGetLastError());

    // Wait for vector update of input_vector to complete
    input_vector.scatter_fwd_end();

    const std::int32_t* b_cell_list = boundary_cells.data();
    ncells = boundary_cells.size();
    LOG(INFO) << "From " << num_cell_dofs_Q1 << " dofs/cell to " << num_cell_dofs_Q2 << " on "
              << ncells << "(boundary) cells";

    hipLaunchKernelGGL(interpolate_Q1Q2<T>, grid_size, block_size, 0, 0, ncells, b_cell_list,
                       input_dofmap.data(), num_cell_dofs_Q1, output_dofmap.data(),
                       num_cell_dofs_Q2, input_values, output_values, _mat_csr->device_matrix());

    err_check(hipDeviceSynchronize());
    err_check(hipGetLastError());
  }

  std::shared_ptr<SmallCSR<T>> matrix() { return _mat_csr; }

private:
  // Dofmap widths
  int num_cell_dofs_Q1;
  int num_cell_dofs_Q2;

  // Per-cell CSR interpolation matrix
  std::shared_ptr<SmallCSR<T>> _mat_csr;

  // Dofmaps (on device).
  std::span<const std::int32_t> input_dofmap;
  std::span<const std::int32_t> output_dofmap;

  // List of cells which are in the "boundary region" which need to wait for a Vector update
  // before interpolation (on device)
  std::span<const std::int32_t> boundary_cells;

  // List of local cells, which can be updated before a Vector update
  std::span<const std::int32_t> local_cells;
};
