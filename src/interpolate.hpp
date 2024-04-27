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
                                 const std::int32_t* mat_row_offset, const std::int32_t* mat_column,
                                 const T* mat_value)
{
  // Calculate the cell index for this thread.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the row index is out of bounds.
  if (i < N)
  {
    const std::int32_t cell = cell_list[i];
    const std::int32_t* cellQ1 = Q1dofmap + cell * Q1_dofs_per_cell;
    const std::int32_t* cellQ2 = Q2dofmap + cell * Q2_dofs_per_cell;

    for (std::int32_t j = 0; j < Q2_dofs_per_cell; j++)
    {
      T vj = 0;
      // Use small CSR matrix on cell-local dofs
      for (std::int32_t k = mat_row_offset[j]; k < mat_row_offset[j + 1]; ++k)
        vj += mat_value[k] * valuesQ1[cellQ1[mat_column[k]]];
      valuesQ2[cellQ2[j]] = vj;
    }
  }
}

// Compress a dense matrix to a CSR sparse matrix
// Values less than tol are set to zero.
// Input: matrix in RowMajor order, shape
// Output: {row_offset, columns, values} of CSR matrix
template <std::floating_point T>
std::tuple<std::vector<std::int32_t>, std::vector<std::int32_t>, std::vector<T>>
compress_to_csr(const std::vector<T>& mat, std::array<std::size_t, 2> shape, T tol = 1e-12)
{
  std::vector<std::int32_t> row_ptr = {0};
  std::vector<std::int32_t> cols;
  std::vector<T> vals;

  for (std::size_t row = 0; row < shape[0]; ++row)
  {
    for (std::size_t col = 0; col < shape[1]; ++col)
    {
      T val = mat[row * shape[1] + col];
      if (std::abs(val) > tol)
      {
        cols.push_back(col);
        vals.push_back(val);
      }
    }
    row_ptr.push_back(cols.size());
  }

  LOG(INFO) << "Compressed dense matrix from " << mat.size() << " to " << vals.size()
            << " CSR values";
  return {std::move(row_ptr), std::move(cols), std::move(vals)};
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
    std::tie(_row_offset, _cols, _vals) = compress_to_csr(mat, shape);

    // Copy small CSR matrix to device
    mat_row_offset.resize(_row_offset.size());
    thrust::copy(_row_offset.begin(), _row_offset.end(), mat_row_offset.begin());
    mat_column.resize(_cols.size());
    thrust::copy(_cols.begin(), _cols.end(), mat_column.begin());
    mat_value.resize(_vals.size());
    thrust::copy(_vals.begin(), _vals.end(), mat_value.begin());

    err_check(hipDeviceSynchronize());
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

    const std::int32_t* _mat_row_ptr = thrust::raw_pointer_cast(mat_row_offset.data());
    const std::int32_t* _mat_col = thrust::raw_pointer_cast(mat_column.data());
    const T* _mat_val = thrust::raw_pointer_cast(mat_value.data());

    hipLaunchKernelGGL(interpolate_Q1Q2<T>, grid_size, block_size, 0, 0, ncells, cell_list,
                       input_dofmap.data(), num_cell_dofs_Q1, output_dofmap.data(),
                       num_cell_dofs_Q2, input_values, output_values, _mat_row_ptr, _mat_col,
                       _mat_val);
    err_check(hipGetLastError());

    // Wait for vector update of input_vector to complete
    input_vector.scatter_fwd_end();

    const std::int32_t* b_cell_list = boundary_cells.data();
    ncells = boundary_cells.size();
    LOG(INFO) << "From " << num_cell_dofs_Q1 << " dofs/cell to " << num_cell_dofs_Q2 << " on "
              << ncells << "(boundary) cells";

    hipLaunchKernelGGL(interpolate_Q1Q2<T>, grid_size, block_size, 0, 0, ncells, b_cell_list,
                       input_dofmap.data(), num_cell_dofs_Q1, output_dofmap.data(),
                       num_cell_dofs_Q2, input_values, output_values, _mat_row_ptr, _mat_col,
                       _mat_val);

    err_check(hipDeviceSynchronize());
    err_check(hipGetLastError());
  }

private:
  // Dofmap widths
  int num_cell_dofs_Q1;
  int num_cell_dofs_Q2;

  // Per-cell CSR interpolation matrix (on device)
  thrust::device_vector<std::int32_t> mat_row_offset;
  thrust::device_vector<std::int32_t> mat_column;
  thrust::device_vector<T> mat_value;

  // Dofmaps (on device).
  std::span<const std::int32_t> input_dofmap;
  std::span<const std::int32_t> output_dofmap;

  // List of cells which are in the "boundary region" which need to wait for a Vector update
  // before interpolation (on device)
  std::span<const std::int32_t> boundary_cells;

  // List of local cells, which can be updated before a Vector update
  std::span<const std::int32_t> local_cells;
};
