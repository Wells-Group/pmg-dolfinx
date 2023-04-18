#include "hip/hip_runtime.h"
#include <dolfinx.h>
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/MatrixCSR.h>

namespace test
{
/// Computes y += A*x for a local CSR matrix A and local dense vectors x,y
/// @param[in] values Nonzero values of A
/// @param[in] row_begin First index of each row in the arrays values and
/// indices.
/// @param[in] row_end Last index of each row in the arrays values and indices.
/// @param[in] indices Column indices for each non-zero element of the matrix A
/// @param[in] x Input vector
/// @param[in, out] x Output vector
template <typename T>
void spmv_impl(std::span<const T> values, std::span<const std::int32_t> row_begin,
               std::span<const std::int32_t> row_end, std::span<const std::int32_t> indices,
               std::span<const T> x, std::span<T> y)
{
  assert(row_begin.size() == row_end.size());
  for (std::size_t i = 0; i < row_begin.size(); i++)
  {
    double vi{0};
    for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
      vi += values[j] * x[indices[j]];
    y[i] += vi;
  }
}

// The matrix A is distributed across P  processes by blocks of rows:
//  A = |   A_0  |
//      |   A_1  |
//      |   ...  |
//      |  A_P-1 |
//
// Each submatrix A_i is owned by a single process "i" and can be further
// decomposed into diagonal (Ai[0]) and off diagonal (Ai[1]) blocks:
//  Ai = |Ai[0] Ai[1]|
//
// If A is square, the diagonal block Ai[0] is also square and contains
// only owned columns and rows. The block Ai[1] contains ghost columns
// (unowned dofs).

// Likewise, a local vector x can be decomposed into owned and ghost blocks:
// xi = |   x[0]  |
//      |   x[1]  |
//
// So the product y = Ax can be computed into two separate steps:
//  y[0] = |Ai[0] Ai[1]| |   x[0]  | = Ai[0] x[0] + Ai[1] x[1]
//                       |   x[1]  |
//
/// Computes y += A*x for a parallel CSR matrix A and parallel dense vectors x,y
/// @param[in] A Parallel CSR matrix
/// @param[in] x Input vector
/// @param[in, out] y Output vector
template <typename T, typename Vector>
void spmv(la::MatrixCSR<T>& A, Vector& x, Vector& y)

{
  // start communication (update ghosts)
  x.scatter_fwd_begin();

  const std::int32_t nrowslocal = A.num_owned_rows();
  std::span<const std::int32_t> row_ptr(A.row_ptr().data(), nrowslocal + 1);
  std::span<const std::int32_t> cols(A.cols().data(), row_ptr[nrowslocal]);
  std::span<const std::int32_t> off_diag_offset(A.off_diag_offset().data(), nrowslocal);
  std::span<const T> values(A.values().data(), row_ptr[nrowslocal]);

  std::span<const T> _x = x.array();
  std::span<T> _y = y.mutable_array();

  std::span<const std::int32_t> row_begin(row_ptr.data(), nrowslocal);
  std::span<const std::int32_t> row_end(row_ptr.data() + 1, nrowslocal);

  // First stage:  spmv - diagonal
  // yi[0] += Ai[0] * xi[0]
  spmv_impl<T>(values, row_begin, off_diag_offset, cols, _x, _y);

  // finalize ghost update
  x.scatter_fwd_end();

  // Second stage:  spmv - off-diagonal
  // yi[0] += Ai[1] * xi[1]
  spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
}
} // namespace test

namespace
{
// // /// Computes y += A*x for a local CSR matrix A and local dense vectors x,y
/// @param[in] values Nonzero values of A
/// @param[in] row_begin First index of each row in the arrays values and
/// indices.
/// @param[in] row_end Last index of each row in the arrays values and indices.
/// @param[in] indices Column indices for each non-zero element of the matrix A
/// @param[in] x Input vector
/// @param[in, out] x Output vector
template <typename T>
__global__ void spmv_impl(const T* values, const std::int32_t* row_begin,
                          const std::int32_t* row_end, const std::int32_t* indices, const T* x,
                          T* y, std::int32_t num_rows)
{
  // Calculate the row index for this thread.
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Check if the row index is out of bounds.
  if (i < num_rows)
  {
    // Perform the sparse matrix-vector multiplication for this row.
    T vi{0};
    for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
      vi += values[j] * x[indices[j]];
    y[i] += vi;
  }
}

} // namespace

#define err_check(command)                                                                         \
  {                                                                                                \
    hipError_t status = command;                                                                   \
    if (status != hipSuccess)                                                                      \
    {                                                                                              \
      printf("(%s:%d) Error: Hip reports %s\n", __FILE__, __LINE__, hipGetErrorString(status));    \
      exit(1);                                                                                     \
    }                                                                                              \
  }

namespace dolfinx::acc
{
template <typename T>
class MatrixOperator
{
public:
  /// The value type
  using value_type = T;

  /// Create a distributed vector
  MatrixOperator(std::shared_ptr<fem::Form<T, T>> a,
                 const std::vector<std::shared_ptr<const fem::DirichletBC<T, T>>>& bcs)
  {

    dolfinx::common::Timer t0("~setup phase MatrixOperator");

    if (a->rank() != 2)
      throw std::runtime_error("Form should have rank be 2.");

    auto V = a->function_spaces()[0];
    la::SparsityPattern pattern = fem::create_sparsity_pattern(*a);
    pattern.assemble();
    _map = std::make_shared<const common::IndexMap>(pattern.column_index_map());

    auto _A = std::make_unique<la::MatrixCSR<T>>(pattern);
    fem::assemble_matrix(_A->mat_add_values(), *a, bcs);
    _A->finalize();
    fem::set_diagonal<T>(_A->mat_set_values(), *V, bcs, T(1.0));

    // Allocate data on device
    err_check(hipMalloc((void**)&_row_ptr, _A->row_ptr().size() * sizeof(std::int32_t)));
    err_check(hipMalloc((void**)&_cols, _A->cols().size() * sizeof(std::int32_t)));
    err_check(
        hipMalloc((void**)&_off_diag_offset, _A->off_diag_offset().size() * sizeof(std::int32_t)));
    err_check(hipMalloc((void**)&_values, _A->values().size() * sizeof(T)));

    // Copy data from host to device
    err_check(hipMemcpy(_row_ptr, _A->row_ptr().data(), _A->row_ptr().size() * sizeof(std::int32_t),
                        hipMemcpyHostToDevice));
    err_check(hipMemcpy(_cols, _A->cols().data(), _A->cols().size() * sizeof(std::int32_t),
                        hipMemcpyHostToDevice));
    err_check(hipMemcpy(_off_diag_offset, _A->off_diag_offset().data(),
                        _A->off_diag_offset().size() * sizeof(std::int32_t),
                        hipMemcpyHostToDevice));
    err_check(hipMemcpy(_values, _A->values().data(), _A->values().size() * sizeof(T),
                        hipMemcpyHostToDevice));

    // Get communicator from mesh
    _comm = V->mesh()->comm();
  }

  /**
   * @brief The matrix-vector multiplication operator, which multiplies the
   * matrix with the input vector and stores the result in the output vector.
   *
   * @tparam Vector  The type of the input and output vector.
   *
   * @param x        The input vector.
   * @param y        The output vector.
   */
  template <typename Vector>
  void operator()(Vector& x, Vector& y, bool transpose = false)
  {
    dolfinx::common::Timer t0("~MatrixOperator application");
    y.set(T{0});
    T* _x = x.mutable_array().data();
    T* _y = y.mutable_array().data();

    int num_rows = _map->size_local();
    dim3 block_size(512);
    dim3 grid_size((num_rows + block_size.x - 1) / block_size.x);

    x.scatter_fwd_begin();
    hipLaunchKernelGGL(spmv_impl<T>, block_size, grid_size, 0, 0, _values, _row_ptr,
                       _off_diag_offset, _cols, _x, _y, num_rows);
    
    x.scatter_fwd_end();
    hipLaunchKernelGGL(spmv_impl<T>, block_size, grid_size, 0, 0, _values, _off_diag_offset,
                       _row_ptr + 1, _cols, _x, _y, num_rows);
  }

  template <typename Vector>
  void apply_host(Vector& x, Vector& y, bool transpose = false)
  {
    test::spmv(*_A, x, y);
  }

  std::shared_ptr<const common::IndexMap> index_map() { return _map; };

  ~MatrixOperator()
  {
    err_check(hipFree(_values));
    err_check(hipFree(_row_ptr));
    err_check(hipFree(_cols));
    err_check(hipFree(_off_diag_offset));
  }

private:
  T* _values;
  std::int32_t* _row_ptr;
  std::int32_t* _cols;
  std::int32_t* _off_diag_offset;
  std::shared_ptr<const common::IndexMap> _map;
  std::unique_ptr<la::MatrixCSR<T>> _A;
  MPI_Comm _comm;
};
} // namespace dolfinx::acc

