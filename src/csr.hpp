#include <dolfinx.h>
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/MatrixCSR.h>

#ifdef USE_HIP
#include "hip/hip_runtime.h"
#include <hipsparse.h>
#elif USE_CUDA
#endif

namespace test
{

/// Computes y += A*x for a local CSR matrix A and local dense vectors x,y
/// @param[in] values Nonzero values of A
/// @param[in] row_begin First index of each row in the arrays values and
/// indices.
/// @param[in] row_end Last index of each row in the arrays values and indices.
/// @param[in] indices Column indices for each non-zero element of the matrix A
/// @param[in] x Input vector
/// @param[in, out] y Output vector
template <typename T>
void spmv_impl(std::span<const T> values, std::span<const std::int32_t> row_begin,
               std::span<const std::int32_t> row_end, std::span<const std::int32_t> indices,
               std::span<const T> x, std::span<T> y)
{
  assert(row_begin.size() == row_end.size());
  for (std::size_t i = 0; i < row_begin.size(); i++)
  {
    T vi{0};
    for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
      vi += values[j] * x[indices[j]];
    y[i] += vi;
  }
}

/// Computes y += A^T*x for a local CSR matrix A and local dense vectors x,y
/// @param[in] values Nonzero values of A
/// @param[in] row_begin First index of each row in the arrays values and
/// indices.
/// @param[in] row_end Last index of each row in the arrays values and indices.
/// @param[in] indices Column indices for each non-zero element of the matrix A
/// @param[in] x Input vector
/// @param[in, out] y Output vector
template <typename T>
void spmvT_impl(std::span<const T> values, std::span<const std::int32_t> row_begin,
                std::span<const std::int32_t> row_end, std::span<const std::int32_t> indices,
                std::span<const T> x, std::span<T> y)
{
  assert(row_begin.size() == row_end.size());
  for (std::size_t i = 0; i < row_begin.size(); i++)
  {
    for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
      y[indices[j]] += values[j] * x[i];
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
/// @param[in, out] y Output vector
template <typename T>
__global__ void spmv_impl(int N, const T* values, const std::int32_t* row_begin,
                          const std::int32_t* row_end, const std::int32_t* indices, const T* x,
                          T* y)
{
  // Calculate the row index for this thread.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the row index is out of bounds.
  if (i < N)
  {
    // Perform the sparse matrix-vector multiplication for this row.
    T vi{0};
    for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
      vi += values[j] * x[indices[j]];
    y[i] += vi;
  }
}

template <typename T>
__global__ void spmvT_impl(int N, const T* values, const std::int32_t* row_begin,
                           const std::int32_t* row_end, const std::int32_t* indices, const T* x,
                           T* y)
{
  // Calculate the row index for this thread.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the row index is out of bounds.
  if (i < N)
  {
    // Perform the transpose sparse matrix-vector multiplication for this row.
    for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
      atomicAdd(&y[indices[j]], values[j] * x[i]);
  }
}

} // namespace

#ifdef USE_HIP
#define err_check(command)                                                                         \
  {                                                                                                \
    hipError_t status = command;                                                                   \
    if (status != hipSuccess)                                                                      \
    {                                                                                              \
      printf("(%s:%d) Error: Hip reports %s\n", __FILE__, __LINE__, hipGetErrorString(status));    \
      exit(1);                                                                                     \
    }                                                                                              \
  }
#elif USE_CUDA
#define err_check(command)                                                                         \
  {                                                                                                \
    cudaError_t status = command;                                                                  \
    if (status != cudaSuccess)                                                                     \
    {                                                                                              \
      printf("(%s:%d) Error: CUDA reports %s\n", __FILE__, __LINE__, cudaGetErrorString(status));  \
      exit(1);                                                                                     \
    }                                                                                              \
  }
#elif CPU
#define err_check(command)                                                                         \
  {                                                                                                \
    int status = command;                                                                          \
    if (status != 0)                                                                               \
    {                                                                                              \
      printf("(%s:%d) Error: Report %s\n", __FILE__, __LINE__, perror());                          \
      exit(1);                                                                                     \
    }                                                                                              \
  }
#endif

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
                 const std::vector<std::shared_ptr<const fem::DirichletBC<T, double>>>& bcs)
  {

    dolfinx::common::Timer t0("~setup phase MatrixOperator");

    if (a->rank() != 2)
      throw std::runtime_error("Form should have rank be 2.");

    auto V = a->function_spaces()[0];
    la::SparsityPattern pattern = fem::create_sparsity_pattern(*a);
    pattern.finalize();
    _col_map = std::make_shared<const common::IndexMap>(pattern.column_index_map());
    _row_map = V->dofmap()->index_map;

    _A = std::make_unique<
        la::MatrixCSR<T, std::vector<T>, std::vector<std::int32_t>, std::vector<std::int32_t>>>(
        pattern);
    fem::assemble_matrix(_A->mat_add_values(), *a, bcs);
    _A->scatter_rev();
    fem::set_diagonal<T>(_A->mat_set_values(), *V, bcs, T(1.0));

    // Get communicator from mesh
    _comm = V->mesh()->comm();

    std::int32_t num_rows = _row_map->size_local();
    std::int32_t nnz = _A->row_ptr()[num_rows];
    _nnz = nnz;

    // Allocate data on device
#ifdef USE_HIP
    err_check(hipMalloc((void**)&_row_ptr, (num_rows + 1) * sizeof(std::int32_t)));
    err_check(hipMalloc((void**)&_off_diag_offset, num_rows * sizeof(std::int32_t)));
    err_check(hipMalloc((void**)&_cols, nnz * sizeof(std::int32_t)));
    err_check(hipMalloc((void**)&_values, nnz * sizeof(T)));

    // Copy data from host to device
    err_check(hipMemcpy(_row_ptr, _A->row_ptr().data(), (num_rows + 1) * sizeof(std::int32_t),
                        hipMemcpyHostToDevice));
    err_check(hipMemcpy(_off_diag_offset, _A->off_diag_offset().data(),
                        num_rows * sizeof(std::int32_t), hipMemcpyHostToDevice));

    err_check(
        hipMemcpy(_cols, _A->cols().data(), nnz * sizeof(std::int32_t), hipMemcpyHostToDevice));
    err_check(hipMemcpy(_values, _A->values().data(), nnz * sizeof(T), hipMemcpyHostToDevice));
    err_check(hipDeviceSynchronize());
#elif USE_CUDA
    err_check(cudaMalloc((void**)&_row_ptr, (num_rows + 1) * sizeof(std::int32_t)));
    err_check(cudaMalloc((void**)&_off_diag_offset, num_rows * sizeof(std::int32_t)));
    err_check(cudaMalloc((void**)&_cols, nnz * sizeof(std::int32_t)));
    err_check(cudaMalloc((void**)&_values, nnz * sizeof(T)));

    // Copy data from host to device
    err_check(cudaMemcpy(_row_ptr, _A->row_ptr().data(), (num_rows + 1) * sizeof(std::int32_t),
                         cudaMemcpyHostToDevice));
    err_check(cudaMemcpy(_off_diag_offset, _A->off_diag_offset().data(),
                         num_rows * sizeof(std::int32_t), cudaMemcpyHostToDevice));

    err_check(
        cudaMemcpy(_cols, _A->cols().data(), nnz * sizeof(std::int32_t), cudaMemcpyHostToDevice));
    err_check(cudaMemcpy(_values, _A->values().data(), nnz * sizeof(T), cudaMemcpyHostToDevice));
    err_check(cudaDeviceSynchronize());
#endif
  }

  MatrixOperator(const fem::FunctionSpace<T>& V0, const fem::FunctionSpace<T>& V1)
  {
    dolfinx::common::Timer t0("~setup phase Interpolation Operators");
    _comm = V0.mesh()->comm();
    assert(V0.mesh());
    auto mesh = V0.mesh();
    assert(V1.mesh());
    assert(mesh == V1.mesh());

    std::shared_ptr<const fem::DofMap> dofmap0 = V0.dofmap();
    assert(dofmap0);
    std::shared_ptr<const fem::DofMap> dofmap1 = V1.dofmap();
    assert(dofmap1);

    // Create and build  sparsity pattern
    assert(dofmap0->index_map);
    assert(dofmap1->index_map);

    la::SparsityPattern pattern(_comm, {dofmap1->index_map, dofmap0->index_map},
                                {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

    int tdim = mesh->topology()->dim();
    auto map = mesh->topology()->index_map(tdim);
    assert(map);
    std::vector<std::int32_t> c(map->size_local(), 0);
    std::iota(c.begin(), c.end(), 0);
    fem::sparsitybuild::cells(pattern, {c, c}, {*dofmap1, *dofmap0});
    pattern.finalize();

    // Build operator
    _A = std::make_unique<
        la::MatrixCSR<T, std::vector<T>, std::vector<std::int32_t>, std::vector<std::int32_t>>>(
        pattern);

    // FIXME: should this be mat_add or mat_set?
    fem::interpolation_matrix<T>(V0, V1, _A->mat_set_values());
    _A->scatter_rev();

    // Create HIP matrix
    _col_map = std::make_shared<const common::IndexMap>(pattern.column_index_map());
    _row_map = V1.dofmap()->index_map;

    // Create hip sparse matrix
    std::int32_t num_rows = _row_map->size_local();
    std::int32_t nnz = _A->row_ptr()[num_rows];
    _nnz = nnz;

    LOG(WARNING) << "Operator Number of non zeros " << _nnz;
    LOG(WARNING) << "Operator Number of rows " << num_rows;
    LOG(WARNING) << "Operator dm0 size " << V0.dofmap()->index_map->size_global();
    LOG(WARNING) << "Operator dm1 size " << V1.dofmap()->index_map->size_global();
    LOG(WARNING) << "Max column = " << *std::max_element(_A->cols().begin(), _A->cols().end());

#ifdef USE_HIP
    // Allocate data on device
    err_check(hipMalloc((void**)&_row_ptr, (num_rows + 1) * sizeof(std::int32_t)));
    err_check(hipMalloc((void**)&_off_diag_offset, num_rows * sizeof(std::int32_t)));
    err_check(hipMalloc((void**)&_cols, nnz * sizeof(std::int32_t)));
    err_check(hipMalloc((void**)&_values, nnz * sizeof(T)));

    // Copy data from host to device
    err_check(hipMemcpy(_row_ptr, _A->row_ptr().data(), (num_rows + 1) * sizeof(std::int32_t),
                        hipMemcpyHostToDevice));
    err_check(hipMemcpy(_off_diag_offset, _A->off_diag_offset().data(),
                        num_rows * sizeof(std::int32_t), hipMemcpyHostToDevice));

    err_check(
        hipMemcpy(_cols, _A->cols().data(), nnz * sizeof(std::int32_t), hipMemcpyHostToDevice));
    err_check(hipMemcpy(_values, _A->values().data(), nnz * sizeof(T), hipMemcpyHostToDevice));
    err_check(hipDeviceSynchronize());
#elif USE_CUDA
    // Allocate data on device
    err_check(hipMalloc((void**)&_row_ptr, (num_rows + 1) * sizeof(std::int32_t)));
    err_check(hipMalloc((void**)&_off_diag_offset, num_rows * sizeof(std::int32_t)));
    err_check(hipMalloc((void**)&_cols, nnz * sizeof(std::int32_t)));
    err_check(hipMalloc((void**)&_values, nnz * sizeof(T)));

    // Copy data from host to device
    err_check(hipMemcpy(_row_ptr, _A->row_ptr().data(), (num_rows + 1) * sizeof(std::int32_t),
                        hipMemcpyHostToDevice));
    err_check(hipMemcpy(_off_diag_offset, _A->off_diag_offset().data(),
                        num_rows * sizeof(std::int32_t), hipMemcpyHostToDevice));

    err_check(
        hipMemcpy(_cols, _A->cols().data(), nnz * sizeof(std::int32_t), hipMemcpyHostToDevice));
    err_check(hipMemcpy(_values, _A->values().data(), nnz * sizeof(T), hipMemcpyHostToDevice));
    err_check(hipDeviceSynchronize());
#endif
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
    dolfinx::common::Timer t0("% MatrixOperator application");

    y.set(T{0});
    T* _x = x.mutable_array().data();
    T* _y = y.mutable_array().data();

    if (transpose)
    {
#ifdef USE_HIPSPARSE
      int num_cols = _col_map->size_local() + _col_map->num_ghosts();
      int num_rows = _row_map->size_local();
      x.scatter_fwd();
      T alpha = 1.0;
      T beta = 0.0;
      hipsparseDcsrmv(handle, HIPSPARSE_OPERATION_TRANSPOSE, num_rows, num_cols, _nnz, &alpha,
                      descrA, _values, _row_ptr, _cols, _x, &beta, _y);
      err_check(hipGetLastError());
      err_check(hipDeviceSynchronize());
#else
      int num_rows = _row_map->size_local();
      dim3 block_size(256);
      dim3 grid_size((num_rows + block_size.x - 1) / block_size.x);
      x.scatter_fwd_begin();
#ifdef USE_HIP
      hipLaunchKernelGGL(spmvT_impl<T>, grid_size, block_size, 0, 0, num_rows, _values, _row_ptr,
                         _off_diag_offset, _cols, _x, _y);
      err_check(hipGetLastError());
#elif USE_CUDA
      spmv_impl<T> << grid_size, block_size, 0, 0,
          num_rows >> (_values, _row_ptr, _off_diag_offset, _cols, _x, _y);
      err_check(cudaGetLastError());
#elif CPU
      spmv_impl<T>(A->values().data(), A->row_ptr().data(), A->off_diag_offset().data(),
                   A->cols().data(), _x, _y);
#endif
      x.scatter_fwd_end();
#ifdef USE_HIP
      hipLaunchKernelGGL(spmvT_impl<T>, grid_size, block_size, 0, 0, num_rows, _values,
                         _off_diag_offset, _row_ptr + 1, _cols, _x, _y);
      err_check(hipGetLastError());
#elif USE_CUDA
      spmv_impl<T> << grid_size, block_size, 0, 0,
          num_rows >> (_values, _off_diag_offset, _row_ptr + 1, _cols, _x, _y);
      err_check(cudaGetLastError());
#elif CPU
      spmv_impl<T>(A->values().data(), A->off_diag_offset().data(), A->row_ptr().data() + 1,
                   A->cols().data(), _x, _y);
#endif
#endif
    }
    else
    {
#ifdef USE_HIPSPARSE
      int num_cols = _col_map->size_local() + _col_map->num_ghosts();
      int num_rows = _row_map->size_local();
      x.scatter_fwd();
      T alpha = 1.0;
      T beta = 0.0;
      hipsparseDcsrmv(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, num_rows, num_cols, _nnz, &alpha,
                      descrA, _values, _row_ptr, _cols, _x, &beta, _y);
      err_check(hipGetLastError());
      err_check(hipDeviceSynchronize());
#else
      int num_rows = _row_map->size_local();
      dim3 block_size(256);
      dim3 grid_size((num_rows + block_size.x - 1) / block_size.x);
      x.scatter_fwd_begin();
#ifdef USE_HIP
      hipLaunchKernelGGL(spmv_impl<T>, grid_size, block_size, 0, 0, num_rows, _values, _row_ptr,
                         _off_diag_offset, _cols, _x, _y);
      err_check(hipGetLastError());
#elif USE_CUDA
      spmv_impl<T> << grid_size, block_size, 0, 0,
          num_rows >> (_values, _row_ptr, _off_diag_offset, _cols, _x, _y);
      err_check(cudaGetLastError());
#elif CPU
      spmv_impl<T>(A->values().data(), A->row_ptr().data(), A->off_diag_offset().data(),
                   A->cols().data(), _x, _y);
#endif
      x.scatter_fwd_end();
#ifdef USE_HIP
      hipLaunchKernelGGL(spmv_impl<T>, grid_size, block_size, 0, 0, num_rows, _values,
                         _off_diag_offset, _row_ptr + 1, _cols, _x, _y);
      err_check(hipGetLastError());
#elif USE_CUDA
      spmv_impl<T> << grid_size, block_size, 0, 0,
          num_rows >> (_values, _off_diag_offset, _row_ptr + 1, _cols, _x, _y);
      err_check(cudaGetLastError());
#elif CPU
      spmv_impl<T>(A->values().data(), A->off_diag_offset().data(), A->row_ptr().data() + 1,
                   A->cols().data(), _x, _y);
#endif
#endif
    }
  }

  template <typename Vector>
  void apply_host(Vector& x, Vector& y, bool transpose = false)
  {
    test::spmv(*_A, x, y);
  }

  std::shared_ptr<const common::IndexMap> column_index_map() { return _col_map; }

  std::shared_ptr<const common::IndexMap> row_index_map() { return _row_map; }

  std::size_t nnz() { return _nnz; }

  ~MatrixOperator()
  {
#ifdef USE_HIP
    err_check(hipFree(_values));
    err_check(hipFree(_row_ptr));
    err_check(hipFree(_cols));
    err_check(hipFree(_off_diag_offset));
#endif

#ifdef USE_HIPSPARSE
    hipsparseDestroyMatDescr(descrA);
    hipsparseDestroy(handle);
#endif
  }

private:
  std::size_t _nnz;
  T* _values;
  std::int32_t* _row_ptr;
  std::int32_t* _cols;
  std::int32_t* _off_diag_offset;
  std::shared_ptr<const common::IndexMap> _col_map, _row_map;
  std::unique_ptr<
      la::MatrixCSR<T, std::vector<T>, std::vector<std::int32_t>, std::vector<std::int32_t>>>
      _A;

#ifdef USE_HIPSPARSE
  hipsparseMatDescr_t descrA;
  hipsparseHandle_t handle;
#endif

  MPI_Comm _comm;
};
} // namespace dolfinx::acc
