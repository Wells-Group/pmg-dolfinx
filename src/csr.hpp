#include <dolfinx.h>
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/MatrixCSR.h>

#include "hip/hip_runtime.h"
#include <hipsparse.h>
// #elif USE_CUDA
// #include <cuda_runtime.h>
// #include <cusparse.h>
// #endif

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

    _row_ptr = thrust::device_vector<std::int32_t>(num_rows + 1);
    _off_diag_offset = thrust::device_vector<std::int32_t>(num_rows);
    _cols = thrust::device_vector<std::int32_t>(nnz);
    _values = thrust::device_vector<T>(nnz);

    // Copy data from host to device
    LOG(WARNING) << "Creating Device matrix with  " << _nnz << " non zeros";
    thrust::copy(_A->row_ptr().begin(), _A->row_ptr().end(), _row_ptr.begin());
    thrust::copy(_A->off_diag_offset().begin(), _A->off_diag_offset().begin() + num_rows, _off_diag_offset.begin());
    thrust::copy(_A->cols().begin(), _A->cols().begin() + nnz, _cols.begin());
    thrust::copy(_A->values().begin(), _A->values().begin() + nnz, _values.begin());

    hipsparseCreate(&handle);
    hipsparseCreateMatDescr(&descrA);
    hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL);
    hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO);
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

    _row_ptr = thrust::device_vector<std::int32_t>(num_rows + 1);
    _off_diag_offset = thrust::device_vector<std::int32_t>(num_rows);
    _cols = thrust::device_vector<std::int32_t>(nnz);
    _values = thrust::device_vector<T>(nnz);

    // Copy data from host to device
    thrust::copy(_A->row_ptr().begin(), _A->row_ptr().begin() + num_rows + 1, _row_ptr.begin());
    thrust::copy(_A->off_diag_offset().begin(), _A->off_diag_offset().begin() + num_rows, _off_diag_offset.begin());
    thrust::copy(_A->cols().begin(), _A->cols().begin() + nnz, _cols.begin());
    thrust::copy(_A->values().begin(), _A->values().begin() + nnz, _values.begin());

    hipsparseCreate(&handle);
    hipsparseCreateMatDescr(&descrA);
    hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL);
    hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO);
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
    LOG(WARNING) << "MatrixOperator application";
    dolfinx::common::Timer t0("% MatrixOperator application");

    x.scatter_fwd_begin();
    y.set(T{0}); // FIXME: This should be done automatically when beta is 0
    T* _x = x.mutable_array().data();
    T* _y = y.mutable_array().data();

    int num_cols = _col_map->size_local() + _col_map->num_ghosts();
    int num_rows = _row_map->size_local();
    T alpha = 1.0;
    T beta = 0.0;
    x.scatter_fwd_end();

    hipsparseOperation_t transa = transpose ? HIPSPARSE_OPERATION_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE;
    T* values = thrust::raw_pointer_cast(_values.data());
    std::int32_t* row_ptr = thrust::raw_pointer_cast(_row_ptr.data());
    std::int32_t* cols = thrust::raw_pointer_cast(_cols.data());
    hipsparseDcsrmv(handle, transa, num_rows, num_cols, _nnz, &alpha,
                    descrA, values, row_ptr, cols, _x, &beta, _y);
    err_check(hipGetLastError());
    err_check(hipDeviceSynchronize());
  }

  // template <typename Vector>
  // void apply_host(Vector& x, Vector& y, bool transpose = false)
  // {
  //   test::spmv(*_A, x, y);
  // }

  std::shared_ptr<const common::IndexMap> column_index_map() { return _col_map; }

  std::shared_ptr<const common::IndexMap> row_index_map() { return _row_map; }

  std::size_t nnz() { return _nnz; }

  ~MatrixOperator()
  {
    hipsparseDestroyMatDescr(descrA);
    hipsparseDestroy(handle);
  }

private:
  std::size_t _nnz;
  thrust::device_vector<T> _values;
  thrust::device_vector<std::int32_t> _row_ptr;
  thrust::device_vector<std::int32_t> _cols;
  thrust::device_vector<std::int32_t> _off_diag_offset;
  std::shared_ptr<const common::IndexMap> _col_map, _row_map;
  std::unique_ptr<
      la::MatrixCSR<T, std::vector<T>, std::vector<std::int32_t>, std::vector<std::int32_t>>>
      _A;

  // HIP specific
  hipsparseMatDescr_t descrA;
  hipsparseHandle_t handle;

  MPI_Comm _comm;
};
} // namespace dolfinx::acc
