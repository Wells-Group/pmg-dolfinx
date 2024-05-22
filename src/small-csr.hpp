#include <hip/hip_runtime.h>
#include <thrust/device_vector.h>

// Simple class for a CSR matrix stored on-device
//
template <typename T>
class SmallCSRDevice
{
public:
  SmallCSRDevice(std::span<std::int32_t> _row_ptr, std::span<std::int32_t> _cols,
                 std::span<T> _vals)
      : row_ptr(_row_ptr), cols(_cols), vals(_vals)
  {
  }

  // Apply matrix direct to input data
  __device__ void apply(const T* data_in, T* data_out) const
  {
    for (std::int32_t j = 0; j < row_ptr.size() - 1; j++)
    {
      T vj = 0;
      for (std::int32_t k = row_ptr[j]; k < row_ptr[j + 1]; ++k)
        vj += vals[k] * data_in[cols[k]];
      data_out[j] = vj;
    }
  }

  // Apply matrix to indirect values in arrays with given mappings in map_in and map_out
  __device__ void apply_indirect(const std::int32_t* map_in, const std::int32_t* map_out,
                                 const T* data_in, T* data_out) const
  {
    for (std::int32_t j = 0; j < row_ptr.size() - 1; j++)
    {
      T vj = 0;
      for (std::int32_t k = row_ptr[j]; k < row_ptr[j + 1]; ++k)
        vj += vals[k] * data_in[map_in[cols[k]]];
      data_out[map_out[j]] = vj;
    }
  }

  // Pointers to row offsets, columns and values, already allocated on device
  std::span<std::int32_t> row_ptr;
  std::span<std::int32_t> cols;
  std::span<T> vals;
};

template <typename T>
class SmallCSR
{
public:
  SmallCSR() {}

  ~SmallCSR()
  {
    spdlog::info("Deallocating SmallCSR");
    if (mat_device)
      err_check(hipFree(mat_device));
  }

  SmallCSR(const std::vector<std::int32_t>& row_ptr, const std::vector<std::int32_t>& columns,
           const std::vector<T>& values)
  {
    cols.resize(columns.size());
    thrust::copy(columns.begin(), columns.end(), cols.begin());
    row_offset.resize(row_ptr.size());
    thrust::copy(row_ptr.begin(), row_ptr.end(), row_offset.begin());
    vals.resize(values.size());
    thrust::copy(values.begin(), values.end(), vals.begin());
    SmallCSRDevice<T> m(
        std::span<std::int32_t>(thrust::raw_pointer_cast(row_offset.data()), row_offset.size()),
        std::span<std::int32_t>(thrust::raw_pointer_cast(cols.data()), cols.size()),
        std::span<T>(thrust::raw_pointer_cast(vals.data()), vals.size()));
    err_check(hipMalloc((void**)&mat_device, sizeof(SmallCSRDevice<T>)));
    err_check(hipMemcpy(mat_device, &m, sizeof(SmallCSRDevice<T>), hipMemcpyHostToDevice));
  }

  // Compress a dense matrix to a CSR sparse matrix
  // Values less than tol are set to zero.
  // Input: matrix in RowMajor order, shape
  SmallCSR(const std::vector<T>& mat, std::array<std::size_t, 2> shape, bool use_transpose,
           T tol = 1e-12)
  {
    std::vector<std::int32_t> row_ptr = {0};
    std::vector<std::int32_t> columns;
    std::vector<T> values;

    if (use_transpose)
    {
      for (std::size_t row = 0; row < shape[1]; ++row)
      {
        for (std::size_t col = 0; col < shape[0]; ++col)
        {
          T val = mat[col * shape[1] + row];
          if (std::abs(val) > tol)
          {
            columns.push_back(col);
            values.push_back(val);
          }
        }
        row_ptr.push_back(columns.size());
      }
    }
    else
    {
      for (std::size_t row = 0; row < shape[0]; ++row)
      {
        for (std::size_t col = 0; col < shape[1]; ++col)
        {
          T val = mat[row * shape[1] + col];
          if (std::abs(val) > tol)
          {
            columns.push_back(col);
            values.push_back(val);
          }
        }
        row_ptr.push_back(columns.size());
      }
    }

    spdlog::info("Compressed dense matrix from {} to {} CSR values", mat.size(), values.size());

    cols.resize(columns.size());
    thrust::copy(columns.begin(), columns.end(), cols.begin());
    row_offset.resize(row_ptr.size());
    thrust::copy(row_ptr.begin(), row_ptr.end(), row_offset.begin());
    vals.resize(values.size());
    thrust::copy(values.begin(), values.end(), vals.begin());
    SmallCSRDevice<T> m(
        std::span<std::int32_t>(thrust::raw_pointer_cast(row_offset.data()), row_offset.size()),
        std::span<std::int32_t>(thrust::raw_pointer_cast(cols.data()), cols.size()),
        std::span<T>(thrust::raw_pointer_cast(vals.data()), vals.size()));
    err_check(hipMalloc((void**)&mat_device, sizeof(SmallCSRDevice<T>)));
    err_check(hipMemcpy(mat_device, &m, sizeof(SmallCSRDevice<T>), hipMemcpyHostToDevice));
  }

  const SmallCSRDevice<T>* device_matrix() const { return mat_device; }

private:
  // On-device storage for CSR data
  thrust::device_vector<std::int32_t> row_offset;
  thrust::device_vector<std::int32_t> cols;
  thrust::device_vector<T> vals;

  // Simple struct allocated on device
  SmallCSRDevice<T>* mat_device;
};
