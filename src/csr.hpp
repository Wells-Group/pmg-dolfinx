#include "hip/hip_runtime.h"
#include <dolfinx/la/MatrixCSR.h>

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
class Matrix
{
public:
  /// The value type
  using value_type = T;

  /// Create a distributed vector
  Matrix(dolfinx::la::MatrixCSR<T>& A)
  {
    // Allocate data on device
    err_check(hipMalloc((void**)&row_ptr, A.row_ptr().size() * sizeof(std::int32_t)));
    err_check(hipMalloc((void**)&cols, A.cols().size() * sizeof(std::int32_t)));
    err_check(
        hipMalloc((void**)&off_diag_offset, A.off_diag_offset().size() * sizeof(std::int32_t)));
    err_check(hipMalloc((void**)&values, A.values().size() * sizeof(T)));

    // Copy data from host to device
    err_check(hipMemcpy(row_ptr, A.row_ptr().data(), A.row_ptr().size() * sizeof(std::int32_t),
                        hipMemcpyHostToDevice));
    err_check(hipMemcpy(cols, A.cols().data(), A.cols().size() * sizeof(std::int32_t),
                        hipMemcpyHostToDevice));
    err_check(hipMemcpy(off_diag_offset, A.off_diag_offset().data(),
                        A.off_diag_offset().size() * sizeof(std::int32_t), hipMemcpyHostToDevice));
    err_check(
        hipMemcpy(values, A.values().data(), A.values().size() * sizeof(T), hipMemcpyHostToDevice));
  }

  ~Matrix()
  {
    err_check(hipFree(values));
    err_check(hipFree(row_ptr));
    err_check(hipFree(cols));
    err_check(hipFree(off_diag_offset));
  }

private:
  T* values;
  std::int32_t* row_ptr;
  std::int32_t* cols;
  std::int32_t* off_diag_offset;
};
} // namespace dolfinx::acc

// // /// Computes y += A*x for a local CSR matrix A and local dense vectors x,y
// // /// @param[in] values Nonzero values of A
// // /// @param[in] row_begin First index of each row in the arrays values and
// // /// indices.
// // /// @param[in] row_end Last index of each row in the arrays values and
// // indices.
// // /// @param[in] indices Column indices for each non-zero element of the matrix
// // A
// // /// @param[in] x Input vector
// // /// @param[in, out] x Output vector
// // template <typename T>
// // __global__ void spmv_impl(std::span<const T> values,
// //                           std::span<const std::int32_t> row_begin,
// //                           std::span<const std::int32_t> row_end,
// //                           std::span<const std::int32_t> indices,
// //                           std::span<const T> x, std::span<T> y)
// // {
// //   // Calculate the row index for this thread.
// //   int i = blockIdx.x * blockDim.x + threadIdx.x;

// //   // Check if the row index is out of bounds.
// //   if (i >= row_begin.size())
// //     return;

// //   // Perform the sparse matrix-vector multiplication for this row.
// //   T vi{0};
// //   for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
// //     vi += values[j] * x[indices[j]];
// //   y[i] += vi;
// // }

// // // The matrix A is distributed across P  processes by blocks of rows:
// // //  A = |   A_0  |
// // //      |   A_1  |
// // //      |   ...  |
// // //      |  A_P-1 |
// // //
// // // Each submatrix A_i is owned by a single process "i" and can be further
// // // decomposed into diagonal (Ai[0]) and off diagonal (Ai[1]) blocks:
// // //  Ai = |Ai[0] Ai[1]|
// // //
// // // If A is square, the diagonal block Ai[0] is also square and contains
// // // only owned columns and rows. The block Ai[1] contains ghost columns
// // // (unowned dofs).

// // // Likewise, a local vector x can be decomposed into owned and ghost blocks:
// // // xi = |   x[0]  |
// // //      |   x[1]  |
// // //
// // // So the product y = Ax can be computed into two separate steps:
// // //  y[0] = |Ai[0] Ai[1]| |   x[0]  | = Ai[0] x[0] + Ai[1] x[1]
// // //                       |   x[1]  |
// // //
// // /// Computes y += A*x for a parallel CSR matrix A and parallel dense vectors
// // x,y
// // /// @param[in] A Parallel CSR matrix
// // /// @param[in] x Input vector
// // /// @param[in, out] y Output vector
// // template <typename T>
// // void spmv(la::MatrixCSR<T>& A, la::Vector<T>& x, la::Vector<T>& y)

// // {
// //   // start communication (update ghosts)
// //   // x.scatter_fwd_begin();

// //   const std::int32_t nrowslocal = A.num_owned_rows();
// //   std::span<const std::int32_t> row_ptr(A.row_ptr().data(), nrowslocal + 1);
// //   std::span<const std::int32_t> cols(A.cols().data(), row_ptr[nrowslocal]);
// //   std::span<const std::int32_t> off_diag_offset(A.off_diag_offset().data(),
// //                                                 nrowslocal);
// //   std::span<const T> values(A.values().data(), row_ptr[nrowslocal]);

// //   std::span<const std::int32_t> row_begin(row_ptr.data(), nrowslocal);
// //   std::span<const std::int32_t> row_end(
// //       row_ptr.data() + 1, nrowslocal); // First stage:  sp:v - diagonal

// //   // yi[0] += Ai[0] * xi[0]
// //   // spmv_impl<T><<>>(values, row_begin, off_diag_offset, cols, _x, _y);
// //   dim3 block_size(1024);
// //   dim3 grid_size((row_begin.size() + block_size.x - 1) / block_size.x);

// //   hipLaunchKernelGGL(spmv_impl<T>, block_size, grid_size, 0, 0, values,
// //                      row_begin, off_diag_offset, cols, _x, _y);

// //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
// //   // row_endcols, _x, _y);
// //   x.scatter_fwd_end();

// //   // Second stage:  spmv - off-diagonal
// //   // yi[0] += Ai[1] * xi[1]
// //   spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
// // }/
// //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
// //   // row_endcols, _x, _y);
// //  ]
// // finalize ghost updsgnd stage:  spmv - off-diagonal
// //} yi[0] += Ai[1] * xi[1]
// spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
// // Second stage:  spmv - off-diagonal
// //   // yi[0] += Ai[1] * xi[1]
// //   spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
// // }/
// //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
// //   // row_endcols, _x, _y);
// //  ]
// // finalize ghost updsgnd stage:  spmv - off-diagonal
// //} yi[0] += Ai[1] * xi[1]
// spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
// /
//     //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
//     //   // row_endcols, _x, _y);
//     //  ]
//     // finalize ghost updsgnd stage:  spmv - off-diagonal
//     //} yi[0] += Ai[1] * xi[1]
//     // Second stage:  spmv - off-diagonal
//     //   // yi[0] += Ai[1] * xi[1]
//     //   spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
//     // }/
//     //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
//     //   // row_endcols, _x, _y);
//     //  ]
//     // finalize ghost updsgnd stage:  spmv - off-diagonal
//     //} yi[0] += Ai[1] * xi[1]
//     spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
// /
//     //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
//     //   // row_endcols, _x, _y);
//     //  ]
//     // finalize ghost updsgnd stage:  spmv - off-diagonal
//     //} yi[0] += Ai[1] * xi[1]
//     // Second stage:  spmv - off-diagonal
//     //   // yi[0] += Ai[1] * xi[1]
//     //   spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
//     // }/
//     //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
//     //   // row_endcols, _x, _y);
//     //  ]
//     // finalize ghost updsgnd stage:  spmv - off-diagonal
//     //} yi[0] += Ai[1] * xi[1]
//     spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
// /
//     //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
//     //   // row_endcols, _x, _y);
//     //  ]
//     // finalize ghost updsgnd stage:  spmv - off-diagonal
//     //} yi[0] += Ai[1] * xi[1]
//     // Second stage:  spmv - off-diagonal
//     //   // yi[0] += Ai[1] * xi[1]
//     //   spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
//     // }/
//     //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
//     //   // row_endcols, _x, _y);
//     //  ]
//     // finalize ghost updsgnd stage:  spmv - off-diagonal
//     //} yi[0] += Ai[1] * xi[1]
//     spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
// /
//     //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
//     //   // row_endcols, _x, _y);
//     //  ]
//     // finalize ghost updsgnd stage:  spmv - off-diagonal
//     //} yi[0] += Ai[1] * xi[1]
//     // Second stage:  spmv - off-diagonal
//     //   // yi[0] += Ai[1] * xi[1]
//     //   spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
//     // }/
//     //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
//     //   // row_endcols, _x, _y);
//     //  ]
//     // finalize ghost updsgnd stage:  spmv - off-diagonal
//     //} yi[0] += Ai[1] * xi[1]
//     spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
// /
//     //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
//     //   // row_endcols, _x, _y);
//     //  ]
//     // finalize ghost updsgnd stage:  spmv - off-diagonal
//     //} yi[0] += Ai[1] * xi[1]
//     // Second stage:  spmv - off-diagonal
//     //   // yi[0] += Ai[1] * xi[1]
//     //   spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
//     // }/
//     //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
//     //   // row_endcols, _x, _y);
//     //  ]
//     // finalize ghost updsgnd stage:  spmv - off-diagonal
//     //} yi[0] += Ai[1] * xi[1]
//     spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
// /
//     //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
//     //   // row_endcols, _x, _y);
//     //  ]
//     // finalize ghost updsgnd stage:  spmv - off-diagonal
//     //} yi[0] += Ai[1] * xi[1]
//     // Second stage:  spmv - off-diagonal
//     //   // yi[0] += Ai[1] * xi[1]
//     //   spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
//     // }/
//     //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
//     //   // row_endcols, _x, _y);
//     //  ]
//     // finalize ghost updsgnd stage:  spmv - off-diagonal
//     //} yi[0] += Ai[1] * xi[1]
//     spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
// /
//     //   // Second stage:  spmv - off-diagonal4);]+=iAi[0o//fspmv_offedt, ,
//     //   // row_endcols, _x, _y);
//     //  ]
//     // finalize ghost updsgnd stage:  spmv - off-diagonal
//     //} yi[0] += Ai[1] * xi[1]
