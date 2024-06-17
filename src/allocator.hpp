// Copyright (C) 2023 Igor A. Baratta
// SPDX-License-Identifier:    MIT

// TODO Is this needed?

#pragma once

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#elif USE_CUDA
#include <cuda/cuda_runtime.h>
#endif
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace dolfinx::acc
{
template <class T>
class allocator
{
public:
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;

  allocator() {}

  template <class U>
  allocator(const allocator<U>&)
  {
  }

  /// Allocates memory that will be automatically managed by the Unified Memory
  /// system
  T* allocate(size_t size)
  {
    T* result = nullptr;

#ifdef USE_HIP
    hipError_t e = hipMallocManaged(&result, size * sizeof(T), hipMemAttachGlobal);
    std::string error_msg = hipGetErrorString(e);

    if (e != hipSuccess)
      throw std::runtime_error("Unable to allocate memory. " + error_msg);
#elif USE_CUDA
    cudaError_t e = cudaMallocManaged(&result, size * sizeof(T), cudaMemAttachGlobal);
    std::string error_msg = cudaGetErrorString(e);

    if (e != cudaSuccess)
      throw std::runtime_error("Unable to allocate memory. " + error_msg);
#elif CPU
    result = malloc(size * sizeof(T));
#endif

    return result;
  }

  void deallocate(T* ptr, size_t)
  {
#ifdef USE_HIP
    hipError_t e = hipFree(ptr);
    std::string error_msg = hipGetErrorString(e);
    if (e != hipSuccess)
      throw std::runtime_error("Unable to deallocate memory" + error_msg);
#elif USE_CUDA
    cudaError_t e = cudaFree(ptr);
    std::string error_msg = cudaGetErrorString(e);
    if (e != cudaSuccess)
      throw std::runtime_error("Unable to deallocate memory" + error_msg);
#elif CPU
    int e = free(ptr);
    if (e != 0)
      throw std::runtime_error("Unable to deallocate memory" + error_msg);
#endif

  }
};

template <class T1, class T2>
bool operator==(const allocator<T1>&, const allocator<T2>&)
{
  return true;
}

template <class T1, class T2>
bool operator!=(const allocator<T1>& lhs, const allocator<T2>& rhs)
{
  return !(lhs == rhs);
}
} // namespace dolfinx::acc
