
// Some useful utilities for error checking and synchronisation
// for each hardware type

#pragma once

#ifdef USE_HIP
#include <hip/hip_runtime.h>
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

// Check last reported device error
void check_device_last_error()
{
#ifdef USE_HIP
  err_check(hipGetLastError());
#elif USE_CUDA
  err_check(cudaGetLastError());
#endif
}

void device_synchronize()
{
#ifdef USE_HIP
  err_check(hipDeviceSynchronize());
#elif USE_CUDA
  err_check(cudaDeviceSynchronize());
#endif
}
