// CUDA runtime error checking.
//
// cudaMalloc/cudaMemcpy/cudaDeviceSynchronize return a cudaError_t that is
// silently discardable, and a kernel launch's own configuration errors only
// surface through a separate cudaGetLastError() call after the launch. This
// header centralizes checking all of the above so call sites stay small
// (AGENTS.md #9: CUDA syntax confined to backend/cuda/).
//
// UNVERIFIED: see cuda_backend.cuh -- no CUDA toolkit/hardware was available
// to compile this file during this review pass.
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CFE_CUDA_CHECK(expr)                                                        \
  do {                                                                              \
    const cudaError_t cfe_cuda_check_status_ = (expr);                              \
    if (cfe_cuda_check_status_ != cudaSuccess) {                                    \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" +     \
                               std::to_string(__LINE__) + " (" + #expr +            \
                               "): " + cudaGetErrorString(cfe_cuda_check_status_)); \
    }                                                                               \
  } while (0)
