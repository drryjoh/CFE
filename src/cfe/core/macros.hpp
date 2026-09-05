// CMU-CFE host/device annotation abstraction.
//
// Physics and numerics code must never spell out raw `__host__`/`__device__`
// syntax directly (see AGENTS.md section 9). This header is the only place
// that is allowed to know whether the translation unit is being compiled by
// nvcc.
#pragma once

#if defined(__CUDACC__)
#define CFE_HOST_DEVICE __host__ __device__
#define CFE_DEVICE __device__
#define CFE_HOST __host__
#define CFE_GLOBAL __global__
#define CFE_COMPILING_CUDA 1
#else
#define CFE_HOST_DEVICE
#define CFE_DEVICE
#define CFE_HOST
#define CFE_GLOBAL
#define CFE_COMPILING_CUDA 0
#endif

#if defined(__CUDACC__)
#define CFE_FORCEINLINE __forceinline__
#elif defined(__GNUC__) || defined(__clang__)
#define CFE_FORCEINLINE inline __attribute__((always_inline))
#else
#define CFE_FORCEINLINE inline
#endif
