// CUDA execution backend (task spec item 5).
//
// NOTE ON VERIFICATION: this file could not be compiled or run in the
// Phase 0 development environment because no CUDA toolkit/NVIDIA GPU was
// available (see docs/adr/0001-execution-backend.md and
// docs/performance/0001-phase0-results.md). It is written to the same
// standard as the rest of the backend and is wired into CMake behind
// `CFE_ENABLE_CUDA`, but it is unverified. Treat it as a reviewable proposal
// until it has been built and profiled on real hardware.
//
// This header must only be included from `.cu` translation units compiled
// by nvcc (triple-chevron kernel launch syntax is not valid in ordinary
// C++ translation units). Keeping CUDA syntax confined to files under
// `backend/cuda/` follows AGENTS.md #9.
#pragma once

#include <cstddef>

#include "cfe/core/macros.hpp"

namespace cfe {
namespace backend {
namespace cuda {

template <class Index, class Functor>
CFE_GLOBAL void parallel_for_kernel(Index n, Functor f)
{
  const Index i = static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x) +
                  static_cast<Index>(threadIdx.x);
  if (i < n) {
    f(i);
  }
}

// Launches one thread per index in [0, n) and blocks until completion.
// `block_size` is a simple, unmeasured default (256); it should be revisited
// once occupancy/register data is available (see scripts/profile_cuda.sh).
template <class Index, class Functor>
void parallel_for(Index n,
                  Functor f,
                  int block_size = 256)
{
  if (n <= 0) return;
  const int grid_size = static_cast<int>((static_cast<long long>(n) + block_size - 1) / block_size);
  parallel_for_kernel<Index, Functor><<<grid_size, block_size>>>(n, f);
  cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace backend
}  // namespace cfe
