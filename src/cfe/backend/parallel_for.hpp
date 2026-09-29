// Minimal execution abstraction (task spec item 5, ADR 0001).
//
// `cfe::backend::serial::parallel_for` and `cfe::backend::threaded::parallel_for`
// are always available. `cfe::parallel_for` is an alias for whichever CPU
// backend was selected as the project default via the CMake cache variable
// `CFE_DEFAULT_BACKEND` (serial|threaded); it exists so physics/numerics code
// can write a single, backend-agnostic call site (AGENTS.md #9) without a
// runtime branch. Code that specifically wants to compare backends (as the
// Phase 0 benchmark does) should call the namespaced versions directly.
//
// The CUDA backend (backend/cuda/cuda_backend.cuh) is deliberately not
// aliased here: it can only be invoked from a `.cu` translation unit, so
// call sites that want CUDA select it explicitly and are compiled
// conditionally on `CFE_ENABLE_CUDA`.
#pragma once

#include "cfe/backend/cpu/serial.hpp"
#include "cfe/backend/cpu/threaded.hpp"

namespace cfe {

#if defined(CFE_DEFAULT_BACKEND_THREADED)
using backend::threaded::parallel_for;
#else
using backend::serial::parallel_for;
#endif

// Backend-selection tag, so kernel-launching code written once (grid/
// boundary conditions, solvers, time integrators) can dispatch to either
// CPU or CUDA by a template parameter, rather than being duplicated
// per-backend the way Phase 0's much simpler kernels were.
// `backend/cuda/cuda_backend.cuh` provides the CUDA counterpart
// (`CudaParallelFor`), usable only from a `.cu` translation unit.
struct CpuParallelFor
{
  template <class Index, class Functor>
  static void run(Index n, Functor f)
  {
    cfe::parallel_for(n, f);
  }
};

}  // namespace cfe
