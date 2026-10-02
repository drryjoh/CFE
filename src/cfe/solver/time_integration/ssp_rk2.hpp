// SSP-RK2 (Heun's method / improved Euler) explicit time integration
// (task spec item 6, ARCHITECTURE.md #13).
//
// Chosen over SSP-RK3: the spatial scheme this phase pairs it with
// (central-difference-style FVM reconstruction, see
// numerics/fvm/interface_value.hpp) is 2nd-order accurate, so a 3rd-order
// time integrator would not improve the overall convergence rate proven
// in tests/unit/test_scalar_advection_convergence.cpp -- global error is
// bounded by the lower-order term regardless, so RK3's third stage would
// only add cost, not accuracy, here. See docs/adr/0007-interface-flux-and-time-integration.md.
//
// Generic over a Residual callable: `residual(state_in, out)` must compute
// `out := dQ/dt` given `state_in`, including any ghost-cell fill the residual
// needs internally. Each combine step only visits the `n_active` cells
// named by `index_map` -- ghost cells are never read or written here,
// matching standard FVM practice (fill ghosts -> compute residual on
// interior cells -> integrate interior cells only). This stepper still
// has no knowledge of what a "ghost cell" *is*: `index_map` is an opaque
// `std::size_t -> std::size_t` translation from "the r-th active cell"
// to a storage index, supplied by the caller (`FvmSolver::active_cell_index_map()`
// for a Cartesian grid; the default `IdentityIndexMap` for a caller with
// no ghost cells at all, e.g. a plain ODE system). This used to iterate
// every cell in `state`'s full storage, ghost cells included, which
// required `FvmSolver::residual()` to invent a defined-but-meaningless
// value for every ghost-cell residual entry -- a real, measured
// performance cost for no benefit, since nothing meaningful ever read
// those values anyway (caught in code review; see agent_history.md).
//
// Each combine step is dispatched through `Backend::run` (default
// `CpuParallelFor`; pass `CudaParallelFor` from a `.cu` translation unit)
// with a `CFE_HOST_DEVICE`-annotated lambda (not `CFE_DEVICE`: these
// kernels must also be host-callable, since the default `CpuParallelFor`
// backend invokes them from a plain host loop even when this file is
// compiled by nvcc), exactly like every other Backend-generic kernel in
// this codebase (see backend/parallel_for.hpp, backend/cuda/cuda_backend.cuh,
// grid/boundary/boundary_condition.hpp).
#pragma once

#include <cstddef>

#include "cfe/backend/parallel_for.hpp"
#include "cfe/core/macros.hpp"

namespace cfe {

// Default index map: the r-th active cell IS storage index r. Correct
// whenever there are no ghost cells at all (e.g. integrating a plain ODE
// system with a single-component, no-grid Field), and the reason
// `ssp_rk2_step` can still be used without a grid in play.
struct IdentityIndexMap
{
  CFE_HOST_DEVICE
  CFE_FORCEINLINE
  std::size_t operator()(std::size_t r) const { return r; }
};

// `stage1` and `residual_scratch` are caller-provided scratch storage,
// the same (full, padded if applicable) shape as `state`, reused across
// calls -- never allocated here (AGENTS.md #10: no allocation inside a
// per-timestep hot path). `n_active` is how many cells `index_map`
// actually names (e.g. `grid.nx*grid.ny*grid.nz` for a Cartesian grid,
// `state.n_cells()` for a plain no-ghost caller).
template <class Scalar, class FieldViewT, class Residual, class Backend = CpuParallelFor,
          class IndexMap = IdentityIndexMap>
void ssp_rk2_step(FieldViewT state, FieldViewT stage1, FieldViewT residual_scratch, Scalar dt,
                   Residual residual, std::size_t n_active, IndexMap index_map = IndexMap{})
{
  constexpr std::size_t n_components = FieldViewT::n_components();

  // Stage 1: stage1 = state + dt * residual(state)
  residual(state, residual_scratch);
  Backend::run(n_active, [=] CFE_HOST_DEVICE(std::size_t r) mutable {
    const std::size_t cell = index_map(r);
    for (std::size_t c = 0; c < n_components; ++c) {
      stage1(cell, c) = state(cell, c) + dt * residual_scratch(cell, c);
    }
  });

  // Stage 2: state = 0.5*state + 0.5*(stage1 + dt*residual(stage1))
  residual(stage1, residual_scratch);
  Backend::run(n_active, [=] CFE_HOST_DEVICE(std::size_t r) mutable {
    const std::size_t cell = index_map(r);
    for (std::size_t c = 0; c < n_components; ++c) {
      state(cell, c) =
          Scalar(0.5) * state(cell, c) + Scalar(0.5) * (stage1(cell, c) + dt * residual_scratch(cell, c));
    }
  });
}

}  // namespace cfe
