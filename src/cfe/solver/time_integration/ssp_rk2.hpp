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
// Generic over a Residual callable: `residual(q_in, out)` must compute
// `out := dQ/dt` given `q_in`, including any ghost-cell fill the residual
// needs internally -- this stepper has no knowledge of grids or boundary
// conditions, only of FieldView. Each combine step is dispatched via
// cfe::parallel_for with a lambda, exactly like every other kernel in
// this codebase (see backend/parallel_for.hpp).
#pragma once

#include <cstddef>

#include "cfe/backend/parallel_for.hpp"

namespace cfe {

// `stage1` and `r_buf` are caller-provided scratch storage, the same
// shape as `q`, reused across calls -- never allocated here (AGENTS.md
// #10: no allocation inside a per-timestep hot path).
template <class Scalar, class FieldViewT, class Residual>
void ssp_rk2_step(FieldViewT q, FieldViewT stage1, FieldViewT r_buf, Scalar dt, Residual residual)
{
  const std::size_t n_cells = q.n_cells();
  constexpr std::size_t n_components = FieldViewT::n_components();

  // Stage 1: stage1 = q + dt * residual(q)
  residual(q, r_buf);
  cfe::parallel_for(n_cells, [=](std::size_t cell) mutable {
    for (std::size_t c = 0; c < n_components; ++c) {
      stage1(cell, c) = q(cell, c) + dt * r_buf(cell, c);
    }
  });

  // Stage 2: q = 0.5*q + 0.5*(stage1 + dt*residual(stage1))
  residual(stage1, r_buf);
  cfe::parallel_for(n_cells, [=](std::size_t cell) mutable {
    for (std::size_t c = 0; c < n_components; ++c) {
      q(cell, c) = Scalar(0.5) * q(cell, c) + Scalar(0.5) * (stage1(cell, c) + dt * r_buf(cell, c));
    }
  });
}

}  // namespace cfe
