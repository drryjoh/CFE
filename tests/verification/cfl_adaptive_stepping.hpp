// Shared CFL-adaptive SSP-RK2 time-advance loop: recomputes `dt` every
// step from the field's own CURRENT max wave speed (not a fixed a
// priori bound, since the maximum signal speed changes as a shock/
// rarefaction forms), clamping the final step so it lands exactly on
// `t_target`.
//
// Factored out after this ~12-line block had been copy-pasted into a
// THIRD canonical-case driver (sod_shock_tube_runner.hpp,
// shu_osher_setup.hpp, propagating_shock_setup.hpp all carried their
// own copy independently) -- Snorlax's review finding on tasks 0012/
// 0013: duplication this codebase already has precedent for noticing
// and fixing once (see sod_shock_tube_runner.hpp's own
// `contact_width_cells`), but which had crept back in three times over
// before anyone factored out the LOOP itself rather than just the
// reporting logic around it.
#pragma once

#include <cmath>
#include <cstddef>

#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/math/fixed_array.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace cfe_verification {

// Advances a 1D (ny=nz=1) FieldView from `t_current` to `t_target`,
// returning the time actually reached (exactly `t_target`, barring
// floating-point roundoff in the final clamped step -- returned rather
// than assumed so a caller can assert on it). `field` must provide
// `velocity_component(state, Axis::X)` and `sound_speed(state)`
// (every `EulerField` Calculator does); `residual` is the same
// `(FieldView in, FieldView out) -> void` callable `ssp_rk2_step`
// already takes.
template <class Scalar, std::size_t N, class Layout, class Field, class Residual, class IndexMap>
Scalar advance_cfl_adaptive(cfe::FieldView<Scalar, N, Layout> state, cfe::FieldView<Scalar, N, Layout> stage1,
                             cfe::FieldView<Scalar, N, Layout> scratch, const cfe::CartesianGrid<Scalar>& grid,
                             const Field& field, Residual residual, cfe::Index n_active, const IndexMap& index_map,
                             Scalar t_current, Scalar t_target, Scalar cfl)
{
  Scalar t = t_current;
  while (t < t_target) {
    Scalar max_speed = Scalar(0.0);
    for (cfe::Index i = 0; i < grid.nx; ++i) {
      cfe::State<Scalar, N> q;
      const cfe::Index cell = grid.flat_index(grid.ngx + i, 0, 0);
      for (cfe::Index c = 0; c < N; ++c) q[c] = state(cell, c);
      const Scalar speed = std::abs(field.velocity_component(q, cfe::Axis::X)) + field.sound_speed(q);
      max_speed = speed > max_speed ? speed : max_speed;
    }
    Scalar dt = cfl * grid.dx / max_speed;
    if (t + dt > t_target) dt = t_target - t;
    cfe::ssp_rk2_step<Scalar>(state, stage1, scratch, dt, residual, n_active, index_map);
    t += dt;
  }
  return t;
}

}  // namespace cfe_verification
