// Standalone 1D propagating shock: a single right-moving shock in
// otherwise-uniform, at-rest gas, with multi-snapshot capture for
// feature-tracking/propagation-speed verification (ROADMAP.md Phase 4,
// tasks/0013-phase4-propagating-shock-tracking.md).
//
// Deliberately NOT a reuse of Sod's own right shock (that problem's
// shock sits adjacent to a contact discontinuity whose numerical
// smearing differs materially between RusanovFlux and HllcFlux --
// docs/adr/0012-hllc-numerical-flux.md's own 17-vs-13-cell contact-width
// result -- which would risk bleeding into a shock-POSITION estimate as
// noise, confounding the one thing this file's own driver measures).
// This is a genuinely simpler IC than Sod's, not merely a "cleaner"
// one: a standalone normal-shock jump is pure closed-form
// Rankine-Hugoniot algebra, no Newton iteration at all -- unlike
// sod_shock_tube_exact.hpp's general iterative Riemann solver, which
// this file does not use and is not a replacement for (that solver
// handles an ARBITRARY left/right state pair with any mix of
// shock/rarefaction/contact; this file handles exactly one normal-shock
// jump at a chosen Mach number, ahead state at rest).
//
// Same shared-driver shape as sod_shock_tube_runner.hpp (templated on
// Scalar and NumericalFlux, default cfe::RusanovFlux, so this case can
// be re-run with cfe::HllcFlux through identical code) -- but ALSO
// captures a profile at each of several REQUESTED output times, not
// just the final one, since feature-tracking needs multiple snapshots
// over the run to estimate a propagation speed.
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/euler/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"
#include "cfl_adaptive_stepping.hpp"
#include "rankine_hugoniot_jump.hpp"  // NormalShockJump / normal_shock_jump, shared with shu_osher_setup.hpp

namespace cfe_verification {

template <class Scalar>
struct ShockSnapshot
{
  Scalar t;
  std::vector<Scalar> x, rho, u, p;
};

// Runs the standalone propagating shock on an `nx`-cell grid over
// `x in [0,1]`, shock initially at `x0`, capturing a profile at each
// time in `output_times` (ascending, all <= the last entry which is
// treated as t_final). Snapshot times are the ACTUAL simulation time
// reached (the timestep is clamped to land exactly on each requested
// output time, generalizing sod_shock_tube_runner.hpp's own
// `if (t + dt > t_final) dt = t_final - t;` clamp to every requested
// output time, not just the last) -- required because the CFL-adaptive
// timestep (recomputed every step from the current max wave speed)
// makes step count alone an unreliable proxy for elapsed time.
template <class Scalar, class NumericalFlux = cfe::RusanovFlux>
std::vector<ShockSnapshot<Scalar>> run_propagating_shock(std::size_t nx, Scalar gamma,
                                                          const NormalShockJump& jump, Scalar x0,
                                                          const std::vector<Scalar>& output_times, Scalar cfl)
{
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = Scalar(1.0) / static_cast<Scalar>(nx);
  grid.origin_x = Scalar(0.0);

  cfe::Field<Scalar, 3> state(grid.n_cells_total());
  cfe::Field<Scalar, 3> stage1(grid.n_cells_total());
  cfe::Field<Scalar, 3> scratch(grid.n_cells_total());

  cfe::EulerField<Scalar, 1> field{};
  field.gamma = gamma;

  const cfe::State<Scalar, 3> behind_q = field.conserved_from_primitive(
      Scalar(jump.behind.rho), cfe::Vector<Scalar, 1>(Scalar(jump.behind.u)), Scalar(jump.behind.p));
  const cfe::State<Scalar, 3> ahead_q = field.conserved_from_primitive(
      Scalar(jump.ahead.rho), cfe::Vector<Scalar, 1>(Scalar(jump.ahead.u)), Scalar(jump.ahead.p));

  for (std::size_t i = 0; i < grid.nx; ++i) {
    const Scalar x = grid.x_center(grid.ngx + i);
    const cfe::State<Scalar, 3>& q = (x < x0) ? behind_q : ahead_q;
    for (std::size_t c = 0; c < 3; ++c) state(grid.flat_index(grid.ngx + i, 0, 0), c) = q[c];
  }

  cfe::StaticBoundary<Scalar, 3> boundary{behind_q, ahead_q};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::EulerField<Scalar, 1>, cfe::StaticBoundary<Scalar, 3>,
                 cfe::StaticBoundary<Scalar, 3>, cfe::StaticBoundary<Scalar, 3>,
                 cfe::fvm::MusclMinmodReconstruction, NumericalFlux>
      solver{grid, field, boundary};
  auto residual = [&](cfe::FieldView<Scalar, 3> in, cfe::FieldView<Scalar, 3> out) {
    solver.residual(in, out);
  };

  auto capture = [&](Scalar t_actual) {
    ShockSnapshot<Scalar> snap;
    snap.t = t_actual;
    snap.x.resize(grid.nx);
    snap.rho.resize(grid.nx);
    snap.u.resize(grid.nx);
    snap.p.resize(grid.nx);
    for (std::size_t i = 0; i < grid.nx; ++i) {
      const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
      cfe::State<Scalar, 3> q;
      for (std::size_t c = 0; c < 3; ++c) q[c] = state(cell, c);
      snap.x[i] = grid.x_center(grid.ngx + i);
      snap.rho[i] = field.density(q);
      snap.u[i] = field.velocity_component(q, cfe::Axis::X);
      snap.p[i] = field.pressure(q);
    }
    return snap;
  };

  std::vector<ShockSnapshot<Scalar>> snapshots;
  snapshots.reserve(output_times.size());

  Scalar t = Scalar(0.0);
  for (Scalar t_target : output_times) {
    t = advance_cfl_adaptive<Scalar, 3>(state.view(), stage1.view(), scratch.view(), grid, field, residual,
                                         solver.active_cell_count(), solver.active_cell_index_map(), t, t_target,
                                         cfl);
    snapshots.push_back(capture(t));
  }

  return snapshots;
}

}  // namespace cfe_verification
