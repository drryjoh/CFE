// Shared Sod-shock-tube driver: builds the grid/field/solver stack and
// runs it to a given final time, used by BOTH
// tests/verification/test_euler_sod_shock_tube.cpp and
// tutorials/euler_1d_sod_shock_tube/sod_shock_tube.cpp so the actual
// simulation setup exists in exactly one place (Snorlax review finding:
// the two previously duplicated this ~50-line block). The test and the
// tutorial still do their OWN, different things with the result (the
// test checks conservation/exact-solution agreement with CFE_CHECK*;
// the tutorial writes CSVs and an L1-error summary) -- only the
// "simulate it" part is shared.
//
// Templated on `Scalar` (not hardcoded `double`) so the tutorial can run
// it at `cfe::scalar` (switchable via `-DCFE_SCALAR_TYPE`) while the
// test runs it at a fixed `double` for oracle-comparison precision --
// the same split every other dual-precision-aware file in this codebase
// uses.
//
// Also templated on `NumericalFlux` (default `cfe::RusanovFlux`, every
// existing call site unaffected) so task 0011's HLLC re-verification
// (tests/verification/test_euler_sod_shock_tube_hllc.cpp) reuses this
// SAME driver with `cfe::HllcFlux` instead of a second, duplicated
// simulation setup -- the same "one driver, shared by every caller"
// shape this file's own header already established between the test and
// the tutorial.
#pragma once

#include <array>
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
#include "sod_shock_tube_exact.hpp"

namespace cfe_verification {

template <class Scalar>
struct SodProfile
{
  std::vector<Scalar> x, rho, u, p;
  // Conserved-variable domain integrals (mass, momentum, energy) at the
  // start and end of the run, always accumulated in `double` regardless
  // of `Scalar` -- a reported/checked diagnostic, not a state value
  // (same convention tests/unit/test_burgers_*.cpp uses).
  double total_mass_initial, total_momentum_initial, total_energy_initial;
  double total_mass_final, total_momentum_final, total_energy_final;
};

// Number of cells spanned crossing from 5% to 95% of the density jump
// across a known discontinuity location (typically the contact) --
// shared (Snorlax review finding: originally duplicated verbatim between
// test_euler_sod_shock_tube_hllc.cpp and the tutorial) by every caller
// wanting to quantify how much a scheme has numerically smeared a
// discontinuity, given the discontinuity's own exact location and the
// exact plateau densities on either side of it.
template <class Scalar>
std::size_t contact_width_cells(const SodProfile<Scalar>& profile, double x_contact, double rho_plateau_left,
                                 double rho_plateau_right)
{
  const double dx = static_cast<double>(profile.x[1]) - static_cast<double>(profile.x[0]);
  const std::size_t center =
      static_cast<std::size_t>(std::lround((x_contact - static_cast<double>(profile.x[0])) / dx));

  const double lo = rho_plateau_right + 0.05 * (rho_plateau_left - rho_plateau_right);
  const double hi = rho_plateau_right + 0.95 * (rho_plateau_left - rho_plateau_right);

  std::size_t left_edge = center;
  while (left_edge > 0 && static_cast<double>(profile.rho[left_edge]) < hi) --left_edge;
  std::size_t right_edge = center;
  while (right_edge + 1 < profile.rho.size() && static_cast<double>(profile.rho[right_edge]) > lo) ++right_edge;
  return right_edge - left_edge;
}

// Runs the Sod tube (classic diaphragm-at-x0 setup) on an `nx`-cell
// grid from `t=0` to `t_final`, with both ends held at their own
// far-field primitive state (`StaticBoundary`) -- valid whenever
// `t_final` is small enough that neither the rarefaction's head nor the
// shock reaches the domain edges (see sod_shock_tube_exact.hpp's own
// comment on the classic parameters' exact wave positions at t=0.2).
// MusclMinmodReconstruction + RusanovFlux + SSP-RK2, with the CFL
// timestep recomputed every step from the CURRENT field's own max wave
// speed (not a fixed a priori bound), since the maximum signal speed
// changes as the shock/rarefaction form.
template <class Scalar, class NumericalFlux = cfe::RusanovFlux>
SodProfile<Scalar> run_sod_shock_tube(std::size_t nx, Scalar gamma, const PrimitiveState& left,
                                       const PrimitiveState& right, Scalar x0, Scalar t_final, Scalar cfl)
{
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = Scalar(1.0) / static_cast<Scalar>(nx);

  cfe::Field<Scalar, 3> state(grid.n_cells_total());
  cfe::Field<Scalar, 3> stage1(grid.n_cells_total());
  cfe::Field<Scalar, 3> scratch(grid.n_cells_total());

  cfe::EulerField<Scalar, 1> field{};
  field.gamma = gamma;

  const cfe::State<Scalar, 3> left_q =
      field.conserved_from_primitive(Scalar(left.rho), cfe::Vector<Scalar, 1>(Scalar(left.u)), Scalar(left.p));
  const cfe::State<Scalar, 3> right_q = field.conserved_from_primitive(
      Scalar(right.rho), cfe::Vector<Scalar, 1>(Scalar(right.u)), Scalar(right.p));

  for (std::size_t i = 0; i < grid.nx; ++i) {
    const Scalar x = grid.x_center(grid.ngx + i);
    const cfe::State<Scalar, 3>& q = (x < x0) ? left_q : right_q;
    for (std::size_t c = 0; c < 3; ++c) state(grid.flat_index(grid.ngx + i, 0, 0), c) = q[c];
  }

  cfe::StaticBoundary<Scalar, 3> boundary{left_q, right_q};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::EulerField<Scalar, 1>, cfe::StaticBoundary<Scalar, 3>,
                 cfe::StaticBoundary<Scalar, 3>, cfe::StaticBoundary<Scalar, 3>,
                 cfe::fvm::MusclMinmodReconstruction, NumericalFlux>
      solver{grid, field, boundary};
  auto residual = [&](cfe::FieldView<Scalar, 3> in, cfe::FieldView<Scalar, 3> out) {
    solver.residual(in, out);
  };

  auto total_integrals = [&]() {
    double mass = 0.0, momentum = 0.0, energy = 0.0;
    for (std::size_t i = 0; i < grid.nx; ++i) {
      const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
      mass += static_cast<double>(state(cell, 0)) * static_cast<double>(grid.dx);
      momentum += static_cast<double>(state(cell, 1)) * static_cast<double>(grid.dx);
      energy += static_cast<double>(state(cell, 2)) * static_cast<double>(grid.dx);
    }
    return std::array<double, 3>{mass, momentum, energy};
  };

  const auto initial = total_integrals();

  Scalar t = Scalar(0.0);
  while (t < t_final) {
    Scalar max_speed = Scalar(0.0);
    for (std::size_t i = 0; i < grid.nx; ++i) {
      cfe::State<Scalar, 3> q;
      const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
      for (std::size_t c = 0; c < 3; ++c) q[c] = state(cell, c);
      const Scalar speed = std::abs(field.velocity_component(q, cfe::Axis::X)) + field.sound_speed(q);
      max_speed = speed > max_speed ? speed : max_speed;
    }
    Scalar dt = cfl * grid.dx / max_speed;
    if (t + dt > t_final) dt = t_final - t;
    cfe::ssp_rk2_step<Scalar>(state.view(), stage1.view(), scratch.view(), dt, residual,
                               solver.active_cell_count(), solver.active_cell_index_map());
    t += dt;
  }

  const auto final_integrals = total_integrals();

  SodProfile<Scalar> result;
  result.x.resize(grid.nx);
  result.rho.resize(grid.nx);
  result.u.resize(grid.nx);
  result.p.resize(grid.nx);
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
    cfe::State<Scalar, 3> q;
    for (std::size_t c = 0; c < 3; ++c) q[c] = state(cell, c);
    result.x[i] = grid.x_center(grid.ngx + i);
    result.rho[i] = field.density(q);
    result.u[i] = field.velocity_component(q, cfe::Axis::X);
    result.p[i] = field.pressure(q);
  }
  result.total_mass_initial = initial[0];
  result.total_momentum_initial = initial[1];
  result.total_energy_initial = initial[2];
  result.total_mass_final = final_integrals[0];
  result.total_momentum_final = final_integrals[1];
  result.total_energy_final = final_integrals[2];
  return result;
}

}  // namespace cfe_verification
