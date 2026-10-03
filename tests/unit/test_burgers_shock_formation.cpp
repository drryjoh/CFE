// "Burgers shock formation" (VERIFICATION.md's own canonical-problem
// name; task spec 0003) -- the actual shock-capturing acceptance bar.
//
// Setup: a Riemann-type step with u_left > u_right on a non-periodic
// domain, boundary ghost cells fixed to those same two states
// (StaticBoundary) so neither boundary ever sees anything other than the
// exact far-field value for the whole run. Because the flux u^2/2 is
// convex and u_left > u_right, the Lax entropy condition is satisfied
// and the EXACT solution is a single shock travelling at the
// Rankine-Hugoniot speed s = (u_left+u_right)/2 -- no rarefaction fan,
// closed form everywhere, not just at the shock itself:
//   u_exact(x, t) = u_left  for x <  x0 + s*t
//                 = u_right for x >= x0 + s*t
// (LeVeque, "Finite Volume Methods for Hyperbolic Problems," Sec. 11.4;
// Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics," Sec.
// 2.4, for the Burgers/convex-scalar Riemann problem solution.)
//
// This lets one L1-error check against the exact solution stand in for
// "the shock speed is right AND both bounding states are right" at
// once, rather than separately hunting for the shock's numerical
// location. What it does NOT claim: a formal convergence *order* for
// the captured shock itself -- any limited/TVD scheme smears a
// discontinuity over an O(1) number of cells regardless of resolution,
// so the mean absolute error is expected to shrink only like O(1/nx)
// (not O(1/nx^2)), a well-known, accepted property of shock-capturing
// schemes at an actual discontinuity -- distinct from
// test_burgers_convergence.cpp's *smooth*-solution 2nd-order claim,
// which this file is not attempting to reproduce.
//
// Templated on `Scalar` (matching
// test_scalar_advection_convergence_variants.cpp's own convention) so
// both double and float precision are actually exercised, per
// VERIFICATION.md section 4 -- not just asserted generic by the
// production code's own templates.
#include <algorithm>
#include <cmath>
#include <cstddef>

#include "cfe/field/field.hpp"
#include "cfe/fields/burgers/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"
#include "test_framework.hpp"

namespace {

struct RunResult
{
  double mean_abs_error;
  double max_overshoot;   // max(0, value - u_left) over all real cells
  double max_undershoot;  // max(0, u_right - value) over all real cells
  double total_variation_initial;
  double total_variation_final;
  double integral_initial;
  double integral_final;
};

template <class Scalar>
RunResult run_burgers_shock(std::size_t nx, Scalar u_left, Scalar u_right, Scalar shock_origin,
                             Scalar domain_length, Scalar final_time, Scalar cfl)
{
  const Scalar shock_speed = Scalar(0.5) * (u_left + u_right);  // Rankine-Hugoniot, exact for Burgers
  auto exact_riemann_solution = [&](Scalar x) {
    return (x < shock_origin + shock_speed * final_time) ? u_left : u_right;
  };

  cfe::CartesianGrid<Scalar> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = domain_length / static_cast<Scalar>(nx);
  grid.origin_x = Scalar(0.0);

  cfe::Field<Scalar, 1> state(grid.n_cells_total());
  cfe::Field<Scalar, 1> stage1(grid.n_cells_total());
  cfe::Field<Scalar, 1> residual_scratch(grid.n_cells_total());

  for (std::size_t i = 0; i < grid.nx; ++i) {
    const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
    const Scalar x = grid.x_center(grid.ngx + i);
    state(cell, 0) = (x < shock_origin) ? u_left : u_right;
  }

  RunResult result{};
  result.total_variation_initial = 0.0;
  result.integral_initial = 0.0;
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const double value = static_cast<double>(state(grid.flat_index(grid.ngx + i, 0, 0), 0));
    result.integral_initial += value * static_cast<double>(grid.dx);
    if (i > 0) {
      const double previous = static_cast<double>(state(grid.flat_index(grid.ngx + i - 1, 0, 0), 0));
      result.total_variation_initial += std::abs(value - previous);
    }
  }

  cfe::BurgersField<Scalar, 1> field{};
  cfe::StaticBoundary<Scalar, 1> boundary{cfe::State<Scalar, 1>(u_left), cfe::State<Scalar, 1>(u_right)};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::BurgersField<Scalar, 1>, cfe::StaticBoundary<Scalar, 1>,
                 cfe::StaticBoundary<Scalar, 1>, cfe::StaticBoundary<Scalar, 1>,
                 cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux>
      solver{grid, field, boundary};

  // dt sized once from the initial max|u| (=u_left here): the sup norm
  // of a scalar conservation law's entropy solution is non-increasing in
  // time, so this bound stays valid for the whole run, same reasoning
  // test_burgers_convergence.cpp uses.
  const Scalar dt_target = cfl * grid.dx / u_left;
  const int n_steps = static_cast<int>(std::ceil(static_cast<double>(final_time / dt_target)));
  const Scalar dt = final_time / static_cast<Scalar>(n_steps);

  auto residual = [&](cfe::FieldView<Scalar, 1> in, cfe::FieldView<Scalar, 1> out) {
    solver.residual(in, out);
  };
  for (int step = 0; step < n_steps; ++step) {
    cfe::ssp_rk2_step<Scalar>(state.view(), stage1.view(), residual_scratch.view(), dt, residual,
                               solver.active_cell_count(), solver.active_cell_index_map());
  }

  double sum_abs_error = 0.0;
  result.max_overshoot = 0.0;
  result.max_undershoot = 0.0;
  result.total_variation_final = 0.0;
  result.integral_final = 0.0;
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const Scalar value = state(grid.flat_index(grid.ngx + i, 0, 0), 0);
    const Scalar x = grid.x_center(grid.ngx + i);
    sum_abs_error += static_cast<double>(std::abs(value - exact_riemann_solution(x)));
    result.max_overshoot = std::max(result.max_overshoot, static_cast<double>(value - u_left));
    result.max_undershoot = std::max(result.max_undershoot, static_cast<double>(u_right - value));
    result.integral_final += static_cast<double>(value) * static_cast<double>(grid.dx);
    if (i > 0) {
      const Scalar previous = state(grid.flat_index(grid.ngx + i - 1, 0, 0), 0);
      result.total_variation_final += static_cast<double>(std::abs(value - previous));
    }
  }
  result.mean_abs_error = sum_abs_error / static_cast<double>(grid.nx);
  return result;
}

}  // namespace

CFE_TEST(test_burgers_shock_speed_and_states_match_exact_riemann_solution)
{
  // u_left=2, u_right=1, shock_origin=2.5, domain=10, final_time=2.0 ->
  // shock ends at 2.5+1.5*2.0=5.5, well clear of both boundaries.
  const RunResult result = run_burgers_shock<double>(200, 2.0, 1.0, 2.5, 10.0, 2.0, 0.4);
  // A captured shock is smeared over a handful of cells regardless of
  // resolution, so the mean absolute error over ALL cells should be
  // small (most cells are far from the shock and match exactly), not
  // machine-zero.
  CFE_CHECK(result.mean_abs_error < 0.05);
}

CFE_TEST(test_burgers_shock_capturing_error_shrinks_under_grid_refinement)
{
  // Not a formal convergence-order claim (see this file's header
  // comment) -- just confirms the error is actually resolution-
  // dependent (a finer grid genuinely captures the shock better), the
  // basic sanity check a fixed, resolution-independent bug would fail.
  const RunResult coarse = run_burgers_shock<double>(200, 2.0, 1.0, 2.5, 10.0, 2.0, 0.4);
  const RunResult fine = run_burgers_shock<double>(800, 2.0, 1.0, 2.5, 10.0, 2.0, 0.4);
  CFE_CHECK(fine.mean_abs_error < coarse.mean_abs_error);
}

CFE_TEST(test_burgers_shock_capturing_produces_no_overshoot_or_undershoot)
{
  // The actual "shock capturing succeeded" criterion a central-
  // difference (unlimited) reconstruction would fail: the solution must
  // never leave [u_right, u_left], the range spanned by the two bounding
  // states, beyond ordinary floating-point roundoff.
  const RunResult result = run_burgers_shock<double>(400, 2.0, 1.0, 2.5, 10.0, 2.0, 0.4);
  CFE_CHECK(result.max_overshoot < 1e-9);
  CFE_CHECK(result.max_undershoot < 1e-9);
}

CFE_TEST(test_burgers_shock_capturing_total_variation_does_not_increase)
{
  // Direct TVD check (Harten, 1983): total variation of the final state
  // must not exceed that of the initial state. The initial step's total
  // variation is exactly |u_left - u_right| (a single monotone jump);
  // any new oscillation anywhere in the domain would increase it.
  const RunResult result = run_burgers_shock<double>(400, 2.0, 1.0, 2.5, 10.0, 2.0, 0.4);
  CFE_CHECK_NEAR(result.total_variation_initial, 2.0 - 1.0, 1e-9);
  CFE_CHECK(result.total_variation_final <= result.total_variation_initial + 1e-9);
}

CFE_TEST(test_burgers_shock_capturing_obeys_exact_flux_balance_conservation)
{
  // This domain is NOT periodic -- mass genuinely flows in at the left
  // boundary (carrying flux F(u_left)) and out at the right (carrying
  // flux F(u_right)), so the domain integral is not constant in time.
  // What the flux-form FVM scheme DOES guarantee exactly (up to time-
  // integration truncation error, since the flux-form residual is an
  // exact telescoping sum regardless of reconstruction) is that the
  // integral's total change equals the net boundary flux over the run:
  //   integral_final - integral_initial ~= (F(u_left) - F(u_right)) * T
  // Both boundary ghost cells are held at the exact, unchanging far-
  // field state for the whole run (StaticBoundary), so there is no
  // boundary-approximation error to account for separately.
  const double u_left = 2.0;
  const double u_right = 1.0;
  const double final_time = 2.0;
  const RunResult result = run_burgers_shock<double>(400, u_left, u_right, 2.5, 10.0, final_time, 0.4);
  const double flux_left = 0.5 * u_left * u_left;
  const double flux_right = 0.5 * u_right * u_right;
  const double expected_change = (flux_left - flux_right) * final_time;
  const double actual_change = result.integral_final - result.integral_initial;
  CFE_CHECK_NEAR(actual_change, expected_change, 1e-6);
}

CFE_TEST(test_burgers_shock_capturing_holds_at_float_precision)
{
  // Confirms the solver's Scalar-genericity claim actually holds at
  // reduced precision, not just for double (same reasoning
  // test_scalar_advection_second_order_convergence_float_precision
  // uses). Coarser resolutions and the overshoot/TVD tolerances loosened
  // from 1e-9 to 1e-5 -- float's ~7 decimal digits means roundoff alone
  // can account for differences at the 1e-9 level that have nothing to
  // do with the scheme itself.
  const RunResult coarse = run_burgers_shock<float>(100, 2.0f, 1.0f, 2.5f, 10.0f, 2.0f, 0.4f);
  const RunResult fine = run_burgers_shock<float>(400, 2.0f, 1.0f, 2.5f, 10.0f, 2.0f, 0.4f);
  CFE_CHECK(fine.mean_abs_error < coarse.mean_abs_error);
  CFE_CHECK(fine.max_overshoot < 1e-5);
  CFE_CHECK(fine.max_undershoot < 1e-5);
  CFE_CHECK(fine.total_variation_final <= fine.total_variation_initial + 1e-5);
}
