// Transonic rarefaction verification for Burgers (review finding on
// PR #3): `u_left=-1 < u_right=1` means the Lax entropy condition
// selects a smooth RAREFACTION FAN, not a shock -- the opposite
// ordering from test_burgers_shock_formation.cpp's `u_left > u_right`.
// This specific choice is TRANSONIC: the fan's own characteristic
// speed xi=(x-x0)/t passes through exactly zero at the fan's center, a
// genuine sonic point sitting inside the domain, not just near a
// boundary. This is EXACTLY the case numerics/numerical_flux/
// upwind.hpp's own header comment names as the gap `UpwindFlux` cannot
// handle correctly (it switches on a single scalar wave speed's sign,
// which is ill-defined right at a sonic point) and `RusanovFlux` exists
// to close (its dissipation depends on the LOCAL MAXIMUM characteristic
// speed magnitude, never on a sign at all) -- so this test is the
// direct, end-to-end confirmation of that specific design claim, not
// just another Riemann problem.
//
// Exact solution (self-similar, xi = (x-x0)/t):
//   u(x,t) = u_left                for xi <  u_left
//          = xi                    for u_left <= xi <= u_right
//          = u_right                for xi >  u_right
// (LeVeque, "Finite Volume Methods for Hyperbolic Problems," Sec. 11.4;
// Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics," Sec.
// 2.4 -- the convex-scalar Riemann problem's rarefaction branch,
// u_left < u_right.)
//
// Exact cell average via a closed-form antiderivative (continuous
// across both fan edges by construction -- verified by matching the
// three pieces' values at s=u_left*t and s=u_right*t, s := x-x0):
//   F(s,t) = u_left*s  - u_left^2 *t/2   for s <= u_left*t
//          = s^2/(2t)                     for u_left*t <= s <= u_right*t
//          = u_right*s - u_right^2*t/2   for s >= u_right*t
// cell average over [x_lo,x_hi] = (F(x_hi-x0,t) - F(x_lo-x0,t)) /
// (x_hi-x_lo) -- correct even for a cell that straddles a fan edge (or,
// in principle, both), not just a cell-center point sample. `t=0` is
// handled as a separate, simpler 2-piece step average (same construction
// test_burgers_shock_formation.cpp's `shock_exact_cell_average`-style
// helpers use) since the closed form above divides by `t`.
//
// Templated on `Scalar` (matching every other Burgers test in this
// repo) so both double and float precision are actually exercised, per
// VERIFICATION.md section 4.
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

template <class Scalar>
Scalar rarefaction_exact_cell_average(Scalar x_lo, Scalar x_hi, Scalar x0, Scalar t, Scalar u_left,
                                       Scalar u_right)
{
  if (t == Scalar(0)) {
    // Degenerate (zero-width fan): a sharp step at x0, same cut-cell
    // construction test_burgers_shock_formation.cpp's own IC uses.
    if (x_hi <= x0) return u_left;
    if (x_lo >= x0) return u_right;
    const Scalar frac_left = (x0 - x_lo) / (x_hi - x_lo);
    return frac_left * u_left + (Scalar(1) - frac_left) * u_right;
  }
  auto antideriv = [&](Scalar s) {
    if (s <= u_left * t) return u_left * s - Scalar(0.5) * u_left * u_left * t;
    if (s >= u_right * t) return u_right * s - Scalar(0.5) * u_right * u_right * t;
    return s * s / (Scalar(2) * t);
  };
  return (antideriv(x_hi - x0) - antideriv(x_lo - x0)) / (x_hi - x_lo);
}

struct RunResult
{
  double l1_error;
  double max_overshoot;   // max(0, value - max(u_left,u_right)) over all real cells
  double max_undershoot;  // max(0, min(u_left,u_right) - value) over all real cells
  double integral_initial;
  double integral_final;
};

template <class Scalar>
RunResult run_burgers_rarefaction(std::size_t nx, Scalar u_left, Scalar u_right, Scalar x0,
                                   Scalar domain_length, Scalar final_time, Scalar cfl)
{
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = domain_length / static_cast<Scalar>(nx);
  grid.origin_x = Scalar(0.0);

  cfe::Field<Scalar, 1> state(grid.n_cells_total());
  cfe::Field<Scalar, 1> stage1(grid.n_cells_total());
  cfe::Field<Scalar, 1> residual_scratch(grid.n_cells_total());

  RunResult result{};
  result.integral_initial = 0.0;
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const Scalar x_lo = static_cast<Scalar>(i) * grid.dx;
    const Scalar x_hi = x_lo + grid.dx;
    const Scalar value = rarefaction_exact_cell_average(x_lo, x_hi, x0, Scalar(0.0), u_left, u_right);
    state(grid.flat_index(grid.ngx + i, 0, 0), 0) = value;
    result.integral_initial += static_cast<double>(value) * static_cast<double>(grid.dx);
  }

  cfe::BurgersField<Scalar, 1> field{};
  // Fixed far-field values on BOTH ends -- unlike the moving-shock test
  // (fixed inflow + extrapolated outflow), the rarefaction's exact
  // solution has a genuinely constant, known value on each side for as
  // long as the fan stays clear of the boundary, so StaticBoundary (not
  // InflowOutflowBoundary) is the faithful choice here.
  cfe::StaticBoundary<Scalar, 1> boundary{cfe::State<Scalar, 1>(u_left), cfe::State<Scalar, 1>(u_right)};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::BurgersField<Scalar, 1>, cfe::StaticBoundary<Scalar, 1>,
                 cfe::StaticBoundary<Scalar, 1>, cfe::StaticBoundary<Scalar, 1>,
                 cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux>
      solver{grid, field, boundary};

  // dt sized from max(|u_left|,|u_right|) -- the KNOWN, fixed far-field
  // magnitudes -- not from a signed state value (review finding): with
  // u_left negative here, `cfl*dx/u_left` (the pattern
  // test_burgers_shock_formation.cpp uses, safe only because that
  // test's own u_left happens to be positive and the larger of its two
  // states) would give a NEGATIVE dt. The fan's own interior values
  // span exactly [u_left,u_right] (no overshoot, by the TVD property
  // this test also checks), so max(|u_left|,|u_right|) bounds the
  // fastest characteristic speed for the entire run, not just at t=0.
  using std::abs;
  const Scalar max_speed = abs(u_left) > abs(u_right) ? abs(u_left) : abs(u_right);
  const Scalar dt_target = cfl * grid.dx / max_speed;
  const int n_steps = static_cast<int>(std::ceil(static_cast<double>(final_time / dt_target)));
  const Scalar dt = final_time / static_cast<Scalar>(n_steps);

  auto residual = [&](cfe::FieldView<Scalar, 1> in, cfe::FieldView<Scalar, 1> out) {
    solver.residual(in, out);
  };
  for (int step = 0; step < n_steps; ++step) {
    cfe::ssp_rk2_step<Scalar>(state.view(), stage1.view(), residual_scratch.view(), dt, residual,
                               solver.active_cell_count(), solver.active_cell_index_map());
  }

  const Scalar bound_hi = u_left > u_right ? u_left : u_right;
  const Scalar bound_lo = u_left < u_right ? u_left : u_right;
  double sum_abs_error = 0.0;
  result.max_overshoot = 0.0;
  result.max_undershoot = 0.0;
  result.integral_final = 0.0;
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const Scalar value = state(grid.flat_index(grid.ngx + i, 0, 0), 0);
    const Scalar x_lo = static_cast<Scalar>(i) * grid.dx;
    const Scalar x_hi = x_lo + grid.dx;
    const Scalar exact = rarefaction_exact_cell_average(x_lo, x_hi, x0, final_time, u_left, u_right);
    sum_abs_error += static_cast<double>(std::abs(value - exact)) * static_cast<double>(grid.dx);
    result.max_overshoot = std::max(result.max_overshoot, static_cast<double>(value - bound_hi));
    result.max_undershoot = std::max(result.max_undershoot, static_cast<double>(bound_lo - value));
    result.integral_final += static_cast<double>(value) * static_cast<double>(grid.dx);
  }
  result.l1_error = sum_abs_error;
  return result;
}

}  // namespace

CFE_TEST(test_burgers_rarefaction_matches_exact_entropy_solution)
{
  // u_left=-1 < u_right=1: a transonic rarefaction fan, x0=5.0 on a
  // domain=10.0 -> fan spans [x0+u_left*t, x0+u_right*t] = [3,7] at
  // final_time=2.0, well clear of both boundaries (3.0 of margin either
  // side, far more than this 2-ghost-layer scheme's stencil reach).
  const RunResult result = run_burgers_rarefaction<double>(200, -1.0, 1.0, 5.0, 10.0, 2.0, 0.4);
  CFE_CHECK(result.l1_error < 0.05);
}

CFE_TEST(test_burgers_rarefaction_l1_error_decreases_under_grid_refinement)
{
  const RunResult coarse = run_burgers_rarefaction<double>(100, -1.0, 1.0, 5.0, 10.0, 2.0, 0.4);
  const RunResult medium = run_burgers_rarefaction<double>(200, -1.0, 1.0, 5.0, 10.0, 2.0, 0.4);
  const RunResult fine = run_burgers_rarefaction<double>(400, -1.0, 1.0, 5.0, 10.0, 2.0, 0.4);
  CFE_CHECK(medium.l1_error < coarse.l1_error);
  CFE_CHECK(fine.l1_error < medium.l1_error);
}

CFE_TEST(test_burgers_rarefaction_stays_bounded_within_far_field_states)
{
  // What this DOES catch: spurious overshoot/undershoot beyond
  // [u_left,u_right] -- the TVD guarantee failing, e.g. ringing from a
  // sign bug that makes the scheme add dissipation of the wrong sign
  // somewhere in the fan.
  //
  // What this does NOT catch, corrected on review: an entropy-violating
  // "stationary expansion jump" (the classic non-physical weak solution
  // at a transonic point -- a scheme with no entropy fix can get stuck
  // reproducing the initial step, u_left for x<x0 and u_right for
  // x>x0, instead of spreading it into the fan) stays entirely within
  // [u_left,u_right] -- no value ever exceeds either bound, so this
  // check alone would NOT flag it. That failure mode is what
  // test_burgers_rarefaction_matches_exact_entropy_solution and
  // test_burgers_rarefaction_l1_error_decreases_under_grid_refinement
  // are actually for: a stationary jump has an O(1) L1 error against
  // the correct spreading-fan solution that does NOT shrink under
  // refinement (a stationary jump is self-sustaining, not a resolution
  // artifact), which those two tests would catch directly. This test's
  // own job is narrower -- just the TVD bound -- and is checked here on
  // its own terms, not stretched to cover entropy-correctness too.
  const RunResult result = run_burgers_rarefaction<double>(400, -1.0, 1.0, 5.0, 10.0, 2.0, 0.4);
  CFE_CHECK(result.max_overshoot < 1e-9);
  CFE_CHECK(result.max_undershoot < 1e-9);
}

CFE_TEST(test_burgers_rarefaction_obeys_exact_flux_balance_conservation)
{
  // Not periodic -- StaticBoundary fixes both far-field ends for the
  // whole run, so the domain integral's change is exactly
  // (F(u_left)-F(u_right))*T (same flux-form guarantee
  // test_burgers_shock_formation.cpp's own conservation test uses). The
  // specific choice u_left=-1, u_right=1 makes F(u_left)=F(u_right)=0.5
  // (Burgers' flux is u^2/2, symmetric in sign) -- so the expected
  // change is exactly ZERO despite this being a non-periodic domain, a
  // clean, elegant property of this particular symmetric choice of
  // far-field states, not a coincidence of periodicity.
  const double u_left = -1.0;
  const double u_right = 1.0;
  const double final_time = 2.0;
  const RunResult result = run_burgers_rarefaction<double>(400, u_left, u_right, 5.0, 10.0, final_time, 0.4);
  const double flux_left = 0.5 * u_left * u_left;
  const double flux_right = 0.5 * u_right * u_right;
  const double expected_change = (flux_left - flux_right) * final_time;
  CFE_CHECK_NEAR(expected_change, 0.0, 1e-12);  // sanity on the claim above, not the solver
  const double actual_change = result.integral_final - result.integral_initial;
  CFE_CHECK_NEAR(actual_change, expected_change, 1e-6);
}

CFE_TEST(test_burgers_rarefaction_holds_at_float_precision)
{
  // Confirms the transonic case specifically (not just the shock case
  // test_burgers_shock_capturing_holds_at_float_precision already
  // covers) holds at reduced precision. Coarser resolutions and
  // loosened bounds-tolerance, same reasoning as that test.
  const RunResult coarse = run_burgers_rarefaction<float>(100, -1.0f, 1.0f, 5.0f, 10.0f, 2.0f, 0.4f);
  const RunResult fine = run_burgers_rarefaction<float>(400, -1.0f, 1.0f, 5.0f, 10.0f, 2.0f, 0.4f);
  CFE_CHECK(fine.l1_error < coarse.l1_error);
  CFE_CHECK(fine.max_overshoot < 1e-5);
  CFE_CHECK(fine.max_undershoot < 1e-5);
}
