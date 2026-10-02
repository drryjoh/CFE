// "Burgers smooth convergence" (VERIFICATION.md's own canonical-problem
// name; task spec 0003). A smooth periodic initial condition
// u0(x) = A + B*sin(2*pi*x/L), run strictly before the characteristics
// first cross (no shock exists yet, so a 2nd-order-in-smooth-regions
// scheme should still show close to 2nd-order global error).
//
// Reference solution: the method of characteristics gives the EXACT
// solution implicitly -- a fluid "particle" starting at x0 moves at its
// own initial speed u0(x0) forever (Burgers has no pressure/forcing
// term), so x(t) = x0 + u0(x0)*t, u(x,t) = u0(x0). Solving for x0 given
// (x,t) is a 1D root-find; this file does it with a few steps of
// Newton's method (g(x0) = x0 + u0(x0)*t - x = 0), which is an analytic
// solution per VERIFICATION.md's verification-hierarchy rung 3, not a
// numerical proxy. Newton converges reliably here specifically because
// `kFinalTime` is kept strictly below the analytic breaking time
// t_break = -1/min_x0(u0'(x0)) = L/(2*pi*B) (LeVeque, "Finite Volume
// Methods for Hyperbolic Problems," Sec. 3.9/11.1) -- below t_break,
// g'(x0) = 1 + u0'(x0)*t stays strictly positive everywhere, so g is
// monotone and has a unique root; at or past t_break this stops holding
// (multiple x0 map to the same x -- the shock has formed) and Newton
// would not reliably converge to the single-valued answer this test
// needs. `dt` is sized once from the IC's max|u0| (sound because a
// scalar conservation law's entropy solution has non-increasing sup
// norm over time -- a standard maximum-principle fact -- so this bound
// stays valid for the whole, pre-shock run).
//
// Convergence-order acceptance band is loosened relative to
// test_scalar_advection_convergence.cpp's `[3.5, 4.5]` window (a ratio
// of ~4 is a clean 2nd order): minmod clips the slope to exactly zero at
// this IC's two smooth extrema (max/min of the sine), a known,
// documented order-reduction mechanism for TVD limiters at smooth
// extrema (Sweby, SIAM J. Numer. Anal. 21, 1984) -- see
// numerics/fvm/muscl_minmod.hpp's own header comment. Measured here
// (reported, not assumed): the ratio stays above ~3.0 at every
// refinement in this sweep, i.e. still close to 2nd order, since only
// 2 cells out of `nx` are ever affected by the clip and their
// contribution to the global L2 norm shrinks as resolution increases.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

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

constexpr double kPi = 3.14159265358979323846;
constexpr double kDomainLength = 1.0;
constexpr double kAmplitudeOffset = 1.0;  // A: kept positive so u0 never changes sign
constexpr double kAmplitudeWave = 0.5;    // B
constexpr double kBreakingTime = kDomainLength / (2.0 * kPi * kAmplitudeWave);  // L/(2*pi*B)
constexpr double kFinalTime = 0.5 * kBreakingTime;  // strictly before breaking, with 2x margin
constexpr double kCfl = 0.4;

double initial_condition(double x0)
{
  return kAmplitudeOffset + kAmplitudeWave * std::sin(2.0 * kPi * x0 / kDomainLength);
}

double initial_condition_derivative(double x0)
{
  return kAmplitudeWave * (2.0 * kPi / kDomainLength) * std::cos(2.0 * kPi * x0 / kDomainLength);
}

// Exact method-of-characteristics solution u(x,t) via Newton's method on
// g(x0) = x0 + u0(x0)*t - x = 0. Valid only for t < kBreakingTime (see
// this file's header comment for why).
double exact_solution(double x, double t)
{
  double x0 = x;  // initial guess: the displacement u0*t is modest this
                   // far before breaking, so "no displacement yet" is a
                   // good enough starting point for Newton to converge
                   // from.
  for (int iteration = 0; iteration < 50; ++iteration) {
    const double g = x0 + initial_condition(x0) * t - x;
    const double g_prime = 1.0 + initial_condition_derivative(x0) * t;
    const double step = g / g_prime;
    x0 -= step;
    if (std::abs(step) < 1e-14) break;
  }
  return initial_condition(x0);
}

// Runs the Burgers solver (MusclMinmodReconstruction + RusanovFlux) on
// an `nx`-cell periodic grid from t=0 to t=kFinalTime and returns the L2
// error against the exact characteristics solution.
double run_and_measure_l2_error(std::size_t nx)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = kDomainLength / static_cast<double>(nx);
  grid.origin_x = 0.0;

  cfe::Field<double, 1> state(grid.n_cells_total());
  cfe::Field<double, 1> stage1(grid.n_cells_total());
  cfe::Field<double, 1> residual_scratch(grid.n_cells_total());

  double max_abs_u0 = 0.0;
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
    const double value = initial_condition(grid.x_center(grid.ngx + i));
    state(cell, 0) = value;
    max_abs_u0 = std::max(max_abs_u0, std::abs(value));
  }

  cfe::BurgersField<double, 1> field{};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::BurgersField<double, 1>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux>
      solver{grid, field, cfe::PeriodicBoundary{}};

  const double dt_target = kCfl * grid.dx / max_abs_u0;
  const int n_steps = static_cast<int>(std::ceil(kFinalTime / dt_target));
  const double dt = kFinalTime / static_cast<double>(n_steps);

  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver.residual(in, out);
  };
  for (int step = 0; step < n_steps; ++step) {
    cfe::ssp_rk2_step<double>(state.view(), stage1.view(), residual_scratch.view(), dt, residual,
                               solver.active_cell_count(), solver.active_cell_index_map());
  }

  double sum_sq_error = 0.0;
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
    const double exact = exact_solution(grid.x_center(grid.ngx + i), kFinalTime);
    const double error = state(cell, 0) - exact;
    sum_sq_error += error * error;
  }
  return std::sqrt(sum_sq_error / static_cast<double>(grid.nx));
}

}  // namespace

CFE_TEST(test_burgers_characteristics_breaking_time_is_positive_and_finite)
{
  // Sanity check on the test's own setup, not the solver: if this ever
  // failed, kFinalTime's "strictly before breaking" premise would be
  // meaningless.
  CFE_CHECK(kBreakingTime > 0.0);
  CFE_CHECK(kFinalTime < kBreakingTime);
}

CFE_TEST(test_burgers_smooth_second_order_convergence)
{
  const std::vector<std::size_t> resolutions = {40, 80, 160, 320};
  std::vector<double> errors;
  errors.reserve(resolutions.size());
  for (std::size_t nx : resolutions) {
    errors.push_back(run_and_measure_l2_error(nx));
  }

  // Loosened lower bound vs. the linear-advection case's [3.5, 4.5] --
  // see this file's header comment for why (minmod clips at the IC's
  // smooth extrema). Upper bound kept at 4.5: nothing about a limiter
  // should make convergence appear BETTER than clean 2nd order.
  for (std::size_t k = 0; k + 1 < errors.size(); ++k) {
    CFE_CHECK(errors[k + 1] > 0.0);
    const double ratio = errors[k] / errors[k + 1];
    CFE_CHECK(ratio > 3.0);
    CFE_CHECK(ratio < 4.5);
  }
}
