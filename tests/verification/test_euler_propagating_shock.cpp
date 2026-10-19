// Canonical verification: 1D propagating shock with feature tracking
// and a periodic propagation-speed estimate (ROADMAP.md Phase 4,
// tasks/0013-phase4-propagating-shock-tracking.md).
//
// A standalone, Mach-2 right-moving shock in otherwise-uniform, at-rest
// gas (gamma=1.4) -- deliberately NOT a reuse of Sod's own shock, which
// sits adjacent to a contact discontinuity that smears differently
// between RusanovFlux/HllcFlux and would risk contaminating a
// shock-POSITION measurement (see propagating_shock_setup.hpp's own
// header comment and this task's file for the full rationale, from
// Amelia's/Violet's pre-implementation consult).
//
// Exact answer: the Rankine-Hugoniot jump conditions
// (rankine_hugoniot_jump.hpp's normal_shock_jump, pure closed-form
// algebra shared with task 0012's Shu-Osher left state, not the general
// iterative Riemann solver sod_shock_tube_exact.hpp implements) give an
// exact analytic shock speed for this IC with NO numerical solve at all
// -- unlike Sod, whose exact wave speeds come from a Newton-iterated
// star-region pressure.
//
// Feature tracking: shock_feature_tracking.hpp's interpolated
// midpoint-density-crossing locator, chosen over steepest-gradient
// specifically to avoid conflating "differently-smeared by flux X" with
// "actually moved" -- see that header's own comment.
//
// Acceptance: a RESOLUTION SWEEP (not a single fixed-resolution
// pass/fail), following this codebase's own established convergence-
// check convention (test_sod_shock_tube_density_l1_error_decreases_
// with_resolution): the measured-speed error against the analytic
// value must not increase as the grid is refined, for EACH flux
// independently, plus a direct check that both fluxes agree with each
// other (same physical shock, should converge to the same speed,
// verified rather than assumed).
#include <cmath>
#include <cstdio>
#include <vector>

#include "cfe/core/types.hpp"
#include "cfe/numerics/numerical_flux/hllc.hpp"
#include "propagating_shock_setup.hpp"
#include "shock_feature_tracking.hpp"
#include "test_framework.hpp"

namespace {

using Scalar = double;
constexpr Scalar kGamma = Scalar(1.4);
constexpr Scalar kMach = Scalar(2.0);
constexpr Scalar kX0 = Scalar(0.2);
constexpr Scalar kCfl = Scalar(0.4);

const cfe_verification::NormalShockJump kJump =
    cfe_verification::normal_shock_jump(kGamma, /*rho_ahead=*/1.0, /*p_ahead=*/1.0, kMach);

// 8 snapshots spread over the run, ascending, last one is t_final. The
// shock (speed ~2.3664) starts at x=0.2 and reaches ~0.2+2.3664*0.2 =
// 0.6733 by t_final=0.2 -- comfortably inside [0,1] with margin on both
// sides for the whole run.
std::vector<Scalar> output_times()
{
  std::vector<Scalar> times;
  for (int i = 1; i <= 8; ++i) times.push_back(Scalar(0.025) * Scalar(i));
  return times;
}

// Fits a propagation-speed estimate from a full set of snapshots,
// locating the shock position in each via the midpoint-density
// crossing, then a least-squares linear fit of position vs. the
// snapshots' own actually-recorded time (not an assumed uniform
// spacing -- see propagating_shock_setup.hpp's header comment on why
// CFL-adaptive stepping makes that necessary).
template <class NumericalFlux>
double measured_shock_speed(cfe::Index nx)
{
  const auto snapshots =
      cfe_verification::run_propagating_shock<Scalar, NumericalFlux>(nx, kGamma, kJump, kX0, output_times(), kCfl);

  std::vector<double> times, positions;
  times.reserve(snapshots.size());
  positions.reserve(snapshots.size());
  for (const auto& snap : snapshots) {
    times.push_back(static_cast<double>(snap.t));
    positions.push_back(
        cfe_verification::locate_shock_position(snap.x, snap.rho, kJump.behind.rho, kJump.ahead.rho));
  }
  return cfe_verification::least_squares_slope(times, positions);
}

}  // namespace

CFE_TEST(test_propagating_shock_rankine_hugoniot_jump_matches_hand_derivation)
{
  // Cross-check normal_shock_jump's own formulas against the
  // independently hand-derived values recorded in
  // tasks/0013-phase4-propagating-shock-tracking.md before trusting it
  // as this test's oracle.
  CFE_CHECK_NEAR(kJump.behind.rho, 2.666667, 1e-5);
  CFE_CHECK_NEAR(kJump.behind.u, 1.479020, 1e-5);
  CFE_CHECK_NEAR(kJump.behind.p, 4.5, 1e-10);
  CFE_CHECK_NEAR(kJump.shock_speed, 2.366432, 1e-5);
}

CFE_TEST(test_propagating_shock_rusanov_speed_error_decreases_with_resolution)
{
  const double error_coarse = std::fabs(measured_shock_speed<cfe::RusanovFlux>(100) - kJump.shock_speed);
  const double error_mid = std::fabs(measured_shock_speed<cfe::RusanovFlux>(200) - kJump.shock_speed);
  const double error_fine = std::fabs(measured_shock_speed<cfe::RusanovFlux>(400) - kJump.shock_speed);
  const double error_finest = std::fabs(measured_shock_speed<cfe::RusanovFlux>(800) - kJump.shock_speed);
  std::printf(
      "[info] Rusanov propagating-shock speed error vs nx: 100=%.6f 200=%.6f 400=%.6f 800=%.6f "
      "(analytic=%.6f)\n",
      error_coarse, error_mid, error_fine, error_finest, kJump.shock_speed);

  CFE_CHECK(error_mid <= error_coarse);
  CFE_CHECK(error_fine <= error_mid);
  CFE_CHECK(error_finest <= error_fine);
  // Absolute floor at the finest resolution tested, set with margin
  // over the measured ~0.000044.
  CFE_CHECK(error_finest < 0.001);
}

CFE_TEST(test_propagating_shock_hllc_speed_error_decreases_with_resolution)
{
  const double error_coarse = std::fabs(measured_shock_speed<cfe::HllcFlux>(100) - kJump.shock_speed);
  const double error_mid = std::fabs(measured_shock_speed<cfe::HllcFlux>(200) - kJump.shock_speed);
  const double error_fine = std::fabs(measured_shock_speed<cfe::HllcFlux>(400) - kJump.shock_speed);
  const double error_finest = std::fabs(measured_shock_speed<cfe::HllcFlux>(800) - kJump.shock_speed);
  std::printf(
      "[info] HLLC propagating-shock speed error vs nx: 100=%.6f 200=%.6f 400=%.6f 800=%.6f "
      "(analytic=%.6f)\n",
      error_coarse, error_mid, error_fine, error_finest, kJump.shock_speed);

  CFE_CHECK(error_mid <= error_coarse);
  CFE_CHECK(error_fine <= error_mid);
  CFE_CHECK(error_finest <= error_fine);
  // Absolute floor at the finest resolution tested, set with margin
  // over the measured ~0.000071.
  CFE_CHECK(error_finest < 0.001);
}

CFE_TEST(test_propagating_shock_rusanov_and_hllc_measure_the_same_physical_speed)
{
  // Both fluxes model the SAME physical shock, so their measured speeds
  // should agree with each other closely at a shared resolution --
  // checked directly rather than assumed, per the task's own explicit
  // "verify this for real" instruction. Measured difference at nx=400:
  // ~0.000085 (Rusanov=2.366554, HLLC=2.366639) -- both converge to the
  // same physical speed, as expected.
  const cfe::Index nx = 400;
  const double rusanov_speed = measured_shock_speed<cfe::RusanovFlux>(nx);
  const double hllc_speed = measured_shock_speed<cfe::HllcFlux>(nx);
  std::printf("[info] Propagating-shock measured speed at nx=%zu: Rusanov=%.6f, HLLC=%.6f (analytic=%.6f)\n", nx,
              rusanov_speed, hllc_speed, kJump.shock_speed);
  CFE_CHECK(std::fabs(rusanov_speed - hllc_speed) < 0.001);
}
