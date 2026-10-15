// Canonical verification: Sod (1978) shock tube, the companion 1D
// benchmark for Phase 3's compressible Euler slice (ROADMAP.md Phase 3
// "Sod shock tube" acceptance item; AGENTS.md #25's acceptance sequence
// "canonical verification" stage -- this is independent verification
// against an exact solution, not a regression against a prior run).
//
// Classic parameters (Sod, "A survey of several finite difference
// methods for systems of nonlinear hyperbolic conservation laws," JCP
// 27, 1978): gamma=1.4, left (rho,u,p)=(1,0,1), right
// (rho,u,p)=(0.125,0,0.1), diaphragm at x=0.5 in a [0,1] tube, compared
// at t=0.2 against the exact Riemann solution
// (tests/verification/sod_shock_tube_exact.hpp). At t=0.2 the
// rarefaction head/shock stay well inside [0,1] (see that header's
// comment for the exact wave positions), so a fixed grid with the far-
// field state held at both ends (StaticBoundary) is physically correct
// -- no moving-frame/grid-recycling machinery needed (explicitly out of
// scope for this slice; that is Phase 4).
//
// Stack: EulerField<Scalar,1> (fields/euler/field.hpp) +
// MusclMinmodReconstruction + RusanovFlux + SSP-RK2, the same
// FvmSolver/ssp_rk2_step assembly every other field in this codebase
// uses -- this test is itself the first end-to-end evidence that the
// Phase-3 generalization of FvmSolver (N=1 -> Field::n_components) and
// the vector RusanovFlux overload actually work together, not just that
// each compiles in isolation.
//
// Known, documented limitation: Rusanov (local Lax-Friedrichs) is the
// most dissipative common numerical flux specifically at a CONTACT
// discontinuity (a characteristic field with zero eigenvalue relative
// to the fastest wave speed used for its dissipation coefficient) --
// the contact is visibly smeared over several dx here, well beyond the
// shock's own width. The star-region density tests below therefore use
// looser tolerance near the contact side and sample points with enough
// margin from every wave to avoid the smeared transition itself; this
// is the scheme behaving exactly as documented
// (numerics/numerical_flux/rusanov.hpp), not a bug. HLLC (tracked as
// still-open against Phase 3's "Rusanov and/or HLLC" acceptance
// criterion) would resolve the contact far more sharply.
#include <cmath>

#include "cfe/core/types.hpp"

#include "sod_shock_tube_exact.hpp"
#include "sod_shock_tube_runner.hpp"
#include "test_framework.hpp"

namespace {

using Scalar = double;
constexpr Scalar kGamma = Scalar(1.4);
constexpr Scalar kX0 = Scalar(0.5);
constexpr Scalar kTFinal = Scalar(0.2);
constexpr Scalar kCfl = Scalar(0.4);

const cfe_verification::PrimitiveState kLeft{1.0, 0.0, 1.0};
const cfe_verification::PrimitiveState kRight{0.125, 0.0, 0.1};

// Thin wrapper fixing this file's own classic-Sod-problem parameters,
// so call sites below read `run_sod(nx)` rather than repeating all six
// arguments every time. The actual simulation is
// cfe_verification::run_sod_shock_tube (tests/verification/
// sod_shock_tube_runner.hpp), shared with tutorials/euler_1d_sod_shock_tube/.
cfe_verification::SodProfile<Scalar> run_sod(cfe::Index nx)
{
  return cfe_verification::run_sod_shock_tube<Scalar>(nx, kGamma, kLeft, kRight, kX0, kTFinal, kCfl);
}

double density_at(const cfe_verification::SodProfile<Scalar>& r, Scalar x_query)
{
  // Nearest-cell lookup (grid is uniform and x_query is always chosen
  // well inside a plateau region, never within one dx of a wave) --
  // exact interpolation is not needed for the tolerances used below.
  cfe::Index best = 0;
  double best_dist = 1e300;
  for (cfe::Index i = 0; i < r.x.size(); ++i) {
    const double d = std::fabs(static_cast<double>(r.x[i]) - static_cast<double>(x_query));
    if (d < best_dist) {
      best_dist = d;
      best = i;
    }
  }
  return r.rho[best];
}

}  // namespace

CFE_TEST(test_sod_shock_tube_exact_wave_positions_match_known_toro_values)
{
  // Cross-check this test's own oracle (sod_shock_tube_exact.hpp)
  // against the standard published values for this exact problem
  // (Toro Table 4.1) before trusting it as a reference for the
  // numerical checks below.
  const auto star = cfe_verification::solve_star_region(kGamma, kLeft, kRight);
  CFE_CHECK_NEAR(star.p_star, 0.30313, 1e-4);
  CFE_CHECK_NEAR(star.u_star, 0.92745, 1e-4);
}

CFE_TEST(test_sod_shock_tube_conserves_mass_and_energy_exactly)
{
  // Mass/energy flux at both domain ends is exactly rho*u=0 /
  // (E+p)*u=0 throughout the run (the near-boundary real cells are
  // never reached by either wave by t=0.2 -- see this file's header
  // comment), so flux-form FVM's exact discrete conservation (the
  // per-cell residual sum telescopes to the boundary flux difference)
  // means total mass/energy must be unchanged to near machine
  // precision, independent of grid resolution or time-step count.
  const cfe_verification::SodProfile<Scalar> r = run_sod(400);
  CFE_CHECK_NEAR(r.total_mass_final, r.total_mass_initial, 1e-10);
  CFE_CHECK_NEAR(r.total_energy_final, r.total_energy_initial, 1e-10);
}

CFE_TEST(test_sod_shock_tube_momentum_change_matches_exact_boundary_impulse)
{
  // Momentum is NOT conserved (physically correct: the left/right
  // pressure difference is a net force on the gas column) -- but by
  // the same telescoping argument, the TOTAL momentum change over the
  // whole run must equal exactly the time integral of the (constant,
  // since u=0 at both undisturbed ends throughout) boundary momentum
  // flux difference: d(total momentum)/dt = -(p_right - p_left) =
  // (p_left - p_right), so total change = (p_left - p_right)*t_final.
  // This is exact in a flux-form discretization regardless of spatial
  // truncation error -- a stronger, more specific check than "roughly
  // conserved".
  const cfe_verification::SodProfile<Scalar> r = run_sod(400);
  const double expected_change = (kLeft.p - kRight.p) * static_cast<double>(kTFinal);
  const double actual_change = r.total_momentum_final - r.total_momentum_initial;
  CFE_CHECK_NEAR(actual_change, expected_change, 1e-8);
}

CFE_TEST(test_sod_shock_tube_matches_exact_solution_away_from_smeared_waves)
{
  const auto star = cfe_verification::solve_star_region(kGamma, kLeft, kRight);
  auto exact_at = [&](double x) {
    const double s = (x - static_cast<double>(kX0)) / static_cast<double>(kTFinal);
    return cfe_verification::sample_exact_solution(kGamma, kLeft, kRight, star, s);
  };

  const cfe_verification::SodProfile<Scalar> r = run_sod(400);

  // Undisturbed left/right far field: tight tolerance, these cells are
  // literally untouched by any wave.
  CFE_CHECK_NEAR(density_at(r, 0.1), exact_at(0.1).rho, 1e-6);
  CFE_CHECK_NEAR(density_at(r, 0.95), exact_at(0.95).rho, 1e-6);

  // Inside the (smooth) rarefaction fan: the scheme has no discontinuity
  // to smear here, so this should track the exact self-similar profile
  // closely. Measured error at nx=400 is ~1.3e-3 (reconstruction error
  // from the fan's own curvature, not wave-smearing); 5e-3 keeps a ~4x
  // margin without being so loose it would pass a genuinely broken
  // reconstruction.
  CFE_CHECK_NEAR(density_at(r, 0.35), exact_at(0.35).rho, 5e-3);

  // Star-left plateau (behind the rarefaction, ahead of the contact):
  // sampled with margin from both neighboring waves. Measured error at
  // nx=400 is ~2e-4 (this plateau is flat and far from both its
  // neighboring waves); 1e-3 keeps a ~5x margin.
  CFE_CHECK_NEAR(density_at(r, 0.55), exact_at(0.55).rho, 1e-3);

  // Star-right plateau (behind the contact, ahead of the shock): this
  // is the side of the plateau nearest the Rusanov-smeared contact (see
  // this file's header comment), but at this sample point (margin
  // ~0.095 from the contact, ~0.07 from the shock) the smearing has not
  // reached it at nx=400 -- measured error is ~3e-5. 1e-3 still keeps a
  // deliberately generous (~30x) margin given this point's proximity to
  // the known-worst-case wave, without using the 0.08 placeholder a
  // prior version of this test used before the actual measured error
  // was checked.
  CFE_CHECK_NEAR(density_at(r, 0.78), exact_at(0.78).rho, 1e-3);
}

CFE_TEST(test_sod_shock_tube_density_l1_error_decreases_with_resolution)
{
  // A weak convergence signal, not a formal order-of-accuracy claim:
  // the exact solution has genuine discontinuities (shock, contact), so
  // no single scheme recovers a clean formal convergence rate globally
  // (first-order behavior at discontinuities is expected and
  // documented, not a defect -- see MusclMinmodReconstruction's own
  // header comment on TVD limiters). What IS a meaningful correctness
  // check is that refining the grid does not make the solution worse.
  const auto star = cfe_verification::solve_star_region(kGamma, kLeft, kRight);
  auto exact_density_at = [&](double x) {
    const double s = (x - static_cast<double>(kX0)) / static_cast<double>(kTFinal);
    return cfe_verification::sample_exact_solution(kGamma, kLeft, kRight, star, s).rho;
  };

  auto l1_error = [&](cfe::Index nx) {
    const cfe_verification::SodProfile<Scalar> r = run_sod(nx);
    double err = 0.0;
    const double dx = 1.0 / static_cast<double>(nx);
    for (cfe::Index i = 0; i < r.x.size(); ++i) {
      err += std::fabs(static_cast<double>(r.rho[i]) - exact_density_at(static_cast<double>(r.x[i]))) * dx;
    }
    return err;
  };

  const double err_coarse = l1_error(100);
  const double err_fine = l1_error(400);
  CFE_CHECK(err_fine < err_coarse);
}
