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
//
// Dual-precision instantiation (task 0017): every check below is a
// `template <class Scalar>` function, instantiated at both `double`
// (unchanged tolerances) and `float`. The oracle
// (`sod_shock_tube_exact.hpp`'s `solve_star_region`/
// `sample_exact_solution`) stays `double` regardless of `Scalar` -- it
// is a reference solution, not simulation state, same convention used
// project-wide.
//
// Measured (this task's own calibration, same toolchain this file's
// tolerances were themselves measured on): the TRUNCATION-dominated
// checks below (the exact-solution density samples, the L1-convergence
// trend) come out NEARLY IDENTICAL at float and double -- spatial
// discretization error, not floating-point roundoff, dominates these,
// so they need no retuning at float. Only the two EXACT-conservation
// checks (which rely on roundoff-level cancellation, not truncation)
// need a widened, float-epsilon-scaled tolerance: measured mass/energy
// drift ~8.6e-10/~4.1e-8 (vs. double's ~7.8e-15/~1.6e-14) and momentum-
// impulse error ~9.2e-8 (vs. double's ~8.3e-17) -- both still many
// orders of magnitude tighter than a 1e-6 tolerance, which is used at
// float for both (>=10x measured margin in every case).
#include <cmath>
#include <cstddef>

#include "sod_shock_tube_exact.hpp"
#include "sod_shock_tube_runner.hpp"
#include "test_framework.hpp"

namespace {

template <class Scalar>
struct SodClassic
{
  static constexpr Scalar kGamma = Scalar(1.4);
  static constexpr Scalar kX0 = Scalar(0.5);
  static constexpr Scalar kTFinal = Scalar(0.2);
  static constexpr Scalar kCfl = Scalar(0.4);
};

const cfe_verification::PrimitiveState kLeft{1.0, 0.0, 1.0};
const cfe_verification::PrimitiveState kRight{0.125, 0.0, 0.1};

// Thin wrapper fixing this file's own classic-Sod-problem parameters,
// so call sites below read `run_sod<Scalar>(nx)` rather than repeating
// all six arguments every time. The actual simulation is
// cfe_verification::run_sod_shock_tube (tests/verification/
// sod_shock_tube_runner.hpp), shared with tutorials/euler_1d_sod_shock_tube/.
template <class Scalar>
cfe_verification::SodProfile<Scalar> run_sod(std::size_t nx)
{
  using P = SodClassic<Scalar>;
  return cfe_verification::run_sod_shock_tube<Scalar>(nx, P::kGamma, kLeft, kRight, P::kX0, P::kTFinal, P::kCfl);
}

template <class Scalar>
double density_at(const cfe_verification::SodProfile<Scalar>& r, Scalar x_query)
{
  // Nearest-cell lookup (grid is uniform and x_query is always chosen
  // well inside a plateau region, never within one dx of a wave) --
  // exact interpolation is not needed for the tolerances used below.
  std::size_t best = 0;
  double best_dist = 1e300;
  for (std::size_t i = 0; i < r.x.size(); ++i) {
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
  // numerical checks below. The oracle is always `double` -- no `float`
  // counterpart needed, nothing here is `Scalar`-templated.
  const auto star = cfe_verification::solve_star_region(1.4, kLeft, kRight);
  CFE_CHECK_NEAR(star.p_star, 0.30313, 1e-4);
  CFE_CHECK_NEAR(star.u_star, 0.92745, 1e-4);
}

template <class Scalar>
void check_sod_shock_tube_conserves_mass_and_energy(double tol)
{
  // Mass/energy flux at both domain ends is exactly rho*u=0 /
  // (E+p)*u=0 throughout the run (the near-boundary real cells are
  // never reached by either wave by t=0.2 -- see this file's header
  // comment), so flux-form FVM's exact discrete conservation (the
  // per-cell residual sum telescopes to the boundary flux difference)
  // means total mass/energy must be unchanged to near machine
  // precision, independent of grid resolution or time-step count.
  const cfe_verification::SodProfile<Scalar> r = run_sod<Scalar>(400);
  CFE_CHECK_NEAR(r.total_mass_final, r.total_mass_initial, tol);
  CFE_CHECK_NEAR(r.total_energy_final, r.total_energy_initial, tol);
}

CFE_TEST(test_sod_shock_tube_conserves_mass_and_energy_exactly_double)
{
  check_sod_shock_tube_conserves_mass_and_energy<double>(1e-10);
}
CFE_TEST(test_sod_shock_tube_conserves_mass_and_energy_exactly_float)
{
  check_sod_shock_tube_conserves_mass_and_energy<float>(1e-6);
}

template <class Scalar>
void check_sod_shock_tube_momentum_change_matches_exact_boundary_impulse(double tol)
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
  using P = SodClassic<Scalar>;
  const cfe_verification::SodProfile<Scalar> r = run_sod<Scalar>(400);
  const double expected_change = (kLeft.p - kRight.p) * static_cast<double>(P::kTFinal);
  const double actual_change = r.total_momentum_final - r.total_momentum_initial;
  CFE_CHECK_NEAR(actual_change, expected_change, tol);
}

CFE_TEST(test_sod_shock_tube_momentum_change_matches_exact_boundary_impulse_double)
{
  check_sod_shock_tube_momentum_change_matches_exact_boundary_impulse<double>(1e-8);
}
CFE_TEST(test_sod_shock_tube_momentum_change_matches_exact_boundary_impulse_float)
{
  check_sod_shock_tube_momentum_change_matches_exact_boundary_impulse<float>(1e-6);
}

template <class Scalar>
void check_sod_shock_tube_matches_exact_solution_away_from_smeared_waves()
{
  using P = SodClassic<Scalar>;
  const auto star = cfe_verification::solve_star_region(1.4, kLeft, kRight);
  auto exact_at = [&](double x) {
    const double s = (x - static_cast<double>(P::kX0)) / static_cast<double>(P::kTFinal);
    return cfe_verification::sample_exact_solution(1.4, kLeft, kRight, star, s);
  };

  const cfe_verification::SodProfile<Scalar> r = run_sod<Scalar>(400);

  // Measured identical (to displayed precision) at float and double:
  // these are dominated by spatial truncation error, not floating-point
  // roundoff, so every tolerance below is unchanged from double.

  // Undisturbed left/right far field: tight tolerance, these cells are
  // literally untouched by any wave.
  CFE_CHECK_NEAR(density_at(r, Scalar(0.1)), exact_at(0.1).rho, 1e-6);
  CFE_CHECK_NEAR(density_at(r, Scalar(0.95)), exact_at(0.95).rho, 1e-6);

  // Inside the (smooth) rarefaction fan: the scheme has no discontinuity
  // to smear here, so this should track the exact self-similar profile
  // closely. Measured error at nx=400 is ~1.3e-3 (reconstruction error
  // from the fan's own curvature, not wave-smearing); 5e-3 keeps a ~4x
  // margin without being so loose it would pass a genuinely broken
  // reconstruction.
  CFE_CHECK_NEAR(density_at(r, Scalar(0.35)), exact_at(0.35).rho, 5e-3);

  // Star-left plateau (behind the rarefaction, ahead of the contact):
  // sampled with margin from both neighboring waves. Measured error at
  // nx=400 is ~2e-4 (this plateau is flat and far from both its
  // neighboring waves); 1e-3 keeps a ~5x margin.
  CFE_CHECK_NEAR(density_at(r, Scalar(0.55)), exact_at(0.55).rho, 1e-3);

  // Star-right plateau (behind the contact, ahead of the shock): this
  // is the side of the plateau nearest the Rusanov-smeared contact (see
  // this file's header comment), but at this sample point (margin
  // ~0.095 from the contact, ~0.07 from the shock) the smearing has not
  // reached it at nx=400 -- measured error is ~3e-5. 1e-3 still keeps a
  // deliberately generous (~30x) margin given this point's proximity to
  // the known-worst-case wave, without using the 0.08 placeholder a
  // prior version of this test used before the actual measured error
  // was checked.
  CFE_CHECK_NEAR(density_at(r, Scalar(0.78)), exact_at(0.78).rho, 1e-3);
}

CFE_TEST(test_sod_shock_tube_matches_exact_solution_away_from_smeared_waves_double)
{
  check_sod_shock_tube_matches_exact_solution_away_from_smeared_waves<double>();
}
CFE_TEST(test_sod_shock_tube_matches_exact_solution_away_from_smeared_waves_float)
{
  check_sod_shock_tube_matches_exact_solution_away_from_smeared_waves<float>();
}

template <class Scalar>
void check_sod_shock_tube_density_l1_error_decreases_with_resolution()
{
  // A weak convergence signal, not a formal order-of-accuracy claim:
  // the exact solution has genuine discontinuities (shock, contact), so
  // no single scheme recovers a clean formal convergence rate globally
  // (first-order behavior at discontinuities is expected and
  // documented, not a defect -- see MusclMinmodReconstruction's own
  // header comment on TVD limiters). What IS a meaningful correctness
  // check is that refining the grid does not make the solution worse --
  // true at float just as at double (measured L1 error at float: 100 ->
  // 0.01055, 400 -> 0.00319, matching double to 5 decimal places).
  const auto star = cfe_verification::solve_star_region(1.4, kLeft, kRight);
  auto exact_density_at = [&](double x) {
    const double s = (x - static_cast<double>(SodClassic<Scalar>::kX0)) / static_cast<double>(SodClassic<Scalar>::kTFinal);
    return cfe_verification::sample_exact_solution(1.4, kLeft, kRight, star, s).rho;
  };

  auto l1_error = [&](std::size_t nx) {
    const cfe_verification::SodProfile<Scalar> r = run_sod<Scalar>(nx);
    double err = 0.0;
    const double dx = 1.0 / static_cast<double>(nx);
    for (std::size_t i = 0; i < r.x.size(); ++i) {
      err += std::fabs(static_cast<double>(r.rho[i]) - exact_density_at(static_cast<double>(r.x[i]))) * dx;
    }
    return err;
  };

  const double err_coarse = l1_error(100);
  const double err_fine = l1_error(400);
  CFE_CHECK(err_fine < err_coarse);
}

CFE_TEST(test_sod_shock_tube_density_l1_error_decreases_with_resolution_double)
{
  check_sod_shock_tube_density_l1_error_decreases_with_resolution<double>();
}
CFE_TEST(test_sod_shock_tube_density_l1_error_decreases_with_resolution_float)
{
  check_sod_shock_tube_density_l1_error_decreases_with_resolution<float>();
}
