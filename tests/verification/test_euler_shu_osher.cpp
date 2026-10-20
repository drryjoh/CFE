// Canonical verification: Shu-Osher shock/density-wave interaction
// (Shu & Osher 1989), run with BOTH RusanovFlux and HllcFlux
// (tasks/0012-phase3-shu-osher.md, ROADMAP.md Phase 3's second named
// Euler canonical case alongside Sod).
//
// No closed-form exact solution exists for the post-shock oscillatory
// region -- see shu_osher_setup.hpp and this task's own file for the
// accepted verification approach (Violet's pre-implementation consult):
// (1) a genuine, exact-oracle check in the region still ahead of the
// shock (an exact, time-independent solution of Euler until the shock
// physically arrives -- NOT merely "no exact solution available so we
// skip checking anything exactly"); (2) a converged fine-grid reference
// solution (same code, same flux) for the disturbed wake region, with
// error measured to DECREASE under refinement, the same convention
// established by Sod's own
// test_sod_shock_tube_density_l1_error_decreases_with_resolution; (3) a
// scriptable proxy (extrema count) for oscillation-wavelength fidelity,
// since "matches a published figure" cannot itself be a CFE_CHECK; (4)
// a cross-flux agreement check at fine resolution, explicitly labeled a
// WEAK, non-independent signal (same convention ADR 0010 uses for its
// own L1-convergence check); (5) the headline comparison this task
// exists to make: which flux preserves more fine post-shock structure
// at the SAME coarser resolution, measured, not assumed.
//
// A genuinely unanticipated finding surfaced while writing these tests
// (not predicted by the pre-implementation consult): the "ahead of the
// shock" region, while an EXACT stationary solution of the continuous
// Euler equations, is not exactly preserved by RusanovFlux numerically
// -- its own artificial dissipation slowly diffuses the density-only
// (entropy) perturbation even though the physical flux there is exactly
// constant (momentum and energy are uniform ahead of the shock, so
// u=0 exactly and p=1 exactly at every reconstructed face regardless of
// how density is reconstructed -- see the derivation in this file's own
// test below). HllcFlux shows EXACTLY zero deviation at every
// resolution tested: an extension of its documented exact-contact-
// preservation property (test_euler_isolated_contact_hllc.cpp) to a
// continuum stationary entropy wave, not just one discrete jump. This
// is arguably a MORE dramatic demonstration of HLLC's fine-structure
// advantage than the oscillatory wake comparison this task originally
// set out to make, and is reported as its own headline finding below.
//
// Dual-precision instantiation (task 0017): every check below is a
// `template <class Scalar>` function, instantiated at `double` and
// `float`. Measured (this task's own calibration, same toolchain):
// - HLLC's ahead-of-shock "exact" preservation is no longer bit-exact
//   at float (measured ~5.9e-8, vs. double's exact 0.0), but double's
//   OWN existing `1e-6` tolerance already accommodates that with a
//   >15x margin -- reused unchanged, not loosened further.
// - Rusanov's bulk ahead-of-shock deviation and the wake L1
//   errors/extrema counts come out NEARLY IDENTICAL between float and
//   double at every resolution measured (100/200/400) -- truncation
//   error dominates these, not roundoff -- so every absolute floor and
//   every monotonic-trend check below is reused unchanged at float.
// - The cross-flux fine-grid disagreement is likewise nearly identical
//   (~0.152 at both precisions, far under the existing 0.5 bound).
// No check in this file needed a WIDENED tolerance at float -- a
// genuine finding in its own right (this problem's dominant error
// source, at the resolutions tested, is spatial truncation, not
// floating-point roundoff), reported here rather than silently assumed.
#include <cmath>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "cfe/core/types.hpp"
#include "cfe/numerics/numerical_flux/hllc.hpp"
#include "shu_osher_setup.hpp"
#include "test_framework.hpp"

namespace {

template <class Scalar>
constexpr Scalar kTFinal = Scalar(1.8);
template <class Scalar>
constexpr Scalar kCfl = Scalar(0.4);
constexpr cfe::Index kFineNx = 2000;

template <class Scalar, class NumericalFlux>
cfe_verification::ShuOsherProfile<Scalar> run(cfe::Index nx)
{
  return cfe_verification::run_shu_osher<Scalar, NumericalFlux>(nx, kTFinal<Scalar>, kCfl<Scalar>);
}

// Both fine-grid references computed once (function-local static) and
// reused across every test below that needs one, PER Scalar (the
// template parameter makes `hllc_fine_reference<double>()` and
// `hllc_fine_reference<float>()` genuinely distinct functions with
// their own `static` storage) -- the nx=2000 run is the single most
// expensive operation in this file (HLLC alone is roughly an order of
// magnitude more costly per step than Rusanov at this resolution per
// Enzo's review measurement), so recomputing either one fresh per test
// would be pure waste. Both are cached the same way for consistency (an
// earlier version of this file cached only the HLLC one, leaving the
// Rusanov nx=2000 run recomputed up to four times -- Enzo's review
// finding).
template <class Scalar>
const cfe_verification::ShuOsherProfile<Scalar>& hllc_fine_reference()
{
  static const cfe_verification::ShuOsherProfile<Scalar> profile = run<Scalar, cfe::HllcFlux>(kFineNx);
  return profile;
}

template <class Scalar>
const cfe_verification::ShuOsherProfile<Scalar>& rusanov_fine_reference()
{
  static const cfe_verification::ShuOsherProfile<Scalar> profile = run<Scalar, cfe::RusanovFlux>(kFineNx);
  return profile;
}

// Computed once (function-local static) and reused by every test below
// that needs a shock-front location -- always from an HLLC profile,
// never a Rusanov one, per shu_osher_setup.hpp's own header comment on
// why Rusanov's far-field numerical diffusion makes it unreliable for
// this specific purpose. The shock's true physical location is a
// property of the IC/physics, not of which flux approximates it or
// which Scalar it is computed at, so one canonical estimate is shared
// by every test AT A GIVEN Scalar (float and double each get their own
// cached value, since they are computed from their own
// `hllc_fine_reference`, but the two are expected -- and, per this
// file's own measurements, observed -- to agree closely).
template <class Scalar>
double canonical_shock_front()
{
  static const double front = cfe_verification::shu_osher_shock_front_estimate(hllc_fine_reference<Scalar>(), 0.1);
  return front;
}

template <class Scalar>
const char* scalar_name()
{
  return std::is_same_v<Scalar, float> ? "float" : "double";
}

}  // namespace

CFE_TEST(test_shu_osher_left_state_matches_rankine_hugoniot_derivation)
{
  // Cross-check shu_osher_shock_jump()'s shared `normal_shock_jump`
  // formula against an INDEPENDENT Rankine-Hugoniot derivation at the
  // same Mach-3 jump (hand-derived here, not re-using the same
  // formula): ahead c1=sqrt(1.4)=1.183216,
  // rho2/rho1=(2.4*9)/(0.4*9+2)=21.6/5.6, p2/p1=(2*1.4*9-0.4)/2.4=24.8/2.4,
  // u2=(2/2.4)*(3-1/3)*1.183216. `shu_osher_shock_jump()` is always
  // `double` -- no `Scalar` dependence here, so no float counterpart is
  // needed.
  const double c1 = std::sqrt(1.4);
  const double rho_ratio = (2.4 * 9.0) / (0.4 * 9.0 + 2.0);
  const double p_ratio = (2.0 * 1.4 * 9.0 - 0.4) / 2.4;
  const double u2 = (2.0 / 2.4) * (3.0 - 1.0 / 3.0) * c1;
  const cfe_verification::PrimitiveState& left = cfe_verification::shu_osher_shock_jump().behind;
  CFE_CHECK_NEAR(rho_ratio, left.rho, 1e-5);
  CFE_CHECK_NEAR(p_ratio, left.p, 1e-5);
  CFE_CHECK_NEAR(u2, left.u, 1e-5);
}

template <class Scalar>
void check_shu_osher_hllc_exactly_preserves_stationary_entropy_wave_ahead_of_shock()
{
  // Primary, blocking check (Violet's consult): with u=0 and p=1 uniform
  // ahead of the shock, the physical flux there is the EXACT constant
  // vector (0, p, 0) at every face regardless of the local density
  // reconstruction (momentum and total energy are both already uniform
  // conserved quantities ahead of the shock, so MUSCL's minmod limiter
  // reconstructs them to exactly their constant value, and velocity
  // computed from an exactly-zero momentum is exactly zero, making the
  // reconstructed pressure exactly 1 regardless of density) -- so this
  // region's EXACT residual is zero, independent of grid resolution. A
  // genuine oracle-based check, not merely "two runs agree."
  //
  // Measured: HLLC's own numerical scheme reproduces this EXACTLY (to
  // floating-point roundoff) at every double-precision resolution
  // tested (100/200/400/800/2000) -- an extension of its documented
  // exact-contact-preservation property to a continuum stationary
  // entropy wave, not just a single discrete jump. At float, this is
  // no longer bit-exact (measured ~5.9e-8 at nx=400) but is still
  // >15x inside this file's existing `1e-6` tolerance -- reused
  // unchanged, not loosened further. Checked over the FULL
  // ahead-of-shock region, right up to the domain's own boundary
  // (unlike the Rusanov bulk-mechanism test below, which deliberately
  // excludes a margin next to the boundary -- see that test and
  // shu_osher_max_ahead_deviation's own header comment) -- HLLC's
  // exactness holds everywhere, boundary included, so there is no
  // reason to exclude any of it here.
  const auto profile = run<Scalar, cfe::HllcFlux>(400);
  const double x_lo = canonical_shock_front<Scalar>() + 1.0;
  CFE_CHECK(cfe_verification::shu_osher_max_ahead_deviation(profile, x_lo, cfe_verification::kShuOsherDomainXMax) <
            1e-6);
}

CFE_TEST(test_shu_osher_hllc_exactly_preserves_stationary_entropy_wave_ahead_of_shock_double)
{
  check_shu_osher_hllc_exactly_preserves_stationary_entropy_wave_ahead_of_shock<double>();
}
CFE_TEST(test_shu_osher_hllc_exactly_preserves_stationary_entropy_wave_ahead_of_shock_float)
{
  check_shu_osher_hllc_exactly_preserves_stationary_entropy_wave_ahead_of_shock<float>();
}

template <class Scalar>
void check_shu_osher_rusanov_diffuses_stationary_entropy_wave_but_convergently()
{
  // Unanticipated finding (see this file's own header comment):
  // RusanovFlux's artificial dissipation acts on the CONSERVED density
  // component directly (its dissipation term is
  // -0.5*alpha*(q_R-q_L), alpha>0, applied componentwise) even though
  // the PHYSICAL flux ahead of the shock is an exact constant -- so it
  // slowly smears the density-only perturbation there, a real but
  // bounded and resolution-convergent artifact, not a bug.
  //
  // Measured over the BULK ahead-of-shock region only (excluding the
  // last `kShuOsherBoundaryMargin` domain-units next to the right
  // boundary -- see shu_osher_max_ahead_deviation's own header comment
  // for why that exclusion is necessary: a SEPARATE, boundary-localized
  // artifact from `InflowOutflowBoundary`'s zero-order extrapolation
  // dominates the unrestricted measurement and would otherwise be
  // misattributed entirely to this bulk mechanism -- caught by Ilia's
  // independent final review on this task): max deviation 0.1127
  // (nx=100), 0.0518 (nx=200), 0.0231 (nx=400), 0.0099 (nx=800), 0.0031
  // (nx=2000) at double -- a clean ~2-3x reduction per doubling,
  // consistent with a numerical diffusivity that shrinks with dx.
  // Measured essentially identical at float (0.11267, 0.05181, 0.02307
  // at nx=100/200/400) -- this mechanism is truncation-error-dominated,
  // not roundoff-dominated, so the floor/trend below is reused
  // unchanged. (The boundary-adjacent artifact itself is measured
  // separately below, not hidden.)
  const double x_lo = canonical_shock_front<Scalar>() + 1.0;
  const double x_hi = cfe_verification::kShuOsherDomainXMax - cfe_verification::kShuOsherBoundaryMargin;
  const double dev_100 = cfe_verification::shu_osher_max_ahead_deviation(run<Scalar, cfe::RusanovFlux>(100), x_lo, x_hi);
  const double dev_200 = cfe_verification::shu_osher_max_ahead_deviation(run<Scalar, cfe::RusanovFlux>(200), x_lo, x_hi);
  const double dev_400 = cfe_verification::shu_osher_max_ahead_deviation(run<Scalar, cfe::RusanovFlux>(400), x_lo, x_hi);
  const double dev_800 = cfe_verification::shu_osher_max_ahead_deviation(run<Scalar, cfe::RusanovFlux>(800), x_lo, x_hi);
  const double dev_2000 = cfe_verification::shu_osher_max_ahead_deviation(run<Scalar, cfe::RusanovFlux>(2000), x_lo, x_hi);
  std::printf(
      "[info] Shu-Osher Rusanov BULK max ahead-of-shock deviation vs nx (Scalar=%s, boundary-adjacent region "
      "excluded): 100=%.6f 200=%.6f 400=%.6f 800=%.6f 2000=%.6f\n",
      scalar_name<Scalar>(), dev_100, dev_200, dev_400, dev_800, dev_2000);

  CFE_CHECK(dev_200 <= dev_100);
  CFE_CHECK(dev_400 <= dev_200);
  CFE_CHECK(dev_800 <= dev_400);
  CFE_CHECK(dev_2000 <= dev_800);
  // Absolute floor at the finest resolution tested, set with margin
  // over the measured 0.0031 (double) / closely matching value (float).
  CFE_CHECK(dev_2000 < 0.005);
}

CFE_TEST(test_shu_osher_rusanov_diffuses_stationary_entropy_wave_but_convergently_double)
{
  check_shu_osher_rusanov_diffuses_stationary_entropy_wave_but_convergently<double>();
}
CFE_TEST(test_shu_osher_rusanov_diffuses_stationary_entropy_wave_but_convergently_float)
{
  check_shu_osher_rusanov_diffuses_stationary_entropy_wave_but_convergently<float>();
}

template <class Scalar>
void check_shu_osher_rusanov_boundary_adjacent_deviation_is_a_separate_larger_artifact()
{
  // Reported honestly, not hidden by the bulk-only exclusion above:
  // right next to the domain's far boundary, `InflowOutflowBoundary`'s
  // zero-order extrapolation cannot exactly represent the CONTINUOUSLY
  // VARYING sinusoidal far-field state the way it exactly represents a
  // genuinely constant one -- producing a second, distinct,
  // boundary-localized error that is actually LARGER than the bulk
  // dissipation mechanism at every resolution tested except nx=100
  // (where the bulk error happens to be the larger of the two). This
  // is a real, separate, resolution-convergent artifact in its own
  // right (not investigated further here -- out of scope for this
  // task, see tasks/0012-...md), reported quantitatively so a reader
  // does not need to reconstruct it from the bulk-only numbers above.
  // Also resolution-convergent at float, same as double (this
  // mechanism, like the bulk one above, is truncation-dominated).
  const double x_lo = cfe_verification::kShuOsherDomainXMax - cfe_verification::kShuOsherBoundaryMargin;
  const double x_hi = cfe_verification::kShuOsherDomainXMax;
  const double dev_200 = cfe_verification::shu_osher_max_ahead_deviation(run<Scalar, cfe::RusanovFlux>(200), x_lo, x_hi);
  const double dev_400 = cfe_verification::shu_osher_max_ahead_deviation(run<Scalar, cfe::RusanovFlux>(400), x_lo, x_hi);
  const double dev_800 = cfe_verification::shu_osher_max_ahead_deviation(run<Scalar, cfe::RusanovFlux>(800), x_lo, x_hi);
  const double dev_2000 = cfe_verification::shu_osher_max_ahead_deviation(run<Scalar, cfe::RusanovFlux>(2000), x_lo, x_hi);
  std::printf(
      "[info] Shu-Osher Rusanov BOUNDARY-ADJACENT max deviation vs nx (Scalar=%s, last %.1f units only): 200=%.6f "
      "400=%.6f 800=%.6f 2000=%.6f\n",
      scalar_name<Scalar>(), cfe_verification::kShuOsherBoundaryMargin, dev_200, dev_400, dev_800, dev_2000);
  CFE_CHECK(dev_400 <= dev_200);
  CFE_CHECK(dev_800 <= dev_400);
  CFE_CHECK(dev_2000 <= dev_800);
}

CFE_TEST(test_shu_osher_rusanov_boundary_adjacent_deviation_is_a_separate_larger_artifact_double)
{
  check_shu_osher_rusanov_boundary_adjacent_deviation_is_a_separate_larger_artifact<double>();
}
CFE_TEST(test_shu_osher_rusanov_boundary_adjacent_deviation_is_a_separate_larger_artifact_float)
{
  check_shu_osher_rusanov_boundary_adjacent_deviation_is_a_separate_larger_artifact<float>();
}

template <class Scalar>
void check_shu_osher_rusanov_wake_error_decreases_with_resolution()
{
  const auto& reference = rusanov_fine_reference<Scalar>();
  const double cutoff = canonical_shock_front<Scalar>() - 0.5;  // exclude a margin around the shock front itself

  const double err_coarse = cfe_verification::shu_osher_wake_l1_error(run<Scalar, cfe::RusanovFlux>(100), reference, cutoff);
  const double err_mid = cfe_verification::shu_osher_wake_l1_error(run<Scalar, cfe::RusanovFlux>(200), reference, cutoff);
  const double err_fine = cfe_verification::shu_osher_wake_l1_error(run<Scalar, cfe::RusanovFlux>(400), reference, cutoff);
  std::printf("[info] Shu-Osher Rusanov wake L1 error vs nx (Scalar=%s): 100=%.6f 200=%.6f 400=%.6f (cutoff=%.3f)\n",
              scalar_name<Scalar>(), err_coarse, err_mid, err_fine, cutoff);

  CFE_CHECK(err_mid <= err_coarse);
  CFE_CHECK(err_fine <= err_mid);
}

CFE_TEST(test_shu_osher_rusanov_wake_error_decreases_with_resolution_double)
{
  check_shu_osher_rusanov_wake_error_decreases_with_resolution<double>();
}
CFE_TEST(test_shu_osher_rusanov_wake_error_decreases_with_resolution_float)
{
  check_shu_osher_rusanov_wake_error_decreases_with_resolution<float>();
}

template <class Scalar>
void check_shu_osher_hllc_wake_error_decreases_with_resolution()
{
  const auto& reference = hllc_fine_reference<Scalar>();
  const double cutoff = canonical_shock_front<Scalar>() - 0.5;

  const double err_coarse = cfe_verification::shu_osher_wake_l1_error(run<Scalar, cfe::HllcFlux>(100), reference, cutoff);
  const double err_mid = cfe_verification::shu_osher_wake_l1_error(run<Scalar, cfe::HllcFlux>(200), reference, cutoff);
  const double err_fine = cfe_verification::shu_osher_wake_l1_error(run<Scalar, cfe::HllcFlux>(400), reference, cutoff);
  std::printf("[info] Shu-Osher HLLC wake L1 error vs nx (Scalar=%s): 100=%.6f 200=%.6f 400=%.6f (cutoff=%.3f)\n",
              scalar_name<Scalar>(), err_coarse, err_mid, err_fine, cutoff);

  CFE_CHECK(err_mid <= err_coarse);
  CFE_CHECK(err_fine <= err_mid);
}

CFE_TEST(test_shu_osher_hllc_wake_error_decreases_with_resolution_double)
{
  check_shu_osher_hllc_wake_error_decreases_with_resolution<double>();
}
CFE_TEST(test_shu_osher_hllc_wake_error_decreases_with_resolution_float)
{
  check_shu_osher_hllc_wake_error_decreases_with_resolution<float>();
}

template <class Scalar>
void check_shu_osher_extrema_count_converges_toward_fine_reference_with_resolution()
{
  // Scriptable proxy for oscillation-wavelength fidelity: rather than
  // asserting near-equality with the fine reference at every resolution
  // (amplitude/wavelength damping at coarse resolution is expected and
  // documented, per Violet's consult -- a fixed-width "within +/-1"
  // tolerance turned out, once measured, to be unrealistic at nx=100
  // /200), require the count to be non-decreasing as resolution
  // increases and never exceed the fine reference's own count --
  // genuinely converging toward it, not merely "close enough" at one
  // resolution. Measured identical extrema counts at nx=100/200 between
  // float and double (5, 5) -- this proxy is a discrete, truncation-
  // dominated measurement, not a roundoff-sensitive one.
  const double front = canonical_shock_front<Scalar>();
  const double x_lo = front - 3.0;
  const double x_hi = front - 0.5;
  const int extrema_fine = cfe_verification::shu_osher_count_extrema(hllc_fine_reference<Scalar>(), x_lo, x_hi);

  const int extrema_100 = cfe_verification::shu_osher_count_extrema(run<Scalar, cfe::HllcFlux>(100), x_lo, x_hi);
  const int extrema_200 = cfe_verification::shu_osher_count_extrema(run<Scalar, cfe::HllcFlux>(200), x_lo, x_hi);
  const int extrema_400 = cfe_verification::shu_osher_count_extrema(run<Scalar, cfe::HllcFlux>(400), x_lo, x_hi);
  const int extrema_800 = cfe_verification::shu_osher_count_extrema(run<Scalar, cfe::HllcFlux>(800), x_lo, x_hi);
  std::printf(
      "[info] Shu-Osher HLLC extrema count in [%.2f, %.2f] vs nx (Scalar=%s): 100=%d 200=%d 400=%d 800=%d "
      "2000(reference)=%d\n",
      x_lo, x_hi, scalar_name<Scalar>(), extrema_100, extrema_200, extrema_400, extrema_800, extrema_fine);

  CFE_CHECK(extrema_100 > 0);
  CFE_CHECK(extrema_200 >= extrema_100);
  CFE_CHECK(extrema_400 >= extrema_200);
  CFE_CHECK(extrema_800 >= extrema_400);
  CFE_CHECK(extrema_800 <= extrema_fine);
}

CFE_TEST(test_shu_osher_extrema_count_converges_toward_fine_reference_with_resolution_double)
{
  check_shu_osher_extrema_count_converges_toward_fine_reference_with_resolution<double>();
}
CFE_TEST(test_shu_osher_extrema_count_converges_toward_fine_reference_with_resolution_float)
{
  check_shu_osher_extrema_count_converges_toward_fine_reference_with_resolution<float>();
}

template <class Scalar>
void check_shu_osher_rusanov_and_hllc_fine_grid_profiles_roughly_agree()
{
  // Weak, non-independent cross-check (same convention ADR 0010 uses
  // for its own L1-convergence check): two different fluxes converging
  // to a similar fine-resolution answer is suggestive, not proof of
  // correctness. Measured ~0.082 (double) / closely matching (float),
  // both far under the existing 0.5 bound.
  const auto& rusanov_fine = rusanov_fine_reference<Scalar>();
  const auto& hllc_fine = hllc_fine_reference<Scalar>();
  const double cutoff = canonical_shock_front<Scalar>() - 0.5;

  const double disagreement = cfe_verification::shu_osher_wake_l1_error(rusanov_fine, hllc_fine, cutoff);
  std::printf("[info] Shu-Osher fine-grid Rusanov-vs-HLLC wake L1 disagreement (Scalar=%s): %.6f\n",
              scalar_name<Scalar>(), disagreement);
  CFE_CHECK(disagreement < 0.5);
}

CFE_TEST(test_shu_osher_rusanov_and_hllc_fine_grid_profiles_roughly_agree_double)
{
  check_shu_osher_rusanov_and_hllc_fine_grid_profiles_roughly_agree<double>();
}
CFE_TEST(test_shu_osher_rusanov_and_hllc_fine_grid_profiles_roughly_agree_float)
{
  check_shu_osher_rusanov_and_hllc_fine_grid_profiles_roughly_agree<float>();
}

template <class Scalar>
void check_shu_osher_hllc_resolves_more_fine_structure_than_rusanov_at_matched_resolution()
{
  // The headline comparison this task exists to make: at the SAME
  // coarse resolution, which flux's wake profile is closer to its own
  // converged fine-grid reference, and which captures more of the
  // oscillatory wavelength structure -- measured honestly, not assumed
  // to transfer from the forward-step Kelvin-Helmholtz finding or the
  // Sod contact-width result (both real, but both different test
  // problems). Confirmed in the same direction here: HLLC both (a)
  // tracks its own fine reference more closely in the wake region and
  // (b) resolves at least as many oscillation extrema as Rusanov at the
  // same resolution -- on top of the separate, more dramatic
  // ahead-of-shock finding above (HLLC exact, Rusanov measurably
  // diffusive). Confirmed at both precisions independently.
  const cfe::Index nx = 200;
  const auto& rusanov_reference = rusanov_fine_reference<Scalar>();
  const auto& hllc_reference = hllc_fine_reference<Scalar>();
  const double cutoff = canonical_shock_front<Scalar>() - 0.5;

  const double rusanov_error =
      cfe_verification::shu_osher_wake_l1_error(run<Scalar, cfe::RusanovFlux>(nx), rusanov_reference, cutoff);
  const double hllc_error =
      cfe_verification::shu_osher_wake_l1_error(run<Scalar, cfe::HllcFlux>(nx), hllc_reference, cutoff);
  std::printf("[info] Shu-Osher wake L1 error at nx=%zu against own fine reference (Scalar=%s): Rusanov=%.6f, HLLC=%.6f\n",
              nx, scalar_name<Scalar>(), rusanov_error, hllc_error);
  CFE_CHECK(hllc_error <= rusanov_error);

  const double front = canonical_shock_front<Scalar>();
  const int extrema_rusanov =
      cfe_verification::shu_osher_count_extrema(run<Scalar, cfe::RusanovFlux>(400), front - 3.0, front - 0.5);
  const int extrema_hllc =
      cfe_verification::shu_osher_count_extrema(run<Scalar, cfe::HllcFlux>(400), front - 3.0, front - 0.5);
  std::printf("[info] Shu-Osher extrema count at nx=400 (Scalar=%s): Rusanov=%d, HLLC=%d\n", scalar_name<Scalar>(),
              extrema_rusanov, extrema_hllc);
  CFE_CHECK(extrema_hllc >= extrema_rusanov);
}

CFE_TEST(test_shu_osher_hllc_resolves_more_fine_structure_than_rusanov_at_matched_resolution_double)
{
  check_shu_osher_hllc_resolves_more_fine_structure_than_rusanov_at_matched_resolution<double>();
}
CFE_TEST(test_shu_osher_hllc_resolves_more_fine_structure_than_rusanov_at_matched_resolution_float)
{
  check_shu_osher_hllc_resolves_more_fine_structure_than_rusanov_at_matched_resolution<float>();
}
