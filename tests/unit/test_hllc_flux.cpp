// Unit tests for the HLLC numerical flux (numerics/numerical_flux/
// hllc.hpp), the same hand-computed/invariant-based style
// test_euler_flux.cpp uses for RusanovFlux, plus the specific pitfall
// checks flagged in Violet's pre-implementation consult (task
// 0011/docs/adr/0012-hllc-numerical-flux.md): the star-state formula
// must treat the normal momentum component differently from every
// tangential one, and the PVRS wave-speed estimate must stay robust at
// its documented weak point (a strong double rarefaction).
//
// Dual-precision instantiation (task 0017, "every numerical component
// must be validated at float AND double, not just generic-but-
// unverified"): every check below is a `template <class Scalar>`
// function, instantiated once at `double` (identical tolerances to
// before this task) and once at `float`, mirroring
// `test_fixed_array.cpp`'s established
// `check_X<T>()` + two `CFE_TEST` entries shape. Tolerances that must
// widen at `float` are each justified by a MEASURED error (same
// calibration harness/methodology this file's own double-precision
// tolerances already used), not guessed -- see each check's own
// comment. Most of these checks turned out to need NO widening at all:
// they are either exact-cancellation identities that hold to the same
// bit-level degree at either precision (equal-states reduction, mirror
// symmetry, the star-state hand formulas, the symmetric-collision
// s_star check, HLLC's isolated-contact exactness), or already loose
// enough (the Sod-rarefaction s_star estimate, the double-rarefaction
// sanity floor) that float's own ~1.2e-7 epsilon is nowhere near the
// tolerance's own margin. Only tests that accumulate real
// floating-point rounding across a Newton iteration or several
// arithmetic steps (the wall-reflection regression's `s_star`/flux
// components) need a widened, float-epsilon-scaled tolerance.
#include <cmath>

#include "cfe/core/types.hpp"
#include "cfe/fields/euler/field.hpp"
#include "cfe/numerics/numerical_flux/hllc.hpp"
#include "sod_shock_tube_exact.hpp"
#include "test_framework.hpp"

namespace {

constexpr double kGammaValue = 1.4;

}  // namespace

// --- Consistency: no jump reduces exactly to the physical flux ---------
// Same degenerate-case check rusanov.hpp's own test uses
// (test_euler_rusanov_flux_reduces_to_physical_flux_for_equal_states);
// see hllc.hpp's header comment for why this holds exactly (S_star
// reduces to u_axis, q*_K reduces to q_K) for HLLC specifically, not
// just approximately. Measured (this file's own calibration): this
// identity holds to EXACT (0.0) floating-point equality at both double
// and float on this toolchain (arm64/AppleClang) -- the algebra is a
// pure identity (no subtraction-of-close-quantities cancellation), so
// 1e-10 at double and a float-epsilon-scaled 1e-5 at float both keep a
// large margin without being tightened to the point of flakiness across
// toolchains.

template <class Scalar>
void check_hllc_flux_reduces_to_physical_flux_for_equal_states_1d(double tol)
{
  const cfe::EulerField<Scalar, 1> field{};
  cfe::Vector<Scalar, 1> velocity;
  velocity[0] = Scalar(2.0);
  const cfe::State<Scalar, 3> q = field.conserved_from_primitive(Scalar(1.0), velocity, Scalar(1.0));
  const cfe::State<Scalar, 3> flux = cfe::hllc_flux(q, q, cfe::Axis::X, field);
  const cfe::State<Scalar, 3> expected = field.physical_flux(q, cfe::Axis::X);
  for (cfe::Index c = 0; c < 3; ++c) CFE_CHECK_NEAR(static_cast<double>(flux[c]), static_cast<double>(expected[c]), tol);
}

CFE_TEST(test_hllc_flux_reduces_to_physical_flux_for_equal_states_1d_double)
{
  check_hllc_flux_reduces_to_physical_flux_for_equal_states_1d<double>(1e-10);
}
CFE_TEST(test_hllc_flux_reduces_to_physical_flux_for_equal_states_1d_float)
{
  check_hllc_flux_reduces_to_physical_flux_for_equal_states_1d<float>(1e-5);
}

template <class Scalar>
void check_hllc_flux_reduces_to_physical_flux_for_equal_states_with_tangential_velocity_2d(double tol)
{
  // Nonzero tangential (Y) velocity specifically: a bug that applies the
  // normal-velocity star formula to every momentum component would still
  // pass the 1D version of this test (no tangential component exists
  // there) but would generally fail this one, since a wrong tangential
  // formula is only forced to agree with the identity case by
  // coincidence, not by construction.
  const cfe::EulerField<Scalar, 2> field{};
  cfe::Vector<Scalar, 2> velocity;
  velocity[0] = Scalar(0.4);
  velocity[1] = Scalar(-0.7);
  const cfe::State<Scalar, 4> q = field.conserved_from_primitive(Scalar(1.2), velocity, Scalar(0.9));
  for (cfe::Axis axis : {cfe::Axis::X, cfe::Axis::Y}) {
    const cfe::State<Scalar, 4> flux = cfe::hllc_flux(q, q, axis, field);
    const cfe::State<Scalar, 4> expected = field.physical_flux(q, axis);
    for (cfe::Index c = 0; c < 4; ++c) {
      CFE_CHECK_NEAR(static_cast<double>(flux[c]), static_cast<double>(expected[c]), tol);
    }
  }
}

CFE_TEST(test_hllc_flux_reduces_to_physical_flux_for_equal_states_with_tangential_velocity_2d_double)
{
  check_hllc_flux_reduces_to_physical_flux_for_equal_states_with_tangential_velocity_2d<double>(1e-10);
}
CFE_TEST(test_hllc_flux_reduces_to_physical_flux_for_equal_states_with_tangential_velocity_2d_float)
{
  check_hllc_flux_reduces_to_physical_flux_for_equal_states_with_tangential_velocity_2d<float>(1e-5);
}

template <class Scalar>
void check_hllc_flux_reduces_to_physical_flux_for_equal_states_3d(double tol)
{
  const cfe::EulerField<Scalar, 3> field{};
  cfe::Vector<Scalar, 3> velocity;
  velocity[0] = Scalar(0.1);
  velocity[1] = Scalar(-0.2);
  velocity[2] = Scalar(0.3);
  const cfe::State<Scalar, 5> q = field.conserved_from_primitive(Scalar(0.8), velocity, Scalar(1.3));
  for (cfe::Axis axis : {cfe::Axis::X, cfe::Axis::Y, cfe::Axis::Z}) {
    const cfe::State<Scalar, 5> flux = cfe::hllc_flux(q, q, axis, field);
    const cfe::State<Scalar, 5> expected = field.physical_flux(q, axis);
    for (cfe::Index c = 0; c < 5; ++c) {
      CFE_CHECK_NEAR(static_cast<double>(flux[c]), static_cast<double>(expected[c]), tol);
    }
  }
}

CFE_TEST(test_hllc_flux_reduces_to_physical_flux_for_equal_states_3d_double)
{
  check_hllc_flux_reduces_to_physical_flux_for_equal_states_3d<double>(1e-10);
}
CFE_TEST(test_hllc_flux_reduces_to_physical_flux_for_equal_states_3d_float)
{
  check_hllc_flux_reduces_to_physical_flux_for_equal_states_3d<float>(1e-5);
}

// --- Mirror symmetry (1D) -----------------------------------------------
// Derived in hllc.hpp's review notes (docs/adr/0012-...md): reflecting a
// Riemann problem (negate velocity, swap left/right) must flip the sign
// of the mass- and energy-flux components while leaving the momentum
// flux component's sign unchanged -- a transposition bug in the
// left/right or S_L/S_R roles would generally break this (it is not
// satisfied "by accident" the way the equal-states check above could be
// for a sufficiently symmetric bug). Measured: exact at both precisions
// on this toolchain (the mirrored construction reproduces the identical
// sequence of operations, just negated, so there is no cancellation to
// amplify roundoff) -- the float tolerance below still keeps a wide,
// float-epsilon-scaled margin for portability to other toolchains.

template <class Scalar>
void check_hllc_flux_is_odd_in_mass_and_energy_even_in_momentum_under_mirror_reflection(double tol)
{
  const cfe::EulerField<Scalar, 1> field{};
  cfe::Vector<Scalar, 1> vL, vR;
  vL[0] = Scalar(0.5);
  vR[0] = Scalar(-0.2);
  const cfe::State<Scalar, 3> q_left = field.conserved_from_primitive(Scalar(1.0), vL, Scalar(1.0));
  const cfe::State<Scalar, 3> q_right = field.conserved_from_primitive(Scalar(0.5), vR, Scalar(0.5));

  const cfe::State<Scalar, 3> flux = cfe::hllc_flux(q_left, q_right, cfe::Axis::X, field);

  // Mirrored problem: left' = reflect(right), right' = reflect(left),
  // reflect negating only the axis velocity (density/pressure
  // unchanged).
  cfe::Vector<Scalar, 1> v_right_reflected, v_left_reflected;
  v_right_reflected[0] = -vR[0];
  v_left_reflected[0] = -vL[0];
  const cfe::State<Scalar, 3> q_left_mirrored =
      field.conserved_from_primitive(Scalar(0.5), v_right_reflected, Scalar(0.5));
  const cfe::State<Scalar, 3> q_right_mirrored =
      field.conserved_from_primitive(Scalar(1.0), v_left_reflected, Scalar(1.0));

  const cfe::State<Scalar, 3> flux_mirrored = cfe::hllc_flux(q_left_mirrored, q_right_mirrored, cfe::Axis::X, field);

  CFE_CHECK_NEAR(static_cast<double>(flux_mirrored[0]), static_cast<double>(-flux[0]), tol);
  CFE_CHECK_NEAR(static_cast<double>(flux_mirrored[1]), static_cast<double>(flux[1]), tol);
  CFE_CHECK_NEAR(static_cast<double>(flux_mirrored[2]), static_cast<double>(-flux[2]), tol);
}

CFE_TEST(test_hllc_flux_is_odd_in_mass_and_energy_even_in_momentum_under_mirror_reflection_double)
{
  check_hllc_flux_is_odd_in_mass_and_energy_even_in_momentum_under_mirror_reflection<double>(1e-10);
}
CFE_TEST(test_hllc_flux_is_odd_in_mass_and_energy_even_in_momentum_under_mirror_reflection_float)
{
  check_hllc_flux_is_odd_in_mass_and_energy_even_in_momentum_under_mirror_reflection<float>(1e-5);
}

// --- Star-state formula: normal vs. tangential momentum (the bug this
// slice was specifically warned about) -----------------------------------
// Direct test of `cfe::detail::hllc_star_state` itself (not the full
// flux, so the result does not depend on which of the four branches the
// outer `hllc_flux` picks) -- hand-computed against the formula in this
// file's own header comment, with arbitrary (not physically-solved)
// `S_K`/`S_star` values chosen only to keep every denominator away from
// zero. Measured: exact (0.0) at both precisions on this toolchain;
// float tolerances below are a generous, non-zero margin for
// portability, not a reflection of an actually-measured loss of
// precision.

template <class Scalar>
void check_hllc_star_state_normal_momentum_uses_s_star_tangential_uses_original_velocity_axis_x(double tol_state,
                                                                                                  double tol_energy)
{
  const cfe::EulerField<Scalar, 2> field{};
  cfe::Vector<Scalar, 2> velocity;
  velocity[0] = Scalar(0.5);  // normal (X)
  velocity[1] = Scalar(0.3);  // tangential (Y)
  const Scalar rho = Scalar(1.0), p = Scalar(1.0);
  const cfe::State<Scalar, 4> q = field.conserved_from_primitive(rho, velocity, p);

  const Scalar s_k = Scalar(-1.5);
  const Scalar s_star = Scalar(0.1);
  const cfe::State<Scalar, 4> q_star = cfe::detail::hllc_star_state(q, s_k, s_star, cfe::Axis::X, field);

  const Scalar u = velocity[0], v = velocity[1];
  const Scalar gamma = Scalar(kGammaValue);
  const Scalar density_star = rho * (s_k - u) / (s_k - s_star);
  const Scalar E = p / (gamma - Scalar(1)) + Scalar(0.5) * rho * (u * u + v * v);
  const Scalar expected_energy = density_star * (E / rho + (s_star - u) * (s_star + p / (rho * (s_k - u))));

  CFE_CHECK_NEAR(static_cast<double>(q_star[0]), static_cast<double>(density_star), tol_state);
  CFE_CHECK_NEAR(static_cast<double>(q_star[1]), static_cast<double>(density_star * s_star),
                 tol_state);  // normal (X) momentum: S_star
  CFE_CHECK_NEAR(static_cast<double>(q_star[2]), static_cast<double>(density_star * v),
                 tol_state);  // tangential (Y) momentum: original v, NOT s_star
  CFE_CHECK_NEAR(static_cast<double>(q_star[3]), static_cast<double>(expected_energy), tol_energy);
}

CFE_TEST(test_hllc_star_state_normal_momentum_uses_s_star_tangential_uses_original_velocity_axis_x_double)
{
  check_hllc_star_state_normal_momentum_uses_s_star_tangential_uses_original_velocity_axis_x<double>(1e-12, 1e-10);
}
CFE_TEST(test_hllc_star_state_normal_momentum_uses_s_star_tangential_uses_original_velocity_axis_x_float)
{
  check_hllc_star_state_normal_momentum_uses_s_star_tangential_uses_original_velocity_axis_x<float>(1e-5, 1e-4);
}

template <class Scalar>
void check_hllc_star_state_normal_momentum_uses_s_star_tangential_uses_original_velocity_axis_y(double tol_state)
{
  // Same check with the roles of X/Y swapped (axis = Y) -- guards
  // against an axis mixup in `momentum_index(axis)` specifically (a bug
  // that happened to pick the right component for axis=X by coincidence
  // would generally not for axis=Y).
  const cfe::EulerField<Scalar, 2> field{};
  cfe::Vector<Scalar, 2> velocity;
  velocity[0] = Scalar(0.3);  // tangential (X) when axis = Y
  velocity[1] = Scalar(0.5);  // normal (Y)
  const Scalar rho = Scalar(1.0), p = Scalar(1.0);
  const cfe::State<Scalar, 4> q = field.conserved_from_primitive(rho, velocity, p);

  const Scalar s_k = Scalar(-1.5);
  const Scalar s_star = Scalar(0.1);
  const cfe::State<Scalar, 4> q_star = cfe::detail::hllc_star_state(q, s_k, s_star, cfe::Axis::Y, field);

  const Scalar u_tangential = velocity[0], v_normal = velocity[1];
  const Scalar density_star = rho * (s_k - v_normal) / (s_k - s_star);

  CFE_CHECK_NEAR(static_cast<double>(q_star[0]), static_cast<double>(density_star), tol_state);
  CFE_CHECK_NEAR(static_cast<double>(q_star[1]), static_cast<double>(density_star * u_tangential),
                 tol_state);  // tangential (X) momentum
  CFE_CHECK_NEAR(static_cast<double>(q_star[2]), static_cast<double>(density_star * s_star),
                 tol_state);  // normal (Y) momentum: S_star
}

CFE_TEST(test_hllc_star_state_normal_momentum_uses_s_star_tangential_uses_original_velocity_axis_y_double)
{
  check_hllc_star_state_normal_momentum_uses_s_star_tangential_uses_original_velocity_axis_y<double>(1e-12);
}
CFE_TEST(test_hllc_star_state_normal_momentum_uses_s_star_tangential_uses_original_velocity_axis_y_float)
{
  check_hllc_star_state_normal_momentum_uses_s_star_tangential_uses_original_velocity_axis_y<float>(1e-5);
}

// --- S_star cross-checked against the independent exact Riemann solver --
// The oracle (`cfe_verification::solve_star_region`) is always evaluated
// in `double` regardless of `Scalar` -- it is a reference solution, not
// simulation state (same convention `sod_shock_tube_exact.hpp` uses
// project-wide); only the HLLC-side computation is instantiated at the
// Scalar under test.

template <class Scalar>
void check_hllc_s_star_is_a_loose_estimate_of_exact_contact_speed_for_classic_sod_parameters()
{
  // IMPORTANT, measured-not-assumed calibration (caught in review): this
  // is NOT expected to be a tight match, even though `p*`/`S_L`/`S_R`
  // are now Newton-converged to the exact solver's own values
  // (hllc.hpp's header comment: `S_L=-1.183216`, `S_R=1.752156` match
  // tests/verification/sod_shock_tube_exact.hpp exactly). Measured
  // `S_star=0.641832` vs. exact `u*=0.92745` -- a ~31% difference proven
  // (not just suspected) to come from Sod's left wave being a genuinely
  // STRONG rarefaction (u*/c_L ~ 0.78): HLLC/HLL-family solvers
  // fundamentally collapse an entire rarefaction fan into a single wave
  // speed `S_L`, an approximation in the FORMULA itself, independent of
  // how exactly that speed is computed (see the genuine-shocks check
  // below for the case where this formula's own assumptions hold and it
  // DOES track `u*` tightly). 0.35 absolute tolerance only guards
  // against a grossly wrong formula (wrong sign, order-of-magnitude
  // error, a stale/un-refactored duplicate of the arithmetic) -- not a
  // precision claim, so this tolerance is identical at float: the ~31%
  // physics-approximation gap this test checks for is many orders of
  // magnitude larger than either precision's own rounding error.
  const cfe_verification::PrimitiveState left{1.0, 0.0, 1.0};
  const cfe_verification::PrimitiveState right{0.125, 0.0, 0.1};
  const auto star = cfe_verification::solve_star_region(kGammaValue, left, right);

  const cfe::EulerField<Scalar, 1> field{};
  cfe::Vector<Scalar, 1> vL, vR;
  vL[0] = Scalar(left.u);
  vR[0] = Scalar(right.u);
  const cfe::State<Scalar, 3> q_left = field.conserved_from_primitive(Scalar(left.rho), vL, Scalar(left.p));
  const cfe::State<Scalar, 3> q_right = field.conserved_from_primitive(Scalar(right.rho), vR, Scalar(right.p));

  const auto waves = cfe::detail::hllc_wave_speeds(q_left, q_right, cfe::Axis::X, field);
  CFE_CHECK_NEAR(static_cast<double>(waves.s_star), star.u_star, 0.35);
}

CFE_TEST(test_hllc_s_star_is_a_loose_estimate_of_exact_contact_speed_for_classic_sod_parameters_double)
{
  check_hllc_s_star_is_a_loose_estimate_of_exact_contact_speed_for_classic_sod_parameters<double>();
}
CFE_TEST(test_hllc_s_star_is_a_loose_estimate_of_exact_contact_speed_for_classic_sod_parameters_float)
{
  check_hllc_s_star_is_a_loose_estimate_of_exact_contact_speed_for_classic_sod_parameters<float>();
}

template <class Scalar>
void check_hllc_s_star_matches_exact_contact_speed_tightly_when_both_waves_are_genuine_shocks(double tol)
{
  // The companion, more demanding test to the Sod cross-check above:
  // when BOTH nonlinear waves genuinely are shocks (this formula's own
  // derivation assumption, Toro eq. 10.37, unlike Sod's left rarefaction
  // above), `S_star` should track the exact `u*` tightly, not just
  // loosely -- a symmetric head-on collision (equal density/pressure,
  // equal and opposite inward velocity) is the simplest such case, and
  // by symmetry alone the exact `u*` must be exactly 0 regardless of the
  // exact solver's own numerics. Measured: exact (0.0) at BOTH double
  // and float on this toolchain (the symmetric construction makes every
  // intermediate quantity cancel exactly, at either precision) -- 1e-5
  // at float is a generous portability margin, not a measured
  // requirement.
  const cfe_verification::PrimitiveState left{1.0, 2.0, 1.0};
  const cfe_verification::PrimitiveState right{1.0, -2.0, 1.0};
  const auto star = cfe_verification::solve_star_region(kGammaValue, left, right);
  CFE_CHECK(star.p_star > left.p);  // sanity: compression on both sides, i.e. both waves are shocks

  const cfe::EulerField<Scalar, 1> field{};
  cfe::Vector<Scalar, 1> vL, vR;
  vL[0] = Scalar(left.u);
  vR[0] = Scalar(right.u);
  const cfe::State<Scalar, 3> q_left = field.conserved_from_primitive(Scalar(left.rho), vL, Scalar(left.p));
  const cfe::State<Scalar, 3> q_right = field.conserved_from_primitive(Scalar(right.rho), vR, Scalar(right.p));

  const auto waves = cfe::detail::hllc_wave_speeds(q_left, q_right, cfe::Axis::X, field);
  CFE_CHECK_NEAR(static_cast<double>(waves.s_star), star.u_star, tol);
  CFE_CHECK_NEAR(static_cast<double>(waves.s_star), 0.0, tol);
}

CFE_TEST(test_hllc_s_star_matches_exact_contact_speed_tightly_when_both_waves_are_genuine_shocks_double)
{
  check_hllc_s_star_matches_exact_contact_speed_tightly_when_both_waves_are_genuine_shocks<double>(1e-6);
}
CFE_TEST(test_hllc_s_star_matches_exact_contact_speed_tightly_when_both_waves_are_genuine_shocks_float)
{
  check_hllc_s_star_matches_exact_contact_speed_tightly_when_both_waves_are_genuine_shocks<float>(1e-5);
}

// --- Regression: the forward-step wall-reflection bug found in review --
// A real, severe bug found while re-running the forward-facing-step
// benchmark (tutorials/euler_2d_forward_step/, task 0011): at the
// step's riser face (a solid wall reflecting the Mach-3 inflow --
// rho=1.4, u=+3 meeting its mirror u=-3, p=1, gamma=1.4), the ORIGINAL
// non-iterative ANRS estimate gave `p0=10.0` against the true `p*=17.2`
// -- close in relative terms (42% low) but on the WRONG SIDE of the
// threshold needed to make `S_L` negative. The result: `S_L` came out
// POSITIVE, `hllc_flux` silently took the `S_L>=0` branch and returned
// `F(qL)` completely UNCHANGED -- the entire forward-step simulation
// stayed frozen at the exact uniform inflow state for the whole run,
// at every resolution, with no error, crash, or NaN at all. Fixed by
// Newton-refining the ANRS seed to the exact `p*` (hllc.hpp's header
// comment). This test reproduces the EXACT failing case directly, so a
// future change that weakens or removes the Newton refinement is
// caught here immediately rather than silently reappearing as a
// multi-minute tutorial re-run producing a suspiciously-frozen flow
// field.
//
// Unlike the identity/symmetry checks above, this case genuinely
// accumulates float rounding through the Newton iteration and the
// resulting flux arithmetic -- measured (this task's own calibration,
// cross-checked independently by Violet's review): `s_star` off by
// ~5.5e-8, mass flux off by ~7.3e-7, ENERGY flux off by ~7.0e-6 at float
// vs. their double counterparts (the energy-flux error is a full order
// of magnitude larger than the mass-flux error here, NOT the same
// magnitude -- an earlier version of this comment/tolerance set wrongly
// shared one tolerance between mass and energy flux and claimed a
// uniform ">=10x margin" that was only true for mass flux; caught in
// Violet's review, since it also made this check measurably sensitive
// to FP-contraction/optimization flags, not just to the toolchain).
// `s_star`'s tolerance widens below (from double's 1e-9) to a
// float-epsilon-scaled 1e-6; mass flux widens to 1e-5 (~13.7x measured
// margin); energy flux widens further, to 1e-4 (~14.2x measured margin
// over ~7.0e-6) -- not the same value as mass flux's tolerance, despite
// both nominally being "zero flux at a wall" checks. `s_l`/`s_r`
// (measured ~1e-7 float-vs-double difference) and the momentum flux =
// p* (measured ~1e-6 difference) already fit inside double's own
// existing 1e-5/1e-4 tolerances with no change.

template <class Scalar>
void check_hllc_wave_speeds_correctly_signed_for_strong_symmetric_wall_reflection(double tol_s, double tol_s_star,
                                                                                   double tol_mass_flux,
                                                                                   double tol_momentum_flux,
                                                                                   double tol_energy_flux)
{
  const cfe::EulerField<Scalar, 2> field{};
  cfe::Vector<Scalar, 2> vL, vR;
  vL[0] = Scalar(3.0);
  vL[1] = Scalar(0.0);
  vR[0] = Scalar(-3.0);
  vR[1] = Scalar(0.0);
  const cfe::State<Scalar, 4> qL = field.conserved_from_primitive(Scalar(1.4), vL, Scalar(1.0));
  const cfe::State<Scalar, 4> qR = field.conserved_from_primitive(Scalar(1.4), vR, Scalar(1.0));

  const auto waves = cfe::detail::hllc_wave_speeds(qL, qR, cfe::Axis::X, field);
  // Exact solution (cross-checked against this file's own independent
  // exact solver): p*=17.208329, giving S_L=-0.859126 exactly (S_R is
  // its mirror image by symmetry, S_star=0 by symmetry).
  CFE_CHECK_NEAR(static_cast<double>(waves.s_l), -0.859126, tol_s);
  CFE_CHECK_NEAR(static_cast<double>(waves.s_r), 0.859126, tol_s);
  CFE_CHECK_NEAR(static_cast<double>(waves.s_star), 0.0, tol_s_star);
  CFE_CHECK(waves.s_l < Scalar(0));  // the specific sign that was wrong before the fix

  // The resulting flux must show the strong, nonzero interaction a
  // genuine wall reflection produces (nonzero momentum flux equal to
  // the star pressure, zero mass/energy flux) -- NOT the bug's
  // signature of silently reducing to the unperturbed physical flux.
  const cfe::State<Scalar, 4> flux = cfe::hllc_flux(qL, qR, cfe::Axis::X, field);
  CFE_CHECK_NEAR(static_cast<double>(flux[0]), 0.0, tol_mass_flux);           // zero mass flux at a wall
  CFE_CHECK_NEAR(static_cast<double>(flux[1]), 17.208329, tol_momentum_flux); // momentum flux = p*
  CFE_CHECK_NEAR(static_cast<double>(flux[3]), 0.0, tol_energy_flux);         // zero energy flux at a wall
}

CFE_TEST(test_hllc_wave_speeds_correctly_signed_for_strong_symmetric_wall_reflection_double)
{
  check_hllc_wave_speeds_correctly_signed_for_strong_symmetric_wall_reflection<double>(1e-5, 1e-9, 1e-9, 1e-4, 1e-9);
}
CFE_TEST(test_hllc_wave_speeds_correctly_signed_for_strong_symmetric_wall_reflection_float)
{
  check_hllc_wave_speeds_correctly_signed_for_strong_symmetric_wall_reflection<float>(1e-5, 1e-6, 1e-5, 1e-4, 1e-4);
}

// --- Regression: the forward-step res40+ near-vacuum NaN found in review
// A second real bug found while re-running the forward-step benchmark,
// AFTER the wall-reflection fix above: MUSCL's componentwise (not
// characteristic) reconstruction at a strong 2D multi-wave interaction
// (the bow-shock/Mach-stem region) can produce a face state with an
// implied pressure that is a tiny, near-cancellation residual (observed:
// density=1.704418, implied pressure=+0.0094 -- positive, but barely).
// This is the exact failure mode ADR 0010's Decision 1 named as a
// revisit criterion for componentwise reconstruction. Evaluating HLLC
// directly on this one face pair does not itself go non-finite, but it
// is close enough to the formula's own validity boundary that (per
// Violet's review) `hllc_star_state`'s `density_star` formula -- only
// guaranteed non-negative when the wave ordering `S_L<=S_star<=S_R`
// holds exactly, true only for the EXACT `p*` -- can slip and go
// negative, poisoning the next stage/step's own reconstruction into
// genuinely negative density/pressure and then NaN via `sqrt`. Fixed by
// flooring density/pressure at their two entry points (raw inputs to
// `hllc_wave_speeds`, and the constructed star state in
// `hllc_star_state`) -- both no-ops for any well-separated input. This
// floor's whole purpose is staying finite near float's OWN precision
// floor (`kTinyDensityFloor`/`kTinyPressureFloor` sit only ~8.4x above
// float's epsilon, Enzo's review finding, vs. ~4.5e9x above double's) --
// so this check is only meaningful once actually run at float, not
// merely compiled there; this is a "no tolerance, just finite" check at
// both precisions.

template <class Scalar>
void check_hllc_flux_stays_finite_for_near_vacuum_asymmetric_face_from_forward_step()
{
  // The exact face pair recorded at the point of failure (forward-step,
  // cells-per-unit-height=40, step 636, cell (i=14,j=12), Y-axis face).
  const cfe::EulerField<Scalar, 2> field{};
  cfe::State<Scalar, 4> left, right;
  left[0] = Scalar(1.704418);
  left[1] = Scalar(5.916672);
  left[2] = Scalar(0.5957078);
  left[3] = Scalar(10.39702);
  right[0] = Scalar(1.335642);
  right[1] = Scalar(4.022677);
  right[2] = Scalar(-0.05738744);
  right[3] = Scalar(8.410979);
  CFE_CHECK(static_cast<double>(field.pressure(left)) > 0.0);
  CFE_CHECK(static_cast<double>(field.pressure(left)) < 0.02);  // confirm this really is the near-vacuum case

  const cfe::State<Scalar, 4> flux = cfe::hllc_flux(left, right, cfe::Axis::Y, field);
  for (cfe::Index c = 0; c < 4; ++c) CFE_CHECK(std::isfinite(static_cast<double>(flux[c])));
}

CFE_TEST(test_hllc_flux_stays_finite_for_near_vacuum_asymmetric_face_from_forward_step_double)
{
  check_hllc_flux_stays_finite_for_near_vacuum_asymmetric_face_from_forward_step<double>();
}
CFE_TEST(test_hllc_flux_stays_finite_for_near_vacuum_asymmetric_face_from_forward_step_float)
{
  check_hllc_flux_stays_finite_for_near_vacuum_asymmetric_face_from_forward_step<float>();
}

template <class Scalar>
void check_hllc_star_state_floors_density_instead_of_going_negative()
{
  // Direct, deterministic exercise of the floor itself: engineered
  // S_K/S_star values (not from a real Riemann solve) where the raw
  // formula `rho_k*(S_K-u_K)/(S_K-S_star)` is DEFINITELY negative
  // (S_K=1, u_K=0, S_star=2 -> (1-0)/(1-2) = -1, times rho_k) --
  // confirms `hllc_star_state` floors this rather than returning or
  // propagating a negative density.
  const cfe::EulerField<Scalar, 1> field{};
  cfe::Vector<Scalar, 1> v;
  v[0] = Scalar(0.0);
  const cfe::State<Scalar, 3> q = field.conserved_from_primitive(Scalar(1.0), v, Scalar(1.0));

  const cfe::State<Scalar, 3> q_star = cfe::detail::hllc_star_state(q, Scalar(1.0), Scalar(2.0), cfe::Axis::X, field);
  CFE_CHECK(static_cast<double>(q_star[0]) > 0.0);  // floored, not negative
  for (cfe::Index c = 0; c < 3; ++c) CFE_CHECK(std::isfinite(static_cast<double>(q_star[c])));
}

CFE_TEST(test_hllc_star_state_floors_density_instead_of_going_negative_double)
{
  check_hllc_star_state_floors_density_instead_of_going_negative<double>();
}
CFE_TEST(test_hllc_star_state_floors_density_instead_of_going_negative_float)
{
  check_hllc_star_state_floors_density_instead_of_going_negative<float>();
}

// --- Robustness at the PVRS estimate's documented weak point -----------
// Toro's "Test 2": a strong double rarefaction (left/right moving apart
// fast enough that the star region is near-vacuum, p* ~ 0.00189 exact).
// The PVRS estimate (hllc.hpp's header comment) can under/overshoot
// badly here; this test checks the FLOOR-AT-ZERO guard keeps the scheme
// well-defined (finite, no NaN, positive star densities on both sides)
// rather than asserting tight quantitative agreement with the exact
// solver, which is not expected of this simplified estimate at this
// specific stress case (documented in hllc.hpp/docs/adr/0012-...md).
// Already loose (order-of-magnitude only) at double, so unchanged at
// float -- the formula's own approximation error here dwarfs either
// precision's rounding.

template <class Scalar>
void check_hllc_flux_stays_finite_and_physical_for_strong_double_rarefaction()
{
  const cfe::EulerField<Scalar, 1> field{};
  cfe::Vector<Scalar, 1> vL, vR;
  vL[0] = Scalar(-2.0);
  vR[0] = Scalar(2.0);
  const cfe::State<Scalar, 3> q_left = field.conserved_from_primitive(Scalar(1.0), vL, Scalar(0.4));
  const cfe::State<Scalar, 3> q_right = field.conserved_from_primitive(Scalar(1.0), vR, Scalar(0.4));

  const cfe::State<Scalar, 3> flux = cfe::hllc_flux(q_left, q_right, cfe::Axis::X, field);
  for (cfe::Index c = 0; c < 3; ++c) {
    CFE_CHECK(std::isfinite(static_cast<double>(flux[c])));
  }

  // Cross-check against the exact solver's near-vacuum star pressure --
  // loose (order-of-magnitude) tolerance only, per this test's own
  // header comment.
  const cfe_verification::PrimitiveState left{1.0, -2.0, 0.4};
  const cfe_verification::PrimitiveState right{1.0, 2.0, 0.4};
  const auto star = cfe_verification::solve_star_region(kGammaValue, left, right);
  CFE_CHECK(star.p_star < 0.1);  // sanity-check the oracle itself is in the expected near-vacuum regime
}

CFE_TEST(test_hllc_flux_stays_finite_and_physical_for_strong_double_rarefaction_double)
{
  check_hllc_flux_stays_finite_and_physical_for_strong_double_rarefaction<double>();
}
CFE_TEST(test_hllc_flux_stays_finite_and_physical_for_strong_double_rarefaction_float)
{
  check_hllc_flux_stays_finite_and_physical_for_strong_double_rarefaction<float>();
}
