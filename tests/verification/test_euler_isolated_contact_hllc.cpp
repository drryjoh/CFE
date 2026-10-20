// The single cleanest, decoupled demonstration of HLLC's claimed
// advantage over Rusanov (task 0011/docs/adr/0012-hllc-numerical-flux.md,
// per Violet's pre-implementation consult): an ISOLATED pure contact
// discontinuity -- no shock, no rarefaction anywhere in the problem --
// left/right primitive states sharing the SAME pressure and the SAME
// (zero) velocity, differing only in density. The exact solution of the
// Euler equations for this initial condition is a perfectly STATIONARY
// density step for all time (a genuine, if degenerate, steady state: the
// one characteristic family that carries a density jump with no
// pressure/velocity signature has zero wave speed here).
//
// Analytically (see docs/adr/0012-...md's evidence section for the full
// derivation), HLLC gives an EXACTLY zero mass flux at any face where
// the two neighboring states share pressure and velocity, regardless of
// how different their densities are: the ANRS `p*` estimate reduces
// exactly to the shared `p`, `S_star` reduces exactly to the shared
// (zero) velocity via Toro eq. 10.37's numerator vanishing, and the
// resulting star state matches the respective original state exactly
// (the same degenerate-equality mechanics as this file's unit-test
// cousin's `hllc_flux(q,q,...)` consistency check, here applied
// one-sidedly). Rusanov has no such property: its dissipation term
// `-0.5*alpha*(uR-uL)` is nonzero on the mass-flux component whenever
// density differs, REGARDLESS of pressure/velocity agreement -- so it
// visibly smears this step over several cells, even though nothing is
// physically moving.
//
// Dual-precision instantiation (task 0017): both checks below are
// `template <class Scalar>` functions, instantiated at `double` and
// `float`. Measured (this task's own calibration): HLLC's analytic
// zero-mass-flux property holds to EXACT (0.0) floating-point equality
// at BOTH precisions on this toolchain (it is a structural algebraic
// identity -- every term in the star-state formula reduces exactly when
// pressure/velocity already agree, with no cancellation of nearly-equal
// quantities to amplify roundoff) -- so the same `< 1e-10` tolerance
// would in fact also pass at float, but a float-epsilon-scaled `< 1e-6`
// is used instead for cross-toolchain portability (the same reasoning
// test_hllc_flux.cpp's identity checks use), not because 1e-10 was
// observed to fail. Rusanov's own smearing magnitude (the negative
// control) is likewise measured identical at both precisions (~0.2237),
// so its `> 0.05` floor is unchanged.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <type_traits>

#include "cfe/core/types.hpp"
#include "cfe/numerics/numerical_flux/hllc.hpp"
#include "sod_shock_tube_runner.hpp"
#include "test_framework.hpp"

namespace {

template <class Scalar>
struct IsolatedContact
{
  static constexpr Scalar kGamma = Scalar(1.4);
  static constexpr Scalar kX0 = Scalar(0.5);
  static constexpr Scalar kTFinal = Scalar(0.5);
  static constexpr Scalar kCfl = Scalar(0.4);
  static constexpr cfe::Index kNx = 100;
};

// Equal pressure, equal (zero) velocity, different density -- the one
// isolated contact wave, nothing else.
const cfe_verification::PrimitiveState kLeft{1.0, 0.0, 1.0};
const cfe_verification::PrimitiveState kRight{0.5, 0.0, 1.0};

template <class Scalar>
double max_deviation_from_initial_step(const cfe_verification::SodProfile<Scalar>& r)
{
  double max_abs_deviation = 0.0;
  for (cfe::Index i = 0; i < r.x.size(); ++i) {
    const double expected =
        (static_cast<double>(r.x[i]) < static_cast<double>(IsolatedContact<Scalar>::kX0)) ? kLeft.rho : kRight.rho;
    max_abs_deviation = std::max(max_abs_deviation, std::fabs(static_cast<double>(r.rho[i]) - expected));
  }
  return max_abs_deviation;
}

}  // namespace

template <class Scalar>
void check_isolated_contact_hllc_keeps_density_step_exactly_in_place(double tol)
{
  using P = IsolatedContact<Scalar>;
  const cfe_verification::SodProfile<Scalar> r =
      cfe_verification::run_sod_shock_tube<Scalar, cfe::HllcFlux>(P::kNx, P::kGamma, kLeft, kRight, P::kX0, P::kTFinal,
                                                                    P::kCfl);

  const double max_abs_deviation = max_deviation_from_initial_step(r);
  std::printf("[info] Isolated contact, HLLC, Scalar=%s, t=%.2f: max density deviation from initial step = %.3e\n",
              std::is_same_v<Scalar, float> ? "float" : "double", static_cast<double>(P::kTFinal), max_abs_deviation);

  // Expected EXACTLY zero (to floating-point roundoff) per this file's
  // header comment's analytic argument -- not an approximate/loose
  // claim.
  CFE_CHECK(max_abs_deviation < tol);
}

CFE_TEST(test_isolated_contact_hllc_keeps_density_step_exactly_in_place_double)
{
  check_isolated_contact_hllc_keeps_density_step_exactly_in_place<double>(1e-10);
}
CFE_TEST(test_isolated_contact_hllc_keeps_density_step_exactly_in_place_float)
{
  check_isolated_contact_hllc_keeps_density_step_exactly_in_place<float>(1e-6);
}

template <class Scalar>
void check_isolated_contact_rusanov_visibly_smears_the_same_density_step()
{
  using P = IsolatedContact<Scalar>;
  const cfe_verification::SodProfile<Scalar> r =
      cfe_verification::run_sod_shock_tube<Scalar>(P::kNx, P::kGamma, kLeft, kRight, P::kX0, P::kTFinal, P::kCfl);

  const double max_abs_deviation = max_deviation_from_initial_step(r);
  std::printf(
      "[info] Isolated contact, Rusanov, Scalar=%s, t=%.2f: max density deviation from initial step = %.3e\n",
      std::is_same_v<Scalar, float> ? "float" : "double", static_cast<double>(P::kTFinal), max_abs_deviation);

  // Rusanov is expected to visibly smear this (a real, nonzero
  // deviation at some interior cell near x0) -- a direct negative
  // control proving the HLLC test above is not vacuous (i.e. that THIS
  // problem/harness is actually capable of showing smearing at all).
  // Measured identical (~0.2237) at float and double, so the floor is
  // unchanged.
  CFE_CHECK(max_abs_deviation > 0.05);
}

CFE_TEST(test_isolated_contact_rusanov_visibly_smears_the_same_density_step_double)
{
  check_isolated_contact_rusanov_visibly_smears_the_same_density_step<double>();
}
CFE_TEST(test_isolated_contact_rusanov_visibly_smears_the_same_density_step_float)
{
  check_isolated_contact_rusanov_visibly_smears_the_same_density_step<float>();
}
