// Unit tests for BurgersField's Calculators and RusanovFlux (task spec
// 0003 tests: "RusanovFlux matches hand-computed reference values for a
// shock case, a rarefaction case, and a degenerate equal-state case").
#include "cfe/core/types.hpp"
#include "cfe/fields/burgers/field.hpp"
#include "cfe/fields/scalar_advection/field.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/numerics/numerical_flux/upwind.hpp"
#include "test_framework.hpp"

CFE_TEST(test_burgers_physical_flux_matches_hand_computed_value)
{
  // F(Q) = Q^2 / 2 -> F(3) = 4.5
  const cfe::BurgersField<double, 1> field{};
  CFE_CHECK_NEAR(field.physical_flux(3.0, cfe::Axis::X), 4.5, 1e-12);
  // Sign-independent: F(-3) = 4.5 too.
  CFE_CHECK_NEAR(field.physical_flux(-3.0, cfe::Axis::X), 4.5, 1e-12);
}

CFE_TEST(test_burgers_wave_speed_is_max_absolute_value_of_either_side)
{
  const cfe::BurgersField<double, 1> field{};
  CFE_CHECK_NEAR(field.wave_speed(2.0, -5.0, cfe::Axis::X), 5.0, 1e-12);
  CFE_CHECK_NEAR(field.wave_speed(-5.0, 2.0, cfe::Axis::X), 5.0, 1e-12);
  CFE_CHECK_NEAR(field.wave_speed(3.0, 3.0, cfe::Axis::X), 3.0, 1e-12);
}

CFE_TEST(test_rusanov_flux_matches_hand_computed_shock_case)
{
  // uL=2, uR=1 (uL > uR: a shock, since characteristics converge).
  // F(uL)=2, F(uR)=0.5, alpha=max(|2|,|1|)=2.
  // F_rusanov = 0.5*(2+0.5) - 0.5*2*(1-2) = 1.25 - (-1.0) = 2.25
  const cfe::BurgersField<double, 1> field{};
  CFE_CHECK_NEAR(cfe::rusanov_flux(2.0, 1.0, cfe::Axis::X, field), 2.25, 1e-12);
}

CFE_TEST(test_rusanov_flux_matches_hand_computed_rarefaction_case)
{
  // uL=-1, uR=2 (uL < uR: a rarefaction, including the sonic point
  // u=0, exactly the case UpwindFlux's own header comment documents as
  // not entropy-correct for Burgers -- Rusanov must still produce a
  // sane, hand-checkable value here).
  // F(uL)=0.5, F(uR)=2, alpha=max(|-1|,|2|)=2.
  // F_rusanov = 0.5*(0.5+2) - 0.5*2*(2-(-1)) = 1.25 - 3.0 = -1.75
  const cfe::BurgersField<double, 1> field{};
  CFE_CHECK_NEAR(cfe::rusanov_flux(-1.0, 2.0, cfe::Axis::X, field), -1.75, 1e-12);
}

CFE_TEST(test_rusanov_flux_reduces_to_physical_flux_for_equal_states)
{
  // uL=uR=u: no jump at all, so the dissipation term vanishes and the
  // flux must reduce to exactly F(u) -- the degenerate case every
  // consistent numerical flux must satisfy by construction.
  // F(2.5) = 3.125
  const cfe::BurgersField<double, 1> field{};
  CFE_CHECK_NEAR(cfe::rusanov_flux(2.5, 2.5, cfe::Axis::X, field), 3.125, 1e-12);
}

CFE_TEST(test_rusanov_flux_functor_matches_free_function)
{
  const cfe::BurgersField<double, 1> field{};
  const cfe::RusanovFlux flux{};
  CFE_CHECK_NEAR(flux(2.0, 1.0, cfe::Axis::X, field), cfe::rusanov_flux(2.0, 1.0, cfe::Axis::X, field),
                 1e-12);
}

// Regression test (review finding on PR #3, commit 1d3654f):
// ScalarAdvectionField::wave_speed returns a *signed* velocity (for
// UpwindFlux's sign-based direction selection), not a magnitude.
// rusanov_flux must take |wave_speed| before using it as a dissipation
// coefficient -- using the raw signed value flips the dissipation
// term's sign for a negative velocity, silently selecting the wrong
// upwind state. Reproduces the exact case the review reported: v=-1,
// (uL,uR)=(1,2) must give -2 (matching upwind, see below), not -1.
CFE_TEST(test_rusanov_flux_matches_upwind_flux_for_linear_advection_with_negative_velocity)
{
  const cfe::ScalarAdvectionField<double, 1> field{cfe::Vector<double, 1>(-1.0)};
  CFE_CHECK_NEAR(cfe::rusanov_flux(1.0, 2.0, cfe::Axis::X, field), -2.0, 1e-12);
  CFE_CHECK_NEAR(cfe::rusanov_flux(1.0, 2.0, cfe::Axis::X, field), cfe::upwind_flux(1.0, 2.0, cfe::Axis::X, field),
                 1e-12);
}

// General property, not just this one case: for ANY constant-velocity
// linear flux (positive or negative), Rusanov's dissipation term is
// exactly strong enough to collapse the scheme to pure upwinding --
// algebraically, F_rusanov = 0.5*v*(uL+uR) -/+ 0.5*v*(uR-uL) reduces to
// v*uL or v*uR depending on v's sign. Checked at both signs so a
// regression specific to either branch would be caught.
CFE_TEST(test_rusanov_flux_exactly_matches_upwind_flux_for_linear_advection_at_both_velocity_signs)
{
  const cfe::ScalarAdvectionField<double, 1> field_positive{cfe::Vector<double, 1>(3.0)};
  CFE_CHECK_NEAR(cfe::rusanov_flux(2.75, 1.25, cfe::Axis::X, field_positive),
                 cfe::upwind_flux(2.75, 1.25, cfe::Axis::X, field_positive), 1e-12);

  const cfe::ScalarAdvectionField<double, 1> field_negative{cfe::Vector<double, 1>(-3.0)};
  CFE_CHECK_NEAR(cfe::rusanov_flux(2.75, 1.25, cfe::Axis::X, field_negative),
                 cfe::upwind_flux(2.75, 1.25, cfe::Axis::X, field_negative), 1e-12);
}
