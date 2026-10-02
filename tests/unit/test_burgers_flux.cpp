// Unit tests for BurgersField's Calculators and RusanovFlux (task spec
// 0003 tests: "RusanovFlux matches hand-computed reference values for a
// shock case, a rarefaction case, and a degenerate equal-state case").
#include "cfe/core/types.hpp"
#include "cfe/fields/burgers/field.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
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
