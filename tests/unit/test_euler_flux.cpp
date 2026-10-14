// Unit tests for EulerField's Calculators (fields/euler/field.hpp) and
// the vector (system) RusanovFlux overload (numerics/numerical_flux/
// rusanov.hpp) -- hand-computed reference values, the same style
// test_burgers_flux.cpp uses for BurgersField.
#include "cfe/core/types.hpp"
#include "cfe/fields/euler/field.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "test_framework.hpp"

CFE_TEST(test_euler_conserved_from_primitive_matches_hand_computed_state)
{
  // rho=1, u=2, p=1, gamma=1.4 (1D): momentum=2,
  // KE=0.5*rho*u^2=0.5*1*4=2, E=p/(gamma-1)+KE=1/0.4+2=4.5.
  const cfe::EulerField<double, 1> field{};
  const cfe::State<double, 3> q = field.conserved_from_primitive(1.0, cfe::Vector<double, 1>(2.0), 1.0);
  CFE_CHECK_NEAR(q[0], 1.0, 1e-12);
  CFE_CHECK_NEAR(q[1], 2.0, 1e-12);
  CFE_CHECK_NEAR(q[2], 4.5, 1e-12);
}

CFE_TEST(test_euler_pressure_and_sound_speed_recover_primitive_inputs)
{
  // Round trip: build Q from (rho=1,u=2,p=1), recover p and c from Q.
  const cfe::EulerField<double, 1> field{};
  const cfe::State<double, 3> q = field.conserved_from_primitive(1.0, cfe::Vector<double, 1>(2.0), 1.0);
  CFE_CHECK_NEAR(field.pressure(q), 1.0, 1e-12);
  // c = sqrt(gamma*p/rho) = sqrt(1.4*1/1) = sqrt(1.4)
  CFE_CHECK_NEAR(field.sound_speed(q), 1.1832159566199232, 1e-10);
  CFE_CHECK_NEAR(field.velocity_component(q, cfe::Axis::X), 2.0, 1e-12);
}

CFE_TEST(test_euler_physical_flux_matches_hand_computed_value)
{
  // Q=(1,2,4.5) (rho=1,u=2,p=1 as above). F(Q) = (rho*u, rho*u^2+p, (E+p)*u)
  //   = (2, 1*4+1, (4.5+1)*2) = (2, 5, 11).
  const cfe::EulerField<double, 1> field{};
  const cfe::State<double, 3> q = field.conserved_from_primitive(1.0, cfe::Vector<double, 1>(2.0), 1.0);
  const cfe::State<double, 3> flux = field.physical_flux(q, cfe::Axis::X);
  CFE_CHECK_NEAR(flux[0], 2.0, 1e-12);
  CFE_CHECK_NEAR(flux[1], 5.0, 1e-12);
  CFE_CHECK_NEAR(flux[2], 11.0, 1e-12);
}

CFE_TEST(test_euler_wave_speed_is_max_abs_u_plus_sound_speed_across_both_states)
{
  // Classic Sod initial states: left (rho=1,u=0,p=1) -> c=sqrt(1.4)=1.18322,
  // right (rho=0.125,u=0,p=0.1) -> c=sqrt(1.4*0.1/0.125)=sqrt(1.12)=1.05830.
  // Both have u=0, so wave_speed = max(c_left, c_right) = c_left.
  const cfe::EulerField<double, 1> field{};
  const cfe::State<double, 3> left = field.conserved_from_primitive(1.0, cfe::Vector<double, 1>(0.0), 1.0);
  const cfe::State<double, 3> right =
      field.conserved_from_primitive(0.125, cfe::Vector<double, 1>(0.0), 0.1);
  CFE_CHECK_NEAR(field.wave_speed(left, right, cfe::Axis::X), 1.1832159566199232, 1e-10);
}

CFE_TEST(test_euler_rusanov_flux_reduces_to_physical_flux_for_equal_states)
{
  // No jump at all -> dissipation term vanishes -> flux reduces to
  // exactly F(Q), the same degenerate-case check test_burgers_flux.cpp
  // performs for the scalar overload, here exercising the vector
  // (system) overload instead.
  const cfe::EulerField<double, 1> field{};
  const cfe::State<double, 3> q = field.conserved_from_primitive(1.0, cfe::Vector<double, 1>(2.0), 1.0);
  const cfe::State<double, 3> flux = cfe::rusanov_flux(q, q, cfe::Axis::X, field);
  const cfe::State<double, 3> expected = field.physical_flux(q, cfe::Axis::X);
  for (std::size_t c = 0; c < 3; ++c) CFE_CHECK_NEAR(flux[c], expected[c], 1e-12);
}

CFE_TEST(test_euler_rusanov_flux_functor_matches_free_function_for_vector_state)
{
  const cfe::EulerField<double, 1> field{};
  const cfe::State<double, 3> left = field.conserved_from_primitive(1.0, cfe::Vector<double, 1>(0.0), 1.0);
  const cfe::State<double, 3> right =
      field.conserved_from_primitive(0.125, cfe::Vector<double, 1>(0.0), 0.1);
  const cfe::RusanovFlux functor{};
  const cfe::State<double, 3> via_functor = functor(left, right, cfe::Axis::X, field);
  const cfe::State<double, 3> via_free_function = cfe::rusanov_flux(left, right, cfe::Axis::X, field);
  for (std::size_t c = 0; c < 3; ++c) CFE_CHECK_NEAR(via_functor[c], via_free_function[c], 1e-12);
}
