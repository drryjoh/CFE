// Unit tests for interface-value reconstruction and the numerical flux
// (task spec item 4/tests: "the interface-value/numerical-flux
// calculation matches a hand-computed reference for a simple case").
#include "cfe/core/types.hpp"
#include "cfe/fields/scalar_advection/field.hpp"
#include "cfe/numerics/fvm/interface_value.hpp"
#include "cfe/numerics/numerical_flux/upwind.hpp"
#include "test_framework.hpp"

CFE_TEST(test_interface_value_right_matches_hand_computed_linear_extrapolation)
{
  // q_{i-1}=1, q_i=2, q_{i+1}=4 -> Q_face = 2 + (4-1)/4 = 2.75
  CFE_CHECK_NEAR(cfe::fvm::interface_value_right(1.0, 2.0, 4.0), 2.75, 1e-12);
}

CFE_TEST(test_interface_value_left_matches_hand_computed_linear_extrapolation)
{
  // Same stencil, opposite sign: Q_face = 2 - (4-1)/4 = 1.25
  CFE_CHECK_NEAR(cfe::fvm::interface_value_left(1.0, 2.0, 4.0), 1.25, 1e-12);
}

CFE_TEST(test_interface_value_reduces_to_cell_average_for_a_uniform_field)
{
  // A spatially constant field has zero slope everywhere: every face
  // value should equal the cell value itself, regardless of side.
  CFE_CHECK_NEAR(cfe::fvm::interface_value_right(5.0, 5.0, 5.0), 5.0, 1e-12);
  CFE_CHECK_NEAR(cfe::fvm::interface_value_left(5.0, 5.0, 5.0), 5.0, 1e-12);
}

CFE_TEST(test_upwind_flux_selects_left_value_for_positive_advection_speed)
{
  // a=2 (positive): flux = a * left_value = 2 * 2.75 = 5.5, regardless of
  // what the right value is.
  const cfe::ScalarAdvectionField<double, 1> field{cfe::Vector<double, 1>(2.0)};
  CFE_CHECK_NEAR(cfe::upwind_flux(2.75, 1.25, cfe::Axis::X, field), 5.5, 1e-12);
}

CFE_TEST(test_upwind_flux_selects_right_value_for_negative_advection_speed)
{
  // a=-2 (negative): flux = a * right_value = -2 * 1.25 = -2.5.
  const cfe::ScalarAdvectionField<double, 1> field{cfe::Vector<double, 1>(-2.0)};
  CFE_CHECK_NEAR(cfe::upwind_flux(2.75, 1.25, cfe::Axis::X, field), -2.5, 1e-12);
}

CFE_TEST(test_upwind_flux_is_zero_when_advection_speed_is_zero)
{
  const cfe::ScalarAdvectionField<double, 1> field{cfe::Vector<double, 1>(0.0)};
  CFE_CHECK_NEAR(cfe::upwind_flux(2.75, 1.25, cfe::Axis::X, field), 0.0, 1e-12);
}
