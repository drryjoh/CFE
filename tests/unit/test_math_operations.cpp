// Unit tests for cfe::math operations (task spec items 7/13): componentwise
// operations, contract, and weight.
#include "cfe/core/component_counts.hpp"
#include "cfe/math/fixed_array.hpp"
#include "cfe/math/operations.hpp"
#include "test_framework.hpp"

CFE_TEST(test_operator_plus_matches_manual_componentwise_sum)
{
  cfe::FixedArray<double, 3> a;
  a[0] = 1.0;
  a[1] = 2.0;
  a[2] = 3.0;
  cfe::FixedArray<double, 3> b;
  b[0] = 4.0;
  b[1] = 5.0;
  b[2] = 6.0;

  auto c = a + b;
  CFE_CHECK_NEAR(c[0], 5.0, 1e-12);
  CFE_CHECK_NEAR(c[1], 7.0, 1e-12);
  CFE_CHECK_NEAR(c[2], 9.0, 1e-12);
}

CFE_TEST(test_scalar_multiply_scales_every_component)
{
  cfe::FixedArray<double, 4> a(2.0);
  auto scaled = a * 3.0;
  for (std::size_t k = 0; k < scaled.size(); ++k) {
    CFE_CHECK_NEAR(scaled[k], 6.0, 1e-12);
  }
  auto scaled_commutative = 3.0 * a;
  for (std::size_t k = 0; k < scaled_commutative.size(); ++k) {
    CFE_CHECK_NEAR(scaled_commutative[k], 6.0, 1e-12);
  }
}

CFE_TEST(test_contract_reduces_to_dot_product_for_orthogonal_unit_vectors)
{
  cfe::FixedArray<double, 3> ex;
  ex[0] = 1.0;
  ex[1] = 0.0;
  ex[2] = 0.0;
  cfe::FixedArray<double, 3> ey;
  ey[0] = 0.0;
  ey[1] = 1.0;
  ey[2] = 0.0;

  CFE_CHECK_NEAR(cfe::contract(ex, ey), 0.0, 1e-12);
  CFE_CHECK_NEAR(cfe::contract(ex, ex), 1.0, 1e-12);
}

CFE_TEST(test_contract_matches_hand_computed_sum_of_products)
{
  cfe::FixedArray<double, 4> a;
  cfe::FixedArray<double, 4> b;
  double expected = 0.0;
  for (std::size_t k = 0; k < 4; ++k) {
    a[k] = static_cast<double>(k + 1);  // 1,2,3,4
    b[k] = static_cast<double>(4 - k);  // 4,3,2,1
    expected += a[k] * b[k];
  }
  CFE_CHECK_NEAR(cfe::contract(a, b), expected, 1e-12);
}

CFE_TEST(test_weight_applies_componentwise_scaling_not_reduction)
{
  cfe::FixedArray<double, 3> a;
  a[0] = 1.0;
  a[1] = 2.0;
  a[2] = 3.0;
  cfe::FixedArray<double, 3> w;
  w[0] = 10.0;
  w[1] = 0.5;
  w[2] = 2.0;

  auto weighted = cfe::weight(a, w);
  CFE_CHECK_NEAR(weighted[0], 10.0, 1e-12);
  CFE_CHECK_NEAR(weighted[1], 1.0, 1e-12);
  CFE_CHECK_NEAR(weighted[2], 6.0, 1e-12);
}

CFE_TEST(test_weight_with_unit_weights_is_identity)
{
  cfe::FixedArray<double, 5> a(2.5);
  cfe::FixedArray<double, 5> ones(1.0);
  auto weighted = cfe::weight(a, ones);
  for (std::size_t k = 0; k < weighted.size(); ++k) {
    CFE_CHECK_NEAR(weighted[k], a[k], 1e-12);
  }
}

CFE_TEST(test_math_operations_compile_for_all_required_component_counts)
{
  cfe::for_each_component_count([](auto n_components) {
    constexpr std::size_t N = decltype(n_components)::value;
    cfe::FixedArray<double, N> a(2.0);
    cfe::FixedArray<double, N> b(3.0);
    const double c = cfe::contract(a, b);
    CFE_CHECK_NEAR(c, 6.0 * static_cast<double>(N), 1e-9);
    auto w = cfe::weight(a, b);
    CFE_CHECK_NEAR(w[0], 6.0, 1e-9);
  });
}

CFE_TEST(test_math_operations_compile_in_float_and_double)
{
  cfe::FixedArray<float, 5> af(1.0f);
  cfe::FixedArray<float, 5> bf(2.0f);
  CFE_CHECK_NEAR(cfe::contract(af, bf), 10.0f, 1e-5f);

  cfe::FixedArray<double, 5> ad(1.0);
  cfe::FixedArray<double, 5> bd(2.0);
  CFE_CHECK_NEAR(cfe::contract(ad, bd), 10.0, 1e-12);
}
