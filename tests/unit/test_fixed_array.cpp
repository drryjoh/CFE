// Unit tests for cfe::FixedArray (task spec item 13: "fixed-size
// containers"). Exercised at N=1 (scalar) and a representative vector size,
// in both float and double.
#include "cfe/core/component_counts.hpp"
#include "cfe/math/fixed_array.hpp"
#include "test_framework.hpp"

namespace {

template <class Scalar>
void check_basic_arithmetic() {
  cfe::FixedArray<Scalar, 3> a;
  a[0] = Scalar(1);
  a[1] = Scalar(2);
  a[2] = Scalar(3);

  cfe::FixedArray<Scalar, 3> b(Scalar(10));

  cfe::FixedArray<Scalar, 3> sum = a;
  sum += b;
  CFE_CHECK_NEAR(sum[0], Scalar(11), 1e-9);
  CFE_CHECK_NEAR(sum[1], Scalar(12), 1e-9);
  CFE_CHECK_NEAR(sum[2], Scalar(13), 1e-9);

  cfe::FixedArray<Scalar, 3> diff = b;
  diff -= a;
  CFE_CHECK_NEAR(diff[0], Scalar(9), 1e-9);
  CFE_CHECK_NEAR(diff[1], Scalar(8), 1e-9);
  CFE_CHECK_NEAR(diff[2], Scalar(7), 1e-9);

  cfe::FixedArray<Scalar, 3> scaled = a;
  scaled *= Scalar(2);
  CFE_CHECK_NEAR(scaled[0], Scalar(2), 1e-9);
  CFE_CHECK_NEAR(scaled[1], Scalar(4), 1e-9);
  CFE_CHECK_NEAR(scaled[2], Scalar(6), 1e-9);

  cfe::FixedArray<Scalar, 3> divided = scaled;
  divided /= Scalar(2);
  CFE_CHECK_NEAR(divided[0], a[0], 1e-9);
  CFE_CHECK_NEAR(divided[1], a[1], 1e-9);
  CFE_CHECK_NEAR(divided[2], a[2], 1e-9);
}

} // namespace

CFE_TEST(test_fixed_array_componentwise_arithmetic_float) { check_basic_arithmetic<float>(); }

CFE_TEST(test_fixed_array_componentwise_arithmetic_double) { check_basic_arithmetic<double>(); }

CFE_TEST(test_fixed_array_fill_constructor_sets_all_components) {
  cfe::FixedArray<double, 5> a(3.5);
  for (std::size_t k = 0; k < a.size(); ++k) {
    CFE_CHECK_NEAR(a[k], 3.5, 1e-12);
  }
}

CFE_TEST(test_fixed_array_all_required_component_counts_compile) {
  // Task spec item 9: components 1, 5, 10, 20, 50, 100 must compile.
  int constructed = 0;
  cfe::for_each_component_count([&](auto n_components) {
    constexpr std::size_t N = decltype(n_components)::value;
    cfe::FixedArray<double, N> state(1.0);
    CFE_CHECK(state.size() == N);
    ++constructed;
  });
  CFE_CHECK(constructed == 6);
}
