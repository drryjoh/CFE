// Self-test of the test framework itself (test_framework.hpp).
// Regression test for a code-review finding (see agent_history.md's
// 2026-09-30 entry): CFE_CHECK_NEAR must reject non-finite values
// explicitly, since `std::fabs(NaN - x) > tol` is always false --
// without this, a NaN or Infinity on either side of a comparison would
// silently PASS a "near" check, exactly the failure mode this project's
// primary CPU/GPU cross-backend acceptance gate must not have.
#include <limits>

#include "test_framework.hpp"

namespace {

// CFE_CHECK_NEAR throws cfe::testing::AssertionFailure on failure --
// catching that here (rather than letting it propagate) is how this
// test verifies "the macro correctly detects bad input" without itself
// failing as a side effect.
bool check_near_throws(double a, double b)
{
  try {
    CFE_CHECK_NEAR(a, b, 1e-6);
  } catch (const cfe::testing::AssertionFailure&) {
    return true;
  }
  return false;
}

}  // namespace

CFE_TEST(test_check_near_rejects_nan)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CFE_CHECK(check_near_throws(nan, 1.0));
  CFE_CHECK(check_near_throws(1.0, nan));
  CFE_CHECK(check_near_throws(nan, nan));
}

CFE_TEST(test_check_near_rejects_infinity)
{
  const double inf = std::numeric_limits<double>::infinity();
  CFE_CHECK(check_near_throws(inf, 1.0));
  CFE_CHECK(check_near_throws(1.0, -inf));
  CFE_CHECK(check_near_throws(inf, inf));  // equal, but still not finite -- still rejected
}

CFE_TEST(test_check_near_still_accepts_ordinary_finite_values)
{
  // Sanity check alongside the two rejection tests above: the fix must
  // not have broken the macro's actual job.
  CFE_CHECK(!check_near_throws(1.0000001, 1.0));
  CFE_CHECK(check_near_throws(2.0, 1.0));  // genuinely too far apart -- should still fail
}
