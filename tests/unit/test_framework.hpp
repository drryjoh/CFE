// Minimal, dependency-free unit test framework for CMU-CFE.
//
// Phase 0 deliberately avoids pulling in an external testing framework
// (GoogleTest/Catch2/etc.): the entire harness needed to satisfy "unit
// tests for math operations, fixed-size containers, field storage, backend
// execution correctness" (task spec item 13) is a couple dozen lines, and
// AGENTS.md #6 asks for the smallest architecture that satisfies the
// requirement. Revisit if test needs grow beyond simple pass/fail checks.
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace cfe {
namespace testing {

struct AssertionFailure
{
  std::string message;
};

using TestFn = std::function<void()>;

struct TestCase
{
  std::string name;
  TestFn fn;
};

inline std::vector<TestCase>& registry()
{
  static std::vector<TestCase> tests;
  return tests;
}

struct Registrar
{
  Registrar(std::string name, TestFn fn) { registry().push_back({std::move(name), std::move(fn)}); }
};

inline int run_all()
{
  int failed = 0;
  for (const auto& test : registry()) {
    try {
      test.fn();
      std::printf("[PASS] %s\n", test.name.c_str());
    } catch (const AssertionFailure& failure) {
      std::printf("[FAIL] %s: %s\n", test.name.c_str(), failure.message.c_str());
      ++failed;
    } catch (const std::exception& e) {
      std::printf("[FAIL] %s: unexpected exception: %s\n", test.name.c_str(), e.what());
      ++failed;
    }
  }
  const int total = static_cast<int>(registry().size());
  std::printf("\n%d/%d tests passed\n", total - failed, total);
  return failed == 0 ? 0 : 1;
}

}  // namespace testing
}  // namespace cfe

#define CFE_CONCAT_(a, b) a##b
#define CFE_CONCAT(a, b) CFE_CONCAT_(a, b)

#define CFE_TEST(test_name)                                                              \
  static void test_name();                                                               \
  namespace {                                                                            \
  const ::cfe::testing::Registrar CFE_CONCAT(cfe_test_registrar_, __LINE__)(#test_name,  \
                                                                            &test_name); \
  }                                                                                      \
  static void test_name()

#define CFE_CHECK(condition)                                                                       \
  do {                                                                                             \
    if (!(condition)) {                                                                            \
      throw ::cfe::testing::AssertionFailure{std::string("CFE_CHECK(" #condition ") failed at ") + \
                                             __FILE__ + ":" + std::to_string(__LINE__)};           \
    }                                                                                              \
  } while (0)

#define CFE_CHECK_NEAR(a, b, tol)                                                             \
  do {                                                                                        \
    const auto cfe_check_near_a_ = (a);                                                       \
    const auto cfe_check_near_b_ = (b);                                                       \
    const auto cfe_check_near_diff_ = std::fabs(static_cast<double>(cfe_check_near_a_) -      \
                                                static_cast<double>(cfe_check_near_b_));      \
    if (cfe_check_near_diff_ > (tol)) {                                                       \
      throw ::cfe::testing::AssertionFailure{                                                 \
          std::string("CFE_CHECK_NEAR(" #a ", " #b ") failed at ") + __FILE__ + ":" +         \
          std::to_string(__LINE__) + " (diff " + std::to_string(cfe_check_near_diff_) + ")"}; \
    }                                                                                         \
  } while (0)
