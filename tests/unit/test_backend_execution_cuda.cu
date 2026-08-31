// CUDA backend correctness test (task spec item 13: "CUDA backend produces
// expected results when available"), expanded per PR #1 review to match the
// CPU backend test's coverage: both precisions, both layouts, and all six
// required component counts.
//
// UNVERIFIED: no CUDA toolkit/NVIDIA GPU was available in the development
// environment (see docs/adr/0001-execution-backend.md). This file is only
// compiled and linked into cfe_unit_tests when CFE_ENABLE_CUDA is on, so it
// does not affect the CPU-only test binary that was actually run.
#include <cmath>
#include <type_traits>
#include <vector>

#include "cfe/backend/cuda/cuda_backend.cuh"
#include "cfe/backend/cuda/device_field.cuh"
#include "cfe/core/component_counts.hpp"
#include "cfe/field/layout.hpp"
#include "test_framework.hpp"

namespace {

// float's ~7 significant decimal digits can't exactly represent every
// squared value at the larger end of this test's input range (up to
// roughly (4095 + 99)^2), unlike double; scale the tolerance accordingly
// instead of using a fixed absolute one.
template <class Scalar>
double tolerance_for(double expected)
{
  if (std::is_same<Scalar, float>::value) {
    return std::fabs(expected) * 1e-5 + 1e-2;
  }
  return 1e-9;
}

template <class Scalar, class Layout>
void run_case()
{
  cfe::for_each_component_count([](auto n_components) {
    constexpr std::size_t N = decltype(n_components)::value;
    constexpr std::size_t n_cells = 4096;

    std::vector<Scalar> host_q(n_cells * N);
    for (std::size_t i = 0; i < n_cells; ++i) {
      for (std::size_t k = 0; k < N; ++k) {
        host_q[Layout::index(i, k, n_cells, N)] = static_cast<Scalar>(i + k + 1);
      }
    }

    cfe::backend::cuda::DeviceField<Scalar, N, Layout> q(n_cells);
    cfe::backend::cuda::DeviceField<Scalar, N, Layout> q_new(n_cells);
    q.copy_from_host(host_q.data());

    auto q_view = q.view();
    auto out_view = q_new.view();
    cfe::backend::cuda::parallel_for(n_cells, [=] CFE_DEVICE(std::size_t i) mutable {
      for (std::size_t k = 0; k < N; ++k) out_view(i, k) = q_view(i, k) * q_view(i, k);
    });
    // parallel_for is asynchronous -- must synchronize before the result is
    // meaningful to copy back and check.
    cfe::backend::cuda::synchronize();

    std::vector<Scalar> host_result(n_cells * N);
    q_new.copy_to_host(host_result.data());

    for (std::size_t i = 0; i < n_cells; ++i) {
      for (std::size_t k = 0; k < N; ++k) {
        const double base = static_cast<double>(host_q[Layout::index(i, k, n_cells, N)]);
        const double expected = base * base;
        CFE_CHECK_NEAR(host_result[Layout::index(i, k, n_cells, N)], expected,
                       tolerance_for<Scalar>(expected));
      }
    }
  });
}

}  // namespace

CFE_TEST(test_cuda_backend_all_cell_square_matches_host_reference_double_aos)
{
  run_case<double, cfe::AoSLayout>();
}

CFE_TEST(test_cuda_backend_all_cell_square_matches_host_reference_double_soa)
{
  run_case<double, cfe::SoALayout>();
}

CFE_TEST(test_cuda_backend_all_cell_square_matches_host_reference_float_aos)
{
  run_case<float, cfe::AoSLayout>();
}

CFE_TEST(test_cuda_backend_all_cell_square_matches_host_reference_float_soa)
{
  run_case<float, cfe::SoALayout>();
}
