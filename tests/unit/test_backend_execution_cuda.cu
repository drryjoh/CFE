// CUDA backend correctness test (task spec item 13: "CUDA backend produces
// expected results when available").
//
// UNVERIFIED: no CUDA toolkit/NVIDIA GPU was available in the Phase 0
// development environment (see docs/adr/0001-execution-backend.md). This
// file is only compiled and linked into cfe_unit_tests when CFE_ENABLE_CUDA
// is on, so it does not affect the CPU-only test binary that was actually
// run for this task.
#include <vector>

#include "cfe/backend/cuda/cuda_backend.cuh"
#include "cfe/backend/cuda/device_field.cuh"
#include "cfe/core/component_counts.hpp"
#include "cfe/field/layout.hpp"
#include "test_framework.hpp"

CFE_TEST(test_cuda_backend_all_cell_square_matches_host_reference)
{
  cfe::for_each_component_count([](auto n_components) {
    constexpr std::size_t N = decltype(n_components)::value;
    constexpr std::size_t n_cells = 4096;

    std::vector<double> host_q(n_cells * N);
    for (std::size_t i = 0; i < n_cells; ++i) {
      for (std::size_t k = 0; k < N; ++k) {
        host_q[cfe::AoSLayout::index(i, k, n_cells, N)] = static_cast<double>(i + k + 1);
      }
    }

    cfe::backend::cuda::DeviceField<double, N, cfe::AoSLayout> q(n_cells);
    cfe::backend::cuda::DeviceField<double, N, cfe::AoSLayout> q_new(n_cells);
    q.copy_from_host(host_q.data());

    auto q_view = q.view();
    auto out_view = q_new.view();
    cfe::backend::cuda::parallel_for(n_cells, [=] CFE_DEVICE(std::size_t i) mutable {
      for (std::size_t k = 0; k < N; ++k) out_view(i, k) = q_view(i, k) * q_view(i, k);
    });

    std::vector<double> host_result(n_cells * N);
    q_new.copy_to_host(host_result.data());

    for (std::size_t i = 0; i < n_cells; ++i) {
      for (std::size_t k = 0; k < N; ++k) {
        const double expected = host_q[cfe::AoSLayout::index(i, k, n_cells, N)] *
                                host_q[cfe::AoSLayout::index(i, k, n_cells, N)];
        CFE_CHECK_NEAR(host_result[cfe::AoSLayout::index(i, k, n_cells, N)], expected, 1e-9);
      }
    }
  });
}
