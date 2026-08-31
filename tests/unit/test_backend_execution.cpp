// Unit tests for backend execution correctness (task spec item 13: "CPU
// backend produces expected results"; "CUDA backend produces expected
// results when available" -- CUDA is exercised separately, see
// benchmarks/memory/bench_field_update_cuda.cu, because it requires a `.cu`
// translation unit and is only built when CFE_ENABLE_CUDA is on).
#include <atomic>
#include <vector>

#include "cfe/backend/cpu/serial.hpp"
#include "cfe/backend/cpu/threaded.hpp"
#include "cfe/core/component_counts.hpp"
#include "cfe/field/field.hpp"
#include "test_framework.hpp"

CFE_TEST(test_serial_parallel_for_visits_every_index_exactly_once)
{
  constexpr int n = 1000;
  std::vector<int> visits(n, 0);
  cfe::backend::serial::parallel_for(n, [&](int i) { visits[i]++; });
  for (int i = 0; i < n; ++i) {
    CFE_CHECK(visits[i] == 1);
  }
}

CFE_TEST(test_threaded_parallel_for_visits_every_index_exactly_once)
{
  constexpr int n = 100000;
  std::vector<std::atomic<int>> visits(n);
  for (auto& v : visits) v = 0;
  cfe::backend::threaded::parallel_for(n, [&](int i) { visits[i]++; });
  for (int i = 0; i < n; ++i) {
    CFE_CHECK(visits[i].load() == 1);
  }
}

CFE_TEST(test_threaded_backend_matches_serial_backend_bitwise)
{
  // The all-cell update from the task spec: q_new(i,k) = q(i,k) * q(i,k).
  // Squaring is associative/commutative-safe per element (no reduction
  // across elements), so serial and threaded execution must agree exactly,
  // not just within a floating-point tolerance.
  constexpr std::size_t n_cells = 5000;
  constexpr std::size_t n_components = 5;

  cfe::Field<double, n_components> q(n_cells);
  for (std::size_t i = 0; i < n_cells; ++i) {
    for (std::size_t k = 0; k < n_components; ++k) {
      q(i, k) = static_cast<double>(i + k) * 0.001;
    }
  }

  cfe::Field<double, n_components> q_new_serial(n_cells);
  cfe::Field<double, n_components> q_new_threaded(n_cells);

  auto q_view = q.view();
  auto out_serial = q_new_serial.view();
  auto out_threaded = q_new_threaded.view();

  cfe::backend::serial::parallel_for(n_cells, [=](std::size_t i) mutable {
    for (std::size_t k = 0; k < n_components; ++k) {
      out_serial(i, k) = q_view(i, k) * q_view(i, k);
    }
  });

  cfe::backend::threaded::parallel_for(n_cells, [=](std::size_t i) mutable {
    for (std::size_t k = 0; k < n_components; ++k) {
      out_threaded(i, k) = q_view(i, k) * q_view(i, k);
    }
  });

  for (std::size_t i = 0; i < n_cells; ++i) {
    for (std::size_t k = 0; k < n_components; ++k) {
      CFE_CHECK(out_serial(i, k) == out_threaded(i, k));
      CFE_CHECK_NEAR(out_serial(i, k), q(i, k) * q(i, k), 1e-15);
    }
  }
}

CFE_TEST(test_all_cell_square_update_compiles_and_runs_for_all_required_component_counts)
{
  cfe::for_each_component_count([](auto n_components) {
    constexpr std::size_t N = decltype(n_components)::value;
    constexpr std::size_t n_cells = 256;

    cfe::Field<double, N> q(n_cells);
    for (std::size_t i = 0; i < n_cells; ++i) {
      for (std::size_t k = 0; k < N; ++k) q(i, k) = static_cast<double>(i + k + 1);
    }
    cfe::Field<double, N> q_new(n_cells);

    auto q_view = q.view();
    auto out_view = q_new.view();
    cfe::backend::serial::parallel_for(n_cells, [=](std::size_t i) mutable {
      for (std::size_t k = 0; k < N; ++k) out_view(i, k) = q_view(i, k) * q_view(i, k);
    });

    for (std::size_t i = 0; i < n_cells; ++i) {
      for (std::size_t k = 0; k < N; ++k) {
        const double expected = q(i, k) * q(i, k);
        CFE_CHECK_NEAR(q_new(i, k), expected, 1e-9);
      }
    }
  });
}
