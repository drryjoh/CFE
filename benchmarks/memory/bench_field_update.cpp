// Phase 0 benchmark infrastructure (task spec items 10/11).
//
// Exercises the simplest possible all-cell kernel, q_new(i,k) = q(i,k) *
// q(i,k), across:
//   - precision:        float, double
//   - component count:  1, 5, 10, 20, 50, 100
//   - backend:          serial, threaded
//   - storage layout:   AoS, SoA
//
// The purpose is to exercise the memory and execution architecture (per the
// task spec), not to solve a PDE. Results are printed as CSV to stdout;
// docs/performance/0001-phase0-results.md records a curated copy of an
// actual run plus the hardware it was measured on.
//
// A CUDA sweep is not part of this binary: kernel launches require a `.cu`
// translation unit, so CUDA is exercised by
// benchmarks/memory/bench_field_update_cuda.cu (built only when
// CFE_ENABLE_CUDA is on).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "cfe/backend/cpu/serial.hpp"
#include "cfe/backend/cpu/threaded.hpp"
#include "cfe/core/component_counts.hpp"
#include "cfe/field/field.hpp"
#include "cfe/field/layout.hpp"

namespace {

enum class BackendKind { Serial, Threaded };

const char* backend_name(BackendKind b) { return b == BackendKind::Serial ? "serial" : "threaded"; }

template <class Layout>
const char* layout_name();
template <>
const char* layout_name<cfe::AoSLayout>()
{
  return "AoS";
}
template <>
const char* layout_name<cfe::SoALayout>()
{
  return "SoA";
}

// Roughly how many bytes each of q/q_new should occupy, so the working set
// comfortably exceeds cache regardless of component count. Not a rigorous
// roofline target -- just large enough that this is a memory-bandwidth
// exercise rather than an L1-resident microbenchmark.
constexpr std::size_t kTargetBytesPerField = 64ull * 1024ull * 1024ull;
constexpr int kRepetitions = 7;

template <class Scalar, std::size_t N, class Layout>
double median_seconds_per_pass(std::size_t n_cells, BackendKind backend)
{
  cfe::Field<Scalar, N, Layout> q(n_cells);
  cfe::Field<Scalar, N, Layout> q_new(n_cells);

  // One-time initialization, outside every timed region.
  for (std::size_t i = 0; i < n_cells; ++i) {
    for (std::size_t k = 0; k < N; ++k) {
      q(i, k) = static_cast<Scalar>((i % 97) + k) * Scalar(0.01);
    }
  }

  auto q_view = q.view();
  auto out_view = q_new.view();

  auto run_pass = [=]() mutable {
    if (backend == BackendKind::Serial) {
      cfe::backend::serial::parallel_for(n_cells, [=](std::size_t i) mutable {
        for (std::size_t k = 0; k < N; ++k) out_view(i, k) = q_view(i, k) * q_view(i, k);
      });
    } else {
      cfe::backend::threaded::parallel_for(n_cells, [=](std::size_t i) mutable {
        for (std::size_t k = 0; k < N; ++k) out_view(i, k) = q_view(i, k) * q_view(i, k);
      });
    }
  };

  // Warm up: first-touch faulting, cache/thread-pool spin-up, etc.
  run_pass();

  std::vector<double> seconds;
  seconds.reserve(kRepetitions);
  for (int r = 0; r < kRepetitions; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    run_pass();
    const auto t1 = std::chrono::steady_clock::now();
    seconds.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(seconds.begin(), seconds.end());
  return seconds[seconds.size() / 2];
}

template <class Scalar, std::size_t N, class Layout>
void run_case(const char* precision_name, BackendKind backend)
{
  const std::size_t n_cells =
      std::max<std::size_t>(1024, kTargetBytesPerField / (N * sizeof(Scalar)));

  const double median_s = median_seconds_per_pass<Scalar, N, Layout>(n_cells, backend);

  const double cell_updates_per_s = static_cast<double>(n_cells) / median_s;
  const double scalar_updates_per_s =
      static_cast<double>(n_cells) * static_cast<double>(N) / median_s;
  // One read of q and one write of q_new per scalar element.
  const double bytes_moved =
      2.0 * static_cast<double>(n_cells) * static_cast<double>(N) * sizeof(Scalar);
  const double bandwidth_gb_s = bytes_moved / median_s / 1.0e9;

  std::printf("%s,%zu,%s,%s,%zu,%d,%.6f,%.3e,%.3e,%.3f\n", precision_name, N, backend_name(backend),
              layout_name<Layout>(), n_cells, kRepetitions, median_s * 1e3, cell_updates_per_s,
              scalar_updates_per_s, bandwidth_gb_s);
}

template <class Scalar>
void run_all_cases_for_precision(const char* precision_name)
{
  const BackendKind backends[] = {BackendKind::Serial, BackendKind::Threaded};
  for (BackendKind backend : backends) {
    cfe::for_each_component_count([&](auto n_components) {
      constexpr std::size_t N = decltype(n_components)::value;
      run_case<Scalar, N, cfe::AoSLayout>(precision_name, backend);
      run_case<Scalar, N, cfe::SoALayout>(precision_name, backend);
    });
  }
}

}  // namespace

int main()
{
  std::printf(
      "precision,n_components,backend,layout,n_cells,repetitions,median_ms,cell_updates_per_s,"
      "scalar_updates_per_s,bandwidth_gb_s\n");
  run_all_cases_for_precision<float>("float");
  run_all_cases_for_precision<double>("double");
  return 0;
}
