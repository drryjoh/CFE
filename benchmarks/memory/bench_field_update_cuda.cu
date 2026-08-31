// CUDA counterpart to bench_field_update.cpp (task spec items 5/9/11/12).
//
// UNVERIFIED: no CUDA toolkit/NVIDIA GPU was available in the Phase 0
// development environment, so this file has not been compiled or run (see
// docs/adr/0001-execution-backend.md and
// docs/performance/0001-phase0-results.md). It is built only when
// `CFE_ENABLE_CUDA` is on (CMake auto-detects `nvcc`; see CMakeLists.txt).
//
// It mirrors bench_field_update.cpp's sweep over precision and component
// count, but only against the CUDA backend (CPU vs. CUDA is compared by
// running both binaries and comparing their CSV output, not by linking them
// into one executable, since backend/cuda/*.cuh may only be included from a
// `.cu` translation unit).
//
// For register/occupancy/spill data, see scripts/profile_cuda.sh; that
// workflow should be run against the `parallel_for_kernel` instantiations
// this binary launches, once real hardware is available.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "cfe/backend/cuda/cuda_backend.cuh"
#include "cfe/backend/cuda/device_field.cuh"
#include "cfe/core/component_counts.hpp"
#include "cfe/field/layout.hpp"

namespace {

template <class Layout>
const char* layout_name();
template <>
const char* layout_name<cfe::AoSLayout>() {
  return "AoS";
}
template <>
const char* layout_name<cfe::SoALayout>() {
  return "SoA";
}

constexpr std::size_t kTargetBytesPerField = 64ull * 1024ull * 1024ull;
constexpr int kRepetitions = 7;

template <class Scalar, std::size_t N, class Layout>
void run_case(const char* precision_name) {
  const std::size_t n_cells =
      std::max<std::size_t>(1024, kTargetBytesPerField / (N * sizeof(Scalar)));

  cfe::backend::cuda::DeviceField<Scalar, N, Layout> q(n_cells);
  cfe::backend::cuda::DeviceField<Scalar, N, Layout> q_new(n_cells);

  std::vector<Scalar> host_q(n_cells * N);
  for (std::size_t i = 0; i < n_cells; ++i) {
    for (std::size_t k = 0; k < N; ++k) {
      host_q[Layout::index(i, k, n_cells, N)] = static_cast<Scalar>((i % 97) + k) * Scalar(0.01);
    }
  }
  q.copy_from_host(host_q.data());

  auto q_view = q.view();
  auto out_view = q_new.view();
  auto run_pass = [=]() mutable {
    cfe::backend::cuda::parallel_for(n_cells, [=] CFE_DEVICE(std::size_t i) mutable {
      for (std::size_t k = 0; k < N; ++k) out_view(i, k) = q_view(i, k) * q_view(i, k);
    });
  };

  run_pass(); // warm up: first launch pays context/JIT costs.

  std::vector<double> seconds;
  seconds.reserve(kRepetitions);
  for (int r = 0; r < kRepetitions; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    run_pass();
    const auto t1 = std::chrono::steady_clock::now();
    seconds.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(seconds.begin(), seconds.end());
  const double median_s = seconds[seconds.size() / 2];

  const double cell_updates_per_s = static_cast<double>(n_cells) / median_s;
  const double scalar_updates_per_s = static_cast<double>(n_cells) * static_cast<double>(N) / median_s;
  const double bytes_moved = 2.0 * static_cast<double>(n_cells) * static_cast<double>(N) * sizeof(Scalar);
  const double bandwidth_gb_s = bytes_moved / median_s / 1.0e9;

  std::printf("%s,%zu,cuda,%s,%zu,%d,%.6f,%.3e,%.3e,%.3f\n", precision_name, N, layout_name<Layout>(),
              n_cells, kRepetitions, median_s * 1e3, cell_updates_per_s, scalar_updates_per_s,
              bandwidth_gb_s);
}

template <class Scalar>
void run_all_cases_for_precision(const char* precision_name) {
  cfe::for_each_component_count([&](auto n_components) {
    constexpr std::size_t N = decltype(n_components)::value;
    run_case<Scalar, N, cfe::AoSLayout>(precision_name);
    run_case<Scalar, N, cfe::SoALayout>(precision_name);
  });
}

} // namespace

int main() {
  std::printf("precision,n_components,backend,layout,n_cells,repetitions,median_ms,cell_updates_per_s,"
              "scalar_updates_per_s,bandwidth_gb_s\n");
  run_all_cases_for_precision<float>("float");
  run_all_cases_for_precision<double>("double");
  return 0;
}
