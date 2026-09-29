// Large-scale CUDA scalar-advection benchmark. Sweeps grid resolutions
// from modest up to genuinely large (10^8 cells) to demonstrate the
// solver actually runs at scale on real GPU hardware, not just at the
// small sizes the correctness/convergence tests use.
//
// Reports median wall-clock time and cell-updates/s over several timed
// SSP-RK2 steps, after one untimed warm-up step (first launch pays
// CUDA context/JIT costs -- see benchmarks/memory/bench_field_update_cuda.cu
// for the same convention).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cfe/backend/cuda/cuda_backend.cuh"
#include "cfe/backend/cuda/device_field.cuh"
#include "cfe/fields/scalar_advection/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSpeed = 1.0;
constexpr int kRepetitions = 10;

void run_case(std::size_t nx)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = 1.0 / static_cast<double>(nx);

  std::vector<double> host_ic(grid.n_cells_total(), 0.0);
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const double x = grid.x_center(grid.ngx + i);
    host_ic[grid.flat_index(grid.ngx + i, 0, 0)] = std::sin(2.0 * kPi * x);
  }

  cfe::backend::cuda::DeviceField<double, 1> q(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, 1> stage1(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, 1> scratch(grid.n_cells_total());
  q.copy_from_host(host_ic.data());

  cfe::ScalarAdvectionField<double, 1> field{cfe::Vector<double, 1>(kSpeed)};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 1>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary,
                 cfe::fvm::CentralDifferenceReconstruction, cfe::UpwindFlux, cfe::CudaParallelFor>
      solver{grid, field, cfe::PeriodicBoundary{}};
  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver.residual(in, out);
  };

  const double dt = 0.4 * grid.dx / kSpeed;
  auto one_step = [&]() {
    cfe::ssp_rk2_step<double, cfe::FieldView<double, 1>, decltype(residual), cfe::CudaParallelFor>(
        q.view(), stage1.view(), scratch.view(), dt, residual);
    cfe::backend::cuda::synchronize();
  };

  one_step();  // warm-up: first launch pays context/JIT costs.

  std::vector<double> seconds;
  seconds.reserve(kRepetitions);
  for (int r = 0; r < kRepetitions; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    one_step();
    const auto t1 = std::chrono::steady_clock::now();
    seconds.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(seconds.begin(), seconds.end());
  const double median_s = seconds[seconds.size() / 2];

  const double cell_updates_per_s = static_cast<double>(grid.nx) / median_s;

  std::printf("cuda,%zu,%d,%.6f,%.3e\n", grid.nx, kRepetitions, median_s * 1e3, cell_updates_per_s);
}

}  // namespace

int main()
{
  std::printf("backend,n_cells,repetitions,median_ms,cell_updates_per_s\n");
  for (std::size_t nx : {std::size_t(1'000'000), std::size_t(10'000'000), std::size_t(50'000'000),
                        std::size_t(100'000'000)}) {
    run_case(nx);
  }
  return 0;
}
