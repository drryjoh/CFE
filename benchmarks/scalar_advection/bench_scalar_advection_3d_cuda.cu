// Large-scale 3D CUDA scalar-advection benchmark (2026-09-29 follow-up
// to bench_scalar_advection_cuda.cu, which only sweeps a 1D line of
// cells). Sweeps cube resolutions up to 512^3 (~134 million cells,
// matching that benchmark's 10^8-cell order of magnitude) to demonstrate
// the fully dimension-generic solver -- all three of X/Y/Z's
// `axis_flux_difference` branches active every step, not just X -- at
// scale on real GPU hardware.
//
// Reports median wall-clock time and cell-updates/s over several timed
// SSP-RK2 steps, after one untimed warm-up step, matching the 1D
// benchmark's convention.
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
constexpr double kUx = 1.0;
constexpr double kUy = 0.6;
constexpr double kUz = 0.3;
constexpr int kRepetitions = 10;

void run_case(std::size_t n_per_axis)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = n_per_axis;
  grid.ny = n_per_axis;
  grid.nz = n_per_axis;
  grid.ngx = 2;
  grid.ngy = 2;
  grid.ngz = 2;
  grid.dx = 1.0 / static_cast<double>(n_per_axis);
  grid.dy = grid.dx;
  grid.dz = grid.dx;

  std::vector<double> host_ic(grid.n_cells_total(), 0.0);
  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const double x = grid.x_center(grid.ngx + i);
        const double y = grid.y_center(grid.ngy + j);
        const double z = grid.z_center(grid.ngz + k);
        host_ic[grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k)] =
            std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y) * std::sin(2.0 * kPi * z);
      }
    }
  }

  cfe::backend::cuda::DeviceField<double, 1> q(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, 1> stage1(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, 1> scratch(grid.n_cells_total());
  q.copy_from_host(host_ic.data());

  cfe::Vector<double, 3> velocity;
  velocity[0] = kUx;
  velocity[1] = kUy;
  velocity[2] = kUz;
  cfe::ScalarAdvectionField<double, 3> field{velocity};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 3>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::CentralDifferenceReconstruction,
                 cfe::UpwindFlux, cfe::CudaParallelFor>
      solver{grid, field, cfe::PeriodicBoundary{}};
  cfe::SolverResidual<decltype(solver)> residual{&solver};

  const double dt = 0.25 * grid.dx / (kUx + kUy + kUz);
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

  const double n_cells = static_cast<double>(grid.nx) * static_cast<double>(grid.ny) *
                          static_cast<double>(grid.nz);
  const double cell_updates_per_s = n_cells / median_s;

  std::printf("cuda,3,%zu,%.0f,%d,%.6f,%.3e\n", n_per_axis, n_cells, kRepetitions, median_s * 1e3,
              cell_updates_per_s);
}

}  // namespace

int main()
{
  std::printf("backend,dim,n_per_axis,n_cells,repetitions,median_ms,cell_updates_per_s\n");
  for (std::size_t n : {std::size_t(64), std::size_t(128), std::size_t(256), std::size_t(400),
                        std::size_t(512)}) {
    run_case(n);
  }
  return 0;
}
