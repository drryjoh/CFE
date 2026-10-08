// Large-scale 3D CUDA Burgers benchmark, mirroring
// bench_scalar_advection_3d_cuda.cu's structure and cube-resolution
// sweep exactly (up to 512^3, ~134M cells), with
// MusclMinmodReconstruction + RusanovFlux in place of
// CentralDifferenceReconstruction + UpwindFlux -- demonstrates the
// fully dimension-generic Burgers solver (all three of X/Y/Z's
// `axis_flux_difference` branches active every step) at scale on real
// GPU hardware.
//
// Simulation precision is `cfe::scalar` (core/types.hpp, project-wide
// via the `CFE_SCALAR_TYPE` CMake cache variable, `double` by default)
// -- NOT a hardcoded `double` -- matching AGENTS.md #11. Wall-clock
// timing (`seconds`/`median_s`) stays `double` regardless.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cfe/backend/cuda/cuda_backend.cuh"
#include "cfe/backend/cuda/device_field.cuh"
#include "cfe/core/types.hpp"
#include "cfe/fields/burgers/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace {

using Scalar = cfe::scalar;

constexpr Scalar kPi = Scalar(3.14159265358979323846);
constexpr Scalar kAmplitudeOffset = Scalar(2.0);  // kept positive (sign never changes) across the whole product IC
constexpr Scalar kAmplitudeWave = Scalar(1.0);
constexpr int kRepetitions = 10;

void run_case(std::size_t n_per_axis)
{
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = n_per_axis;
  grid.ny = n_per_axis;
  grid.nz = n_per_axis;
  grid.ngx = 2;
  grid.ngy = 2;
  grid.ngz = 2;
  grid.dx = Scalar(1.0) / static_cast<Scalar>(n_per_axis);
  grid.dy = grid.dx;
  grid.dz = grid.dx;

  std::vector<Scalar> host_ic(grid.n_cells_total(), Scalar(0.0));
  Scalar max_abs_u0 = Scalar(0.0);
  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const Scalar x = grid.x_center(grid.ngx + i);
        const Scalar y = grid.y_center(grid.ngy + j);
        const Scalar z = grid.z_center(grid.ngz + k);
        const Scalar value = kAmplitudeOffset +
                              kAmplitudeWave * std::sin(Scalar(2.0) * kPi * x) *
                                  std::sin(Scalar(2.0) * kPi * y) * std::sin(Scalar(2.0) * kPi * z);
        host_ic[grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k)] = value;
        max_abs_u0 = std::max(max_abs_u0, std::abs(value));
      }
    }
  }

  cfe::backend::cuda::DeviceField<Scalar, 1> state(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<Scalar, 1> stage1(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<Scalar, 1> scratch(grid.n_cells_total());
  state.copy_from_host(host_ic.data());

  cfe::BurgersField<Scalar, 3> field{};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::BurgersField<Scalar, 3>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux, cfe::CudaParallelFor>
      solver{grid, field, cfe::PeriodicBoundary{}};
  cfe::SolverResidual<decltype(solver)> residual{&solver};

  // Multi-axis CFL: dt must respect the sum of per-axis Courant numbers
  // (same reasoning test_scalar_advection_3d_cuda.cu's dt uses), with
  // the per-axis speed bounded by max|u0| (Burgers' own characteristic
  // speed is the state itself, same axis-independent flux on every
  // axis).
  const Scalar dt = Scalar(0.25) * grid.dx / (Scalar(3.0) * max_abs_u0);
  auto one_step = [&]() {
    cfe::ssp_rk2_step<Scalar, cfe::FieldView<Scalar, 1>, decltype(residual), cfe::CudaParallelFor>(
        state.view(), stage1.view(), scratch.view(), dt, residual, solver.active_cell_count(),
        solver.active_cell_index_map());
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
