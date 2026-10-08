// Large-scale CUDA Burgers benchmark, mirroring
// bench_scalar_advection_cuda.cu's structure and resolution sweep
// exactly (1M through 10^8 cells), with MusclMinmodReconstruction +
// RusanovFlux in place of CentralDifferenceReconstruction + UpwindFlux.
// Same smooth-periodic-IC convention as bench_burgers.cpp -- see that
// file's header comment for why.
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
constexpr Scalar kAmplitudeOffset = Scalar(1.0);
constexpr Scalar kAmplitudeWave = Scalar(0.5);
constexpr int kRepetitions = 10;

void run_case(std::size_t nx)
{
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = Scalar(1.0) / static_cast<Scalar>(nx);

  std::vector<Scalar> host_ic(grid.n_cells_total(), Scalar(0.0));
  Scalar max_abs_u0 = Scalar(0.0);
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const Scalar x = grid.x_center(grid.ngx + i);
    const Scalar value = kAmplitudeOffset + kAmplitudeWave * std::sin(Scalar(2.0) * kPi * x);
    host_ic[grid.flat_index(grid.ngx + i, 0, 0)] = value;
    max_abs_u0 = std::max(max_abs_u0, std::abs(value));
  }

  cfe::backend::cuda::DeviceField<Scalar, 1> state(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<Scalar, 1> stage1(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<Scalar, 1> scratch(grid.n_cells_total());
  state.copy_from_host(host_ic.data());

  cfe::BurgersField<Scalar, 1> field{};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::BurgersField<Scalar, 1>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux, cfe::CudaParallelFor>
      solver{grid, field, cfe::PeriodicBoundary{}};
  cfe::SolverResidual<decltype(solver)> residual{&solver};

  const Scalar dt = Scalar(0.4) * grid.dx / max_abs_u0;
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
