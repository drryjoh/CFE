// CPU benchmark for the Burgers solver (MusclMinmodReconstruction +
// RusanovFlux), mirroring bench_scalar_advection.cpp's own structure and
// conventions exactly (same two local Backend tags, same repetition/
// warm-up convention, same CSV shape) -- grid resolution is the scaling
// axis, component count stays at 1.
//
// Uses a smooth periodic IC (not the Riemann shock
// test_burgers_shock_formation.cpp verifies correctness against): this
// benchmark measures raw per-cell kernel throughput, which the
// correctness tests already established is scheme-correct regardless of
// IC, so a smooth/periodic setup (matching every other benchmark in this
// codebase) avoids introducing StaticBoundary's slightly different
// ghost-fill cost as a confound in the comparison against
// bench_scalar_advection.cpp's own numbers.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cfe/backend/cpu/serial.hpp"
#include "cfe/backend/cpu/threaded.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/burgers/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace {

struct SerialBackend
{
  template <class Index, class Functor>
  static void run(Index n, Functor f)
  {
    cfe::backend::serial::parallel_for(n, f);
  }
};

struct ThreadedBackend
{
  template <class Index, class Functor>
  static void run(Index n, Functor f)
  {
    cfe::backend::threaded::parallel_for(n, f);
  }
};

constexpr double kPi = 3.14159265358979323846;
constexpr double kAmplitudeOffset = 1.0;
constexpr double kAmplitudeWave = 0.5;
constexpr int kRepetitions = 7;

template <class Backend>
void run_case(const char* backend_name, std::size_t nx)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = 1.0 / static_cast<double>(nx);

  cfe::Field<double, 1> state(grid.n_cells_total());
  cfe::Field<double, 1> stage1(grid.n_cells_total());
  cfe::Field<double, 1> scratch(grid.n_cells_total());
  double max_abs_u0 = 0.0;
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const double x = grid.x_center(grid.ngx + i);
    const double value = kAmplitudeOffset + kAmplitudeWave * std::sin(2.0 * kPi * x);
    state(grid.flat_index(grid.ngx + i, 0, 0), 0) = value;
    max_abs_u0 = std::max(max_abs_u0, std::abs(value));
  }

  cfe::BurgersField<double, 1> field{};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::BurgersField<double, 1>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux, Backend>
      solver{grid, field, cfe::PeriodicBoundary{}};
  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver.residual(in, out);
  };

  const double dt = 0.4 * grid.dx / max_abs_u0;
  auto one_step = [&]() {
    cfe::ssp_rk2_step<double, cfe::FieldView<double, 1>, decltype(residual), Backend>(
        state.view(), stage1.view(), scratch.view(), dt, residual, solver.active_cell_count(),
        solver.active_cell_index_map());
  };

  one_step();  // warm-up: first-touch faulting, thread-pool spin-up.

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

  std::printf("%s,%zu,%d,%.6f,%.3e\n", backend_name, grid.nx, kRepetitions, median_s * 1e3,
              cell_updates_per_s);
}

}  // namespace

int main()
{
  std::printf("backend,n_cells,repetitions,median_ms,cell_updates_per_s\n");
  for (std::size_t nx : {std::size_t(10'000), std::size_t(100'000), std::size_t(1'000'000),
                        std::size_t(10'000'000)}) {
    run_case<SerialBackend>("serial", nx);
    run_case<ThreadedBackend>("threaded", nx);
  }
  return 0;
}
