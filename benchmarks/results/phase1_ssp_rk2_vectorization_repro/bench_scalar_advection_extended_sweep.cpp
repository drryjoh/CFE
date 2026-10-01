// CPU benchmark for the scalar-advection solver (task spec item 11:
// "runtime / cell-updates-per-second for the advection kernel, swept
// across a few grid resolutions ... on CPU"). Component count stays at
// 1 for this task (single transported scalar), so grid resolution is
// the scaling axis here, unlike bench_field_update.cpp's component-count
// sweep.
//
// Sweeps both CPU backends directly (cfe::backend::serial /
// cfe::backend::threaded), the same way bench_field_update.cpp does, via
// two small local Backend tags -- not `cfe::CpuParallelFor`, which
// resolves to whichever single backend was picked as the project default
// (see backend/parallel_for.hpp) and so can't sweep both in one binary.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cfe/backend/cpu/serial.hpp"
#include "cfe/backend/cpu/threaded.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/scalar_advection/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
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
constexpr double kSpeed = 1.0;
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
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const double x = grid.x_center(grid.ngx + i);
    state(grid.flat_index(grid.ngx + i, 0, 0), 0) = std::sin(2.0 * kPi * x);
  }

  cfe::ScalarAdvectionField<double, 1> field{cfe::Vector<double, 1>(kSpeed)};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 1>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::CentralDifferenceReconstruction,
                 cfe::UpwindFlux, Backend>
      solver{grid, field, cfe::PeriodicBoundary{}};
  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver.residual(in, out);
  };

  const double dt = 0.4 * grid.dx / kSpeed;
  auto one_step = [&]() {
    cfe::ssp_rk2_step<double, cfe::FieldView<double, 1>, decltype(residual), Backend>(
        state.view(), stage1.view(), scratch.view(), dt, residual);
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
                        std::size_t(10'000'000), std::size_t(50'000'000), std::size_t(100'000'000)}) {
    run_case<SerialBackend>("serial", nx);
    run_case<ThreadedBackend>("threaded", nx);
  }
  return 0;
}
