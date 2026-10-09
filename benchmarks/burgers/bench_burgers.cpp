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
//
// Simulation precision is `cfe::scalar` (core/types.hpp, project-wide
// via the `CFE_SCALAR_TYPE` CMake cache variable, `double` by default)
// -- NOT a hardcoded `double` -- matching AGENTS.md #11. Wall-clock
// timing (`seconds`/`median_s`) stays `double` regardless -- that is a
// measurement precision, not a simulation one.

// Standard C++ library headers (not specific to this project).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

// This project's own headers -- each one brings in one piece of the solver.
#include "cfe/backend/cpu/serial.hpp"
#include "cfe/backend/cpu/threaded.hpp"
#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/burgers/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

// An anonymous namespace: everything below, until the matching `}`, is
// private to this file only -- no other .cpp file can see or call it.
// Closest Python analogy: names that are never exported from a module.
namespace {

// A short, local name standing in for whichever type `cfe::scalar`
// currently is (`float` or `double` -- see this file's header comment).
using Scalar = cfe::scalar;

// A "backend tag": a type with no data, whose only job is to say HOW to
// loop over cells. `run(n, f)` calls `f(i)` once for each `i` in
// `[0, n)`. In Python terms, this plays the role a function argument
// would (e.g. `backend=threaded`); C++ uses a type here instead of a
// value because the rest of the solver is written as a template that
// picks its looping strategy at compile time, not at run time.
struct SerialBackend
{
  template <class Index, class Functor>
  static void run(Index n, Functor f)
  {
    cfe::backend::serial::parallel_for(n, f);
  }
};

// Same idea as SerialBackend, but `run` spreads the `n` calls to `f`
// across multiple CPU threads instead of looping on just one.
struct ThreadedBackend
{
  template <class Index, class Functor>
  static void run(Index n, Functor f)
  {
    cfe::backend::threaded::parallel_for(n, f);
  }
};

// Parameters for the initial wave shape u0(x) = offset + wave*sin(2*pi*x).
constexpr Scalar kPi = Scalar(3.14159265358979323846);
constexpr Scalar kAmplitudeOffset = Scalar(1.0);
constexpr Scalar kAmplitudeWave = Scalar(0.5);
// How many times each case is timed, so a median can be taken (see below).
constexpr int kRepetitions = 7;

// Runs and times ONE benchmark case (one grid size, one backend), then
// prints one CSV row with the result. Called once per (backend, nx)
// combination from main() below.
template <class Backend>
void run_case(const char* backend_name, std::size_t nx)
{
  // Build a 1D grid of `nx` cells, each `grid.dx` wide, covering [0,1].
  // `ngx = 2` ghost cells pad each end -- needed because the solver's
  // stencil reaches 2 cells either side of the one it's updating.
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = Scalar(1.0) / static_cast<Scalar>(nx);

  // Three arrays of `nx` numbers each (plus their ghost cells), the
  // same shape as a 1D numpy array: `state` is the simulation's current
  // values; `stage1` and `scratch` are scratch space the time-stepping
  // algorithm uses internally and never needs to be read directly.
  cfe::Field<Scalar, 1> state(grid.n_cells_total());
  cfe::Field<Scalar, 1> stage1(grid.n_cells_total());
  cfe::Field<Scalar, 1> scratch(grid.n_cells_total());
  Scalar max_abs_u0 = Scalar(0.0);
  // Fill `state` with the initial wave: state[i] = offset + wave*sin(2*pi*x_i),
  // the same formula a numpy one-liner `offset + wave*np.sin(2*np.pi*x)`
  // would compute. Also remember the largest |value| seen, needed below
  // to pick a stable time-step size.
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const Scalar x = grid.x_center(grid.ngx + i);
    const Scalar value = kAmplitudeOffset + kAmplitudeWave * std::sin(Scalar(2.0) * kPi * x);
    state(grid.flat_index(grid.ngx + i, 0, 0), 0) = value;
    max_abs_u0 = std::max(max_abs_u0, std::abs(value));
  }

  // `field` picks WHICH equation is being solved (Burgers' equation).
  // `solver` bundles that equation together with the grid, the boundary
  // rule (`PeriodicBoundary`: the right edge wraps around to the left),
  // the numerical method (limiter + flux scheme), and which `Backend`
  // (serial or threaded, above) should do the actual looping.
  cfe::BurgersField<Scalar, 1> field{};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::BurgersField<Scalar, 1>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux, Backend>
      solver{grid, field, cfe::PeriodicBoundary{}};
  // `residual` is a small function: given the current values (`in`), it
  // writes the instantaneous rate of change into `out` -- the right-hand
  // side `f(u)` of the ODE `du/dt = f(u)` the time-stepper integrates.
  auto residual = [&](cfe::FieldView<Scalar, 1> in, cfe::FieldView<Scalar, 1> out) {
    solver.residual(in, out);
  };

  // The time-step size, chosen small enough for numerical stability
  // (the CFL condition: a wave must not cross more than ~0.4 of one
  // cell per step). `one_step` is a closure (like a Python lambda
  // capturing variables from the enclosing scope) that advances `state`
  // forward by exactly one such step, using a 2-stage Runge-Kutta method.
  const Scalar dt = Scalar(0.4) * grid.dx / max_abs_u0;
  auto one_step = [&]() {
    cfe::ssp_rk2_step<Scalar, cfe::FieldView<Scalar, 1>, decltype(residual), Backend>(
        state.view(), stage1.view(), scratch.view(), dt, residual, solver.active_cell_count(),
        solver.active_cell_index_map());
  };

  // Run (and discard) one step before timing anything: the very first
  // call pays one-time costs (first-touch memory faults, spinning up
  // the thread pool for ThreadedBackend) that have nothing to do with
  // the solver's steady-state speed, and would otherwise make the very
  // first timed repetition look artificially slow.
  one_step();

  // Time `kRepetitions` more steps individually, then take the median
  // (the middle value once sorted) as the reported time -- more robust
  // to a single slow outlier than a plain average would be.
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

  // How many cells this backend updates per second of wall-clock time.
  const double cell_updates_per_s = static_cast<double>(grid.nx) / median_s;

  // Print one CSV row: backend name, grid size, repetitions, median
  // time in milliseconds, throughput.
  std::printf("%s,%zu,%d,%.6f,%.3e\n", backend_name, grid.nx, kRepetitions, median_s * 1e3,
              cell_updates_per_s);
}

}  // namespace

// Entry point: prints a CSV header, then for each grid size in the list
// below, runs the benchmark once single-threaded and once
// multi-threaded, printing one CSV row per (backend, size) combination.
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
