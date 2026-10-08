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

// Standard C++ library headers (not specific to this project).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

// This project's own headers -- each one brings in one piece of the solver.
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

// An anonymous namespace: everything below, until the matching `}`, is
// private to this file only -- no other .cu/.cpp file can see it.
namespace {

// A short, local name standing in for whichever type `cfe::scalar`
// currently is (`float` or `double` -- see this file's header comment).
using Scalar = cfe::scalar;

// Parameters for the initial wave shape u0(x) = offset + wave*sin(2*pi*x).
constexpr Scalar kPi = Scalar(3.14159265358979323846);
constexpr Scalar kAmplitudeOffset = Scalar(1.0);
constexpr Scalar kAmplitudeWave = Scalar(0.5);
// How many times each case is timed, so a median can be taken (see below).
constexpr int kRepetitions = 10;

// Runs and times ONE benchmark case (one grid size, on the GPU), then
// prints one CSV row with the result. Called once per grid size from
// main() below.
void run_case(std::size_t nx)
{
  // Build a 1D grid of `nx` cells, each `grid.dx` wide, covering [0,1].
  // `ngx = 2` ghost cells pad each end -- needed because the solver's
  // stencil reaches 2 cells either side of the one it's updating.
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = Scalar(1.0) / static_cast<Scalar>(nx);

  // Build the initial wave on the CPU first (`host_ic`, an ordinary
  // array in regular RAM): host_ic[i] = offset + wave*sin(2*pi*x_i), the
  // same formula a numpy one-liner `offset + wave*np.sin(2*np.pi*x)`
  // would compute. Also remember the largest |value| seen, needed below
  // to pick a stable time-step size.
  std::vector<Scalar> host_ic(grid.n_cells_total(), Scalar(0.0));
  Scalar max_abs_u0 = Scalar(0.0);
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const Scalar x = grid.x_center(grid.ngx + i);
    const Scalar value = kAmplitudeOffset + kAmplitudeWave * std::sin(Scalar(2.0) * kPi * x);
    host_ic[grid.flat_index(grid.ngx + i, 0, 0)] = value;
    max_abs_u0 = std::max(max_abs_u0, std::abs(value));
  }

  // `DeviceField` is the GPU-memory counterpart of `cfe::Field` --
  // three arrays of `nx` numbers each, but living in the GPU's own
  // memory, not the CPU's. `copy_from_host` copies the initial wave
  // from `host_ic` (CPU RAM) into `state` (GPU memory) once, up front;
  // everything from here on runs on the GPU without further transfers.
  cfe::backend::cuda::DeviceField<Scalar, 1> state(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<Scalar, 1> stage1(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<Scalar, 1> scratch(grid.n_cells_total());
  state.copy_from_host(host_ic.data());

  // `field` picks WHICH equation is being solved (Burgers' equation).
  // `solver` bundles that equation together with the grid, the boundary
  // rule (`PeriodicBoundary`: the right edge wraps around to the left),
  // the numerical method (limiter + flux scheme), and `CudaParallelFor`
  // telling it to run its per-cell work on the GPU.
  cfe::BurgersField<Scalar, 1> field{};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::BurgersField<Scalar, 1>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux, cfe::CudaParallelFor>
      solver{grid, field, cfe::PeriodicBoundary{}};
  // `SolverResidual` wraps `solver.residual(...)` as a callable object
  // -- the GPU equivalent of the plain lambda the CPU benchmark uses
  // (nvcc, the CUDA compiler, does not allow a lambda defined inside a
  // function to be passed where this one is used below).
  cfe::SolverResidual<decltype(solver)> residual{&solver};

  // The time-step size, chosen small enough for numerical stability
  // (the CFL condition: a wave must not cross more than ~0.4 of one
  // cell per step). `one_step` is a closure (like a Python lambda
  // capturing variables from the enclosing scope) that advances `state`
  // forward by exactly one such step, using a 2-stage Runge-Kutta
  // method. `synchronize()` blocks until the GPU has actually finished
  // that work -- GPU launches return immediately otherwise, which would
  // make the timing below measure "time to queue the work", not "time
  // to do it".
  const Scalar dt = Scalar(0.4) * grid.dx / max_abs_u0;
  auto one_step = [&]() {
    cfe::ssp_rk2_step<Scalar, cfe::FieldView<Scalar, 1>, decltype(residual), cfe::CudaParallelFor>(
        state.view(), stage1.view(), scratch.view(), dt, residual, solver.active_cell_count(),
        solver.active_cell_index_map());
    cfe::backend::cuda::synchronize();
  };

  // Run (and discard) one step before timing anything: the very first
  // GPU launch pays a one-time cost (CUDA context setup, just-in-time
  // compilation) that has nothing to do with the solver's steady-state
  // speed, and would otherwise make the first timed repetition look
  // artificially slow.
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

  // How many cells the GPU updates per second of wall-clock time.
  const double cell_updates_per_s = static_cast<double>(grid.nx) / median_s;

  // Print one CSV row: backend name, grid size, repetitions, median
  // time in milliseconds, throughput.
  std::printf("cuda,%zu,%d,%.6f,%.3e\n", grid.nx, kRepetitions, median_s * 1e3, cell_updates_per_s);
}

}  // namespace

// Entry point: prints a CSV header, then runs the benchmark once for
// each grid size in the list below, printing one CSV row per size.
int main()
{
  std::printf("backend,n_cells,repetitions,median_ms,cell_updates_per_s\n");
  for (std::size_t nx : {std::size_t(1'000'000), std::size_t(10'000'000), std::size_t(50'000'000),
                        std::size_t(100'000'000)}) {
    run_case(nx);
  }
  return 0;
}
