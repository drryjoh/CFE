// Large-scale CUDA Euler benchmark (the "at scale, on GPU" evidence
// requested as a direct follow-up to task 0009 -- compressible Euler,
// MUSCL/Rusanov, Sod shock tube, PR review -- mirroring
// bench_burgers_cuda.cu's/bench_scalar_advection_cuda.cu's structure
// and resolution sweep exactly (1M through 10^8 cells). Unlike those
// benchmarks (N=1 fields), `EulerField<Scalar,1>` has
// `n_components = 3` (density, momentum_x, total energy), so this is
// also this codebase's first at-scale GPU throughput measurement for a
// genuinely multi-component Field/vector RusanovFlux.
//
// Smooth, strictly-positive-density/pressure periodic initial
// condition (not the Sod tube's discontinuity) -- same "smooth IC for a
// throughput benchmark, discontinuous IC reserved for correctness
// tests" convention bench_burgers_cuda.cu's own header comment
// explains: rho(x) = 1 + 0.2*sin(2*pi*x), u(x) = 0.1*sin(2*pi*x),
// p(x) = 1 + 0.2*sin(2*pi*x) stays safely positive in both density and
// pressure everywhere (amplitude 0.2 against a background of 1.0), so
// no case-specific equation-of-state failure can occur mid-sweep.
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
#include "cfe/fields/euler/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace {

// A short, local name standing in for whichever type `cfe::scalar`
// currently is (`float` or `double` -- see this file's header comment).
using Scalar = cfe::scalar;
using Field = cfe::EulerField<Scalar, 1>;
constexpr std::size_t kN = Field::n_components;  // 3

constexpr Scalar kPi = Scalar(3.14159265358979323846);
constexpr Scalar kBackground = Scalar(1.0);
constexpr Scalar kAmplitude = Scalar(0.2);
constexpr Scalar kVelocityAmplitude = Scalar(0.1);
constexpr Scalar kGamma = Scalar(1.4);
// How many times each case is timed, so a median can be taken (see below).
constexpr int kRepetitions = 10;

// Runs and times ONE benchmark case (one grid size, on the GPU), then
// prints one CSV row with the result. Called once per grid size from
// main() below.
void run_case(std::size_t nx)
{
  // Build a 1D grid of `nx` cells, each `grid.dx` wide, covering [0,1].
  // `ngx = 2` ghost cells pad each end -- the solver's stencil reaches 2
  // cells either side of the one it's updating.
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = Scalar(1.0) / static_cast<Scalar>(nx);

  // `field` picks which equation is being solved (1D compressible
  // Euler, gamma-law gas). Build the smooth initial condition on the
  // CPU first (`host_ic`), converting primitive (density, velocity,
  // pressure) to conserved variables via the Field's own Calculator --
  // this is also where the actual per-cell max wave speed is measured
  // (not just assumed), for a stable, non-arbitrary time-step choice.
  Field field{};
  field.gamma = kGamma;

  std::vector<Scalar> host_ic(grid.n_cells_total() * kN, Scalar(0.0));
  Scalar max_speed = Scalar(0.0);
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const Scalar x = grid.x_center(grid.ngx + i);
    const Scalar wave = std::sin(Scalar(2.0) * kPi * x);
    const Scalar rho = kBackground + kAmplitude * wave;
    const Scalar u = kVelocityAmplitude * wave;
    const Scalar p = kBackground + kAmplitude * wave;

    cfe::Vector<Scalar, 1> velocity;
    velocity[0] = u;
    const cfe::State<Scalar, kN> q = field.conserved_from_primitive(rho, velocity, p);
    const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
    for (std::size_t c = 0; c < kN; ++c) host_ic[cell * kN + c] = q[c];

    const Scalar c_sound = field.sound_speed(q);
    const Scalar speed = (u < Scalar(0) ? -u : u) + c_sound;
    max_speed = speed > max_speed ? speed : max_speed;
  }

  // `DeviceField` is the GPU-memory counterpart of `cfe::Field` --
  // `n_cells * kN` numbers, but living in the GPU's own memory, not the
  // CPU's. `copy_from_host` copies the initial condition from
  // `host_ic` (CPU RAM) into `state` (GPU memory) once, up front;
  // everything from here on runs on the GPU without further transfers.
  cfe::backend::cuda::DeviceField<Scalar, kN> state(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<Scalar, kN> stage1(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<Scalar, kN> scratch(grid.n_cells_total());
  state.copy_from_host(host_ic.data());

  // `solver` bundles the equation together with the grid, the boundary
  // rule (`PeriodicBoundary`: the right edge wraps around to the left),
  // the numerical method (MUSCL-minmod limiter + Rusanov flux -- the
  // same shock-capturing stack the Sod tube verification test uses),
  // and `CudaParallelFor` telling it to run its per-cell work on the
  // GPU.
  cfe::FvmSolver<Scalar, cfe::AoSLayout, Field, cfe::PeriodicBoundary, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux,
                 cfe::CudaParallelFor>
      solver{grid, field, cfe::PeriodicBoundary{}};
  // `SolverResidual` wraps `solver.residual(...)` as a callable object
  // -- the GPU equivalent of the plain lambda the CPU benchmark uses
  // (nvcc, the CUDA compiler, does not allow a lambda defined inside a
  // function to be passed where this one is used below).
  cfe::SolverResidual<decltype(solver)> residual{&solver};

  // The time-step size, chosen small enough for numerical stability
  // (the CFL condition: a wave must not cross more than ~0.4 of one
  // cell per step) from the ACTUAL measured max wave speed of this
  // initial condition (not a guessed constant). `one_step` is a closure
  // that advances `state` forward by exactly one such step, using a
  // 2-stage Runge-Kutta method. `synchronize()` blocks until the GPU
  // has actually finished that work -- GPU launches return immediately
  // otherwise, which would make the timing below measure "time to queue
  // the work", not "time to do it".
  const Scalar dt = Scalar(0.4) * grid.dx / max_speed;
  auto one_step = [&]() {
    cfe::ssp_rk2_step<Scalar, cfe::FieldView<Scalar, kN>, decltype(residual), cfe::CudaParallelFor>(
        state.view(), stage1.view(), scratch.view(), dt, residual, solver.active_cell_count(),
        solver.active_cell_index_map());
    cfe::backend::cuda::synchronize();
  };

  // Run (and discard) one step before timing anything: the very first
  // GPU launch pays a one-time cost (CUDA context setup, just-in-time
  // compilation) that has nothing to do with the solver's steady-state
  // speed.
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
