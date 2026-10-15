// Large-scale 3D CUDA Euler benchmark, mirroring
// bench_burgers_3d_cuda.cu's/bench_scalar_advection_3d_cuda.cu's
// structure and cube-resolution sweep exactly (up to 512^3, ~134M
// cells), with `EulerField<Scalar,3>` (`n_components = 5`) +
// MusclMinmodReconstruction + RusanovFlux -- demonstrates the fully
// dimension-generic Euler solver (all three of X/Y/Z's
// `axis_flux_difference` branches active every step, on a genuinely
// multi-component Field) at scale on real GPU hardware.
//
// Smooth, strictly-positive-density/pressure periodic initial
// condition (not the Sod tube's discontinuity) -- same convention
// bench_euler_cuda.cu's own header comment explains, extended to a
// product of three sine waves (one per axis, matching
// bench_burgers_3d_cuda.cu's own 3D IC) so all three axes carry
// genuinely non-zero velocity and none degenerates into the inert
// zero-flux special case test_euler_3d_sanity.cpp deliberately uses for
// its own (different) purpose.
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
using Field = cfe::EulerField<Scalar, 3>;
constexpr std::size_t kN = Field::n_components;  // 5

constexpr Scalar kPi = Scalar(3.14159265358979323846);
constexpr Scalar kBackground = Scalar(1.0);
constexpr Scalar kAmplitude = Scalar(0.2);
constexpr Scalar kVelocityAmplitude = Scalar(0.1);
constexpr Scalar kGamma = Scalar(1.4);
// How many times each case is timed, so a median can be taken (see below).
constexpr int kRepetitions = 10;

// Runs and times ONE benchmark case (one cube size, on the GPU), then
// prints one CSV row with the result. Called once per cube size from
// main() below.
void run_case(cfe::Index n_per_axis)
{
  // Build a 3D grid of `n_per_axis^3` cells covering the unit cube
  // [0,1]^3, with 2 ghost cells padding every face in every direction.
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

  Field field{};
  field.gamma = kGamma;

  // Build the smooth initial condition on the CPU first (`host_ic`):
  // rho = 1 + 0.2*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z), a velocity
  // vector with a nonzero component on every axis (same sine-product
  // shape, each axis a different phase so none of X/Y/Z is a disguised
  // copy of another), and pressure using the same sine product --
  // stays safely positive everywhere, same reasoning as the 1D
  // benchmark's own comment. Also measures the actual per-cell max wave
  // speed (|u_axis much larger than the others never assumed -- the
  // full 3-vector's magnitude plus sound speed bounds every axis at
  // once), for a stable, non-arbitrary time-step choice.
  std::vector<Scalar> host_ic(grid.n_cells_total() * kN, Scalar(0.0));
  Scalar max_speed = Scalar(0.0);
  for (cfe::Index k = 0; k < grid.nz; ++k) {
    for (cfe::Index j = 0; j < grid.ny; ++j) {
      for (cfe::Index i = 0; i < grid.nx; ++i) {
        const Scalar x = grid.x_center(grid.ngx + i);
        const Scalar y = grid.y_center(grid.ngy + j);
        const Scalar z = grid.z_center(grid.ngz + k);
        const Scalar wave = std::sin(Scalar(2.0) * kPi * x) * std::sin(Scalar(2.0) * kPi * y) *
                             std::sin(Scalar(2.0) * kPi * z);
        const Scalar rho = kBackground + kAmplitude * wave;
        const Scalar p = kBackground + kAmplitude * wave;

        cfe::Vector<Scalar, 3> velocity;
        velocity[0] = kVelocityAmplitude * wave;
        velocity[1] = Scalar(0.6) * kVelocityAmplitude * wave;
        velocity[2] = Scalar(0.3) * kVelocityAmplitude * wave;

        const cfe::State<Scalar, kN> q = field.conserved_from_primitive(rho, velocity, p);
        const cfe::Index cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
        for (cfe::Index c = 0; c < kN; ++c) host_ic[cell * kN + c] = q[c];

        const Scalar c_sound = field.sound_speed(q);
        const Scalar speed_x = (velocity[0] < Scalar(0) ? -velocity[0] : velocity[0]) + c_sound;
        const Scalar speed_y = (velocity[1] < Scalar(0) ? -velocity[1] : velocity[1]) + c_sound;
        const Scalar speed_z = (velocity[2] < Scalar(0) ? -velocity[2] : velocity[2]) + c_sound;
        const Scalar speed_sum = speed_x + speed_y + speed_z;
        max_speed = speed_sum > max_speed ? speed_sum : max_speed;
      }
    }
  }

  // `DeviceField` is the GPU-memory counterpart of `cfe::Field`.
  // `copy_from_host` copies the initial condition from `host_ic` (CPU
  // RAM) into `state` (GPU memory) once, up front; everything from here
  // on runs on the GPU without further transfers.
  cfe::backend::cuda::DeviceField<Scalar, kN> state(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<Scalar, kN> stage1(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<Scalar, kN> scratch(grid.n_cells_total());
  state.copy_from_host(host_ic.data());

  // `field` picks which equation is being solved (3D compressible
  // Euler). `solver` bundles that equation together with the grid, the
  // boundary rule on all three axes (`PeriodicBoundary`: each face
  // wraps around to the opposite one), the numerical method, and
  // `CudaParallelFor` telling it to run its per-cell work on the GPU.
  cfe::FvmSolver<Scalar, cfe::AoSLayout, Field, cfe::PeriodicBoundary, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux,
                 cfe::CudaParallelFor>
      solver{grid, field, cfe::PeriodicBoundary{}};
  // `SolverResidual` wraps `solver.residual(...)` as a callable object
  // -- the GPU equivalent of a plain lambda (nvcc, the CUDA compiler,
  // does not allow a lambda defined inside a function to be passed
  // where this one is used below).
  cfe::SolverResidual<decltype(solver)> residual{&solver};

  // Multi-axis CFL: dt must respect the SUM of per-axis Courant numbers
  // (same reasoning test_scalar_advection_3d_cuda.cu's/
  // bench_burgers_3d_cuda.cu's dt use), bounded by the actual measured
  // per-cell `speed_x + speed_y + speed_z` above -- a cell's residual
  // combines flux contributions from all three active axes every
  // stage. `one_step` is a closure that advances `state` forward by
  // exactly one such step. `synchronize()` blocks until the GPU has
  // actually finished that work.
  const Scalar dt = Scalar(0.25) * grid.dx / max_speed;
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
  // (the middle value once sorted) as the reported time.
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
  const double n_cells =
      static_cast<double>(grid.nx) * static_cast<double>(grid.ny) * static_cast<double>(grid.nz);
  const double cell_updates_per_s = n_cells / median_s;

  // Print one CSV row: backend name, dimensionality, cells per axis,
  // total cells, repetitions, median time in milliseconds, throughput.
  std::printf("cuda,3,%zu,%.0f,%d,%.6f,%.3e\n", n_per_axis, n_cells, kRepetitions, median_s * 1e3,
              cell_updates_per_s);
}

}  // namespace

// Entry point: prints a CSV header, then runs the benchmark once for
// each cube size in the list below, printing one CSV row per size.
int main()
{
  std::printf("backend,dim,n_per_axis,n_cells,repetitions,median_ms,cell_updates_per_s\n");
  for (cfe::Index n : {cfe::Index(64), cfe::Index(128), cfe::Index(256), cfe::Index(400),
                        cfe::Index(512)}) {
    run_case(n);
  }
  return 0;
}
