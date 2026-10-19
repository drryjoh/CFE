// Large-scale 3D CUDA HLLC benchmark (task 0011,
// docs/adr/0012-hllc-numerical-flux.md), mirroring
// bench_euler_3d_cuda.cu's exact structure and cube-resolution sweep
// (up to 512^3, ~134M cells) with `RusanovFlux` swapped for `HllcFlux`
// -- the standing per-PR requirement ("every PR adding a Field/scheme
// needs an at-scale GPU benchmark, not just a correctness test") applied
// to HLLC specifically, closing the gap left open when
// test_euler_hllc_3d_cuda.cu landed without this companion benchmark
// (caught in Ilia's final review).
//
// Smooth, strictly-positive-density/pressure periodic initial condition
// (not the Sod tube's discontinuity) -- identical IC to
// bench_euler_3d_cuda.cu's own, so the two benchmarks differ in ONLY
// the numerical flux, making the reported throughput ratio a direct,
// apples-to-apples per-face-cost comparison (unlike the forward-step
// tutorial's own wall-clock comparison, which docs/adr/0012-...md's
// Revisit criteria flags as methodologically confounded -- this
// benchmark is the isolated, repeated-median measurement that
// confound called for).
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
#include "cfe/fields/euler/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/hllc.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace {

using Scalar = cfe::scalar;
using Field = cfe::EulerField<Scalar, 3>;
constexpr std::size_t kN = Field::n_components;  // 5

constexpr Scalar kPi = Scalar(3.14159265358979323846);
constexpr Scalar kBackground = Scalar(1.0);
constexpr Scalar kAmplitude = Scalar(0.2);
constexpr Scalar kVelocityAmplitude = Scalar(0.1);
constexpr Scalar kGamma = Scalar(1.4);
constexpr int kRepetitions = 10;

void run_case(cfe::Index n_per_axis)
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

  Field field{};
  field.gamma = kGamma;

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

  cfe::backend::cuda::DeviceField<Scalar, kN> state(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<Scalar, kN> stage1(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<Scalar, kN> scratch(grid.n_cells_total());
  state.copy_from_host(host_ic.data());

  cfe::FvmSolver<Scalar, cfe::AoSLayout, Field, cfe::PeriodicBoundary, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction, cfe::HllcFlux,
                 cfe::CudaParallelFor>
      solver{grid, field, cfe::PeriodicBoundary{}};
  cfe::SolverResidual<decltype(solver)> residual{&solver};

  const Scalar dt = Scalar(0.25) * grid.dx / max_speed;
  auto one_step = [&]() {
    cfe::ssp_rk2_step<Scalar, cfe::FieldView<Scalar, kN>, decltype(residual), cfe::CudaParallelFor>(
        state.view(), stage1.view(), scratch.view(), dt, residual, solver.active_cell_count(),
        solver.active_cell_index_map());
    cfe::backend::cuda::synchronize();
  };

  one_step();

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

  const double n_cells =
      static_cast<double>(grid.nx) * static_cast<double>(grid.ny) * static_cast<double>(grid.nz);
  const double cell_updates_per_s = n_cells / median_s;

  std::printf("cuda,3,%zu,%.0f,%d,%.6f,%.3e\n", n_per_axis, n_cells, kRepetitions, median_s * 1e3,
              cell_updates_per_s);
}

}  // namespace

int main()
{
  std::printf("backend,dim,n_per_axis,n_cells,repetitions,median_ms,cell_updates_per_s\n");
  for (cfe::Index n : {cfe::Index(64), cfe::Index(128), cfe::Index(256), cfe::Index(400),
                        cfe::Index(512)}) {
    run_case(n);
  }
  return 0;
}
