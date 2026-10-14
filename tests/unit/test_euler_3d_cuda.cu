// CUDA correctness test for the compressible Euler solver's 3D path
// (closes the policy gap flagged when task 0009 was otherwise ready to
// commit: every PR adding a Field/scheme needs a 3D GPU correctness
// test, not just 1D -- mirrors test_scalar_advection_3d_cuda.cu/
// test_burgers_3d_cuda.cu). The 1D CUDA test (test_euler_cuda.cu)
// already covers the single-axis flux path and `SlipWallBoundary::
// fill_x` in the degenerate 1D case (grid.ny=grid.nz=1); this exercises
// the X, Y, AND Z `axis_flux_difference` branches together with
// EulerField/RusanovFlux/MusclMinmodReconstruction on real hardware,
// and exercises `SlipWallBoundary::fill_x` with a genuinely non-trivial
// padded_ny()*padded_nz() index decomposition (ny, nz > 1) for the first
// time on GPU -- test_euler_3d_sanity.cpp already established on CPU
// that this configuration (Sod shock tube along X, uniform/periodic
// Y/Z) must match the 1D reference column for column, so this test's
// CPU-vs-GPU comparison is simultaneously a 3D-dimension-generic-
// residual check AND a GPU-port check for both the Euler flux path and
// the slip-wall boundary.
//
// IMPORTANT, found during review (same issue as test_euler_cuda.cu, see
// its header comment for the full probe methodology): the original
// `kSteps=150` (t=0.12) never reached the wall at x=1 either -- a
// standalone CPU-only probe mirroring this exact grid/IC/BCs showed the
// shock arrives at t~0.27, with wall-region |u| peaking at ~0.93 just
// after. `kSteps` below was corrected so the run genuinely exercises a
// full reflection cycle (through t=0.5), not just the inert pre-arrival
// state. The same probe measured the actual post-reflection max |u|+c
// over t in [0,0.6] at 2.21 -- unchanged from the 1D case (Y/Z carry no
// velocity here, so the bound is set by the same X-direction physics),
// so `kMaxSpeedEstimate` did not need to change.
#include <cmath>
#include <vector>

#include "cfe/backend/cuda/cuda_backend.cuh"
#include "cfe/backend/cuda/device_field.cuh"
#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/euler/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/boundary/slip_wall_boundary.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/math/fixed_array.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"
#include "test_framework.hpp"

CFE_TEST(test_euler_3d_cuda_matches_cpu_reference)
{
  using Field = cfe::EulerField<double, 3>;
  constexpr std::size_t kN = Field::n_components;  // 5

  constexpr std::size_t kNx = 100;
  constexpr std::size_t kNy = 8;
  constexpr std::size_t kNz = 6;
  constexpr double kGamma = 1.4;
  constexpr double kX0 = 0.5;
  constexpr double kDx = 1.0 / static_cast<double>(kNx);
  // Same fixed, conservatively-chosen dt convention as test_euler_cuda.cu
  // (a CPU/GPU cross-check, not a convergence/exact-solution check).
  constexpr double kMaxSpeedEstimate = 5.0;
  constexpr double kCfl = 0.4;
  constexpr double kDt = kCfl * kDx / kMaxSpeedEstimate;
  // 625 steps * kDt (0.0008) = t=0.5 -- see this file's header comment:
  // the shock reaches the right wall at t~0.27, so this reaches well
  // past the first reflection (the original 150 steps, t=0.12, did
  // not).
  constexpr int kSteps = 625;

  cfe::CartesianGrid<double> grid;
  grid.nx = kNx;
  grid.ny = kNy;
  grid.nz = kNz;
  grid.ngx = 2;
  grid.ngy = 2;
  grid.ngz = 2;
  grid.dx = kDx;
  grid.dy = 0.3;  // deliberately different from dx/dz -- should have no effect
  grid.dz = 0.7;

  Field field{};
  field.gamma = kGamma;
  const cfe::Vector<double, 3> zero_velocity{};
  const cfe::State<double, kN> left_q = field.conserved_from_primitive(1.0, zero_velocity, 1.0);
  const cfe::State<double, kN> right_q = field.conserved_from_primitive(0.125, zero_velocity, 0.1);

  std::vector<double> host_ic(grid.n_cells_total() * kN, 0.0);
  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const double x = grid.x_center(grid.ngx + i);
        const cfe::State<double, kN>& q = (x < kX0) ? left_q : right_q;
        const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
        for (std::size_t c = 0; c < kN; ++c) host_ic[cell * kN + c] = q[c];
      }
    }
  }

  cfe::SlipWallBoundary<Field> boundary_x{};

  // --- CPU reference ---
  cfe::Field<double, kN> state_cpu(grid.n_cells_total());
  cfe::Field<double, kN> stage1_cpu(grid.n_cells_total());
  cfe::Field<double, kN> scratch_cpu(grid.n_cells_total());
  for (std::size_t idx = 0; idx < host_ic.size(); ++idx) state_cpu.data()[idx] = host_ic[idx];

  cfe::FvmSolver<double, cfe::AoSLayout, Field, cfe::SlipWallBoundary<Field>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux>
      solver_cpu{grid, field, boundary_x};
  // A named functor, not a local lambda: nvcc forbids a locally-defined
  // lambda as a template argument to ssp_rk2_step in a .cu file (see
  // SolverResidual's doc comment in fvm_solver.hpp).
  cfe::SolverResidual<decltype(solver_cpu)> residual_cpu{&solver_cpu};

  // Mass/total-energy BEFORE stepping -- same reasoning as
  // test_euler_cuda.cu's own conservation check (a closed tube on X,
  // periodic on Y/Z, so no flux leaves the domain on any axis; no
  // closed-form solution exists for a doubly-reflected 3D problem, so
  // conservation is the right physical sanity check here, independent
  // of the CPU-vs-GPU agreement check below).
  const double cell_volume = grid.dx * grid.dy * grid.dz;
  auto total_mass = [&](const cfe::Field<double, kN>& s) {
    double m = 0.0;
    for (std::size_t k = 0; k < grid.nz; ++k) {
      for (std::size_t j = 0; j < grid.ny; ++j) {
        for (std::size_t i = 0; i < grid.nx; ++i) {
          m += s(grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k), 0) * cell_volume;
        }
      }
    }
    return m;
  };
  auto total_energy = [&](const cfe::Field<double, kN>& s) {
    double e = 0.0;
    for (std::size_t k = 0; k < grid.nz; ++k) {
      for (std::size_t j = 0; j < grid.ny; ++j) {
        for (std::size_t i = 0; i < grid.nx; ++i) {
          e += s(grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k), Field::energy_index()) * cell_volume;
        }
      }
    }
    return e;
  };
  const double mass_before = total_mass(state_cpu);
  const double energy_before = total_energy(state_cpu);

  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double>(state_cpu.view(), stage1_cpu.view(), scratch_cpu.view(), kDt, residual_cpu,
                               solver_cpu.active_cell_count(), solver_cpu.active_cell_index_map());
  }

  CFE_CHECK_NEAR(total_mass(state_cpu), mass_before, 1e-10);
  CFE_CHECK_NEAR(total_energy(state_cpu), energy_before, 1e-10);

  // --- CUDA ---
  cfe::backend::cuda::DeviceField<double, kN> state_gpu(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, kN> stage1_gpu(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, kN> scratch_gpu(grid.n_cells_total());
  state_gpu.copy_from_host(host_ic.data());

  cfe::FvmSolver<double, cfe::AoSLayout, Field, cfe::SlipWallBoundary<Field>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux,
                 cfe::CudaParallelFor>
      solver_gpu{grid, field, boundary_x};
  cfe::SolverResidual<decltype(solver_gpu)> residual_gpu{&solver_gpu};
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double, cfe::FieldView<double, kN>, decltype(residual_gpu), cfe::CudaParallelFor>(
        state_gpu.view(), stage1_gpu.view(), scratch_gpu.view(), kDt, residual_gpu,
        solver_gpu.active_cell_count(), solver_gpu.active_cell_index_map());
  }
  cfe::backend::cuda::synchronize();

  std::vector<double> host_result(grid.n_cells_total() * kN);
  state_gpu.copy_to_host(host_result.data());

  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
        for (std::size_t c = 0; c < kN; ++c) {
          CFE_CHECK_NEAR(host_result[cell * kN + c], state_cpu.data()[cell * kN + c], 1e-9);
        }
      }
    }
  }
}
