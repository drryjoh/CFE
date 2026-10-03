// CUDA correctness test for the Burgers solver's 3D path (task
// 0003/PR #3 feedback: every PR needs a 3D GPU correctness test, not
// just 1D -- mirrors test_scalar_advection_3d_cuda.cu). Exercises the
// X, Y, and Z `axis_flux_difference` branches together with
// BurgersField/RusanovFlux/MusclMinmodReconstruction: the same Riemann
// shock as test_burgers_cuda.cu, varying only in X, uniform (and
// periodic) in Y/Z -- test_burgers_3d_sanity.cpp already established on
// CPU that this must match the 1D reference column for column, so this
// test's CPU-vs-GPU comparison is simultaneously a 3D-dimension-
// generic-residual check AND a GPU-port check. Runs at a genuinely
// 3D, at-scale resolution (not a token 4^3 cube).
#include <cmath>
#include <vector>

#include "cfe/backend/cuda/cuda_backend.cuh"
#include "cfe/backend/cuda/device_field.cuh"
#include "cfe/field/field.hpp"
#include "cfe/fields/burgers/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"
#include "test_framework.hpp"

CFE_TEST(test_burgers_3d_cuda_matches_cpu_reference)
{
  constexpr std::size_t kN = 96;
  constexpr double kULeft = 2.0;
  constexpr double kURight = 1.0;
  constexpr double kShockOrigin = 2.5;
  constexpr double kDomainLength = 10.0;
  constexpr double kDx = kDomainLength / static_cast<double>(kN);
  constexpr double kDt = 0.4 * kDx / kULeft;
  constexpr int kSteps = 200;

  cfe::CartesianGrid<double> grid;
  grid.nx = kN;
  grid.ny = kN;
  grid.nz = kN;
  grid.ngx = 2;
  grid.ngy = 2;
  grid.ngz = 2;
  grid.dx = kDx;
  grid.dy = kDx;
  grid.dz = kDx;

  std::vector<double> host_ic(grid.n_cells_total(), 0.0);
  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const double x = grid.x_center(grid.ngx + i);
        host_ic[grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k)] =
            (x < kShockOrigin) ? kULeft : kURight;
      }
    }
  }

  cfe::BurgersField<double, 3> field{};
  cfe::StaticBoundary<double, 1> boundary_x{cfe::State<double, 1>(kULeft), cfe::State<double, 1>(kURight)};

  // --- CPU reference ---
  cfe::Field<double, 1> state_cpu(grid.n_cells_total());
  cfe::Field<double, 1> stage1_cpu(grid.n_cells_total());
  cfe::Field<double, 1> scratch_cpu(grid.n_cells_total());
  for (std::size_t idx = 0; idx < host_ic.size(); ++idx) state_cpu.data()[idx] = host_ic[idx];

  cfe::FvmSolver<double, cfe::AoSLayout, cfe::BurgersField<double, 3>, cfe::StaticBoundary<double, 1>,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux>
      solver_cpu{grid, field, boundary_x};
  cfe::SolverResidual<decltype(solver_cpu)> residual_cpu{&solver_cpu};
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double>(state_cpu.view(), stage1_cpu.view(), scratch_cpu.view(), kDt, residual_cpu,
                               solver_cpu.active_cell_count(), solver_cpu.active_cell_index_map());
  }

  // --- CUDA ---
  cfe::backend::cuda::DeviceField<double, 1> state_gpu(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, 1> stage1_gpu(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, 1> scratch_gpu(grid.n_cells_total());
  state_gpu.copy_from_host(host_ic.data());

  cfe::FvmSolver<double, cfe::AoSLayout, cfe::BurgersField<double, 3>, cfe::StaticBoundary<double, 1>,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux, cfe::CudaParallelFor>
      solver_gpu{grid, field, boundary_x};
  cfe::SolverResidual<decltype(solver_gpu)> residual_gpu{&solver_gpu};
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double, cfe::FieldView<double, 1>, decltype(residual_gpu), cfe::CudaParallelFor>(
        state_gpu.view(), stage1_gpu.view(), scratch_gpu.view(), kDt, residual_gpu,
        solver_gpu.active_cell_count(), solver_gpu.active_cell_index_map());
  }
  cfe::backend::cuda::synchronize();

  std::vector<double> host_result(grid.n_cells_total());
  state_gpu.copy_to_host(host_result.data());

  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
        CFE_CHECK_NEAR(host_result[cell], state_cpu.data()[cell], 1e-9);
      }
    }
  }
}

// Review finding on PR #3 (commit 1d3654f): the test above varies only
// in X (uniform, periodic Y/Z), so it cannot distinguish "Y/Z transport
// correct" from "Y/Z transport silently broken" -- with zero Y/Z
// gradient, both look identical (zero flux difference either way). This
// test uses a smooth IC that varies in all three directions at once
// (same IC as
// test_burgers_3d_solve_with_genuine_multi_axis_variation_preserves_
// yz_symmetry_and_conserves_mass in test_burgers_3d_sanity.cpp, which
// checks Y/Z-swap symmetry on CPU alone) and compares CPU vs. GPU on
// it -- a genuine, non-degenerate 3D transport problem, not just a
// backend-agreement check on an X-only case.
CFE_TEST(test_burgers_3d_cuda_matches_cpu_reference_with_genuine_multi_axis_variation)
{
  constexpr std::size_t kN = 64;
  constexpr double kPi = 3.14159265358979323846;
  constexpr double kBackground = 1.0;
  constexpr double kAmplitude = 0.3;
  constexpr double kDx = 1.0 / static_cast<double>(kN);
  constexpr int kSteps = 50;

  cfe::CartesianGrid<double> grid;
  grid.nx = kN;
  grid.ny = kN;
  grid.nz = kN;
  grid.ngx = 2;
  grid.ngy = 2;
  grid.ngz = 2;
  grid.dx = kDx;
  grid.dy = kDx;
  grid.dz = kDx;

  std::vector<double> host_ic(grid.n_cells_total(), 0.0);
  double max_abs_u0 = 0.0;
  double initial_sum = 0.0;
  for (std::size_t k = 0; k < grid.nz; ++k) {
    const double z = grid.z_center(grid.ngz + k);
    for (std::size_t j = 0; j < grid.ny; ++j) {
      const double y = grid.y_center(grid.ngy + j);
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const double x = grid.x_center(grid.ngx + i);
        const double value =
            kBackground + kAmplitude * (std::sin(2.0 * kPi * x) + std::sin(2.0 * kPi * y) +
                                         std::sin(2.0 * kPi * z));
        host_ic[grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k)] = value;
        max_abs_u0 = std::max(max_abs_u0, std::abs(value));
        initial_sum += value;
      }
    }
  }
  const double dt = 0.3 * grid.dx / (3.0 * max_abs_u0);

  cfe::BurgersField<double, 3> field{};

  // --- CPU reference ---
  cfe::Field<double, 1> state_cpu(grid.n_cells_total());
  cfe::Field<double, 1> stage1_cpu(grid.n_cells_total());
  cfe::Field<double, 1> scratch_cpu(grid.n_cells_total());
  for (std::size_t idx = 0; idx < host_ic.size(); ++idx) state_cpu.data()[idx] = host_ic[idx];

  cfe::FvmSolver<double, cfe::AoSLayout, cfe::BurgersField<double, 3>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux>
      solver_cpu{grid, field, cfe::PeriodicBoundary{}};
  cfe::SolverResidual<decltype(solver_cpu)> residual_cpu{&solver_cpu};
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double>(state_cpu.view(), stage1_cpu.view(), scratch_cpu.view(), dt, residual_cpu,
                               solver_cpu.active_cell_count(), solver_cpu.active_cell_index_map());
  }

  // --- CUDA ---
  cfe::backend::cuda::DeviceField<double, 1> state_gpu(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, 1> stage1_gpu(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, 1> scratch_gpu(grid.n_cells_total());
  state_gpu.copy_from_host(host_ic.data());

  cfe::FvmSolver<double, cfe::AoSLayout, cfe::BurgersField<double, 3>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux, cfe::CudaParallelFor>
      solver_gpu{grid, field, cfe::PeriodicBoundary{}};
  cfe::SolverResidual<decltype(solver_gpu)> residual_gpu{&solver_gpu};
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double, cfe::FieldView<double, 1>, decltype(residual_gpu), cfe::CudaParallelFor>(
        state_gpu.view(), stage1_gpu.view(), scratch_gpu.view(), dt, residual_gpu,
        solver_gpu.active_cell_count(), solver_gpu.active_cell_index_map());
  }
  cfe::backend::cuda::synchronize();

  std::vector<double> host_result(grid.n_cells_total());
  state_gpu.copy_to_host(host_result.data());

  double sum_gpu = 0.0;
  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
        CFE_CHECK_NEAR(host_result[cell], state_cpu.data()[cell], 1e-9);
        sum_gpu += host_result[cell];
      }
    }
  }
  // Conservation, checked on the GPU result directly (not inferred from
  // the CPU-agreement check above): a periodic domain's flux-form
  // residual conserves the domain integral exactly, up to time-
  // integration truncation error. Raw cell-value sums, not sums*dx^3 --
  // every cell has the same volume here, so comparing sums directly is
  // equivalent to comparing domain integrals.
  CFE_CHECK_NEAR(sum_gpu, initial_sum, 1e-6 * static_cast<double>(grid.nx * grid.ny * grid.nz));
}
