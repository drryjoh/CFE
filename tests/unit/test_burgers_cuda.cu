// CUDA correctness test for the Burgers solver (task 0003/PR #3 feedback:
// every PR adding a Field/scheme needs a GPU correctness test, not just
// CPU -- same requirement test_scalar_advection_cuda.cu already
// satisfies for linear scalar advection). Runs the exact Riemann-shock
// setup test_burgers_shock_formation.cpp verifies on CPU (same IC, same
// StaticBoundary states, same dt/step count) on both CPU and GPU and
// compares cell-by-cell, matching VERIFICATION.md #3's cross-backend
// comparison guidance. UNVERIFIED until run on real CUDA hardware -- see
// docs/bridges2-setup.md.
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

CFE_TEST(test_burgers_cuda_matches_cpu_reference)
{
  constexpr std::size_t kNx = 400;
  constexpr double kULeft = 2.0;
  constexpr double kURight = 1.0;
  constexpr double kShockOrigin = 2.5;
  constexpr double kDomainLength = 10.0;
  constexpr double kDx = kDomainLength / static_cast<double>(kNx);
  constexpr double kDt = 0.4 * kDx / kULeft;
  constexpr int kSteps = 400;  // ~kFinalTime=2.0, matching the CPU shock-formation test

  cfe::CartesianGrid<double> grid;
  grid.nx = kNx;
  grid.ngx = 2;
  grid.dx = kDx;

  std::vector<double> host_ic(grid.n_cells_total(), 0.0);
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const double x = grid.x_center(grid.ngx + i);
    host_ic[grid.flat_index(grid.ngx + i, 0, 0)] = (x < kShockOrigin) ? kULeft : kURight;
  }

  cfe::BurgersField<double, 1> field{};
  cfe::StaticBoundary<double, 1> boundary{cfe::State<double, 1>(kULeft), cfe::State<double, 1>(kURight)};

  // --- CPU reference ---
  cfe::Field<double, 1> state_cpu(grid.n_cells_total());
  cfe::Field<double, 1> stage1_cpu(grid.n_cells_total());
  cfe::Field<double, 1> scratch_cpu(grid.n_cells_total());
  for (std::size_t idx = 0; idx < host_ic.size(); ++idx) state_cpu.data()[idx] = host_ic[idx];

  cfe::FvmSolver<double, cfe::AoSLayout, cfe::BurgersField<double, 1>, cfe::StaticBoundary<double, 1>,
                 cfe::StaticBoundary<double, 1>, cfe::StaticBoundary<double, 1>,
                 cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux>
      solver_cpu{grid, field, boundary};
  // A named functor, not a local lambda: nvcc forbids a locally-defined
  // lambda as a template argument to ssp_rk2_step in a .cu file (see
  // SolverResidual's doc comment in fvm_solver.hpp).
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

  cfe::FvmSolver<double, cfe::AoSLayout, cfe::BurgersField<double, 1>, cfe::StaticBoundary<double, 1>,
                 cfe::StaticBoundary<double, 1>, cfe::StaticBoundary<double, 1>,
                 cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux, cfe::CudaParallelFor>
      solver_gpu{grid, field, boundary};
  cfe::SolverResidual<decltype(solver_gpu)> residual_gpu{&solver_gpu};
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double, cfe::FieldView<double, 1>, decltype(residual_gpu), cfe::CudaParallelFor>(
        state_gpu.view(), stage1_gpu.view(), scratch_gpu.view(), kDt, residual_gpu,
        solver_gpu.active_cell_count(), solver_gpu.active_cell_index_map());
  }
  cfe::backend::cuda::synchronize();

  std::vector<double> host_result(grid.n_cells_total());
  state_gpu.copy_to_host(host_result.data());

  for (std::size_t i = 0; i < grid.nx; ++i) {
    const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
    CFE_CHECK_NEAR(host_result[cell], state_cpu.data()[cell], 1e-9);
  }
}
