// CUDA correctness test for the scalar-advection solver (task spec item
// 8/tests: "CUDA backend produces expected results when available").
// Runs the identical setup on CPU and GPU (same IC, same dt, same step
// count) and compares cell-by-cell, matching VERIFICATION.md #3's
// cross-backend comparison guidance (a real tolerance, not bitwise
// equality). UNVERIFIED until run on real CUDA hardware -- see
// docs/bridges2-setup.md.
#include <cmath>
#include <vector>

#include "cfe/backend/cuda/cuda_backend.cuh"
#include "cfe/backend/cuda/device_field.cuh"
#include "cfe/field/field.hpp"
#include "cfe/fields/scalar_advection/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"
#include "test_framework.hpp"

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

CFE_TEST(test_scalar_advection_cuda_matches_cpu_reference)
{
  constexpr std::size_t kNx = 256;
  constexpr double kSpeed = 1.0;
  constexpr double kDx = 1.0 / static_cast<double>(kNx);
  constexpr double kDt = 0.4 * kDx / kSpeed;
  constexpr int kSteps = 50;

  cfe::CartesianGrid<double> grid;
  grid.nx = kNx;
  grid.ngx = 2;
  grid.dx = kDx;

  std::vector<double> host_ic(grid.n_cells_total(), 0.0);
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const double x = grid.x_center(grid.ngx + i);
    host_ic[grid.flat_index(grid.ngx + i, 0, 0)] = std::sin(2.0 * kPi * x);
  }

  cfe::ScalarAdvectionField<double, 1> field{cfe::Vector<double, 1>(kSpeed)};

  // --- CPU reference ---
  cfe::Field<double, 1> q_cpu(grid.n_cells_total());
  cfe::Field<double, 1> stage1_cpu(grid.n_cells_total());
  cfe::Field<double, 1> scratch_cpu(grid.n_cells_total());
  for (std::size_t idx = 0; idx < host_ic.size(); ++idx) q_cpu.data()[idx] = host_ic[idx];

  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 1>, cfe::PeriodicBoundary>
      solver_cpu{grid, field, cfe::PeriodicBoundary{}};
  auto residual_cpu = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver_cpu.residual(in, out);
  };
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double>(q_cpu.view(), stage1_cpu.view(), scratch_cpu.view(), kDt, residual_cpu);
  }

  // --- CUDA ---
  cfe::backend::cuda::DeviceField<double, 1> q_gpu(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, 1> stage1_gpu(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, 1> scratch_gpu(grid.n_cells_total());
  q_gpu.copy_from_host(host_ic.data());

  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 1>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary,
                 cfe::fvm::CentralDifferenceReconstruction, cfe::UpwindFlux, cfe::CudaParallelFor>
      solver_gpu{grid, field, cfe::PeriodicBoundary{}};
  auto residual_gpu = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver_gpu.residual(in, out);
  };
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double, cfe::FieldView<double, 1>, decltype(residual_gpu), cfe::CudaParallelFor>(
        q_gpu.view(), stage1_gpu.view(), scratch_gpu.view(), kDt, residual_gpu);
  }
  cfe::backend::cuda::synchronize();

  std::vector<double> host_result(grid.n_cells_total());
  q_gpu.copy_to_host(host_result.data());

  for (std::size_t i = 0; i < grid.nx; ++i) {
    const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
    CFE_CHECK_NEAR(host_result[cell], q_cpu.data()[cell], 1e-9);
  }
}
