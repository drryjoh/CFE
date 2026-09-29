// CUDA correctness test for the scalar-advection solver's 3D path
// (task spec item 8/tests: "CUDA backend produces expected results when
// available"). The 1D CUDA test (test_scalar_advection_cuda.cu) already
// covers the single-axis flux path on real hardware; this exercises the
// X, Y, *and* Z `axis_flux_difference` branches together (Y and Z are
// otherwise only reached by CPU-only tests -- test_scalar_advection_
// 2d_sanity.cpp for Y, nothing yet for Z), with a genuinely
// direction-dependent velocity so none of the three axes degenerates
// into the zero-flux special case that test used deliberately. Runs the
// identical setup on CPU and GPU and compares cell-by-cell, matching
// VERIFICATION.md #3's cross-backend comparison guidance.
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

CFE_TEST(test_scalar_advection_3d_cuda_matches_cpu_reference)
{
  constexpr std::size_t kN = 32;
  constexpr double kUx = 1.0;
  constexpr double kUy = 0.6;
  constexpr double kUz = 0.3;
  constexpr double kDx = 1.0 / static_cast<double>(kN);
  // Multi-axis CFL: dt must respect the *sum* of per-axis Courant
  // numbers, not each axis independently, since a cell's residual
  // combines flux differences from all three active axes every stage.
  constexpr double kDt = 0.25 * kDx / (kUx + kUy + kUz);
  constexpr int kSteps = 30;

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
        const double y = grid.y_center(grid.ngy + j);
        const double z = grid.z_center(grid.ngz + k);
        host_ic[grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k)] =
            std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y) * std::sin(2.0 * kPi * z);
      }
    }
  }

  cfe::Vector<double, 3> velocity;
  velocity[0] = kUx;
  velocity[1] = kUy;
  velocity[2] = kUz;
  cfe::ScalarAdvectionField<double, 3> field{velocity};

  // --- CPU reference ---
  cfe::Field<double, 1> state_cpu(grid.n_cells_total());
  cfe::Field<double, 1> stage1_cpu(grid.n_cells_total());
  cfe::Field<double, 1> scratch_cpu(grid.n_cells_total());
  for (std::size_t idx = 0; idx < host_ic.size(); ++idx) state_cpu.data()[idx] = host_ic[idx];

  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 3>, cfe::PeriodicBoundary>
      solver_cpu{grid, field, cfe::PeriodicBoundary{}};
  cfe::SolverResidual<decltype(solver_cpu)> residual_cpu{&solver_cpu};
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double>(state_cpu.view(), stage1_cpu.view(), scratch_cpu.view(), kDt, residual_cpu);
  }

  // --- CUDA ---
  cfe::backend::cuda::DeviceField<double, 1> state_gpu(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, 1> stage1_gpu(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, 1> scratch_gpu(grid.n_cells_total());
  state_gpu.copy_from_host(host_ic.data());

  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 3>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::CentralDifferenceReconstruction,
                 cfe::UpwindFlux, cfe::CudaParallelFor>
      solver_gpu{grid, field, cfe::PeriodicBoundary{}};
  cfe::SolverResidual<decltype(solver_gpu)> residual_gpu{&solver_gpu};
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double, cfe::FieldView<double, 1>, decltype(residual_gpu), cfe::CudaParallelFor>(
        state_gpu.view(), stage1_gpu.view(), scratch_gpu.view(), kDt, residual_gpu);
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
