// Sanity check for FvmSolver's dimension-generic residual loop applied
// to BurgersField (same philosophy as
// test_scalar_advection_2d_sanity.cpp, extended one dimension further):
// a Riemann-type shock varying only in X, uniform in Y and Z, periodic
// on Y/Z so uniformity is preserved (zero slope on those axes -> zero
// flux difference, same reasoning the 2D sanity test uses), gives every
// (j,k) column an independent copy of the same 1D shock-formation
// problem -- a 3D solve should match the 1D reference exactly, column
// for column. This is the Burgers-specific analogue of task 0003's
// deferred "3D CPU correctness" gap: it directly exercises the Y/Z
// `axis_flux_difference` branches with BurgersField/RusanovFlux/
// MusclMinmodReconstruction, which the 1D-only shock-formation test
// cannot.
#include <cstddef>

#include "cfe/field/field.hpp"
#include "cfe/fields/burgers/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"
#include "test_framework.hpp"

CFE_TEST(test_burgers_3d_solve_matches_1d_solve_column_for_column_when_uniform_in_y_and_z)
{
  constexpr std::size_t kNx = 100;
  constexpr double kULeft = 2.0;
  constexpr double kURight = 1.0;
  constexpr double kShockOrigin = 2.5;
  constexpr double kDomainLength = 10.0;
  constexpr double kDx = kDomainLength / static_cast<double>(kNx);
  constexpr double kDt = 0.4 * kDx / kULeft;
  constexpr int kSteps = 40;

  // --- 1D reference ---
  cfe::CartesianGrid<double> grid_1d;
  grid_1d.nx = kNx;
  grid_1d.ngx = 2;
  grid_1d.dx = kDx;

  cfe::Field<double, 1> state_1d(grid_1d.n_cells_total());
  cfe::Field<double, 1> stage1_1d(grid_1d.n_cells_total());
  cfe::Field<double, 1> scratch_1d(grid_1d.n_cells_total());
  for (std::size_t i = 0; i < grid_1d.nx; ++i) {
    const double x = grid_1d.x_center(grid_1d.ngx + i);
    state_1d(grid_1d.flat_index(grid_1d.ngx + i, 0, 0), 0) = (x < kShockOrigin) ? kULeft : kURight;
  }

  cfe::BurgersField<double, 1> field_1d{};
  cfe::StaticBoundary<double, 1> boundary_1d{cfe::State<double, 1>(kULeft), cfe::State<double, 1>(kURight)};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::BurgersField<double, 1>, cfe::StaticBoundary<double, 1>,
                 cfe::StaticBoundary<double, 1>, cfe::StaticBoundary<double, 1>,
                 cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux>
      solver_1d{grid_1d, field_1d, boundary_1d};
  auto residual_1d = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver_1d.residual(in, out);
  };
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double>(state_1d.view(), stage1_1d.view(), scratch_1d.view(), kDt, residual_1d,
                               solver_1d.active_cell_count(), solver_1d.active_cell_index_map());
  }

  // --- 3D solve: same X profile repeated across every (j,k) column,
  // periodic (inert) Y/Z ---
  cfe::CartesianGrid<double> grid_3d;
  grid_3d.nx = kNx;
  grid_3d.ny = 5;
  grid_3d.nz = 3;
  grid_3d.ngx = 2;
  grid_3d.ngy = 2;
  grid_3d.ngz = 2;
  grid_3d.dx = kDx;
  grid_3d.dy = 0.3;  // deliberately different from dx/dz -- should have no effect
  grid_3d.dz = 0.7;

  cfe::Field<double, 1> state_3d(grid_3d.n_cells_total());
  cfe::Field<double, 1> stage1_3d(grid_3d.n_cells_total());
  cfe::Field<double, 1> scratch_3d(grid_3d.n_cells_total());
  for (std::size_t k = 0; k < grid_3d.nz; ++k) {
    for (std::size_t j = 0; j < grid_3d.ny; ++j) {
      for (std::size_t i = 0; i < grid_3d.nx; ++i) {
        const double x = grid_3d.x_center(grid_3d.ngx + i);
        state_3d(grid_3d.flat_index(grid_3d.ngx + i, grid_3d.ngy + j, grid_3d.ngz + k), 0) =
            (x < kShockOrigin) ? kULeft : kURight;
      }
    }
  }

  cfe::BurgersField<double, 3> field_3d{};
  cfe::StaticBoundary<double, 1> boundary_x_3d{cfe::State<double, 1>(kULeft), cfe::State<double, 1>(kURight)};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::BurgersField<double, 3>, cfe::StaticBoundary<double, 1>,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux>
      solver_3d{grid_3d, field_3d, boundary_x_3d};
  auto residual_3d = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver_3d.residual(in, out);
  };
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double>(state_3d.view(), stage1_3d.view(), scratch_3d.view(), kDt, residual_3d,
                               solver_3d.active_cell_count(), solver_3d.active_cell_index_map());
  }

  // Every (j,k) column of the 3D solve should match the 1D reference
  // exactly (same dt/dx/steps, same arithmetic repeated).
  for (std::size_t k = 0; k < grid_3d.nz; ++k) {
    for (std::size_t j = 0; j < grid_3d.ny; ++j) {
      for (std::size_t i = 0; i < grid_3d.nx; ++i) {
        const double value_1d = state_1d(grid_1d.flat_index(grid_1d.ngx + i, 0, 0), 0);
        const double value_3d =
            state_3d(grid_3d.flat_index(grid_3d.ngx + i, grid_3d.ngy + j, grid_3d.ngz + k), 0);
        CFE_CHECK_NEAR(value_3d, value_1d, 1e-12);
      }
    }
  }
}
