// Sanity check for FvmSolver's dimension-generic residual loop (the Y-axis
// code path is otherwise only exercised by grid/boundary-condition unit
// tests, never by an actual solve). Deliberately does NOT require
// ScalarAdvectionField to support a direction-dependent velocity beyond
// what it already does: velocity has a Y-component of exactly zero, and
// the initial condition doesn't vary in Y, so the Y-direction flux is
// legitimately inert everywhere (zero slope -> zero flux difference, see
// test_interface_value_reduces_to_cell_average_for_a_uniform_field). This
// means every row (fixed j) is physically an independent copy of the same
// 1D problem, which gives a strong, hand-computable-in-spirit check: a 2D
// solve should match a 1D solve at the same resolution, row for row.
#include <cmath>
#include <cstddef>

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

CFE_TEST(test_scalar_advection_2d_solve_matches_1d_solve_row_for_row_when_y_velocity_is_zero)
{
  constexpr std::size_t kNx = 40;
  constexpr double kSpeed = 1.0;
  constexpr double kDx = 1.0 / static_cast<double>(kNx);
  constexpr double kDt = 0.4 * kDx / kSpeed;
  constexpr int kSteps = 25;

  // --- 1D reference ---
  cfe::CartesianGrid<double> grid_1d;
  grid_1d.nx = kNx;
  grid_1d.ngx = 2;
  grid_1d.dx = kDx;

  cfe::Field<double, 1> q_1d(grid_1d.n_cells_total());
  cfe::Field<double, 1> stage1_1d(grid_1d.n_cells_total());
  cfe::Field<double, 1> scratch_1d(grid_1d.n_cells_total());
  for (std::size_t i = 0; i < grid_1d.nx; ++i) {
    const double x = grid_1d.x_center(grid_1d.ngx + i);
    q_1d(grid_1d.flat_index(grid_1d.ngx + i, 0, 0), 0) = std::sin(2.0 * kPi * x);
  }

  cfe::ScalarAdvectionField<double, 1> field_1d{cfe::Vector<double, 1>(kSpeed)};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 1>, cfe::PeriodicBoundary>
      solver_1d{grid_1d, field_1d, cfe::PeriodicBoundary{}};
  auto residual_1d = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver_1d.residual(in, out);
  };
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double>(q_1d.view(), stage1_1d.view(), scratch_1d.view(), kDt, residual_1d);
  }

  // --- 2D solve: same X profile repeated across every row, zero Y velocity ---
  cfe::CartesianGrid<double> grid_2d;
  grid_2d.nx = kNx;
  grid_2d.ny = 6;
  grid_2d.ngx = 2;
  grid_2d.ngy = 2;
  grid_2d.dx = kDx;
  grid_2d.dy = 0.3;  // deliberately different from dx -- should have no effect

  cfe::Field<double, 1> q_2d(grid_2d.n_cells_total());
  cfe::Field<double, 1> stage1_2d(grid_2d.n_cells_total());
  cfe::Field<double, 1> scratch_2d(grid_2d.n_cells_total());
  for (std::size_t j = 0; j < grid_2d.ny; ++j) {
    for (std::size_t i = 0; i < grid_2d.nx; ++i) {
      const double x = grid_2d.x_center(grid_2d.ngx + i);
      q_2d(grid_2d.flat_index(grid_2d.ngx + i, grid_2d.ngy + j, 0), 0) = std::sin(2.0 * kPi * x);
    }
  }

  cfe::Vector<double, 2> velocity_2d;
  velocity_2d[0] = kSpeed;
  velocity_2d[1] = 0.0;
  cfe::ScalarAdvectionField<double, 2> field_2d{velocity_2d};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 2>, cfe::PeriodicBoundary>
      solver_2d{grid_2d, field_2d, cfe::PeriodicBoundary{}};
  auto residual_2d = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver_2d.residual(in, out);
  };
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double>(q_2d.view(), stage1_2d.view(), scratch_2d.view(), kDt, residual_2d);
  }

  // Every row of the 2D solve should match the 1D reference exactly (both
  // used the same dt/dx/steps, so this is not just "close", it's the same
  // arithmetic repeated).
  for (std::size_t j = 0; j < grid_2d.ny; ++j) {
    for (std::size_t i = 0; i < grid_2d.nx; ++i) {
      const double value_1d = q_1d(grid_1d.flat_index(grid_1d.ngx + i, 0, 0), 0);
      const double value_2d = q_2d(grid_2d.flat_index(grid_2d.ngx + i, grid_2d.ngy + j, 0), 0);
      CFE_CHECK_NEAR(value_2d, value_1d, 1e-12);
    }
  }
}
