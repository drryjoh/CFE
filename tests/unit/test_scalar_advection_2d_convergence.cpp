// Genuine multi-axis analytic-solution convergence check (gap flagged in
// code review): test_scalar_advection_2d_sanity.cpp deliberately keeps
// the Y velocity at zero so every row degenerates into an independent
// copy of the 1D problem -- a real bug in how the X and Y flux
// contributions combine (e.g. a sign error, a missed cross term) could
// slip past that test and past every CUDA test (which only check
// CPU-vs-GPU agreement, not either backend against an independent
// reference) without being caught. This test uses a genuinely
// two-dimensional profile with nonzero velocity on both axes, checked
// against the exact analytic solution.
//
// For linear advection dQ/dt + velocity.grad(Q) = 0, any Q(x,y,t) =
// f(x - ux*t, y - uy*t) is an exact solution (the PDE says Q is constant
// along characteristics) -- so a product-of-sines initial profile stays
// an exact product-of-sines at any later time, just phase-shifted by
// the velocity in each direction.
#include <cmath>
#include <cstddef>
#include <vector>

#include "cfe/field/field.hpp"
#include "cfe/fields/scalar_advection/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"
#include "test_framework.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kUx = 1.0;
constexpr double kUy = 0.6;
constexpr double kFinalTime = 0.4;
constexpr double kCfl = 0.4;

double exact_solution(double x, double y, double t)
{
  return std::sin(2.0 * kPi * (x - kUx * t)) * std::sin(2.0 * kPi * (y - kUy * t));
}

double run_and_measure_l2_error(std::size_t n)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = n;
  grid.ny = n;
  grid.ngx = 2;
  grid.ngy = 2;
  grid.dx = 1.0 / static_cast<double>(n);
  grid.dy = grid.dx;

  cfe::Field<double, 1> state(grid.n_cells_total());
  cfe::Field<double, 1> stage1(grid.n_cells_total());
  cfe::Field<double, 1> residual_scratch(grid.n_cells_total());

  for (std::size_t j = 0; j < grid.ny; ++j) {
    for (std::size_t i = 0; i < grid.nx; ++i) {
      const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, 0);
      state(cell, 0) =
          exact_solution(grid.x_center(grid.ngx + i), grid.y_center(grid.ngy + j), 0.0);
    }
  }

  cfe::Vector<double, 2> velocity;
  velocity[0] = kUx;
  velocity[1] = kUy;
  cfe::ScalarAdvectionField<double, 2> field{velocity};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 2>, cfe::PeriodicBoundary>
      solver{grid, field, cfe::PeriodicBoundary{}};

  const double dt_target = kCfl * grid.dx / (kUx + kUy);
  const int n_steps = static_cast<int>(std::ceil(kFinalTime / dt_target));
  const double dt = kFinalTime / static_cast<double>(n_steps);

  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver.residual(in, out);
  };
  for (int step = 0; step < n_steps; ++step) {
    cfe::ssp_rk2_step<double>(state.view(), stage1.view(), residual_scratch.view(), dt, residual);
  }

  double sum_sq_error = 0.0;
  for (std::size_t j = 0; j < grid.ny; ++j) {
    for (std::size_t i = 0; i < grid.nx; ++i) {
      const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, 0);
      const double exact =
          exact_solution(grid.x_center(grid.ngx + i), grid.y_center(grid.ngy + j), kFinalTime);
      const double error = state(cell, 0) - exact;
      sum_sq_error += error * error;
    }
  }
  return std::sqrt(sum_sq_error / static_cast<double>(grid.nx * grid.ny));
}

}  // namespace

CFE_TEST(test_scalar_advection_2d_second_order_convergence_with_nonzero_xy_velocity)
{
  const std::vector<std::size_t> resolutions = {20, 40, 80, 160};
  std::vector<double> errors;
  errors.reserve(resolutions.size());
  for (std::size_t n : resolutions) {
    errors.push_back(run_and_measure_l2_error(n));
  }

  for (std::size_t k = 0; k + 1 < errors.size(); ++k) {
    CFE_CHECK(errors[k + 1] > 0.0);
    const double ratio = errors[k] / errors[k + 1];
    CFE_CHECK(ratio > 3.5);
    CFE_CHECK(ratio < 4.5);
  }
}
