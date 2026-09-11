// The Phase 1 acceptance bar (ROADMAP.md, task spec item 9): "verify the
// expected 2nd-order accuracy via a grid-refinement study... 'It ran and
// looked reasonable' does not satisfy this." Named to match
// VERIFICATION.md's own example test name.
//
// Canonical scalar-translation case (VERIFICATION.md's first listed
// canonical problem): a smooth sine profile advected at constant speed
// around a periodic domain. Exact solution at any time is the initial
// profile shifted by a*t, so the error against it is a direct measure of
// the scheme's truncation error -- not a proxy or a visual check.
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
constexpr double kDomainLength = 1.0;
constexpr double kAdvectionSpeed = 1.0;
constexpr double kFinalTime = 0.7;  // deliberately not a whole number of periods
constexpr double kCfl = 0.4;

double exact_solution(double x, double t)
{
  return std::sin(2.0 * kPi * (x - kAdvectionSpeed * t) / kDomainLength);
}

// Runs the scalar-advection solver on an `nx`-cell periodic grid from
// t=0 to t=kFinalTime and returns the L2 error against the exact
// solution.
double run_and_measure_l2_error(std::size_t nx)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = kDomainLength / static_cast<double>(nx);
  grid.origin_x = 0.0;

  cfe::Field<double, 1> q(grid.n_cells_total());
  cfe::Field<double, 1> stage1(grid.n_cells_total());
  cfe::Field<double, 1> residual_scratch(grid.n_cells_total());

  for (std::size_t i = 0; i < grid.nx; ++i) {
    const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
    q(cell, 0) = exact_solution(grid.x_center(grid.ngx + i), 0.0);
  }

  cfe::ScalarAdvectionField<double, 1> field{cfe::Vector<double, 1>(kAdvectionSpeed)};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 1>, cfe::PeriodicBoundary>
      solver{grid, field, cfe::PeriodicBoundary{}};

  const double dt_target = kCfl * grid.dx / kAdvectionSpeed;
  const int n_steps = static_cast<int>(std::ceil(kFinalTime / dt_target));
  const double dt = kFinalTime / static_cast<double>(n_steps);  // land exactly on kFinalTime

  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver.residual(in, out);
  };
  for (int step = 0; step < n_steps; ++step) {
    cfe::ssp_rk2_step<double>(q.view(), stage1.view(), residual_scratch.view(), dt, residual);
  }

  double sum_sq_error = 0.0;
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
    const double exact = exact_solution(grid.x_center(grid.ngx + i), kFinalTime);
    const double error = q(cell, 0) - exact;
    sum_sq_error += error * error;
  }
  return std::sqrt(sum_sq_error / static_cast<double>(grid.nx));
}

}  // namespace

CFE_TEST(test_scalar_advection_second_order_convergence)
{
  const std::vector<std::size_t> resolutions = {20, 40, 80, 160};
  std::vector<double> errors;
  errors.reserve(resolutions.size());
  for (std::size_t nx : resolutions) {
    errors.push_back(run_and_measure_l2_error(nx));
  }

  // Each doubling of resolution (halving of dx) should quarter the error
  // for a 2nd-order scheme. Check every consecutive pair, not just the
  // endpoints, so a fluke at one resolution can't hide behind an average.
  for (std::size_t k = 0; k + 1 < errors.size(); ++k) {
    CFE_CHECK(errors[k + 1] > 0.0);
    const double ratio = errors[k] / errors[k + 1];
    CFE_CHECK(ratio > 3.5);
    CFE_CHECK(ratio < 4.5);
  }
}
