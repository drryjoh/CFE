// Conservation check over a periodic domain (task spec item 10). Flux
// differencing (residual = -(F_right - F_left)/dx) telescopes exactly to
// zero net change when summed over a periodic domain, regardless of the
// flux's accuracy -- this is a structural property of conservative FVM,
// not something that depends on the scheme being 2nd-order. A break here
// would mean a real bug in the residual assembly, independent of whatever
// the convergence study shows.
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

double total_quantity(const cfe::Field<double, 1>& q, const cfe::CartesianGrid<double>& grid)
{
  double sum = 0.0;
  for (std::size_t i = 0; i < grid.nx; ++i) {
    sum += q(grid.flat_index(grid.ngx + i, 0, 0), 0);
  }
  return sum * grid.dx;
}

}  // namespace

CFE_TEST(test_scalar_advection_conserves_total_quantity_over_periodic_domain)
{
  constexpr double kPi = 3.14159265358979323846;

  cfe::CartesianGrid<double> grid;
  grid.nx = 64;
  grid.ngx = 2;
  grid.dx = 1.0 / static_cast<double>(grid.nx);

  cfe::Field<double, 1> q(grid.n_cells_total());
  cfe::Field<double, 1> stage1(grid.n_cells_total());
  cfe::Field<double, 1> residual_scratch(grid.n_cells_total());

  // An asymmetric (not just a pure sine) smooth profile, so this isn't
  // accidentally testing a special-case symmetry.
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const double x = grid.x_center(grid.ngx + i);
    q(grid.flat_index(grid.ngx + i, 0, 0), 0) = std::sin(2.0 * kPi * x) + 0.3 * std::cos(4.0 * kPi * x);
  }

  const double initial_total = total_quantity(q, grid);

  cfe::ScalarAdvectionField<double, 1> field{cfe::Vector<double, 1>(1.3)};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 1>, cfe::PeriodicBoundary>
      solver{grid, field, cfe::PeriodicBoundary{}};

  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver.residual(in, out);
  };

  const double dt = 0.4 * grid.dx / 1.3;
  for (int step = 0; step < 200; ++step) {
    cfe::ssp_rk2_step<double>(q.view(), stage1.view(), residual_scratch.view(), dt, residual);
  }

  const double final_total = total_quantity(q, grid);
  CFE_CHECK_NEAR(final_total, initial_total, 1e-9);
}
