// Additional 2nd-order convergence coverage beyond
// test_scalar_advection_convergence.cpp's single (double precision,
// positive-velocity) case (gap flagged in code review: CPU/GPU agreement
// alone cannot catch an error present identically on both backends, and
// a positive-velocity-only check cannot catch an upwind-direction sign
// bug that only manifests for negative wave speeds). Same methodology
// as that file (smooth sine profile, exact solution is the initial
// profile shifted by `speed*t`, L2 error against it measures truncation
// error directly) -- templated here on `Scalar` and parametrized on
// `speed` so both variants share one implementation.
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

template <class Scalar>
Scalar exact_solution(Scalar x, Scalar t, Scalar speed)
{
  return std::sin(Scalar(2.0 * kPi) * (x - speed * t));
}

template <class Scalar>
double run_and_measure_l2_error(std::size_t nx, Scalar speed, Scalar final_time, Scalar cfl)
{
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = Scalar(1.0) / static_cast<Scalar>(nx);

  cfe::Field<Scalar, 1> state(grid.n_cells_total());
  cfe::Field<Scalar, 1> stage1(grid.n_cells_total());
  cfe::Field<Scalar, 1> residual_scratch(grid.n_cells_total());

  for (std::size_t i = 0; i < grid.nx; ++i) {
    const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
    state(cell, 0) = exact_solution(grid.x_center(grid.ngx + i), Scalar(0.0), speed);
  }

  cfe::ScalarAdvectionField<Scalar, 1> field{cfe::Vector<Scalar, 1>(speed)};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::ScalarAdvectionField<Scalar, 1>, cfe::PeriodicBoundary>
      solver{grid, field, cfe::PeriodicBoundary{}};

  const Scalar dt_target = cfl * grid.dx / std::fabs(speed);
  const int n_steps = static_cast<int>(std::ceil(static_cast<double>(final_time / dt_target)));
  const Scalar dt = final_time / static_cast<Scalar>(n_steps);

  auto residual = [&](cfe::FieldView<Scalar, 1> in, cfe::FieldView<Scalar, 1> out) {
    solver.residual(in, out);
  };
  for (int step = 0; step < n_steps; ++step) {
    cfe::ssp_rk2_step<Scalar>(state.view(), stage1.view(), residual_scratch.view(), dt, residual,
                              solver.active_cell_count(), solver.active_cell_index_map());
  }

  double sum_sq_error = 0.0;
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const std::size_t cell = grid.flat_index(grid.ngx + i, 0, 0);
    const Scalar exact = exact_solution(grid.x_center(grid.ngx + i), final_time, speed);
    const double error = static_cast<double>(state(cell, 0) - exact);
    sum_sq_error += error * error;
  }
  return std::sqrt(sum_sq_error / static_cast<double>(grid.nx));
}

}  // namespace

CFE_TEST(test_scalar_advection_second_order_convergence_negative_velocity)
{
  // Same case as test_scalar_advection_convergence.cpp's canonical check,
  // but with the advection speed's sign flipped -- UpwindFlux selects the
  // opposite face for a negative wave speed (numerics/numerical_flux/upwind.hpp),
  // so this exercises a genuinely different code path, not just a
  // relabeling of the same one.
  const std::vector<std::size_t> resolutions = {20, 40, 80, 160};
  std::vector<double> errors;
  for (std::size_t nx : resolutions) {
    errors.push_back(run_and_measure_l2_error<double>(nx, -1.0, 0.7, 0.4));
  }
  for (std::size_t k = 0; k + 1 < errors.size(); ++k) {
    CFE_CHECK(errors[k + 1] > 0.0);
    const double ratio = errors[k] / errors[k + 1];
    CFE_CHECK(ratio > 3.5);
    CFE_CHECK(ratio < 4.5);
  }
}

CFE_TEST(test_scalar_advection_second_order_convergence_float_precision)
{
  // Confirms the solver's Scalar-genericity claim actually holds at
  // reduced precision, not just for double. Coarser resolutions and a
  // shorter final time than the double-precision case: float's ~7
  // decimal digits of precision means rounding noise would otherwise
  // start competing with truncation error at the finer resolutions/
  // longer integration times the double case uses, which would corrupt
  // the ratio check for reasons unrelated to the scheme's actual order.
  const std::vector<std::size_t> resolutions = {10, 20, 40, 80};
  std::vector<double> errors;
  for (std::size_t nx : resolutions) {
    errors.push_back(run_and_measure_l2_error<float>(nx, 1.0f, 0.3f, 0.4f));
  }
  for (std::size_t k = 0; k + 1 < errors.size(); ++k) {
    CFE_CHECK(errors[k + 1] > 0.0);
    const double ratio = errors[k] / errors[k + 1];
    CFE_CHECK(ratio > 3.3);
    CFE_CHECK(ratio < 4.7);
  }
}
