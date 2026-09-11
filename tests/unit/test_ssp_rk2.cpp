// Unit test for the SSP-RK2 time integrator (task spec item 6/tests:
// "SSP-RK time integration is correct, e.g. verified against a known
// analytic ODE solution"). Uses a single-cell Field as a trivial
// "grid" -- ssp_rk2_step has no knowledge of grids, only FieldView, so
// this exercises it in complete isolation from anything spatial.
#include <cmath>

#include "cfe/field/field.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"
#include "test_framework.hpp"

namespace {

// Solves dy/dt = -y, y(0) = 1, whose exact solution is y(t) = exp(-t), by
// taking `n_steps` of size `dt` and returning the numerical y(n_steps*dt).
double integrate_exponential_decay(double dt, int n_steps)
{
  cfe::Field<double, 1> q(1);
  cfe::Field<double, 1> stage1(1);
  cfe::Field<double, 1> r_buf(1);
  q(0, 0) = 1.0;

  auto residual = [](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    out(0, 0) = -in(0, 0);
  };

  for (int step = 0; step < n_steps; ++step) {
    cfe::ssp_rk2_step<double>(q.view(), stage1.view(), r_buf.view(), dt, residual);
  }
  return q(0, 0);
}

}  // namespace

CFE_TEST(test_ssp_rk2_matches_exponential_decay_within_second_order_tolerance)
{
  const double t_final = 1.0;
  const double exact = std::exp(-t_final);

  const double y_coarse = integrate_exponential_decay(0.1, 10);
  CFE_CHECK_NEAR(y_coarse, exact, 1e-3);
}

CFE_TEST(test_ssp_rk2_error_drops_fourfold_when_dt_is_halved)
{
  // A direct order check, not just a plausibility check: SSP-RK2 is
  // 2nd-order, so halving dt should quarter the error against the exact
  // exponential-decay solution -- the same "halve and check ~4x" logic
  // used for the spatial convergence study, applied here to the time
  // integrator alone.
  const double t_final = 1.0;
  const double exact = std::exp(-t_final);

  const double err_coarse = std::fabs(integrate_exponential_decay(0.1, 10) - exact);
  const double err_fine = std::fabs(integrate_exponential_decay(0.05, 20) - exact);

  CFE_CHECK(err_fine > 0.0);
  const double ratio = err_coarse / err_fine;
  CFE_CHECK(ratio > 3.5);
  CFE_CHECK(ratio < 4.5);
}
