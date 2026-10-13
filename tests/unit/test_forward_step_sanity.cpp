// End-to-end sanity check for the forward-step two-block assembly
// (tests/verification/forward_step_setup.hpp), at toy scale -- a short
// run confirming the coupled, stage-interleaved SSP-RK2 stepping
// described in that file's header comment actually produces a finite,
// physically sane solution (positive density/pressure everywhere, no
// NaN/Inf), not just that each piece compiles in isolation. The real
// qualitative/quantitative benchmark evidence (bow shock, Mach stem,
// contact waviness, multiple resolutions, t~4) lives in
// tutorials/euler_2d_forward_step/ and
// tutorials/euler_3d_forward_step_extrusion/ -- this is the fast,
// CI-sized regression guard, run at both Dim=2 and Dim=3.
#include <cmath>
#include <cstddef>

#include "forward_step_setup.hpp"
#include "test_framework.hpp"

namespace {

using Scalar = double;

template <std::size_t Dim>
void run_short_sanity_check(std::size_t nz_cells)
{
  cfe_verification::ForwardStepBlocks<Scalar, Dim> blocks(10, nz_cells, Scalar(0.2));
  const double initial_mass = blocks.total_mass();
  CFE_CHECK(initial_mass > 0.0);

  const std::size_t steps = blocks.run_to(Scalar(0.05), Scalar(0.4));
  CFE_CHECK(steps > 0);

  auto check_finite_and_physical = [&](const auto& state, const auto& grid) {
    for (std::size_t k = 0; k < grid.nz; ++k) {
      for (std::size_t j = 0; j < grid.ny; ++j) {
        for (std::size_t i = 0; i < grid.nx; ++i) {
          const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
          cfe::State<Scalar, cfe::EulerField<Scalar, Dim>::n_components> q;
          for (std::size_t c = 0; c < cfe::EulerField<Scalar, Dim>::n_components; ++c) q[c] = state(cell, c);
          const double rho = blocks.field.density(q);
          const double p = blocks.field.pressure(q);
          CFE_CHECK(std::isfinite(rho));
          CFE_CHECK(std::isfinite(p));
          CFE_CHECK(rho > 0.0);
          CFE_CHECK(p > 0.0);
        }
      }
    }
  };
  check_finite_and_physical(blocks.state1, blocks.grid1);
  check_finite_and_physical(blocks.state2, blocks.grid2);

  const double final_mass = blocks.total_mass();
  CFE_CHECK(std::isfinite(final_mass));
  CFE_CHECK(final_mass > 0.0);
}

}  // namespace

CFE_TEST(test_forward_step_2d_short_run_stays_finite_and_physical)
{
  run_short_sanity_check<2>(1);
}

CFE_TEST(test_forward_step_3d_short_run_stays_finite_and_physical)
{
  run_short_sanity_check<3>(4);
}
