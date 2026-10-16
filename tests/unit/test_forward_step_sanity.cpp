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

#include "cfe/core/types.hpp"

#include "forward_step_setup.hpp"
#include "test_framework.hpp"

namespace {

using Scalar = double;

template <std::size_t Dim>
void run_short_sanity_check(cfe::Index nz_cells)
{
  cfe_verification::ForwardStepBlocks<Scalar, Dim> blocks(10, nz_cells, Scalar(0.2));
  const double initial_mass = blocks.total_mass();
  CFE_CHECK(initial_mass > 0.0);

  const cfe::Index steps = blocks.run_to(Scalar(0.05), Scalar(0.4));
  CFE_CHECK(steps > 0);

  auto check_finite_and_physical = [&](const auto& state, const auto& grid) {
    for (cfe::Index k = 0; k < grid.nz; ++k) {
      for (cfe::Index j = 0; j < grid.ny; ++j) {
        for (cfe::Index i = 0; i < grid.nx; ++i) {
          const cfe::Index cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
          cfe::State<Scalar, cfe::EulerField<Scalar, Dim>::n_components> q;
          for (cfe::Index c = 0; c < cfe::EulerField<Scalar, Dim>::n_components; ++c) q[c] = state(cell, c);
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
  check_finite_and_physical(blocks.state_blocks.view(blocks.kUpstream), blocks.grid1);
  check_finite_and_physical(blocks.state_blocks.view(blocks.kDownstream), blocks.grid2);

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

// Directly checks the memory-layout claim this task's `BlockSet` refactor
// rests on -- not just that the two-block solve still produces the right
// ANSWER (the sanity checks above), but that block2's view genuinely sits
// immediately after block1's in ONE allocation, rather than two
// allocations that merely happen to produce the same numbers. Computed
// from `n_cells()`/`n_components()`, not hardcoded, so it stays correct
// at any grid resolution.
CFE_TEST(test_forward_step_block_set_buffers_are_genuinely_one_contiguous_allocation)
{
  cfe_verification::ForwardStepBlocks<Scalar, 2> blocks(10);
  using Field = cfe::EulerField<Scalar, 2>;
  constexpr std::size_t N = Field::n_components;

  auto check_contiguous = [&](auto& block_set) {
    const auto view0 = block_set.view(blocks.kUpstream);
    const auto view1 = block_set.view(blocks.kDownstream);
    CFE_CHECK(view1.data() == view0.data() + view0.n_cells() * N);
  };
  check_contiguous(blocks.state_blocks);
  check_contiguous(blocks.stage1_blocks);
  check_contiguous(blocks.scratch_blocks);
}
