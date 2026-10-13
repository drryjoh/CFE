// Unit tests for BlockInterfaceBoundary
// (grid/boundary/block_interface_boundary.hpp) -- CFE's new same-process
// cross-block ghost-fill mechanism (task 0010, see that file's header
// comment and docs/adr/0011-forward-step-block-interface.md).
//
// Uses a small, hand-computable two-block L-shaped setup mirroring the
// real forward-step geometry at toy scale: block1 (upstream, full
// height, ny=5) sits at x < interface; block2 (downstream, post-step
// height, ny=3) sits at x >= interface, covering only the TOP 3 of
// block1's 5 real rows (local j=2..4 in block1 <-> local j=0..2 in
// block2) -- the bottom 2 rows of block1 (local j=0..1) have no
// downstream counterpart at all, exactly like the real step's riser
// face.
//
// Every real cell gets a distinct, fully-populated state (density,
// momentum_x, momentum_y, energy all different) so a bug that mixed up
// an axis, a component, or a row offset would be caught, not masked by
// coincidentally-equal values.
#include <cstddef>

#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/euler/field.hpp"
#include "cfe/grid/boundary/block_interface_boundary.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "test_framework.hpp"

namespace {

using Field = cfe::EulerField<double, 2>;  // n_components = 4

cfe::CartesianGrid<double> make_block(std::size_t nx, std::size_t ny, double origin_x, double origin_y)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = nx;
  grid.ny = ny;
  grid.ngx = 2;
  grid.ngy = 0;  // this test only exercises fill_x
  grid.dx = 0.1;
  grid.dy = 0.1;
  grid.origin_x = origin_x;
  grid.origin_y = origin_y;
  return grid;
}

void fill_distinct(cfe::Field<double, 4>& state, const cfe::CartesianGrid<double>& grid, double base)
{
  for (std::size_t j = 0; j < grid.ny; ++j) {
    for (std::size_t i = 0; i < grid.nx; ++i) {
      const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, 0);
      const double tag = base + 100.0 * static_cast<double>(j) + static_cast<double>(i);
      state(cell, 0) = 2.0 + tag;    // density
      state(cell, 1) = 3.0 + tag;    // momentum_x (normal to this interface)
      state(cell, 2) = -5.0 - tag;   // momentum_y (tangential)
      state(cell, 3) = 10.0 + tag;   // total energy
    }
  }
}

}  // namespace

CFE_TEST(test_block_interface_boundary_high_side_reads_neighbor_in_shared_band)
{
  // Block1: upstream, full height (ny=5), interface on its HIGH-x end.
  auto grid1 = make_block(3, 5, 0.0, 0.0);
  cfe::Field<double, 4> state1(grid1.n_cells_total());
  fill_distinct(state1, grid1, 0.0);

  // Block2: downstream, post-step height (ny=3), covering block1's
  // local j=2..4 (its own local j=0..2).
  auto grid2 = make_block(4, 3, 0.3, 0.2);
  cfe::Field<double, 4> state2(grid2.n_cells_total());
  fill_distinct(state2, grid2, 1000.0);

  cfe::State<double, 4> inflow;
  inflow[0] = 1.0;
  inflow[1] = 2.0;
  inflow[2] = 0.0;
  inflow[3] = 3.0;

  cfe::BlockInterfaceBoundary<double, Field, cfe::Side::High> boundary{
      .domain_edge_value = inflow,
      .neighbor_field = state2.view(),
      .neighbor_grid = grid2,
      .j_offset_to_neighbor = -2,
      .j_valid_lo = 2,
      .j_valid_hi = 5};
  boundary.fill_x(state1.view(), grid1);

  // Low end: fixed inflow value on every row.
  for (std::size_t j = 0; j < grid1.ny; ++j) {
    for (std::size_t g = 0; g < grid1.ngx; ++g) {
      const std::size_t ghost = grid1.flat_index(grid1.ngx - 1 - g, j, 0);
      for (std::size_t c = 0; c < 4; ++c) {
        CFE_CHECK_NEAR(state1(ghost, c), inflow[c], 1e-12);
      }
    }
  }

  // High end, shared band (block1 local j=2,3,4 <-> block2 local j=0,1,2):
  // ghost layer g reads block2's g-th real cell from ITS own low-x face.
  for (std::size_t local_j = 2; local_j < 5; ++local_j) {
    const std::size_t neighbor_j = local_j - 2;
    for (std::size_t g = 0; g < grid1.ngx; ++g) {
      const std::size_t ghost = grid1.flat_index(grid1.ngx + grid1.nx + g, local_j, 0);
      const std::size_t source = grid2.flat_index(grid2.ngx + g, neighbor_j, 0);
      for (std::size_t c = 0; c < 4; ++c) {
        CFE_CHECK_NEAR(state1(ghost, c), state2(source, c), 1e-12);
      }
    }
  }

  // High end, below the shared band (block1 local j=0,1 -- the riser
  // face): local slip-wall mirror, normal (X) momentum negated.
  for (std::size_t local_j = 0; local_j < 2; ++local_j) {
    for (std::size_t g = 0; g < grid1.ngx; ++g) {
      const std::size_t ghost = grid1.flat_index(grid1.ngx + grid1.nx + g, local_j, 0);
      const std::size_t source = grid1.flat_index(grid1.ngx + grid1.nx - 1 - g, local_j, 0);
      CFE_CHECK_NEAR(state1(ghost, 0), state1(source, 0), 1e-12);
      CFE_CHECK_NEAR(state1(ghost, 1), -state1(source, 1), 1e-12);  // normal: negated
      CFE_CHECK_NEAR(state1(ghost, 2), state1(source, 2), 1e-12);   // tangential: unchanged
      CFE_CHECK_NEAR(state1(ghost, 3), state1(source, 3), 1e-12);
    }
  }
}

CFE_TEST(test_block_interface_boundary_low_side_reads_neighbor_and_extrapolates_high_end)
{
  // Mirror-image setup: block2 (downstream) sees the SAME interface
  // from its own low-x end, which is always fully valid (its entire
  // height lies inside block1's full height) -- no fallback should ever
  // trigger, which this test confirms by checking every row.
  auto grid1 = make_block(3, 5, 0.0, 0.0);
  cfe::Field<double, 4> state1(grid1.n_cells_total());
  fill_distinct(state1, grid1, 0.0);

  auto grid2 = make_block(4, 3, 0.3, 0.2);
  cfe::Field<double, 4> state2(grid2.n_cells_total());
  fill_distinct(state2, grid2, 1000.0);

  cfe::BlockInterfaceBoundary<double, Field, cfe::Side::Low> boundary{
      .domain_edge_value = cfe::State<double, 4>{},
      .neighbor_field = state1.view(),
      .neighbor_grid = grid1,
      .j_offset_to_neighbor = 2,
      .j_valid_lo = 0,
      .j_valid_hi = 3};
  boundary.fill_x(state2.view(), grid2);

  // Low end: every row reads block1's g-th real cell from ITS own
  // high-x face, offset up by 2 rows (block2 local j -> block1 local
  // j+2).
  for (std::size_t local_j = 0; local_j < grid2.ny; ++local_j) {
    const std::size_t neighbor_j = local_j + 2;
    for (std::size_t g = 0; g < grid2.ngx; ++g) {
      const std::size_t ghost = grid2.flat_index(grid2.ngx - 1 - g, local_j, 0);
      const std::size_t source = grid1.flat_index(grid1.ngx + grid1.nx - 1 - g, neighbor_j, 0);
      for (std::size_t c = 0; c < 4; ++c) {
        CFE_CHECK_NEAR(state2(ghost, c), state1(source, c), 1e-12);
      }
    }
  }

  // High end: zero-order extrapolation (outlet), identical to
  // InflowOutflowBoundary's own high end.
  for (std::size_t local_j = 0; local_j < grid2.ny; ++local_j) {
    const std::size_t nearest_real = grid2.ngx + grid2.nx - 1;
    const std::size_t source = grid2.flat_index(nearest_real, local_j, 0);
    for (std::size_t g = 0; g < grid2.ngx; ++g) {
      const std::size_t ghost = grid2.flat_index(grid2.ngx + grid2.nx + g, local_j, 0);
      for (std::size_t c = 0; c < 4; ++c) {
        CFE_CHECK_NEAR(state2(ghost, c), state2(source, c), 1e-12);
      }
    }
  }
}

CFE_TEST(test_block_interface_boundary_low_side_fallback_branch_is_reachable_and_correct)
{
  // The real forward-step geometry never exercises Side::Low's
  // riser-face fallback (the downstream block's entire height always
  // lies inside the upstream block's -- see the test above, "no
  // fallback should ever trigger"). This test forces it anyway with a
  // synthetic partial-overlap setup (a hypothetical neighbor that only
  // covers the TOP 2 of this block's 4 rows), confirming the fallback
  // branch itself -- not just the neighbor-read branch -- is correct
  // for Side::Low too, not only for Side::High (which the first test in
  // this file already covers via the real geometry's own riser face).
  auto grid_self = make_block(3, 4, 0.3, 0.0);
  cfe::Field<double, 4> state_self(grid_self.n_cells_total());
  fill_distinct(state_self, grid_self, 0.0);

  auto grid_neighbor = make_block(3, 2, 0.0, 0.2);
  cfe::Field<double, 4> state_neighbor(grid_neighbor.n_cells_total());
  fill_distinct(state_neighbor, grid_neighbor, 1000.0);

  // Only local rows 2,3 of grid_self (the top half) have a neighbor
  // counterpart; rows 0,1 do not -- those should fall back to a local
  // slip-wall-style mirror.
  cfe::BlockInterfaceBoundary<double, Field, cfe::Side::Low> boundary{
      .domain_edge_value = cfe::State<double, 4>{},
      .neighbor_field = state_neighbor.view(),
      .neighbor_grid = grid_neighbor,
      .j_offset_to_neighbor = -2,
      .j_valid_lo = 2,
      .j_valid_hi = 4};
  boundary.fill_x(state_self.view(), grid_self);

  // Rows 0,1: fallback -- local slip-wall mirror, normal (X) momentum negated.
  for (std::size_t local_j = 0; local_j < 2; ++local_j) {
    for (std::size_t g = 0; g < grid_self.ngx; ++g) {
      const std::size_t ghost = grid_self.flat_index(grid_self.ngx - 1 - g, local_j, 0);
      const std::size_t source = grid_self.flat_index(grid_self.ngx + g, local_j, 0);
      CFE_CHECK_NEAR(state_self(ghost, 0), state_self(source, 0), 1e-12);
      CFE_CHECK_NEAR(state_self(ghost, 1), -state_self(source, 1), 1e-12);  // normal: negated
      CFE_CHECK_NEAR(state_self(ghost, 2), state_self(source, 2), 1e-12);   // tangential: unchanged
      CFE_CHECK_NEAR(state_self(ghost, 3), state_self(source, 3), 1e-12);
    }
  }

  // Rows 2,3: real neighbor read.
  for (std::size_t local_j = 2; local_j < 4; ++local_j) {
    const std::size_t neighbor_j = local_j - 2;
    for (std::size_t g = 0; g < grid_self.ngx; ++g) {
      const std::size_t ghost = grid_self.flat_index(grid_self.ngx - 1 - g, local_j, 0);
      const std::size_t source = grid_neighbor.flat_index(grid_neighbor.ngx + grid_neighbor.nx - 1 - g,
                                                           neighbor_j, 0);
      for (std::size_t c = 0; c < 4; ++c) {
        CFE_CHECK_NEAR(state_self(ghost, c), state_neighbor(source, c), 1e-12);
      }
    }
  }
}
