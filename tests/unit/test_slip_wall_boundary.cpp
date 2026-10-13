// Unit tests for SlipWallBoundary (grid/boundary/slip_wall_boundary.hpp)
// -- CFE's first Field-aware BoundaryCondition (AGENTS.md #17,
// ARCHITECTURE.md #11). Checks the two properties that make it correct:
// (1) density/energy/tangential-momentum mirror the interior value
// unchanged, (2) the single momentum component normal to the wall axis
// is negated -- exercised on EulerField<Scalar,2> specifically so the
// X-wall test can also confirm the tangential (Y) momentum component is
// left untouched, not just that "some" component gets flipped.
#include <cstddef>

#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/euler/field.hpp"
#include "cfe/grid/boundary/slip_wall_boundary.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "test_framework.hpp"

CFE_TEST(test_slip_wall_boundary_fill_x_mirrors_density_and_negates_normal_momentum_only)
{
  using Field = cfe::EulerField<double, 2>;  // n_components = 4: rho, mom_x, mom_y, E
  cfe::CartesianGrid<double> grid;
  grid.nx = 4;
  grid.ny = 1;
  grid.ngx = 2;
  grid.ngy = 0;
  grid.dx = 1.0;
  grid.dy = 1.0;

  cfe::Field<double, 4> state(grid.n_cells_total());
  // Fill every real cell with a distinct, fully-populated state so a
  // bug that mixed up components or axes would be caught.
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy, 0);
    state(cell, 0) = 2.0 + static_cast<double>(i);        // density
    state(cell, 1) = 3.0 + static_cast<double>(i);        // momentum_x (normal to an X wall)
    state(cell, 2) = -5.0 - static_cast<double>(i);       // momentum_y (tangential to an X wall)
    state(cell, 3) = 10.0 + static_cast<double>(i);       // total energy
  }

  cfe::SlipWallBoundary<Field> boundary;
  boundary.fill_x(state.view(), grid);

  // Low-side ghosts: ghost layer g mirrors real cell at local index g
  // (nearest-to-wall real cell is local index 0).
  for (std::size_t g = 0; g < grid.ngx; ++g) {
    const std::size_t ghost_cell = grid.flat_index(grid.ngx - 1 - g, grid.ngy, 0);
    const std::size_t source_cell = grid.flat_index(grid.ngx + g, grid.ngy, 0);
    CFE_CHECK_NEAR(state(ghost_cell, 0), state(source_cell, 0), 1e-12);
    CFE_CHECK_NEAR(state(ghost_cell, 1), -state(source_cell, 1), 1e-12);  // normal: negated
    CFE_CHECK_NEAR(state(ghost_cell, 2), state(source_cell, 2), 1e-12);   // tangential: unchanged
    CFE_CHECK_NEAR(state(ghost_cell, 3), state(source_cell, 3), 1e-12);
  }

  // High-side ghosts: mirrored from the opposite end.
  for (std::size_t g = 0; g < grid.ngx; ++g) {
    const std::size_t ghost_cell = grid.flat_index(grid.ngx + grid.nx + g, grid.ngy, 0);
    const std::size_t source_cell = grid.flat_index(grid.ngx + grid.nx - 1 - g, grid.ngy, 0);
    CFE_CHECK_NEAR(state(ghost_cell, 0), state(source_cell, 0), 1e-12);
    CFE_CHECK_NEAR(state(ghost_cell, 1), -state(source_cell, 1), 1e-12);
    CFE_CHECK_NEAR(state(ghost_cell, 2), state(source_cell, 2), 1e-12);
    CFE_CHECK_NEAR(state(ghost_cell, 3), state(source_cell, 3), 1e-12);
  }
}

CFE_TEST(test_slip_wall_boundary_fill_y_negates_momentum_y_only)
{
  using Field = cfe::EulerField<double, 2>;
  cfe::CartesianGrid<double> grid;
  grid.nx = 1;
  grid.ny = 3;
  grid.ngx = 0;
  grid.ngy = 2;
  grid.dx = 1.0;
  grid.dy = 1.0;

  cfe::Field<double, 4> state(grid.n_cells_total());
  for (std::size_t j = 0; j < grid.ny; ++j) {
    const std::size_t cell = grid.flat_index(grid.ngx, grid.ngy + j, 0);
    state(cell, 0) = 1.0 + static_cast<double>(j);
    state(cell, 1) = 4.0 + static_cast<double>(j);   // momentum_x (tangential to a Y wall)
    state(cell, 2) = 7.0 + static_cast<double>(j);   // momentum_y (normal to a Y wall)
    state(cell, 3) = 9.0 + static_cast<double>(j);
  }

  cfe::SlipWallBoundary<Field> boundary;
  boundary.fill_y(state.view(), grid);

  for (std::size_t g = 0; g < grid.ngy; ++g) {
    const std::size_t ghost_cell = grid.flat_index(grid.ngx, grid.ngy - 1 - g, 0);
    const std::size_t source_cell = grid.flat_index(grid.ngx, grid.ngy + g, 0);
    CFE_CHECK_NEAR(state(ghost_cell, 0), state(source_cell, 0), 1e-12);
    CFE_CHECK_NEAR(state(ghost_cell, 1), state(source_cell, 1), 1e-12);   // tangential: unchanged
    CFE_CHECK_NEAR(state(ghost_cell, 2), -state(source_cell, 2), 1e-12);  // normal: negated
    CFE_CHECK_NEAR(state(ghost_cell, 3), state(source_cell, 3), 1e-12);
  }
}
