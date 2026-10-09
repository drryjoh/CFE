// Unit tests for ghost-cell filling (task spec item 3/tests: "static and
// periodic boundary conditions produce the correct ghost-cell values at
// domain edges").
#include <stdexcept>

#include "cfe/field/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/ghost/ghost_fill.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "test_framework.hpp"

namespace {

cfe::CartesianGrid<double> make_1d_grid(std::size_t nx, std::size_t ngx)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = nx;
  grid.ngx = ngx;
  grid.dx = 1.0;
  return grid;
}

}  // namespace

CFE_TEST(test_periodic_boundary_wraps_ghost_cells_to_opposite_real_boundary)
{
  const auto grid = make_1d_grid(5, 2);
  cfe::Field<double, 1> state(grid.n_cells_total());

  // Real cells at padded index 2..6 get values 10, 20, 30, 40, 50.
  for (std::size_t i = 0; i < grid.nx; ++i) {
    state(grid.flat_index(grid.ngx + i, 0, 0), 0) = static_cast<double>((i + 1) * 10);
  }

  cfe::fill_ghost_cells(state.view(), grid, cfe::Axis::X, cfe::PeriodicBoundary{});

  // Low ghost layer (padded 0,1) should equal the last two real cells
  // (40, 50) in wrap order: nearest-to-boundary ghost (index 1) mirrors
  // the nearest-to-boundary real cell (50).
  CFE_CHECK_NEAR(state(grid.flat_index(1, 0, 0), 0), 50.0, 1e-12);
  CFE_CHECK_NEAR(state(grid.flat_index(0, 0, 0), 0), 40.0, 1e-12);

  // High ghost layer (padded 7,8) should equal the first two real cells
  // (10, 20).
  CFE_CHECK_NEAR(state(grid.flat_index(7, 0, 0), 0), 10.0, 1e-12);
  CFE_CHECK_NEAR(state(grid.flat_index(8, 0, 0), 0), 20.0, 1e-12);
}

CFE_TEST(test_static_boundary_writes_fixed_value_into_every_ghost_cell)
{
  const auto grid = make_1d_grid(4, 2);
  cfe::Field<double, 1> state(grid.n_cells_total());

  cfe::StaticBoundary<double, 1> bc(cfe::State<double, 1>(-1.0), cfe::State<double, 1>(99.0));
  cfe::fill_ghost_cells(state.view(), grid, cfe::Axis::X, bc);

  for (std::size_t g = 0; g < grid.ngx; ++g) {
    CFE_CHECK_NEAR(state(grid.flat_index(g, 0, 0), 0), -1.0, 1e-12);
    CFE_CHECK_NEAR(state(grid.flat_index(grid.ngx + grid.nx + g, 0, 0), 0), 99.0, 1e-12);
  }
}

CFE_TEST(test_static_boundary_is_generic_over_component_count)
{
  const auto grid = make_1d_grid(3, 1);
  cfe::Field<double, 3> state(grid.n_cells_total());

  cfe::State<double, 3> low, high;
  low[0] = 1.0;
  low[1] = 2.0;
  low[2] = 3.0;
  high[0] = 4.0;
  high[1] = 5.0;
  high[2] = 6.0;
  cfe::StaticBoundary<double, 3> bc(low, high);
  cfe::fill_ghost_cells(state.view(), grid, cfe::Axis::X, bc);

  const std::size_t low_ghost = grid.flat_index(0, 0, 0);
  const std::size_t high_ghost = grid.flat_index(grid.ngx + grid.nx, 0, 0);
  for (std::size_t c = 0; c < 3; ++c) {
    CFE_CHECK_NEAR(state(low_ghost, c), low[c], 1e-12);
    CFE_CHECK_NEAR(state(high_ghost, c), high[c], 1e-12);
  }
}

CFE_TEST(test_periodic_boundary_fills_y_axis_ghost_cells_in_2d_grid)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = 3;
  grid.ny = 4;
  grid.ngx = 1;
  grid.ngy = 1;
  cfe::Field<double, 1> state(grid.n_cells_total());

  // Real cells at padded j = 1..4 get values 100*j for a fixed i.
  const std::size_t i = grid.ngx;
  for (std::size_t j = 0; j < grid.ny; ++j) {
    state(grid.flat_index(i, grid.ngy + j, 0), 0) = static_cast<double>((j + 1) * 100);
  }

  cfe::fill_ghost_cells(state.view(), grid, cfe::Axis::Y, cfe::PeriodicBoundary{});

  // Low ghost (padded j=0) mirrors the last real row (400); high ghost
  // (padded j=5) mirrors the first real row (100).
  CFE_CHECK_NEAR(state(grid.flat_index(i, 0, 0), 0), 400.0, 1e-12);
  CFE_CHECK_NEAR(state(grid.flat_index(i, grid.ngy + grid.ny, 0), 0), 100.0, 1e-12);
}

// Regression test for a code-review finding (see agent_history.md's
// 2026-09-30 follow-up entry): PeriodicBoundary::fill_x/y/z must reject
// nx < ngx (ny < ngy, nz < ngz) with an *always-on* runtime check, not an
// `assert` that silently compiles out under `NDEBUG` -- this project's
// own default and benchmarked build type is Release, so a guard that
// only exists in Debug builds would never actually fire where it
// matters. This test exercises the real, as-built binary (whatever
// CMAKE_BUILD_TYPE configured it, Release by default) rather than only a
// Debug configuration, specifically to catch a regression back to
// `assert`.
CFE_TEST(test_periodic_boundary_rejects_narrower_extent_than_ghost_depth_on_every_axis)
{
  {
    cfe::CartesianGrid<double> grid = make_1d_grid(1, 2);  // nx=1 < ngx=2
    cfe::Field<double, 1> state(grid.n_cells_total());
    bool threw = false;
    try {
      cfe::fill_ghost_cells(state.view(), grid, cfe::Axis::X, cfe::PeriodicBoundary{});
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CFE_CHECK(threw);
  }
  {
    cfe::CartesianGrid<double> grid;
    grid.nx = 3;
    grid.ny = 1;
    grid.ngx = 1;
    grid.ngy = 2;  // ny=1 < ngy=2
    cfe::Field<double, 1> state(grid.n_cells_total());
    bool threw = false;
    try {
      cfe::fill_ghost_cells(state.view(), grid, cfe::Axis::Y, cfe::PeriodicBoundary{});
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CFE_CHECK(threw);
  }
  {
    cfe::CartesianGrid<double> grid;
    grid.nx = 3;
    grid.nz = 1;
    grid.ngx = 1;
    grid.ngz = 2;  // nz=1 < ngz=2
    cfe::Field<double, 1> state(grid.n_cells_total());
    bool threw = false;
    try {
      cfe::fill_ghost_cells(state.view(), grid, cfe::Axis::Z, cfe::PeriodicBoundary{});
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CFE_CHECK(threw);
  }
}

CFE_TEST(test_inflow_outflow_boundary_fixes_low_end_and_extrapolates_high_end)
{
  const auto grid = make_1d_grid(4, 2);
  cfe::Field<double, 1> state(grid.n_cells_total());

  // Real cells (padded index 2..5) get values 10, 20, 30, 40 -- the
  // nearest-to-high-boundary real cell is 40.
  for (std::size_t i = 0; i < grid.nx; ++i) {
    state(grid.flat_index(grid.ngx + i, 0, 0), 0) = static_cast<double>((i + 1) * 10);
  }

  cfe::InflowOutflowBoundary<double, 1> bc(cfe::State<double, 1>(-7.0));
  cfe::fill_ghost_cells(state.view(), grid, cfe::Axis::X, bc);

  // Low end: every ghost layer gets the fixed inflow value.
  for (std::size_t g = 0; g < grid.ngx; ++g) {
    CFE_CHECK_NEAR(state(grid.flat_index(g, 0, 0), 0), -7.0, 1e-12);
  }
  // High end: every ghost layer copies the SAME single nearest real cell
  // (40) -- zero-order extrapolation, not a mirror/reflection.
  for (std::size_t g = 0; g < grid.ngx; ++g) {
    CFE_CHECK_NEAR(state(grid.flat_index(grid.ngx + grid.nx + g, 0, 0), 0), 40.0, 1e-12);
  }
}
