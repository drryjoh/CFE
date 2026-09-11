// Unit tests for CartesianGrid indexing (task spec item 1/tests: "grid
// indexing/connectivity is correct in 1D/2D/3D, including that ghost
// cells resolve to the correct neighbor").
#include <vector>

#include "cfe/grid/structured/cartesian_grid.hpp"
#include "test_framework.hpp"

CFE_TEST(test_1d_grid_flat_index_matches_i_when_y_and_z_are_size_one)
{
  cfe::CartesianGrid grid;
  grid.nx = 5;
  grid.ngx = 2;
  grid.dx = 0.1;

  CFE_CHECK(grid.padded_nx() == 9);
  CFE_CHECK(grid.padded_ny() == 1);
  CFE_CHECK(grid.padded_nz() == 1);
  CFE_CHECK(grid.n_cells_total() == 9);

  for (std::size_t i = 0; i < grid.padded_nx(); ++i) {
    CFE_CHECK(grid.flat_index(i, 0, 0) == i);
  }
}

CFE_TEST(test_2d_grid_flat_index_is_row_major_in_x_then_y)
{
  cfe::CartesianGrid grid;
  grid.nx = 4;
  grid.ny = 3;
  grid.ngx = 1;
  grid.ngy = 1;

  CFE_CHECK(grid.padded_nx() == 6);
  CFE_CHECK(grid.padded_ny() == 5);
  CFE_CHECK(grid.n_cells_total() == 30);

  for (std::size_t j = 0; j < grid.padded_ny(); ++j) {
    for (std::size_t i = 0; i < grid.padded_nx(); ++i) {
      CFE_CHECK(grid.flat_index(i, j, 0) == i + j * grid.padded_nx());
    }
  }
}

CFE_TEST(test_3d_grid_flat_index_covers_all_cells_without_gaps_or_overlap)
{
  cfe::CartesianGrid grid;
  grid.nx = grid.ny = grid.nz = 2;
  grid.ngx = grid.ngy = grid.ngz = 1;

  CFE_CHECK(grid.n_cells_total() == 4 * 4 * 4);

  std::vector<bool> seen(grid.n_cells_total(), false);
  for (std::size_t k = 0; k < grid.padded_nz(); ++k) {
    for (std::size_t j = 0; j < grid.padded_ny(); ++j) {
      for (std::size_t i = 0; i < grid.padded_nx(); ++i) {
        const std::size_t idx = grid.flat_index(i, j, k);
        CFE_CHECK(idx < grid.n_cells_total());
        CFE_CHECK(!seen[idx]);
        seen[idx] = true;
      }
    }
  }
  for (bool s : seen) {
    CFE_CHECK(s);
  }
}

CFE_TEST(test_grid_ghost_cell_is_adjacent_to_first_and_last_real_cell)
{
  cfe::CartesianGrid grid;
  grid.nx = 5;
  grid.ngx = 2;

  // Real cells occupy padded indices [ngx, ngx + nx) = [2, 7).
  const std::size_t first_real = grid.ngx;
  const std::size_t last_real = grid.ngx + grid.nx - 1;

  // The ghost cell immediately below the first real cell, and the one
  // immediately above the last real cell, resolve to distinct flat
  // indices adjacent (by exactly one padded-array slot) to those real
  // cells -- i.e. ghost cells sit in the same contiguous array, not a
  // separate storage region.
  CFE_CHECK(grid.flat_index(first_real - 1, 0, 0) + 1 == grid.flat_index(first_real, 0, 0));
  CFE_CHECK(grid.flat_index(last_real, 0, 0) + 1 == grid.flat_index(last_real + 1, 0, 0));

  // Both outer ghost layers (depth ngx=2) are within bounds and distinct
  // from every real cell's flat index.
  for (std::size_t g = 0; g < grid.ngx; ++g) {
    const std::size_t low_ghost = grid.flat_index(g, 0, 0);
    const std::size_t high_ghost = grid.flat_index(grid.ngx + grid.nx + g, 0, 0);
    CFE_CHECK(low_ghost < first_real);
    CFE_CHECK(high_ghost > last_real);
  }
}

CFE_TEST(test_grid_cell_center_coordinates_are_spaced_by_dx)
{
  cfe::CartesianGrid grid;
  grid.nx = 4;
  grid.ngx = 1;
  grid.dx = 0.5;
  grid.origin_x = 0.0;

  // First real cell (padded index ngx) is centered half a cell-width from
  // the origin; subsequent real cells step by exactly dx.
  CFE_CHECK_NEAR(grid.x_center(grid.ngx), 0.25, 1e-12);
  CFE_CHECK_NEAR(grid.x_center(grid.ngx + 1), 0.75, 1e-12);
  CFE_CHECK_NEAR(grid.x_center(grid.ngx + 2), 1.25, 1e-12);
  CFE_CHECK_NEAR(grid.x_center(grid.ngx + 3), 1.75, 1e-12);
}
