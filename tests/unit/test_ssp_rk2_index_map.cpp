// Regression tests for the interior-cells-only ssp_rk2_step design
// (see agent_history.md's 2026-10-01 entry and docs/adr/ for the design
// rationale). Three properties that earlier tests didn't directly
// exercise:
//   1. detail::CartesianRealCellIndexMap is a bijection onto exactly the
//      real cells, on a genuinely RECTANGULAR grid (nx != ny != nz) --
//      every earlier 2D/3D test used a cube, which could not catch an
//      nx/ny-mixup bug in the index decomposition.
//   2. ssp_rk2_step's combine kernels never read or write a ghost cell
//      of EITHER state or residual_scratch -- verified by planting an
//      exact sentinel in every ghost cell before stepping and checking
//      it survives byte-for-byte (not just "isn't NaN"), on a
//      rectangular grid, using a residual callable with no ghost-fill
//      of its own (isolating the combine step's own behavior from
//      FvmSolver's separate fill_ghost_cells calls).
//   3. The whole mechanism (index map + ssp_rk2_step) works for a
//      multi-component field (N=3) under both AoSLayout and SoALayout --
//      untested until now, since FvmSolver/ScalarAdvectionField hardcode
//      N=1.
#include <cstddef>
#include <set>

#include "cfe/field/field.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"
#include "test_framework.hpp"

namespace {

// Exact closed-form solution of one SSP-RK2 (Heun's method) step applied
// to the linear per-cell ODE dy/dt = -y, derived by hand:
//   stage1 = y0 * (1 - dt)
//   y1     = 0.5*y0 + 0.5*(stage1 - dt*stage1) = y0 * (1 - dt + 0.5*dt^2)
// Exact (not an approximation) because the residual is linear and
// decoupled per cell -- lets every real-cell check below use a tight
// tolerance instead of "changed in the right direction."
double exact_decay_step(double y0, double dt) { return y0 * (1.0 - dt + 0.5 * dt * dt); }

}  // namespace

CFE_TEST(test_cartesian_real_cell_index_map_is_a_bijection_for_rectangular_2d_grid)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = 5;  // deliberately nx != ny: a cube could not catch a
  grid.ny = 9;  // local_i/local_j mixup in the index decomposition.
  grid.ngx = 2;
  grid.ngy = 2;

  cfe::detail::CartesianRealCellIndexMap<double, 2> index_map{grid};

  std::set<std::size_t> visited;
  for (std::size_t r = 0; r < grid.nx * grid.ny; ++r) {
    const std::size_t cell = index_map(r);
    CFE_CHECK(visited.insert(cell).second);  // never the same storage index twice
  }

  std::set<std::size_t> expected_real_cells;
  for (std::size_t j = 0; j < grid.ny; ++j) {
    for (std::size_t i = 0; i < grid.nx; ++i) {
      expected_real_cells.insert(grid.flat_index(grid.ngx + i, grid.ngy + j, 0));
    }
  }
  CFE_CHECK(visited == expected_real_cells);  // visits exactly the real cells, nothing else
}

CFE_TEST(test_cartesian_real_cell_index_map_is_a_bijection_for_rectangular_3d_grid)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = 4;  // all three different, same reasoning as the 2D case
  grid.ny = 7;
  grid.nz = 3;
  grid.ngx = 2;
  grid.ngy = 2;
  grid.ngz = 2;

  cfe::detail::CartesianRealCellIndexMap<double, 3> index_map{grid};

  std::set<std::size_t> visited;
  for (std::size_t r = 0; r < grid.nx * grid.ny * grid.nz; ++r) {
    const std::size_t cell = index_map(r);
    CFE_CHECK(visited.insert(cell).second);
  }

  std::set<std::size_t> expected_real_cells;
  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        expected_real_cells.insert(grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k));
      }
    }
  }
  CFE_CHECK(visited == expected_real_cells);
}

CFE_TEST(test_ssp_rk2_step_leaves_ghost_state_and_ghost_residual_exactly_untouched_on_rectangular_grid)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = 5;
  grid.ny = 7;
  grid.ngx = 2;
  grid.ngy = 2;
  grid.dx = 1.0;
  grid.dy = 1.0;
  const std::size_t n_total = grid.n_cells_total();
  const std::size_t n_active = grid.nx * grid.ny;

  cfe::Field<double, 1> state(n_total);
  cfe::Field<double, 1> stage1(n_total);
  cfe::Field<double, 1> residual_scratch(n_total);

  constexpr double kGhostSentinel = -999.0;
  for (std::size_t idx = 0; idx < n_total; ++idx) {
    state.data()[idx] = kGhostSentinel;
    residual_scratch.data()[idx] = kGhostSentinel;
  }
  for (std::size_t j = 0; j < grid.ny; ++j) {
    for (std::size_t i = 0; i < grid.nx; ++i) {
      state(grid.flat_index(grid.ngx + i, grid.ngy + j, 0), 0) =
          static_cast<double>(i * 10 + j + 1);
    }
  }

  cfe::detail::CartesianRealCellIndexMap<double, 2> index_map{grid};

  // Deliberately no ghost-fill at all (unlike FvmSolver::residual()):
  // this isolates ssp_rk2_step's own combine kernels from any other
  // code path that legitimately writes ghost cells, so any ghost-cell
  // change observed below can only have come from ssp_rk2_step itself.
  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    for (std::size_t r = 0; r < n_active; ++r) {
      const std::size_t cell = index_map(r);
      out(cell, 0) = -in(cell, 0);
    }
  };

  constexpr double dt = 0.1;
  cfe::ssp_rk2_step<double>(state.view(), stage1.view(), residual_scratch.view(), dt, residual,
                             n_active, index_map);

  for (std::size_t j = 0; j < grid.padded_ny(); ++j) {
    for (std::size_t i = 0; i < grid.padded_nx(); ++i) {
      const std::size_t idx = grid.flat_index(i, j, 0);
      const bool is_real = (i >= grid.ngx && i < grid.ngx + grid.nx) &&
                            (j >= grid.ngy && j < grid.ngy + grid.ny);
      if (!is_real) {
        CFE_CHECK_NEAR(state.data()[idx], kGhostSentinel, 0.0);
        CFE_CHECK_NEAR(residual_scratch.data()[idx], kGhostSentinel, 0.0);
      } else {
        const double i_local = static_cast<double>(i - grid.ngx);
        const double j_local = static_cast<double>(j - grid.ngy);
        const double y0 = i_local * 10.0 + j_local + 1.0;
        CFE_CHECK_NEAR(state.data()[idx], exact_decay_step(y0, dt), 1e-12);
      }
    }
  }
}

namespace {

template <class Layout>
void run_multi_component_case()
{
  cfe::CartesianGrid<double> grid;
  grid.nx = 6;
  grid.ngx = 2;
  grid.dx = 1.0;
  constexpr std::size_t N = 3;
  const std::size_t n_total = grid.n_cells_total();
  const std::size_t n_active = grid.nx;

  cfe::Field<double, N, Layout> state(n_total);
  cfe::Field<double, N, Layout> stage1(n_total);
  cfe::Field<double, N, Layout> residual_scratch(n_total);

  constexpr double kGhostSentinel = -777.0;
  for (std::size_t idx = 0; idx < n_total; ++idx) {
    for (std::size_t c = 0; c < N; ++c) {
      state(idx, c) = kGhostSentinel;
      residual_scratch(idx, c) = kGhostSentinel;
    }
  }
  for (std::size_t i = 0; i < grid.nx; ++i) {
    for (std::size_t c = 0; c < N; ++c) {
      state(grid.flat_index(grid.ngx + i, 0, 0), c) = static_cast<double>((i + 1) * 10 + c);
    }
  }

  cfe::detail::CartesianRealCellIndexMap<double, 1> index_map{grid};
  auto residual = [&](cfe::FieldView<double, N, Layout> in, cfe::FieldView<double, N, Layout> out) {
    for (std::size_t r = 0; r < n_active; ++r) {
      const std::size_t cell = index_map(r);
      for (std::size_t c = 0; c < N; ++c) out(cell, c) = -in(cell, c);
    }
  };

  constexpr double dt = 0.1;
  cfe::ssp_rk2_step<double>(state.view(), stage1.view(), residual_scratch.view(), dt, residual,
                             n_active, index_map);

  for (std::size_t i = 0; i < grid.padded_nx(); ++i) {
    const bool is_real = (i >= grid.ngx && i < grid.ngx + grid.nx);
    const std::size_t idx = grid.flat_index(i, 0, 0);
    for (std::size_t c = 0; c < N; ++c) {
      if (!is_real) {
        CFE_CHECK_NEAR(state(idx, c), kGhostSentinel, 0.0);
      } else {
        const double i_local = static_cast<double>(i - grid.ngx);
        const double y0 = (i_local + 1.0) * 10.0 + static_cast<double>(c);
        CFE_CHECK_NEAR(state(idx, c), exact_decay_step(y0, dt), 1e-12);
      }
    }
  }
}

}  // namespace

CFE_TEST(test_ssp_rk2_step_handles_multiple_components_under_aos_layout)
{
  run_multi_component_case<cfe::AoSLayout>();
}

CFE_TEST(test_ssp_rk2_step_handles_multiple_components_under_soa_layout)
{
  run_multi_component_case<cfe::SoALayout>();
}
