// Regression test for a code-review finding (see agent_history.md's
// 2026-09-30 entry): FvmSolver::residual() used to write only real-cell
// entries of `out`, while ssp_rk2_step's combine kernels read/write
// every padded cell with no knowledge of which indices are real vs.
// ghost. On CUDA that was a genuine uninitialized-memory read
// (cudaMalloc doesn't zero-initialize); on CPU it was silently masked by
// std::vector's zero-initialization. This test makes the masking
// impossible: it deliberately poisons the residual scratch buffer with
// NaN *before* stepping, so if FvmSolver::residual() ever again leaves a
// ghost-cell residual entry unwritten, that poison survives into the
// combine kernel's output and this test catches it -- on any backend,
// not just CUDA.
#include <cmath>
#include <cstddef>
#include <limits>

#include "cfe/field/field.hpp"
#include "cfe/fields/scalar_advection/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"
#include "test_framework.hpp"

CFE_TEST(test_fvm_solver_residual_never_leaves_nan_poison_in_output_after_ssp_rk2_step)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = 8;
  grid.ngx = 2;
  grid.dx = 1.0;

  cfe::Field<double, 1> state(grid.n_cells_total());
  cfe::Field<double, 1> stage1(grid.n_cells_total());
  cfe::Field<double, 1> residual_scratch(grid.n_cells_total());

  // A constant field: the true residual is exactly zero everywhere,
  // real or ghost, so any NaN in the final state can only have come
  // from the poisoned scratch buffer below, not from real physics.
  for (std::size_t idx = 0; idx < state.n_cells(); ++idx) state.data()[idx] = 7.0;

  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::size_t idx = 0; idx < residual_scratch.n_cells(); ++idx) {
    residual_scratch.data()[idx] = nan;
  }

  cfe::ScalarAdvectionField<double, 1> field{cfe::Vector<double, 1>(1.0)};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 1>, cfe::PeriodicBoundary>
      solver{grid, field, cfe::PeriodicBoundary{}};
  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver.residual(in, out);
  };

  cfe::ssp_rk2_step<double>(state.view(), stage1.view(), residual_scratch.view(), 0.01, residual);

  for (std::size_t i = 0; i < grid.padded_nx(); ++i) {
    const double value = state.data()[grid.flat_index(i, 0, 0)];
    CFE_CHECK(!std::isnan(value));
    CFE_CHECK_NEAR(value, 7.0, 1e-12);
  }
}
