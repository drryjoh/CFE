// MPI correctness test (task spec 0004): decomposes ScalarAdvectionField
// over `kGlobalNx` cells across however many ranks mpirun launched this
// binary with, using MpiHaloBoundary for X-axis halo exchange, and
// checks that the result is BIT-IDENTICAL (tolerance 0) to an
// independently-computed, single-rank reference run of the identical
// global problem -- not just "close" to the analytic exact solution.
//
// Why bit-identical is achievable and the right bar: ssp_rk2_step/
// FvmSolver::residual are fully per-cell-independent (Backend::run, no
// reduction/atomics), and a halo-exchanged ghost value is a literal byte
// copy of the owning rank's real-cell value, so stencil arithmetic at a
// rank boundary is byte-identical to a single-rank run of the SAME
// global problem -- PROVIDED each cell's coordinate is computed
// identically in both cases. This test keeps `grid.origin_x = 0` on
// every rank (CartesianGrid::x_center() is only ever used here for the
// initial condition, never inside a per-timestep kernel -- see that
// type's own header comment) and derives each cell's coordinate directly
// from its GLOBAL index (`(global_i + 0.5) * dx`), the exact same single
// multiply-add `x_center()` itself performs when `origin_x == 0` -- so
// there is no `(a+b)*dx` vs. `a*dx+b*dx` rounding mismatch to worry
// about (see docs/adr/0009-mpi-domain-decomposition.md for the general
// version of this hazard).
//
// The reference run needs no MPI communication of its own: every
// process independently simulates the FULL global domain, single-rank
// style (no decomposition, ordinary PeriodicBoundary), entirely locally
// -- then compares its own decomposed real cells against the matching
// slice of that local reference. No gather, no second binary.
//
// Not part of the single-process cfe_unit_tests binary: an MPI test
// needs its own executable launched under mpirun/mpiexec (see
// tests/mpi/CMakeLists.txt), only built/registered when CFE_ENABLE_MPI
// is ON.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "cfe/backend/mpi/mpi_environment.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/scalar_advection/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/boundary/mpi_halo_boundary.hpp"
#include "cfe/grid/partition/slab_partition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDomainLength = 1.0;
constexpr double kAdvectionSpeed = 1.0;
constexpr double kFinalTime = 0.3;
constexpr double kCfl = 0.4;
constexpr std::size_t kGlobalNx = 160;

double exact_solution(double x, double t)
{
  return std::sin(2.0 * kPi * (x - kAdvectionSpeed * t) / kDomainLength);
}

// Same single multiply-add `CartesianGrid::x_center()` performs when
// `origin_x == 0` -- see this file's header comment for why that
// equivalence is what makes bit-identity achievable.
double cell_center(std::size_t global_i, double dx)
{
  return (static_cast<double>(global_i) + 0.5) * dx;
}

// Lands exactly on kFinalTime with a CFL-respecting step count -- same
// pattern every existing convergence test uses. Depends only on `dx`
// (identical on every rank, since it's derived from the same
// `kGlobalNx`), so every rank/the reference all compute the same
// `n_steps`/`dt`.
void compute_steps(double dx, int& n_steps, double& dt)
{
  const double dt_target = kCfl * dx / kAdvectionSpeed;
  n_steps = static_cast<int>(std::ceil(kFinalTime / dt_target));
  dt = kFinalTime / static_cast<double>(n_steps);
}

// Runs the full global problem on ONE (virtual) rank, entirely locally,
// no MPI involved -- the reference this test's decomposed run must
// match bit-for-bit.
std::vector<double> run_reference(std::size_t global_nx)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = global_nx;
  grid.ngx = 2;
  grid.dx = kDomainLength / static_cast<double>(global_nx);

  cfe::Field<double, 1> state(grid.n_cells_total());
  cfe::Field<double, 1> stage1(grid.n_cells_total());
  cfe::Field<double, 1> scratch(grid.n_cells_total());

  for (std::size_t i = 0; i < grid.nx; ++i) {
    state(grid.flat_index(grid.ngx + i, 0, 0), 0) = exact_solution(cell_center(i, grid.dx), 0.0);
  }

  cfe::ScalarAdvectionField<double, 1> field{cfe::Vector<double, 1>(kAdvectionSpeed)};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 1>, cfe::PeriodicBoundary>
      solver{grid, field, cfe::PeriodicBoundary{}};
  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver.residual(in, out);
  };

  int n_steps;
  double dt;
  compute_steps(grid.dx, n_steps, dt);
  for (int step = 0; step < n_steps; ++step) {
    cfe::ssp_rk2_step<double>(state.view(), stage1.view(), scratch.view(), dt, residual,
                               solver.active_cell_count(), solver.active_cell_index_map());
  }

  std::vector<double> result(global_nx);
  for (std::size_t i = 0; i < global_nx; ++i) {
    result[i] = state(grid.flat_index(grid.ngx + i, 0, 0), 0);
  }
  return result;
}

// Runs this rank's own slice of the SAME global problem, decomposed via
// SlabPartition + MpiHaloBoundary, and returns its local real cells.
std::vector<double> run_decomposed(std::size_t global_nx, const cfe::SlabPartition& partition)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = partition.local_nx;
  grid.ngx = 2;
  grid.dx = kDomainLength / static_cast<double>(global_nx);
  // grid.origin_x deliberately left at its default (0.0) -- see this
  // file's header comment on why that is what keeps the two runs
  // bit-identical.

  cfe::Field<double, 1> state(grid.n_cells_total());
  cfe::Field<double, 1> stage1(grid.n_cells_total());
  cfe::Field<double, 1> scratch(grid.n_cells_total());

  for (std::size_t i = 0; i < grid.nx; ++i) {
    const std::size_t global_i = partition.global_offset_x + i;
    state(grid.flat_index(grid.ngx + i, 0, 0), 0) = exact_solution(cell_center(global_i, grid.dx), 0.0);
  }

  cfe::ScalarAdvectionField<double, 1> field{cfe::Vector<double, 1>(kAdvectionSpeed)};
  cfe::MpiHaloBoundary<double, 1> boundary_x{MPI_COMM_WORLD, partition.left_rank, partition.right_rank};
  // BoundaryY/BoundaryZ explicitly set to PeriodicBoundary (not left at
  // their default, which would be BoundaryX = MpiHaloBoundary): that
  // default-member-initializes `boundary_y{}`/`boundary_z{}`, which
  // requires a default constructor MpiHaloBoundary deliberately does not
  // have (it always needs a communicator + neighbor ranks) -- this is
  // required regardless of Field::dim, since FvmSolver's member
  // declarations are part of the class definition, not conditionally
  // instantiated by `if constexpr (y_active)`.
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 1>, cfe::MpiHaloBoundary<double, 1>,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary>
      solver{grid, field, boundary_x};
  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver.residual(in, out);
  };

  int n_steps;
  double dt;
  compute_steps(grid.dx, n_steps, dt);
  for (int step = 0; step < n_steps; ++step) {
    cfe::ssp_rk2_step<double>(state.view(), stage1.view(), scratch.view(), dt, residual,
                               solver.active_cell_count(), solver.active_cell_index_map());
  }

  std::vector<double> result(grid.nx);
  for (std::size_t i = 0; i < grid.nx; ++i) {
    result[i] = state(grid.flat_index(grid.ngx + i, 0, 0), 0);
  }
  return result;
}

}  // namespace

int main(int argc, char** argv)
{
  cfe::backend::mpi::Environment env(&argc, &argv);
  const int rank = cfe::backend::mpi::rank();
  const int size = cfe::backend::mpi::size();

  const auto partition = cfe::make_slab_partition(kGlobalNx, rank, size, /*periodic=*/true);
  const std::vector<double> reference = run_reference(kGlobalNx);
  const std::vector<double> decomposed = run_decomposed(kGlobalNx, partition);

  int local_failures = 0;
  for (std::size_t i = 0; i < decomposed.size(); ++i) {
    const std::size_t global_i = partition.global_offset_x + i;
    if (decomposed[i] != reference[global_i]) {
      std::fprintf(stderr, "[rank %d] MISMATCH at global cell %zu: decomposed=%.17g reference=%.17g\n", rank,
                   global_i, decomposed[i], reference[global_i]);
      ++local_failures;
    }
  }

  int total_failures = 0;
  MPI_Allreduce(&local_failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (rank == 0) {
    if (total_failures == 0) {
      std::printf(
          "[PASS] test_mpi_halo_exchange: %d ranks, %zu global cells, bit-identical to single-rank "
          "reference\n",
          size, kGlobalNx);
    } else {
      std::printf("[FAIL] test_mpi_halo_exchange: %d mismatched cells across %d ranks\n", total_failures,
                  size);
    }
  }

  return total_failures == 0 ? 0 : 1;
}
