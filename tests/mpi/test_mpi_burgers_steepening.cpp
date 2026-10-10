// MPI correctness test for decomposed BurgersField (task spec 0006):
// proves the `MPI_Allreduce(MAX)` CFL fix (`backend/mpi/mpi_reduce.hpp`)
// actually makes Burgers safe to decompose -- not just that it compiles.
//
// Scenario: the same periodic sinusoidal-steepening-into-shock case
// `tutorials/burgers_1d_shock_and_steepening/`'s Case B uses
// (`u0 = 1.0 + 0.5*sin(2*pi*x)`), run PAST the analytic breaking time
// (`t_s = 1/(2*pi*0.5) ~= 0.318`) so a genuine shock has formed -- this
// is deliberately NOT a uniform-amplitude IC: `sin(2*pi*x)` peaks at
// x=0.25 and troughs at x=0.75, so a rank whose local block does not
// contain a point near x=0.25 sees a meaningfully SMALLER local
// |u|-maximum than the true global one (verified by construction: at
// 4 ranks, the two ranks owning x in [0.5,0.75) and [0.75,1.0) each see
// a local max of ~1.0, vs. the true global max of 1.5 at x=0.25 -- a
// 33% discrepancy, large enough that computing `dt` from the WRONG
// (local-only) maximum would desynchronize those ranks' timestep size
// from the others', not just round differently in the last few bits).
// This is exactly the hazard `docs/adr/0009-mpi-domain-decomposition.md`
// named and task 0004/0005 deferred -- this test is the proof it is now
// closed, not just documented.
//
// Same bit-identical-vs-independently-computed-reference oracle as
// `test_mpi_halo_exchange.cpp` (see that file's header comment for the
// full reasoning), extended to BurgersField + MusclMinmodReconstruction
// + RusanovFlux (the shock-capturing scheme, not the default
// unlimited/central one) and the Allreduce-synchronized `dt`.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "cfe/backend/mpi/mpi_environment.hpp"
#include "cfe/backend/mpi/mpi_reduce.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/burgers/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/boundary/mpi_halo_boundary.hpp"
#include "cfe/grid/partition/slab_partition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDomainLength = 1.0;
constexpr double kBackground = 1.0;
constexpr double kAmplitude = 0.5;
constexpr double kCfl = 0.4;
constexpr double kFinalTime = 0.5;  // past the analytic breaking time (~0.318): a real shock exists
constexpr std::size_t kGlobalNx = 400;

double ic(double x) { return kBackground + kAmplitude * std::sin(2.0 * kPi * x / kDomainLength); }

double cell_center(std::size_t global_i, double dx) { return (static_cast<double>(global_i) + 0.5) * dx; }

// Lands exactly on kFinalTime with a CFL-respecting step count, from
// the GLOBAL (Allreduce-synchronized, or trivially global in the
// single-process reference) max|u|.
void compute_steps(double dx, double global_max_abs_u, int& n_steps, double& dt)
{
  const double dt_target = kCfl * dx / global_max_abs_u;
  n_steps = static_cast<int>(std::ceil(kFinalTime / dt_target));
  dt = kFinalTime / static_cast<double>(n_steps);
}

// Full global problem on ONE (virtual) rank, entirely locally, no MPI
// involved -- the reference the decomposed run must match bit-for-bit.
std::vector<double> run_reference(std::size_t global_nx)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = global_nx;
  grid.ngx = 2;
  grid.dx = kDomainLength / static_cast<double>(global_nx);

  cfe::Field<double, 1> state(grid.n_cells_total());
  cfe::Field<double, 1> stage1(grid.n_cells_total());
  cfe::Field<double, 1> scratch(grid.n_cells_total());

  double max_abs_u = 0.0;
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const double value = ic(cell_center(i, grid.dx));
    state(grid.flat_index(grid.ngx + i, 0, 0), 0) = value;
    max_abs_u = std::max(max_abs_u, std::abs(value));
  }

  cfe::BurgersField<double, 1> field{};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::BurgersField<double, 1>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux>
      solver{grid, field, cfe::PeriodicBoundary{}};
  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver.residual(in, out);
  };

  int n_steps;
  double dt;
  compute_steps(grid.dx, max_abs_u, n_steps, dt);
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

// This rank's own slice of the SAME global problem, decomposed via
// SlabPartition + MpiHaloBoundary. `max_abs_u` is computed over THIS
// rank's own local cells, then synchronized to the true global value
// via `allreduce_max` -- the fix under test. Without that call, a rank
// whose local block does not contain the IC's peak would size its `dt`
// from a smaller, wrong maximum, desynchronizing it from the other
// ranks (see this file's header comment for the concrete numbers).
std::vector<double> run_decomposed(std::size_t global_nx, const cfe::SlabPartition& partition)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = partition.local_nx;
  grid.ngx = 2;
  grid.dx = kDomainLength / static_cast<double>(global_nx);
  // grid.origin_x deliberately left at its default (0.0) -- see
  // test_mpi_halo_exchange.cpp's header comment on why that is what
  // keeps this bit-identical to the single-rank reference.

  cfe::Field<double, 1> state(grid.n_cells_total());
  cfe::Field<double, 1> stage1(grid.n_cells_total());
  cfe::Field<double, 1> scratch(grid.n_cells_total());

  double local_max_abs_u = 0.0;
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const std::size_t global_i = partition.global_offset_x + i;
    const double value = ic(cell_center(global_i, grid.dx));
    state(grid.flat_index(grid.ngx + i, 0, 0), 0) = value;
    local_max_abs_u = std::max(local_max_abs_u, std::abs(value));
  }
  const double global_max_abs_u = cfe::backend::mpi::allreduce_max(local_max_abs_u);

  cfe::BurgersField<double, 1> field{};
  cfe::MpiHaloBoundary<double, 1> boundary_x{MPI_COMM_WORLD, partition.left_rank, partition.right_rank};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::BurgersField<double, 1>, cfe::MpiHaloBoundary<double, 1>,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux>
      solver{grid, field, boundary_x};
  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver.residual(in, out);
  };

  int n_steps;
  double dt;
  compute_steps(grid.dx, global_max_abs_u, n_steps, dt);
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
          "[PASS] test_mpi_burgers_steepening: %d ranks, %zu global cells, post-shock, bit-identical to "
          "single-rank reference\n",
          size, kGlobalNx);
    } else {
      std::printf("[FAIL] test_mpi_burgers_steepening: %d mismatched cells across %d ranks\n",
                  total_failures, size);
    }
  }

  return total_failures == 0 ? 0 : 1;
}
