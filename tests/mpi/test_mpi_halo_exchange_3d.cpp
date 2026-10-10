// MPI 3D block-decomposition correctness test (task spec 0005): same
// bit-identical-vs-independently-computed-reference oracle as
// test_mpi_halo_exchange.cpp (see that file's header comment for the
// full reasoning on why bit-identity is achievable and the right bar),
// extended from 1D-slab-along-X to a full 3D block decomposition via
// CartesianPartition, with all three of MpiHaloBoundary's fill_x/
// fill_y/fill_z genuinely exercised (not just fill_x).
//
// Fixed at a 2x2x2 process grid (np=8 exactly) -- the smallest shape
// that exercises every axis's halo exchange simultaneously for a
// genuinely 3D problem; larger/non-cubic shapes are covered by
// test_cartesian_partition.cpp's pure-math unit tests, not re-verified
// here under real MPI communication.
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "cfe/backend/mpi/mpi_environment.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/scalar_advection/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/boundary/mpi_halo_boundary.hpp"
#include "cfe/grid/partition/cartesian_partition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDomainLength = 1.0;
constexpr double kVelocity = 1.0;  // same speed on all three axes
constexpr double kFinalTime = 0.1;
constexpr double kCfl = 0.4;
constexpr std::size_t kGlobalN = 16;  // 16^3 = 4096 global cells

// Product-of-sines IC (same convention bench_burgers_3d_cuda.cu/
// test_burgers_3d_sanity.cpp use for their own 3D product ICs) advected
// at the same speed on every axis: exact solution is the IC evaluated
// at (x-v*t, y-v*t, z-v*t).
double exact_solution(double x, double y, double z, double t)
{
  const double s = 2.0 * kPi / kDomainLength;
  return std::sin(s * (x - kVelocity * t)) * std::sin(s * (y - kVelocity * t)) *
         std::sin(s * (z - kVelocity * t));
}

double cell_center(std::size_t global_i, double dx) { return (static_cast<double>(global_i) + 0.5) * dx; }

void compute_steps(double dx, int& n_steps, double& dt)
{
  // Multi-axis CFL (same reasoning every existing 3D test/benchmark in
  // this repo uses): dt sized from the sum of per-axis Courant numbers,
  // all three axes sharing the same speed/spacing here.
  const double dt_target = kCfl / (3.0 * kVelocity / dx);
  n_steps = static_cast<int>(std::ceil(kFinalTime / dt_target));
  dt = kFinalTime / static_cast<double>(n_steps);
}

// Full global problem on ONE (virtual) rank, entirely locally, no MPI
// involved -- the reference every rank's decomposed slice must match
// bit-for-bit. Returns a flat (i-fastest, j-next, k-slowest) array of
// size global_n^3, the same ordering CartesianGrid::flat_index uses for
// real cells.
std::vector<double> run_reference(std::size_t global_n)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = grid.ny = grid.nz = global_n;
  grid.ngx = grid.ngy = grid.ngz = 2;
  grid.dx = grid.dy = grid.dz = kDomainLength / static_cast<double>(global_n);

  cfe::Field<double, 1> state(grid.n_cells_total());
  cfe::Field<double, 1> stage1(grid.n_cells_total());
  cfe::Field<double, 1> scratch(grid.n_cells_total());

  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const double value = exact_solution(cell_center(i, grid.dx), cell_center(j, grid.dy),
                                             cell_center(k, grid.dz), 0.0);
        state(grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k), 0) = value;
      }
    }
  }

  cfe::ScalarAdvectionField<double, 3> field{cfe::Vector<double, 3>(kVelocity)};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 3>, cfe::PeriodicBoundary>
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

  std::vector<double> result(global_n * global_n * global_n);
  for (std::size_t k = 0; k < global_n; ++k) {
    for (std::size_t j = 0; j < global_n; ++j) {
      for (std::size_t i = 0; i < global_n; ++i) {
        result[i + global_n * (j + global_n * k)] =
            state(grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k), 0);
      }
    }
  }
  return result;
}

// This rank's own block of the SAME global problem, decomposed via
// CartesianPartition + MpiHaloBoundary on all three axes.
std::vector<double> run_decomposed(std::size_t global_n, const cfe::CartesianPartition& partition)
{
  cfe::CartesianGrid<double> grid;
  grid.nx = partition.local_nx;
  grid.ny = partition.local_ny;
  grid.nz = partition.local_nz;
  grid.ngx = grid.ngy = grid.ngz = 2;
  grid.dx = grid.dy = grid.dz = kDomainLength / static_cast<double>(global_n);
  // grid.origin_x/y/z deliberately left at their default (0.0) -- see
  // test_mpi_halo_exchange.cpp's header comment on why that is what
  // keeps this bit-identical to the single-process reference.

  cfe::Field<double, 1> state(grid.n_cells_total());
  cfe::Field<double, 1> stage1(grid.n_cells_total());
  cfe::Field<double, 1> scratch(grid.n_cells_total());

  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const double value =
            exact_solution(cell_center(partition.global_offset_x + i, grid.dx),
                            cell_center(partition.global_offset_y + j, grid.dy),
                            cell_center(partition.global_offset_z + k, grid.dz), 0.0);
        state(grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k), 0) = value;
      }
    }
  }

  cfe::ScalarAdvectionField<double, 3> field{cfe::Vector<double, 3>(kVelocity)};
  cfe::MpiHaloBoundary<double, 1> boundary_x{MPI_COMM_WORLD, partition.x_left_rank, partition.x_right_rank};
  cfe::MpiHaloBoundary<double, 1> boundary_y{MPI_COMM_WORLD, partition.y_left_rank, partition.y_right_rank};
  cfe::MpiHaloBoundary<double, 1> boundary_z{MPI_COMM_WORLD, partition.z_left_rank, partition.z_right_rank};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 3>, cfe::MpiHaloBoundary<double, 1>>
      solver{grid, field, boundary_x, boundary_y, boundary_z};
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

  std::vector<double> result(grid.nx * grid.ny * grid.nz);
  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        result[i + grid.nx * (j + grid.ny * k)] =
            state(grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k), 0);
      }
    }
  }
  return result;
}

}  // namespace

int main(int argc, char** argv)
{
  cfe::backend::mpi::Environment env(&argc, &argv);
  const int rank = cfe::backend::mpi::rank();
  const int size = cfe::backend::mpi::size();

  if (size != 8) {
    if (rank == 0) {
      std::fprintf(stderr,
                    "test_mpi_halo_exchange_3d requires exactly 8 ranks (2x2x2), got %d -- run with "
                    "mpirun -n 8\n",
                    size);
    }
    return 1;
  }

  const auto partition =
      cfe::make_cartesian_partition(kGlobalN, kGlobalN, kGlobalN, 2, 2, 2, rank, size, true, true, true);
  const std::vector<double> reference = run_reference(kGlobalN);
  const std::vector<double> decomposed = run_decomposed(kGlobalN, partition);

  int local_failures = 0;
  for (std::size_t k = 0; k < partition.local_nz; ++k) {
    for (std::size_t j = 0; j < partition.local_ny; ++j) {
      for (std::size_t i = 0; i < partition.local_nx; ++i) {
        const std::size_t gi = partition.global_offset_x + i;
        const std::size_t gj = partition.global_offset_y + j;
        const std::size_t gk = partition.global_offset_z + k;
        const double got = decomposed[i + partition.local_nx * (j + partition.local_ny * k)];
        const double want = reference[gi + kGlobalN * (gj + kGlobalN * gk)];
        if (got != want) {
          std::fprintf(stderr, "[rank %d] MISMATCH at global (%zu,%zu,%zu): decomposed=%.17g reference=%.17g\n",
                        rank, gi, gj, gk, got, want);
          ++local_failures;
        }
      }
    }
  }

  int total_failures = 0;
  MPI_Allreduce(&local_failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (rank == 0) {
    if (total_failures == 0) {
      std::printf(
          "[PASS] test_mpi_halo_exchange_3d: 8 ranks (2x2x2), %zu^3 global cells, bit-identical to "
          "single-rank reference\n",
          kGlobalN);
    } else {
      std::printf("[FAIL] test_mpi_halo_exchange_3d: %d mismatched cells across 8 ranks\n", total_failures);
    }
  }

  return total_failures == 0 ? 0 : 1;
}
