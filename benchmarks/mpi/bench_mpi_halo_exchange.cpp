// MPI halo-exchange communication benchmark (task spec 0004): isolates
// just the `MpiHaloBoundary::fill_x` step (not a full solver timestep),
// timing it in isolation to produce a communication-only cost curve.
//
// Three independent sweeps, distinguished by a `sweep` CSV column
// rather than three separate binaries:
// - "ranks": this run's own rank count vs. a fixed problem size/ghost
//   depth -- one row per invocation. This one binary cannot itself
//   produce the full rank-count curve (the rank count is fixed by
//   however many processes `mpirun` launched it with); assemble it by
//   running this binary once per rank count and concatenating output
//   (see this task's README for the exact 1/2/4/8 sweep command).
// - "size": vs. the Y-Z cross-section (ny*nz) at fixed rank
//   count/ghost depth. A literal 1D problem's halo message is a fixed
//   few scalars regardless of nx -- message size scales with the
//   cross-section, not nx -- so this sweep uses a genuinely 2D/3D
//   domain decomposed along X only, to produce a meaningful bandwidth
//   curve (unlike tests/mpi/test_mpi_halo_exchange.cpp's correctness
//   oracle, which stays 1D for the cleanest comparison).
// - "ghost_depth": vs. ngx at fixed rank count/problem size, to
//   separate pack/unpack cost from one-time MPI latency.
//
// Calls `MpiHaloBoundary::fill_x` directly on a bare `cfe::Field` --
// not through `FvmSolver`/`ssp_rk2_step` -- since this benchmark
// measures communication cost alone, with no physics or time
// integration involved at all.
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "cfe/backend/mpi/mpi_environment.hpp"
#include "cfe/backend/mpi/mpi_reduce.hpp"
#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/grid/boundary/mpi_halo_boundary.hpp"
#include "cfe/grid/partition/slab_partition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"

namespace {

// Precision is `cfe::scalar` (core/types.hpp, project-wide via
// `-DCFE_SCALAR_TYPE`) -- task 0017 finding: this file previously
// hardcoded `double` here, but this IS the physics/field precision
// (what actually gets packed/sent), not a reported metric -- a float
// build genuinely halves the bytes moved per exchange
// (`bytes_per_exchange`'s `sizeof(Scalar)` below), so a float build's
// own halo-exchange bandwidth number must come from actually moving
// `float`-sized messages, not from a `double` message mislabeled as a
// float measurement. The wall-clock timings themselves
// (`time_one_fill_x_median`'s returned `double` seconds, and the
// bytes/bandwidth values derived from them) stay plain `double`
// regardless of `Scalar` -- those are reported metrics, matching this
// project's existing timing/metric convention.
using Scalar = cfe::scalar;
constexpr int kRepetitions = 20;

// Times `kRepetitions` calls to `boundary.fill_x(...)` (after one
// untimed warm-up call) and returns the median wall-clock time, in
// seconds. Each timed call is preceded by an MPI_Barrier so every rank
// starts together -- comparable repeated measurements need a
// synchronized start.
//
// A blocking Sendrecv-based exchange's true cost, as an event every
// rank must wait through before any of them can move on, is bounded by
// the SLOWEST participating rank, not by whichever rank happens to be
// printing the result (rank 0). Measuring only rank 0's own local
// elapsed time silently reports an optimistic number whenever some
// other rank is the straggler (a real risk on a shared machine or a
// busy cluster node) -- so each repetition's elapsed time is
// MPI_MAX-reduced across every rank BEFORE it is pushed into `seconds`;
// the reduction itself happens after `t1` is captured, i.e. strictly
// outside the timed region, so it never inflates the measurement it is
// correcting. Every rank ends up with the identical (global-max)
// per-repetition value, so the subsequent median and the bandwidth
// calculation that uses it are already the cross-rank-correct number on
// every rank, not just rank 0.
double time_one_fill_x_median(const cfe::MpiHaloBoundary<Scalar, 1>& boundary, cfe::Field<Scalar, 1>& state,
                               const cfe::CartesianGrid<Scalar>& grid)
{
  auto one_call = [&]() { boundary.fill_x(state.view(), grid); };

  one_call();  // warm-up: the first call may pay extra one-time setup cost.

  std::vector<double> seconds;
  seconds.reserve(kRepetitions);
  for (int r = 0; r < kRepetitions; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    const auto t0 = std::chrono::steady_clock::now();
    one_call();
    const auto t1 = std::chrono::steady_clock::now();
    const double local_s = std::chrono::duration<double>(t1 - t0).count();
    seconds.push_back(cfe::backend::mpi::allreduce_max(local_s));
  }
  std::sort(seconds.begin(), seconds.end());
  return seconds[seconds.size() / 2];
}

// Bytes moved in one fill_x call: two directions (low-face send +
// high-face send), each `ghost_depth * padded_ny * padded_nz` scalars.
double bytes_per_exchange(std::size_t ghost_depth, std::size_t padded_ny, std::size_t padded_nz)
{
  return static_cast<double>(2 * ghost_depth * padded_ny * padded_nz) * sizeof(Scalar);
}

void print_row(const char* sweep, int size, std::size_t global_nx, std::size_t ny, std::size_t nz,
               std::size_t ghost_depth, double median_s, double bytes)
{
  std::printf("%s,%d,%zu,%zu,%zu,%zu,%d,%.6f,%.0f,%.3f\n", sweep, size, global_nx, ny, nz, ghost_depth,
              kRepetitions, median_s * 1e6, bytes, bytes / median_s / 1e9);
}

void run_ranks_sweep(int rank, int size)
{
  constexpr std::size_t kGlobalNx = 1024;
  constexpr std::size_t kNy = 64;
  constexpr std::size_t kNz = 64;
  constexpr std::size_t kNgx = 2;

  const auto partition = cfe::make_slab_partition(kGlobalNx, rank, size, /*periodic=*/true);

  cfe::CartesianGrid<Scalar> grid;
  grid.nx = partition.local_nx;
  grid.ny = kNy;
  grid.nz = kNz;
  grid.ngx = kNgx;
  grid.dx = 1.0 / static_cast<Scalar>(kGlobalNx);
  grid.dy = 1.0 / static_cast<Scalar>(kNy);
  grid.dz = 1.0 / static_cast<Scalar>(kNz);

  cfe::Field<Scalar, 1> state(grid.n_cells_total());
  cfe::MpiHaloBoundary<Scalar, 1> boundary(MPI_COMM_WORLD, partition.left_rank, partition.right_rank);

  const double median_s = time_one_fill_x_median(boundary, state, grid);
  const double bytes = bytes_per_exchange(kNgx, grid.padded_ny(), grid.padded_nz());
  if (rank == 0) print_row("ranks", size, kGlobalNx, kNy, kNz, kNgx, median_s, bytes);
}

void run_size_sweep(int rank, int size)
{
  constexpr std::size_t kGlobalNx = 256;
  constexpr std::size_t kNgx = 2;
  const std::vector<std::size_t> cross_sections = {8, 16, 32, 64, 128};

  const auto partition = cfe::make_slab_partition(kGlobalNx, rank, size, /*periodic=*/true);

  for (std::size_t n : cross_sections) {
    cfe::CartesianGrid<Scalar> grid;
    grid.nx = partition.local_nx;
    grid.ny = n;
    grid.nz = n;
    grid.ngx = kNgx;
    grid.dx = 1.0 / static_cast<Scalar>(kGlobalNx);
    grid.dy = 1.0 / static_cast<Scalar>(n);
    grid.dz = 1.0 / static_cast<Scalar>(n);

    cfe::Field<Scalar, 1> state(grid.n_cells_total());
    cfe::MpiHaloBoundary<Scalar, 1> boundary(MPI_COMM_WORLD, partition.left_rank, partition.right_rank);

    const double median_s = time_one_fill_x_median(boundary, state, grid);
    const double bytes = bytes_per_exchange(kNgx, grid.padded_ny(), grid.padded_nz());
    if (rank == 0) print_row("size", size, kGlobalNx, n, n, kNgx, median_s, bytes);
  }
}

void run_ghost_depth_sweep(int rank, int size)
{
  constexpr std::size_t kGlobalNx = 256;
  constexpr std::size_t kNy = 64;
  constexpr std::size_t kNz = 64;
  const std::vector<std::size_t> depths = {2, 4, 8};

  const auto partition = cfe::make_slab_partition(kGlobalNx, rank, size, /*periodic=*/true);

  for (std::size_t ngx : depths) {
    cfe::CartesianGrid<Scalar> grid;
    grid.nx = partition.local_nx;
    grid.ny = kNy;
    grid.nz = kNz;
    grid.ngx = ngx;
    grid.dx = 1.0 / static_cast<Scalar>(kGlobalNx);
    grid.dy = 1.0 / static_cast<Scalar>(kNy);
    grid.dz = 1.0 / static_cast<Scalar>(kNz);

    cfe::Field<Scalar, 1> state(grid.n_cells_total());
    cfe::MpiHaloBoundary<Scalar, 1> boundary(MPI_COMM_WORLD, partition.left_rank, partition.right_rank);

    const double median_s = time_one_fill_x_median(boundary, state, grid);
    const double bytes = bytes_per_exchange(ngx, grid.padded_ny(), grid.padded_nz());
    if (rank == 0) print_row("ghost_depth", size, kGlobalNx, kNy, kNz, ngx, median_s, bytes);
  }
}

}  // namespace

int main(int argc, char** argv)
{
  cfe::backend::mpi::Environment env(&argc, &argv);
  const int rank = cfe::backend::mpi::rank();
  const int size = cfe::backend::mpi::size();

  if (rank == 0) {
    std::printf(
        "sweep,ranks,global_nx,ny,nz,ghost_depth,repetitions,median_exchange_us,bytes_per_exchange,"
        "bandwidth_GBps\n");
  }

  run_ranks_sweep(rank, size);
  run_size_sweep(rank, size);
  run_ghost_depth_sweep(rank, size);

  return 0;
}
