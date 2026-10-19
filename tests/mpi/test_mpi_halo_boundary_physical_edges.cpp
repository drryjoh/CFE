// Regression test for a review finding on CFE PR #4: `MpiHaloBoundary`'s
// unpack step used to write BOTH ghost sides unconditionally, even on a
// side with no neighbor (`MPI_PROC_NULL`, a true non-periodic physical
// boundary). A null-destination/null-source `MPI_Sendrecv` leaves its
// receive buffer untouched, so that unconditional write was clobbering
// whatever value was already sitting in a physical ghost cell with
// stale/zero receive-buffer contents instead of leaving it alone. See
// `src/cfe/grid/boundary/mpi_halo_boundary.hpp`'s `exchange()` for the
// fix: the unpack now skips writing a ghost side whose rank is
// `MPI_PROC_NULL`.
//
// This test exercises `MpiHaloBoundary` directly (no `FvmSolver`, no
// physics field) -- a genuinely non-periodic 2-rank decomposition along
// ONE axis at a time, X/Y/Z all covered, with the OTHER two axes left
// trivial (n=1, ng=0, same "unused dimension" convention
// `CartesianGrid`'s own header comment documents). Every cell (real and
// ghost) starts prefilled with a distinct, component-dependent sentinel
// value; only the real cells are then overwritten with distinct,
// rank/position/component-dependent values. After one `fill_x`/`fill_y`/
// `fill_z` call, this test checks, cell by cell:
//   - the physical-edge-facing ghost cells (the side with no neighbor)
//     are EXACTLY the untouched sentinel -- the bug this test guards
//     against would have overwritten them;
//   - the neighbor-facing ghost cells hold EXACTLY the expected value
//     copied from the neighbor's real cells -- proving the fix did not
//     also silently break the already-working, has-a-neighbor side.
//
// Also doubles as the "direct halo-value check" coverage the same review
// asked for beyond the solver's own single-component/AoS/double-only
// correctness tests: every combination of {double,float} x {N=1,N=3
// multicomponent} x {AoSLayout,SoALayout} is run through the same
// harness, since `MpiHaloBoundary`/`Field`/`FieldView` are generic over
// all three and the existing physics solver (`FvmSolver`) only ever
// instantiates N=1 fields, so there is no way to exercise N=3 through a
// real physics field.
#include <mpi.h>

#include <cstdio>

#include "cfe/backend/mpi/mpi_environment.hpp"
#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/grid/boundary/mpi_halo_boundary.hpp"
#include "cfe/grid/partition/slab_partition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"

namespace {

constexpr cfe::Index kLocalN = 4;
constexpr cfe::Index kGhostDepth = 2;

// Deliberately distinguishable from any `value(...)` below (which never
// goes negative) and component-dependent, so a component-index swap bug
// in the fix would also be caught.
template <class Scalar>
Scalar sentinel(cfe::Index c)
{
  return static_cast<Scalar>(-999 - 7 * static_cast<int>(c));
}

// This rank's real-cell value at local axis index `i`, component `c` --
// also used (with the KNOWN neighbor rank 0 or 1, not `rank`) to compute
// what a neighbor-facing ghost cell must contain after the exchange.
template <class Scalar>
Scalar value(int r, cfe::Index i, cfe::Index c)
{
  return static_cast<Scalar>(1000 * r + 100 * static_cast<int>(c) + static_cast<int>(i));
}

template <class Scalar, std::size_t N, class Layout>
int run_one_case(int rank, cfe::Axis axis, const char* axis_name, const char* scalar_name,
                  const char* layout_name)
{
  cfe::CartesianGrid<Scalar> grid;
  switch (axis) {
    case cfe::Axis::X:
      grid.nx = kLocalN;
      grid.ngx = kGhostDepth;
      break;
    case cfe::Axis::Y:
      grid.ny = kLocalN;
      grid.ngy = kGhostDepth;
      break;
    case cfe::Axis::Z:
      grid.nz = kLocalN;
      grid.ngz = kGhostDepth;
      break;
  }

  auto real_index = [&](cfe::Index i) -> cfe::Index {
    switch (axis) {
      case cfe::Axis::X: return grid.flat_index(kGhostDepth + i, 0, 0);
      case cfe::Axis::Y: return grid.flat_index(0, kGhostDepth + i, 0);
      default: return grid.flat_index(0, 0, kGhostDepth + i);
    }
  };
  auto low_ghost_index = [&](cfe::Index d) -> cfe::Index {
    switch (axis) {
      case cfe::Axis::X: return grid.flat_index(kGhostDepth - 1 - d, 0, 0);
      case cfe::Axis::Y: return grid.flat_index(0, kGhostDepth - 1 - d, 0);
      default: return grid.flat_index(0, 0, kGhostDepth - 1 - d);
    }
  };
  auto high_ghost_index = [&](cfe::Index d) -> cfe::Index {
    switch (axis) {
      case cfe::Axis::X: return grid.flat_index(kGhostDepth + kLocalN + d, 0, 0);
      case cfe::Axis::Y: return grid.flat_index(0, kGhostDepth + kLocalN + d, 0);
      default: return grid.flat_index(0, 0, kGhostDepth + kLocalN + d);
    }
  };

  cfe::Field<Scalar, N, Layout> field(grid.n_cells_total());
  for (cfe::Index cell = 0; cell < grid.n_cells_total(); ++cell) {
    for (cfe::Index c = 0; c < N; ++c) field(cell, c) = sentinel<Scalar>(c);
  }
  for (cfe::Index i = 0; i < kLocalN; ++i) {
    for (cfe::Index c = 0; c < N; ++c) field(real_index(i), c) = value<Scalar>(rank, i, c);
  }

  // np=2, non-periodic: rank 0 has no left neighbor, rank 1 has no right
  // neighbor -- a true physical boundary on exactly one side of each
  // rank, the precise case the review finding was about.
  const int left_rank = (rank == 0) ? cfe::SlabPartition::kNoNeighbor : 0;
  const int right_rank = (rank == 0) ? 1 : cfe::SlabPartition::kNoNeighbor;
  cfe::MpiHaloBoundary<Scalar, N> boundary(MPI_COMM_WORLD, left_rank, right_rank);

  auto view = field.view();
  switch (axis) {
    case cfe::Axis::X: boundary.fill_x(view, grid); break;
    case cfe::Axis::Y: boundary.fill_y(view, grid); break;
    case cfe::Axis::Z: boundary.fill_z(view, grid); break;
  }

  int failures = 0;
  for (cfe::Index d = 0; d < kGhostDepth; ++d) {
    for (cfe::Index c = 0; c < N; ++c) {
      const Scalar got_low = field(low_ghost_index(d), c);
      const Scalar want_low =
          (left_rank == cfe::SlabPartition::kNoNeighbor) ? sentinel<Scalar>(c) : value<Scalar>(0, kLocalN - 1 - d, c);
      if (got_low != want_low) {
        std::fprintf(stderr,
                      "[rank %d][axis %s][%s][N=%zu][%s] LOW ghost d=%zu c=%zu MISMATCH: got=%.17g want=%.17g\n",
                      rank, axis_name, scalar_name, N, layout_name, d, c, static_cast<double>(got_low),
                      static_cast<double>(want_low));
        ++failures;
      }

      const Scalar got_high = field(high_ghost_index(d), c);
      const Scalar want_high =
          (right_rank == cfe::SlabPartition::kNoNeighbor) ? sentinel<Scalar>(c) : value<Scalar>(1, d, c);
      if (got_high != want_high) {
        std::fprintf(stderr,
                      "[rank %d][axis %s][%s][N=%zu][%s] HIGH ghost d=%zu c=%zu MISMATCH: got=%.17g want=%.17g\n",
                      rank, axis_name, scalar_name, N, layout_name, d, c, static_cast<double>(got_high),
                      static_cast<double>(want_high));
        ++failures;
      }
    }
  }
  return failures;
}

template <class Scalar, std::size_t N, class Layout>
int run_all_axes(int rank, const char* scalar_name, const char* layout_name)
{
  int failures = 0;
  failures += run_one_case<Scalar, N, Layout>(rank, cfe::Axis::X, "X", scalar_name, layout_name);
  failures += run_one_case<Scalar, N, Layout>(rank, cfe::Axis::Y, "Y", scalar_name, layout_name);
  failures += run_one_case<Scalar, N, Layout>(rank, cfe::Axis::Z, "Z", scalar_name, layout_name);
  return failures;
}

}  // namespace

int main(int argc, char** argv)
{
  cfe::backend::mpi::Environment env(&argc, &argv);
  const int rank = cfe::backend::mpi::rank();
  const int size = cfe::backend::mpi::size();

  if (size != 2) {
    if (rank == 0) {
      std::fprintf(stderr,
                    "test_mpi_halo_boundary_physical_edges requires exactly 2 ranks, got %d -- run with "
                    "mpirun -n 2\n",
                    size);
    }
    return 1;
  }

  int local_failures = 0;
  local_failures += run_all_axes<double, 1, cfe::AoSLayout>(rank, "double", "AoS");
  local_failures += run_all_axes<double, 1, cfe::SoALayout>(rank, "double", "SoA");
  local_failures += run_all_axes<float, 1, cfe::AoSLayout>(rank, "float", "AoS");
  local_failures += run_all_axes<float, 1, cfe::SoALayout>(rank, "float", "SoA");
  local_failures += run_all_axes<double, 3, cfe::AoSLayout>(rank, "double", "AoS");
  local_failures += run_all_axes<double, 3, cfe::SoALayout>(rank, "double", "SoA");
  local_failures += run_all_axes<float, 3, cfe::AoSLayout>(rank, "float", "AoS");
  local_failures += run_all_axes<float, 3, cfe::SoALayout>(rank, "float", "SoA");

  int total_failures = 0;
  MPI_Allreduce(&local_failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (rank == 0) {
    if (total_failures == 0) {
      std::printf(
          "[PASS] test_mpi_halo_boundary_physical_edges: 2 ranks, X/Y/Z axes, "
          "{double,float}x{N=1,N=3}x{AoS,SoA}: physical-edge ghost cells preserved, "
          "neighbor-facing ghost cells correctly updated\n");
    } else {
      std::printf("[FAIL] test_mpi_halo_boundary_physical_edges: %d mismatches across 2 ranks\n",
                  total_failures);
    }
  }

  return total_failures == 0 ? 0 : 1;
}
