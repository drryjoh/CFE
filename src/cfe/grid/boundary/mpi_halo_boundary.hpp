// MPI halo-exchange boundary condition (task spec 0004, ADR 0009):
// fills one rank's X-axis ghost cells from its left/right neighbor
// ranks' real cells, instead of from local data the way every other
// boundary type in this file does. This is the "swappable neighbor
// provider" ADR 0004 and grid/ghost/ghost_fill.hpp's own header comment
// already named as the intended MPI plug-in point -- same duck-typed
// fill_x/fill_y/fill_z shape as StaticBoundary/PeriodicBoundary/
// InflowOutflowBoundary, so FvmSolver/fill_ghost_cells need zero changes
// to use it (plug it in as BoundaryX; BoundaryY/BoundaryZ stay
// PeriodicBoundary for a 2D/3D problem decomposed only along X).
//
// Scope for this first prototype (see tasks/0004-...md's "Do not
// implement" list for the full exclusion list):
// - X-axis only -- fill_y/fill_z are not implemented. Y/Z-axis slab
//   decomposition and full 2D/3D block decomposition are separate
//   follow-up work.
// - CPU-only (`CpuParallelFor`) -- combining this with CudaParallelFor
//   (GPU-aware MPI) is explicitly out of scope.
// - Blocking `MPI_Sendrecv`, not non-blocking Isend/Irecv with
//   communication/computation overlap (AGENTS.md #16 names overlap as
//   "eventual", not required for a first prototype).
// - A non-periodic SlabPartition's two true edge ranks (left_rank/
//   right_rank == SlabPartition::kNoNeighbor, translated to
//   MPI_PROC_NULL below) get a genuine no-op Sendrecv on that one side --
//   their corresponding ghost cells are simply never written by this
//   type and stay at whatever they were previously (zero, the first
//   time). This type does not compose with a second boundary condition
//   to give a true physical edge a real value; only the fully periodic
//   case is exercised/tested in this task.
//
// Lives in grid/boundary/ (not backend/mpi/), matching every other
// boundary type's location: the actual MPI plumbing (communicator
// bootstrap, datatype mapping) lives in backend/mpi/, exactly as
// PeriodicBoundary/StaticBoundary already call into backend/
// parallel_for.hpp's Backend::run without owning that backend logic
// themselves.
//
// Only ever included from a translation unit built with CFE_ENABLE_MPI.
#pragma once

#ifndef CFE_ENABLE_MPI
#error "mpi_halo_boundary.hpp requires CFE_ENABLE_MPI -- only include it from an MPI-gated translation unit."
#endif

#include <cstddef>
#include <mpi.h>
#include <stdexcept>
#include <vector>

#include "cfe/backend/mpi/mpi_datatype.hpp"
#include "cfe/backend/parallel_for.hpp"
#include "cfe/core/macros.hpp"
#include "cfe/field/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/partition/slab_partition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"

namespace cfe {

template <class Scalar, std::size_t N>
class MpiHaloBoundary
{
 public:
  MpiHaloBoundary(MPI_Comm comm, const SlabPartition& partition)
      : comm_(comm),
        left_rank_(partition.left_rank == SlabPartition::kNoNeighbor ? MPI_PROC_NULL
                                                                      : partition.left_rank),
        right_rank_(partition.right_rank == SlabPartition::kNoNeighbor ? MPI_PROC_NULL
                                                                        : partition.right_rank)
  {
  }

  // fill_y/fill_z: required to exist and type-check for this type to be
  // usable as `BoundaryX` at all, even though they are never actually
  // *called* -- `grid/ghost/ghost_fill.hpp`'s `fill_ghost_cells` is one
  // function template per `Boundary` type, with a runtime `switch` over
  // all three axes inside it; every `case` must compile for whichever
  // `Boundary` the template is instantiated with, regardless of which
  // one its caller's runtime `axis` argument actually selects (same
  // lesson `tutorials/burgers_2d_diagonal_shock/burgers_2d.cpp`'s own
  // `DiagonalShockExactBoundary::fill_z` comment already documents).
  // This type is X-axis-only by design (see header comment): FvmSolver
  // only ever instantiates `fill_ghost_cells` with this type as
  // `BoundaryX`, called only with `Axis::X` at runtime -- so these two
  // are genuinely unreachable, not just unlikely, and throw rather than
  // silently doing nothing.
  template <class Backend = CpuParallelFor, class Layout>
  void fill_y(FieldView<Scalar, N, Layout>, const CartesianGrid<Scalar>) const
  {
    throw std::logic_error(
        "MpiHaloBoundary::fill_y is not implemented -- this type is X-axis-only; use a "
        "different BoundaryY (e.g. PeriodicBoundary) for a problem with an active Y axis.");
  }

  template <class Backend = CpuParallelFor, class Layout>
  void fill_z(FieldView<Scalar, N, Layout>, const CartesianGrid<Scalar>) const
  {
    throw std::logic_error(
        "MpiHaloBoundary::fill_z is not implemented -- this type is X-axis-only; use a "
        "different BoundaryZ (e.g. PeriodicBoundary) for a problem with an active Z axis.");
  }

  template <class Backend = CpuParallelFor, class Layout>
  void fill_x(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngx == 0) return;
    // Same hazard PeriodicBoundary::fill_x already guards against (see
    // that function's own comment): a local extent narrower than the
    // ghost depth would make the pack/unpack index arithmetic below read
    // into the ghost region itself instead of real cells. Reused rather
    // than duplicated.
    detail::require_periodic_extent_covers_ghost_depth(grid.nx, grid.ngx, "x");
    const std::size_t ngx = grid.ngx;
    const std::size_t nx = grid.nx;
    const std::size_t py = grid.padded_ny();
    const std::size_t pz = grid.padded_nz();
    const std::size_t slab_cells = ngx * py * pz;
    const std::size_t count = slab_cells * N;

    if (send_low_.size() != count) {
      send_low_.resize(count);
      send_high_.resize(count);
      recv_low_.resize(count);
      recv_high_.resize(count);
    }

    // Pack: send_low_[g,j,k] <- this rank's real cell at distance `g`
    // from its own LOW face (the face shared with left_rank_) -- what
    // left_rank_ will use to fill ITS high ghost. send_high_[g,j,k] <-
    // this rank's real cell at distance `g` from its own HIGH face (the
    // face shared with right_rank_) -- what right_rank_ will use to
    // fill its low ghost. `g=0` is nearest the shared face in both
    // cases, matching the adjacency order the unpack step below expects.
    Scalar* send_low = send_low_.data();
    Scalar* send_high = send_high_.data();
    Backend::run(slab_cells, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
      const std::size_t g = idx % ngx;
      const std::size_t rem = idx / ngx;
      const std::size_t j = rem % py;
      const std::size_t k = rem / py;

      const std::size_t low_real = grid.flat_index(ngx + g, j, k);
      const std::size_t high_real = grid.flat_index(ngx + nx - 1 - g, j, k);

      for (std::size_t c = 0; c < N; ++c) {
        send_low[idx * N + c] = field(low_real, c);
        send_high[idx * N + c] = field(high_real, c);
      }
    });

    const MPI_Datatype datatype = backend::mpi::mpi_datatype_for<Scalar>();
    constexpr int kTagLeftward = 0;   // data moving in the -X direction
    constexpr int kTagRightward = 1;  // data moving in the +X direction

    // Two blocking Sendrecv calls, one per direction. Each call's send
    // and recv happen concurrently from this rank's point of view, so a
    // ring of these calls cannot deadlock at any rank count -- no
    // odd/even-rank ordering trick needed. A rank with MPI_PROC_NULL as
    // source or destination (the non-periodic physical-boundary case)
    // makes that side of the call a true no-op automatically.
    MPI_Sendrecv(send_low_.data(), static_cast<int>(count), datatype, left_rank_, kTagLeftward,
                 recv_high_.data(), static_cast<int>(count), datatype, right_rank_, kTagLeftward, comm_,
                 MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_high_.data(), static_cast<int>(count), datatype, right_rank_, kTagRightward,
                 recv_low_.data(), static_cast<int>(count), datatype, left_rank_, kTagRightward, comm_,
                 MPI_STATUS_IGNORE);

    // Unpack: recv_low_ -> this rank's low ghost cells (nearest-face-first,
    // matching the pack order above); recv_high_ -> this rank's high
    // ghost cells.
    const Scalar* recv_low = recv_low_.data();
    const Scalar* recv_high = recv_high_.data();
    Backend::run(slab_cells, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
      const std::size_t g = idx % ngx;
      const std::size_t rem = idx / ngx;
      const std::size_t j = rem % py;
      const std::size_t k = rem / py;

      const std::size_t low_ghost = grid.flat_index(ngx - 1 - g, j, k);
      const std::size_t high_ghost = grid.flat_index(ngx + nx + g, j, k);

      for (std::size_t c = 0; c < N; ++c) {
        field(low_ghost, c) = recv_low[idx * N + c];
        field(high_ghost, c) = recv_high[idx * N + c];
      }
    });
  }

 private:
  MPI_Comm comm_;
  int left_rank_;
  int right_rank_;
  // Sized once on first fill_x call for a given grid shape, never shrunk
  // afterward -- satisfies ARCHITECTURE.md #7's "MPI packing should
  // avoid repeated allocation" without needing allocation inside the
  // per-timestep hot loop itself.
  mutable std::vector<Scalar> send_low_, send_high_, recv_low_, recv_high_;
};

}  // namespace cfe
