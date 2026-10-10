// MPI halo-exchange boundary condition (task spec 0004/0005, ADR 0009):
// fills one rank's ghost cells, on whichever ONE axis a given instance
// is responsible for, from its left/right neighbor ranks' real cells --
// instead of from local data the way every other boundary type in this
// file does. This is the "swappable neighbor provider" ADR 0004 and
// grid/ghost/ghost_fill.hpp's own header comment already named as the
// intended MPI plug-in point -- same duck-typed fill_x/fill_y/fill_z
// shape as StaticBoundary/PeriodicBoundary/InflowOutflowBoundary, so
// FvmSolver/fill_ghost_cells need zero changes to use it.
//
// One instance handles exactly one axis's neighbor pair (its
// constructor takes that axis's own left/right rank directly, not a
// whole partition object) -- construct THREE instances (one per axis,
// pulling each axis's neighbor pair out of a `SlabPartition` for a
// 1D-only decomposition, or a `CartesianPartition` for full 3D block
// decomposition) and plug them in as `BoundaryX`/`BoundaryY`/
// `BoundaryZ` simultaneously via explicit aggregate construction
// (`solver{grid, field, boundary_x, boundary_y, boundary_z}` --
// `FvmSolver` already supports this, zero changes needed there). All
// three of `fill_x`/`fill_y`/`fill_z` are real, genuinely-reachable
// implementations (not stubs) now that this type is used on every
// axis, each forwarding to one shared private `exchange<Axis>` template
// so the pack/exchange/unpack logic is written once, not tripled --
// mirrors `detail::axis_flux_difference`'s own `if constexpr`
// axis-dispatch pattern in `solver/explicit/fvm_solver.hpp`.
//
// Deliberately does NOT exchange diagonal/corner neighbors: this
// solver's reconstruction is strictly axis-split (see
// `fvm_solver.hpp`'s own header comment and `axis_flux_difference`) --
// no code path ever reads a ghost cell that is simultaneously a ghost
// on two axes at once, so face-neighbor exchange alone (one axis at a
// time, exactly what this type does) is sufficient for a full 3D block
// decomposition, not just a 1D slab. See ADR 0009's amendment.
//
// Remaining scope limits (see tasks/0005-...md's "Do not implement"
// list for the full exclusion list):
// - CPU-only (`CpuParallelFor`) -- combining this with CudaParallelFor
//   (GPU-aware MPI) is explicitly out of scope.
// - Blocking `MPI_Sendrecv`, not non-blocking Isend/Irecv with
//   communication/computation overlap (AGENTS.md #16 names overlap as
//   "eventual", not required for a first prototype).
// - A non-periodic partition's true physical-boundary ranks get a
//   genuine no-op Sendrecv on that one side -- their corresponding
//   ghost cells are simply never written by this type and stay at
//   whatever they were previously (zero, the first time). This type
//   does not compose with a second boundary condition to give a true
//   physical edge a real value; only the fully periodic case is
//   exercised/tested so far.
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
#include <vector>

#include "cfe/backend/mpi/mpi_datatype.hpp"
#include "cfe/backend/parallel_for.hpp"
#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"

namespace cfe {

template <class Scalar, std::size_t N>
class MpiHaloBoundary
{
 public:
  // `left_rank`/`right_rank` are this axis's own neighbor ranks (one of
  // `SlabPartition::left_rank/right_rank`, or one of
  // `CartesianPartition::x_left_rank/x_right_rank` /
  // `y_left_rank/y_right_rank` / `z_left_rank/z_right_rank`), with the
  // partition's own `kNoNeighbor` sentinel (`-1`) translated to
  // `MPI_PROC_NULL` here -- the one point plain-int partition data
  // crosses into MPI-aware code. `MPI_PROC_NULL`'s actual value is
  // implementation-defined, so neither partition type bakes it in.
  MpiHaloBoundary(MPI_Comm comm, int left_rank, int right_rank)
      : comm_(comm),
        left_rank_(left_rank < 0 ? MPI_PROC_NULL : left_rank),
        right_rank_(right_rank < 0 ? MPI_PROC_NULL : right_rank)
  {
  }

  template <class Backend = CpuParallelFor, class Layout>
  void fill_x(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    exchange<Axis::X, Backend>(field, grid);
  }

  template <class Backend = CpuParallelFor, class Layout>
  void fill_y(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    exchange<Axis::Y, Backend>(field, grid);
  }

  template <class Backend = CpuParallelFor, class Layout>
  void fill_z(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    exchange<Axis::Z, Backend>(field, grid);
  }

 private:
  // The one shared pack/exchange/unpack implementation, parameterized
  // on which axis to operate along (`if constexpr`, resolved at compile
  // time -- same dispatch shape `detail::axis_flux_difference` uses).
  // Packing/unpacking index formulas mirror `PeriodicBoundary::fill_x`/
  // `fill_y`/`fill_z`'s own existing per-axis argument ordering
  // (X: (axis,o1,o2); Y: (o1,axis,o2); Z: (o1,o2,axis)) so this reads as
  // a direct generalization of code already in this file, not a new
  // convention.
  template <Axis A, class Backend, class Layout>
  void exchange(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    std::size_t ghost_depth, axis_extent, p_o1, p_o2;
    if constexpr (A == Axis::X) {
      ghost_depth = grid.ngx;
      axis_extent = grid.nx;
      p_o1 = grid.padded_ny();
      p_o2 = grid.padded_nz();
    } else if constexpr (A == Axis::Y) {
      ghost_depth = grid.ngy;
      axis_extent = grid.ny;
      p_o1 = grid.padded_nx();
      p_o2 = grid.padded_nz();
    } else {
      ghost_depth = grid.ngz;
      axis_extent = grid.nz;
      p_o1 = grid.padded_nx();
      p_o2 = grid.padded_ny();
    }

    if (ghost_depth == 0) return;
    // Same hazard PeriodicBoundary::fill_x/fill_y/fill_z already guard
    // against: a local extent narrower than the ghost depth would make
    // the pack/unpack index arithmetic below read into the ghost region
    // itself instead of real cells. Reused rather than duplicated.
    detail::require_periodic_extent_covers_ghost_depth(
        axis_extent, ghost_depth, A == Axis::X ? "x" : (A == Axis::Y ? "y" : "z"));

    const std::size_t ng = ghost_depth;
    const std::size_t ax_n = axis_extent;
    const std::size_t slab_cells = ng * p_o1 * p_o2;
    const std::size_t count = slab_cells * N;

    if (send_low_.size() != count) {
      send_low_.resize(count);
      send_high_.resize(count);
      recv_low_.resize(count);
      recv_high_.resize(count);
    }

    // Pack: send_low_[g,o1,o2] <- this rank's real cell at distance `g`
    // from its own LOW face on axis A (the face shared with
    // `left_rank_`) -- what `left_rank_` will use to fill ITS high
    // ghost on this axis. send_high_ is the mirror image for the HIGH
    // face / `right_rank_`. `g=0` is nearest the shared face in both
    // cases, matching the adjacency order the unpack step below expects.
    Scalar* send_low = send_low_.data();
    Scalar* send_high = send_high_.data();
    Backend::run(slab_cells, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
      const std::size_t g = idx % ng;
      const std::size_t rem = idx / ng;
      const std::size_t o1 = rem % p_o1;
      const std::size_t o2 = rem / p_o1;

      std::size_t low_real, high_real;
      if constexpr (A == Axis::X) {
        low_real = grid.flat_index(ng + g, o1, o2);
        high_real = grid.flat_index(ng + ax_n - 1 - g, o1, o2);
      } else if constexpr (A == Axis::Y) {
        low_real = grid.flat_index(o1, ng + g, o2);
        high_real = grid.flat_index(o1, ng + ax_n - 1 - g, o2);
      } else {
        low_real = grid.flat_index(o1, o2, ng + g);
        high_real = grid.flat_index(o1, o2, ng + ax_n - 1 - g);
      }

      for (std::size_t c = 0; c < N; ++c) {
        send_low[idx * N + c] = field(low_real, c);
        send_high[idx * N + c] = field(high_real, c);
      }
    });

    const MPI_Datatype datatype = backend::mpi::mpi_datatype_for<Scalar>();
    constexpr int kTagLeftward = 0;   // data moving in the -axis direction
    constexpr int kTagRightward = 1;  // data moving in the +axis direction

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

    // Unpack: recv_low_ -> this rank's low ghost cells on axis A
    // (nearest-face-first, matching the pack order above); recv_high_
    // -> this rank's high ghost cells.
    const Scalar* recv_low = recv_low_.data();
    const Scalar* recv_high = recv_high_.data();
    Backend::run(slab_cells, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
      const std::size_t g = idx % ng;
      const std::size_t rem = idx / ng;
      const std::size_t o1 = rem % p_o1;
      const std::size_t o2 = rem / p_o1;

      std::size_t low_ghost, high_ghost;
      if constexpr (A == Axis::X) {
        low_ghost = grid.flat_index(ng - 1 - g, o1, o2);
        high_ghost = grid.flat_index(ng + ax_n + g, o1, o2);
      } else if constexpr (A == Axis::Y) {
        low_ghost = grid.flat_index(o1, ng - 1 - g, o2);
        high_ghost = grid.flat_index(o1, ng + ax_n + g, o2);
      } else {
        low_ghost = grid.flat_index(o1, o2, ng - 1 - g);
        high_ghost = grid.flat_index(o1, o2, ng + ax_n + g);
      }

      for (std::size_t c = 0; c < N; ++c) {
        field(low_ghost, c) = recv_low[idx * N + c];
        field(high_ghost, c) = recv_high[idx * N + c];
      }
    });
  }

  MPI_Comm comm_;
  int left_rank_;
  int right_rank_;
  // Sized once on first exchange() call for a given grid shape, never
  // shrunk afterward -- satisfies ARCHITECTURE.md #7's "MPI packing
  // should avoid repeated allocation" without needing allocation inside
  // the per-timestep hot loop itself. Shared by whichever ONE axis this
  // instance handles (fill_x/fill_y/fill_z are never all called on the
  // same instance, so there is no cross-axis sizing conflict).
  mutable std::vector<Scalar> send_low_, send_high_, recv_low_, recv_high_;
};

}  // namespace cfe
