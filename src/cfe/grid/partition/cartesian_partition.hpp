// Full 3D block domain decomposition (task spec 0005, ADR 0009's
// amended decision): generalizes `SlabPartition`'s 1D-along-X-only
// split to all three axes at once, computing each rank's local extent/
// offset per axis plus its 6 face-neighbor ranks (+-X, +-Y, +-Z).
//
// Deliberately does NOT need (or compute) diagonal/corner-neighbor
// ranks: `fvm_solver.hpp`'s residual loop is strictly axis-split (each
// axis's flux pass only ever varies that one axis's index, holding the
// other two at the real cell's own value -- see
// `detail::axis_flux_difference` and the interior loop in
// `FvmSolver::residual()`), so no code path ever reads a ghost cell
// that is simultaneously a ghost on two axes at once. Face-neighbor
// exchange alone (what `MpiHaloBoundary` does, once per axis) is
// therefore sufficient -- this was verified by direct inspection, not
// assumed, and is recorded in ADR 0009's amendment.
//
// Zero MPI dependency, same as `SlabPartition` -- pure index
// arithmetic, unit-testable without any MPI toolchain. `px*py*pz` must
// equal the communicator size: this type does not auto-factor a rank
// count into a 3D shape (that choice is left to the caller -- a real
// production run would let the launcher/scheduler decide it, not the
// library; see `tutorials/mpi_scalar_advection_3d_strong_scaling/` for
// a tutorial-local example of picking a reasonable shape).
#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "cfe/grid/partition/slab_partition.hpp"

namespace cfe {

struct CartesianPartition
{
  static constexpr int kNoNeighbor = SlabPartition::kNoNeighbor;

  std::size_t local_nx = 0, local_ny = 0, local_nz = 0;
  std::size_t global_offset_x = 0, global_offset_y = 0, global_offset_z = 0;

  int x_left_rank = kNoNeighbor, x_right_rank = kNoNeighbor;
  int y_left_rank = kNoNeighbor, y_right_rank = kNoNeighbor;
  int z_left_rank = kNoNeighbor, z_right_rank = kNoNeighbor;
};

// `rank` is unraveled into 3D process-grid coordinates `(rx,ry,rz)` in
// row-major order (X fastest-varying, matching `CartesianGrid::
// flat_index`'s own axis-ordering convention): `rz = rank/(px*py);
// ry = (rank%(px*py))/px; rx = rank%px`. Each axis's local extent and
// global offset is obtained by calling the already-proven
// `make_slab_partition` once per axis, treating that axis's own
// process-grid coordinate/count as an independent 1D split -- this
// reuses the remainder-to-the-first-ranks rule without duplicating it.
// Face-neighbor ranks are computed by moving one coordinate at a time
// (holding the other two fixed) and re-flattening via `flatten(rx,ry,rz)
// = rz*px*py + ry*px + rx`, with periodic wraparound or `kNoNeighbor`
// (a true physical boundary) chosen independently per axis.
inline CartesianPartition make_cartesian_partition(std::size_t global_nx, std::size_t global_ny,
                                                    std::size_t global_nz, int px, int py, int pz, int rank,
                                                    int size, bool periodic_x, bool periodic_y,
                                                    bool periodic_z)
{
  assert(px > 0 && py > 0 && pz > 0 && "px, py, pz must each be positive");
  assert(rank >= 0 && rank < size && "rank must be in [0, size)");
  // Always-on, not an assert: this depends on runtime launch
  // configuration (how many ranks were actually started vs. what shape
  // the caller chose), exactly the kind of invariant that must still be
  // checked in a Release build -- same reasoning as
  // `grid/boundary/boundary_condition.hpp`'s own
  // `require_periodic_extent_covers_ghost_depth`.
  if (px * py * pz != size) {
    throw std::invalid_argument("make_cartesian_partition: px*py*pz (" + std::to_string(px) + "*" +
                                 std::to_string(py) + "*" + std::to_string(pz) +
                                 ") must equal the communicator size (" + std::to_string(size) + ")");
  }

  const int rx = rank % px;
  const int ry = (rank / px) % py;
  const int rz = rank / (px * py);

  auto flatten = [&](int x, int y, int z) { return z * px * py + y * px + x; };

  CartesianPartition partition;

  const SlabPartition sx = make_slab_partition(global_nx, rx, px, periodic_x);
  partition.local_nx = sx.local_nx;
  partition.global_offset_x = sx.global_offset_x;

  const SlabPartition sy = make_slab_partition(global_ny, ry, py, periodic_y);
  partition.local_ny = sy.local_nx;
  partition.global_offset_y = sy.global_offset_x;

  const SlabPartition sz = make_slab_partition(global_nz, rz, pz, periodic_z);
  partition.local_nz = sz.local_nx;
  partition.global_offset_z = sz.global_offset_x;

  if (rx > 0) {
    partition.x_left_rank = flatten(rx - 1, ry, rz);
  } else if (periodic_x) {
    partition.x_left_rank = flatten(px - 1, ry, rz);
  }
  if (rx < px - 1) {
    partition.x_right_rank = flatten(rx + 1, ry, rz);
  } else if (periodic_x) {
    partition.x_right_rank = flatten(0, ry, rz);
  }

  if (ry > 0) {
    partition.y_left_rank = flatten(rx, ry - 1, rz);
  } else if (periodic_y) {
    partition.y_left_rank = flatten(rx, py - 1, rz);
  }
  if (ry < py - 1) {
    partition.y_right_rank = flatten(rx, ry + 1, rz);
  } else if (periodic_y) {
    partition.y_right_rank = flatten(rx, 0, rz);
  }

  if (rz > 0) {
    partition.z_left_rank = flatten(rx, ry, rz - 1);
  } else if (periodic_z) {
    partition.z_left_rank = flatten(rx, ry, pz - 1);
  }
  if (rz < pz - 1) {
    partition.z_right_rank = flatten(rx, ry, rz + 1);
  } else if (periodic_z) {
    partition.z_right_rank = flatten(rx, ry, 0);
  }

  return partition;
}

}  // namespace cfe
