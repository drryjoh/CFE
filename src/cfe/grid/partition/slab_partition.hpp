// 1D slab domain decomposition along X (task spec 0004, ARCHITECTURE.md
// #7, ADR 0009): splits a global cell count across MPI ranks and
// computes each rank's local cell count, its offset into the global
// domain, and its left/right neighbor ranks.
//
// Deliberately has NO MPI dependency -- no `<mpi.h>`, no `MPI_Comm`, just
// plain `int` rank/size arguments. This is pure index arithmetic, so it
// is unit-testable today with the ordinary single-process
// `cfe_unit_tests` binary, with no `CFE_ENABLE_MPI` build gating at all.
// `grid/boundary/mpi_halo_boundary.hpp` is the one place a
// `SlabPartition` is consumed by MPI-aware code; it translates
// `kNoNeighbor` to `MPI_PROC_NULL` there, not here, since
// `MPI_PROC_NULL`'s actual value is implementation-defined and must not
// leak into this MPI-independent type.
//
// Only X is decomposed (task 0004's explicit scope -- see that file's
// "Do not implement" list for why Y/Z slab and full 3D block
// decomposition are separate follow-up work): `CartesianGrid::
// flat_index` makes X the unit-stride axis, so halo exchange along X is
// the structurally hardest case (strided packing once ny/nz > 1, not a
// contiguous end-of-array run) -- proving it correct here first, then
// extending to other axes, is the intended order of operations.
#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>

namespace cfe {

struct SlabPartition
{
  // Not a valid MPI rank -- this rank/side has no neighbor at all (a
  // real physical boundary), distinct from "the neighbor is rank 0/N-1
  // via periodic wraparound".
  static constexpr int kNoNeighbor = -1;

  std::size_t local_nx = 0;
  std::size_t global_offset_x = 0;
  int left_rank = kNoNeighbor;
  int right_rank = kNoNeighbor;
};

// Splits `global_nx` cells across `size` ranks as evenly as possible:
// the first `global_nx % size` ranks get one extra cell, so no two
// ranks' `local_nx` ever differ by more than 1 (the standard
// remainder-to-the-first-ranks rule). Requires `global_nx >= size` --
// a rank with zero cells has no valid `CartesianGrid`.
//
// `periodic`: true wraps rank 0's left neighbor to `size-1` and
// `size-1`'s right neighbor to `0` (matching `PeriodicBoundary`'s own
// topology, so a problem's Y/Z axes can stay `PeriodicBoundary` while X
// uses this); false leaves rank 0/`size-1`'s outward side as
// `kNoNeighbor` -- a genuine physical boundary for that one rank only,
// handled by whatever per-rank boundary the caller supplies for the
// non-periodic case (not expressible as a single process-wide boundary
// type, since only the two edge ranks have one).
inline SlabPartition make_slab_partition(std::size_t global_nx, int rank, int size, bool periodic)
{
  assert(size > 0 && "size must be a positive rank count");
  assert(rank >= 0 && rank < size && "rank must be in [0, size)");
  assert(global_nx >= static_cast<std::size_t>(size) &&
         "global_nx must be >= size -- a rank with zero cells has no valid CartesianGrid");

  const std::size_t base = global_nx / static_cast<std::size_t>(size);
  const std::size_t remainder = global_nx % static_cast<std::size_t>(size);
  const std::size_t rank_u = static_cast<std::size_t>(rank);

  SlabPartition partition;
  partition.local_nx = base + (rank_u < remainder ? 1 : 0);
  partition.global_offset_x = rank_u * base + std::min(rank_u, remainder);

  if (size == 1) {
    partition.left_rank = periodic ? 0 : SlabPartition::kNoNeighbor;
    partition.right_rank = periodic ? 0 : SlabPartition::kNoNeighbor;
    return partition;
  }

  if (rank == 0) {
    partition.left_rank = periodic ? size - 1 : SlabPartition::kNoNeighbor;
  } else {
    partition.left_rank = rank - 1;
  }

  if (rank == size - 1) {
    partition.right_rank = periodic ? 0 : SlabPartition::kNoNeighbor;
  } else {
    partition.right_rank = rank + 1;
  }

  return partition;
}

}  // namespace cfe
