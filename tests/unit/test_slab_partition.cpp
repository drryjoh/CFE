// Unit tests for SlabPartition (task spec 0004): pure index arithmetic,
// no MPI toolchain needed -- runs in the ordinary single-process
// cfe_unit_tests binary.
#include "cfe/grid/partition/slab_partition.hpp"
#include "test_framework.hpp"

CFE_TEST(test_slab_partition_splits_evenly_divisible_global_count)
{
  // global_nx=100, size=4 -> every rank gets exactly 25 cells.
  for (int rank = 0; rank < 4; ++rank) {
    const auto p = cfe::make_slab_partition(100, rank, 4, /*periodic=*/true);
    CFE_CHECK(p.local_nx == 25);
    CFE_CHECK(p.global_offset_x == static_cast<std::size_t>(rank) * 25);
  }
}

CFE_TEST(test_slab_partition_gives_remainder_cells_to_the_first_ranks)
{
  // global_nx=10, size=3 -> base=3, remainder=1: rank 0 gets 4, ranks 1/2 get 3.
  const auto p0 = cfe::make_slab_partition(10, 0, 3, /*periodic=*/true);
  const auto p1 = cfe::make_slab_partition(10, 1, 3, /*periodic=*/true);
  const auto p2 = cfe::make_slab_partition(10, 2, 3, /*periodic=*/true);

  CFE_CHECK(p0.local_nx == 4);
  CFE_CHECK(p1.local_nx == 3);
  CFE_CHECK(p2.local_nx == 3);

  // Offsets must be contiguous and span the whole global domain with no
  // gap or overlap.
  CFE_CHECK(p0.global_offset_x == 0);
  CFE_CHECK(p1.global_offset_x == p0.global_offset_x + p0.local_nx);
  CFE_CHECK(p2.global_offset_x == p1.global_offset_x + p1.local_nx);
  CFE_CHECK(p2.global_offset_x + p2.local_nx == 10);
}

CFE_TEST(test_slab_partition_periodic_wraps_neighbor_ranks_at_both_ends)
{
  const auto p0 = cfe::make_slab_partition(100, 0, 4, /*periodic=*/true);
  const auto p1 = cfe::make_slab_partition(100, 1, 4, /*periodic=*/true);
  const auto p2 = cfe::make_slab_partition(100, 2, 4, /*periodic=*/true);
  const auto p3 = cfe::make_slab_partition(100, 3, 4, /*periodic=*/true);

  // Interior ranks: ordinary +-1 neighbors.
  CFE_CHECK(p1.left_rank == 0);
  CFE_CHECK(p1.right_rank == 2);
  CFE_CHECK(p2.left_rank == 1);
  CFE_CHECK(p2.right_rank == 3);

  // Edge ranks: wrap around to the opposite end, matching PeriodicBoundary's topology.
  CFE_CHECK(p0.left_rank == 3);
  CFE_CHECK(p0.right_rank == 1);
  CFE_CHECK(p3.left_rank == 2);
  CFE_CHECK(p3.right_rank == 0);
}

CFE_TEST(test_slab_partition_non_periodic_leaves_edge_ranks_with_no_neighbor)
{
  const auto p0 = cfe::make_slab_partition(100, 0, 4, /*periodic=*/false);
  const auto p3 = cfe::make_slab_partition(100, 3, 4, /*periodic=*/false);

  CFE_CHECK(p0.left_rank == cfe::SlabPartition::kNoNeighbor);
  CFE_CHECK(p0.right_rank == 1);  // interior-facing side is unaffected by periodic/false
  CFE_CHECK(p3.left_rank == 2);
  CFE_CHECK(p3.right_rank == cfe::SlabPartition::kNoNeighbor);
}

CFE_TEST(test_slab_partition_single_rank_periodic_neighbors_itself)
{
  // size=1: a single rank owns the whole domain. Periodic wraparound
  // means its own left/right neighbor is itself (rank 0) -- matching
  // PeriodicBoundary's existing single-rank topology (it already wraps a
  // single block's ghost cells from its own opposite real edge).
  const auto p = cfe::make_slab_partition(50, 0, 1, /*periodic=*/true);
  CFE_CHECK(p.local_nx == 50);
  CFE_CHECK(p.global_offset_x == 0);
  CFE_CHECK(p.left_rank == 0);
  CFE_CHECK(p.right_rank == 0);
}

CFE_TEST(test_slab_partition_single_rank_non_periodic_has_no_neighbors)
{
  const auto p = cfe::make_slab_partition(50, 0, 1, /*periodic=*/false);
  CFE_CHECK(p.local_nx == 50);
  CFE_CHECK(p.left_rank == cfe::SlabPartition::kNoNeighbor);
  CFE_CHECK(p.right_rank == cfe::SlabPartition::kNoNeighbor);
}
