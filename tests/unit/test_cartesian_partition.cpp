// Unit tests for CartesianPartition (task spec 0005): pure index
// arithmetic, no MPI toolchain needed.
#include "cfe/grid/partition/cartesian_partition.hpp"
#include "test_framework.hpp"

CFE_TEST(test_cartesian_partition_single_rank_1x1x1_owns_everything)
{
  const auto p = cfe::make_cartesian_partition(40, 20, 10, 1, 1, 1, 0, 1, true, true, true);
  CFE_CHECK(p.local_nx == 40);
  CFE_CHECK(p.local_ny == 20);
  CFE_CHECK(p.local_nz == 10);
  CFE_CHECK(p.global_offset_x == 0);
  CFE_CHECK(p.global_offset_y == 0);
  CFE_CHECK(p.global_offset_z == 0);
  // Periodic, single rank: every face wraps to itself (rank 0), same
  // topology PeriodicBoundary already uses for a single block.
  CFE_CHECK(p.x_left_rank == 0);
  CFE_CHECK(p.x_right_rank == 0);
  CFE_CHECK(p.y_left_rank == 0);
  CFE_CHECK(p.y_right_rank == 0);
  CFE_CHECK(p.z_left_rank == 0);
  CFE_CHECK(p.z_right_rank == 0);
}

CFE_TEST(test_cartesian_partition_local_extents_and_offsets_at_2x2x2)
{
  // global 16x16x16, 2x2x2 ranks -> 8x8x8 each, offsets at 0 or 8 per axis.
  for (int rank = 0; rank < 8; ++rank) {
    const auto p = cfe::make_cartesian_partition(16, 16, 16, 2, 2, 2, rank, 8, true, true, true);
    CFE_CHECK(p.local_nx == 8);
    CFE_CHECK(p.local_ny == 8);
    CFE_CHECK(p.local_nz == 8);
    CFE_CHECK(p.global_offset_x == 8 * static_cast<std::size_t>(rank % 2));
    CFE_CHECK(p.global_offset_y == 8 * static_cast<std::size_t>((rank / 2) % 2));
    CFE_CHECK(p.global_offset_z == 8 * static_cast<std::size_t>(rank / 4));
  }
}

CFE_TEST(test_cartesian_partition_face_neighbors_at_2x2x2_interior_wiring)
{
  // Rank layout (flatten = rz*4 + ry*2 + rx): rank 0 = (0,0,0).
  // Its +X neighbor is rank 1 = (1,0,0); +Y neighbor is rank 2 = (0,1,0);
  // +Z neighbor is rank 4 = (0,0,1). With periodic=true on all axes, its
  // -X/-Y/-Z neighbors wrap to the opposite coordinate (1,0,0 only has 2
  // values per axis here, so -X from rx=0 wraps to rx=1 too -- the same
  // rank as +X, which is correct for a 2-wide periodic axis).
  const auto p0 = cfe::make_cartesian_partition(16, 16, 16, 2, 2, 2, 0, 8, true, true, true);
  CFE_CHECK(p0.x_right_rank == 1);
  CFE_CHECK(p0.x_left_rank == 1);  // wraps: only 2 ranks along X, so left==right
  CFE_CHECK(p0.y_right_rank == 2);
  CFE_CHECK(p0.y_left_rank == 2);
  CFE_CHECK(p0.z_right_rank == 4);
  CFE_CHECK(p0.z_left_rank == 4);

  // Rank 3 = (1,1,0): -X neighbor is rank 2 = (0,1,0), +X wraps to rank 2
  // as well (2-wide periodic axis); -Y neighbor is rank 1 = (1,0,0), +Y
  // wraps to rank 1 as well; +Z neighbor is rank 7 = (1,1,1), -Z wraps to
  // rank 7 as well (2-wide Z axis).
  const auto p3 = cfe::make_cartesian_partition(16, 16, 16, 2, 2, 2, 3, 8, true, true, true);
  CFE_CHECK(p3.x_left_rank == 2);
  CFE_CHECK(p3.x_right_rank == 2);
  CFE_CHECK(p3.y_left_rank == 1);
  CFE_CHECK(p3.y_right_rank == 1);
  CFE_CHECK(p3.z_left_rank == 7);
  CFE_CHECK(p3.z_right_rank == 7);
}

CFE_TEST(test_cartesian_partition_non_cubic_shape_4x2x1)
{
  // global 40x20x5, process grid 4x2x1 (pz=1 -- Z not decomposed at all).
  const auto p = cfe::make_cartesian_partition(40, 20, 5, 4, 2, 1, 5 /* rx=1,ry=1,rz=0 */, 8, true, true,
                                                true);
  CFE_CHECK(p.local_nx == 10);  // 40/4
  CFE_CHECK(p.local_ny == 10);  // 20/2
  CFE_CHECK(p.local_nz == 5);   // 5/1, whole Z extent, unsplit
  CFE_CHECK(p.global_offset_x == 10);
  CFE_CHECK(p.global_offset_y == 10);
  CFE_CHECK(p.global_offset_z == 0);
  // pz=1: both Z neighbors wrap to the rank itself (periodic, single
  // layer along Z) -- same single-rank-periodic-self topology as the
  // 1x1x1 case above, just restricted to one axis here.
  CFE_CHECK(p.z_left_rank == 5);
  CFE_CHECK(p.z_right_rank == 5);
}

CFE_TEST(test_cartesian_partition_non_periodic_edges_have_no_neighbor)
{
  // rank 0 = (0,0,0) in a 2x2x2 grid, non-periodic on all axes: every
  // low-side neighbor of this corner rank must be kNoNeighbor.
  const auto p0 = cfe::make_cartesian_partition(16, 16, 16, 2, 2, 2, 0, 8, false, false, false);
  CFE_CHECK(p0.x_left_rank == cfe::CartesianPartition::kNoNeighbor);
  CFE_CHECK(p0.y_left_rank == cfe::CartesianPartition::kNoNeighbor);
  CFE_CHECK(p0.z_left_rank == cfe::CartesianPartition::kNoNeighbor);
  // High-side neighbors from this corner are still ordinary interior
  // neighbors (2-wide axes, so "high" is a real rank, not an edge, on
  // every axis from rank 0's perspective).
  CFE_CHECK(p0.x_right_rank == 1);
  CFE_CHECK(p0.y_right_rank == 2);
  CFE_CHECK(p0.z_right_rank == 4);

  // rank 7 = (1,1,1), the opposite corner: every high-side neighbor must
  // be kNoNeighbor instead.
  const auto p7 = cfe::make_cartesian_partition(16, 16, 16, 2, 2, 2, 7, 8, false, false, false);
  CFE_CHECK(p7.x_right_rank == cfe::CartesianPartition::kNoNeighbor);
  CFE_CHECK(p7.y_right_rank == cfe::CartesianPartition::kNoNeighbor);
  CFE_CHECK(p7.z_right_rank == cfe::CartesianPartition::kNoNeighbor);
}

CFE_TEST(test_cartesian_partition_mixed_periodicity_per_axis)
{
  // Periodic in X only; Y and Z are true physical boundaries.
  const auto p0 = cfe::make_cartesian_partition(16, 16, 16, 2, 2, 2, 0, 8, true, false, false);
  CFE_CHECK(p0.x_left_rank == 1);   // X periodic: wraps
  CFE_CHECK(p0.y_left_rank == cfe::CartesianPartition::kNoNeighbor);  // Y not periodic: true edge
  CFE_CHECK(p0.z_left_rank == cfe::CartesianPartition::kNoNeighbor);  // Z not periodic: true edge
}
