// Uniform Cartesian grid, generic over 1D/2D/3D (task spec item 1,
// ARCHITECTURE.md #6, ADR 0004).
//
// A `CartesianGrid` describes ONE block: real (interior) cell counts,
// ghost-layer depth, and cell spacing per dimension, plus the physical
// origin of the interior domain's lower corner. Unused dimensions (for a
// 1D or 2D problem) are represented by `n_cells == 1, n_ghost == 0` --
// there is no separate 1D/2D/3D type, to avoid combinatorial
// specialization for a small block description.
//
// `CartesianGrid` has NO notion of storage: it only converts a padded
// (i,j,k) triple (including ghost layers) to the single flat `cell` index
// that `cfe::Field`/`FieldView` already index by (see field/field.hpp).
// This is deliberate -- Field/FieldView need no changes at all to support
// a grid.
//
// Per AGENTS.md #2 (build for measured/anticipated needs, not
// speculation) and the AMR-readiness direction recorded in
// tasks/0002-phase1-cartesian-grid-scalar-transport.md: `dx`/`dy`/`dz`
// and extents are scoped to this one block/grid object, not a global
// constant, specifically so that fixed, block-based static refinement
// (multiple grids at different resolutions) can be added later without
// redesigning this type.
#pragma once

#include <cstddef>

#include "cfe/core/macros.hpp"

namespace cfe {

// Which face of the block a boundary condition or ghost-fill operation
// applies to. Only the faces relevant to active dimensions (n_cells > 1
// or explicitly enabled) are ever used; see CartesianGrid::n_ghost_*.
enum class Axis
{
  X,
  Y,
  Z
};

enum class Side
{
  Low,
  High
};

struct CartesianGrid
{
  // Real (non-ghost) cell counts per dimension. 1 for an unused dimension.
  std::size_t nx = 1;
  std::size_t ny = 1;
  std::size_t nz = 1;

  // Ghost-layer depth per dimension. 0 for an unused dimension -- there is
  // nothing to be a neighbor of, and a periodic/static boundary condition
  // on a size-1 axis is meaningless.
  std::size_t ngx = 0;
  std::size_t ngy = 0;
  std::size_t ngz = 0;

  // Cell spacing per dimension. Meaningless (and unused) on an axis with
  // n_cells == 1.
  double dx = 1.0;
  double dy = 1.0;
  double dz = 1.0;

  // Physical coordinate of the interior domain's lower corner.
  double origin_x = 0.0;
  double origin_y = 0.0;
  double origin_z = 0.0;

  CFE_HOST_DEVICE
  std::size_t padded_nx() const { return nx + 2 * ngx; }
  CFE_HOST_DEVICE
  std::size_t padded_ny() const { return ny + 2 * ngy; }
  CFE_HOST_DEVICE
  std::size_t padded_nz() const { return nz + 2 * ngz; }

  CFE_HOST_DEVICE
  std::size_t n_cells_total() const { return padded_nx() * padded_ny() * padded_nz(); }

  // i, j, k are PADDED indices (including ghost layers): valid range is
  // [0, padded_nx()) etc. Real interior cells occupy [ngx, ngx + nx), and
  // so on for j, k. This is the one and only place (i,j,k) is converted
  // to the flat `cell` index Field/FieldView expect.
  CFE_HOST_DEVICE
  std::size_t flat_index(std::size_t i, std::size_t j, std::size_t k) const
  {
    return i + j * padded_nx() + k * padded_nx() * padded_ny();
  }

  // Physical cell-center coordinate for a padded index. Host-only: used
  // for setting initial conditions and evaluating exact solutions, never
  // inside a per-timestep kernel.
  double x_center(std::size_t i) const
  {
    return origin_x + (static_cast<double>(i) - static_cast<double>(ngx) + 0.5) * dx;
  }
  double y_center(std::size_t j) const
  {
    return origin_y + (static_cast<double>(j) - static_cast<double>(ngy) + 0.5) * dy;
  }
  double z_center(std::size_t k) const
  {
    return origin_z + (static_cast<double>(k) - static_cast<double>(ngz) + 0.5) * dz;
  }
};

}  // namespace cfe
