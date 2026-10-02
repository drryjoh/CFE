// Uniform Cartesian grid, generic over 1D/2D/3D (task spec item 1,
// ARCHITECTURE.md #6, ADR 0004).
//
// A `CartesianGrid<Scalar>` describes ONE block: real (interior) cell
// counts, ghost-layer depth, and cell spacing per dimension, plus the
// physical origin of the interior domain's lower corner. Unused
// dimensions (for a 1D or 2D problem) are represented by `n_cells == 1,
// n_ghost == 0` -- there is no separate 1D/2D/3D type, to avoid
// combinatorial specialization for a small block description.
//
// `CartesianGrid` has NO notion of storage: it only converts a padded
// (i,j,k) triple (including ghost layers) to the single flat `cell` index
// that `cfe::Field`/`FieldView` already index by (see field/field.hpp).
// This is deliberate -- Field/FieldView need no changes at all to support
// a grid.
//
// Templated on `Scalar` (matching every other type in this codebase --
// `Field`, `FixedArray`, etc.) rather than hardcoding `double`: spacing
// and origin are used directly alongside state values in per-timestep
// code (see solver/explicit/fvm_solver.hpp), so a `float`-precision
// solver should not carry a hidden `double` conversion for its grid
// spacing.
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
#include "cfe/core/types.hpp"

namespace cfe {

template <class Scalar>
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
  Scalar dx = Scalar(1);
  Scalar dy = Scalar(1);
  Scalar dz = Scalar(1);

  // Physical coordinate of the interior domain's lower corner.
  Scalar origin_x = Scalar(0);
  Scalar origin_y = Scalar(0);
  Scalar origin_z = Scalar(0);

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
  Scalar x_center(std::size_t i) const
  {
    return origin_x + (static_cast<Scalar>(i) - static_cast<Scalar>(ngx) + Scalar(0.5)) * dx;
  }
  Scalar y_center(std::size_t j) const
  {
    return origin_y + (static_cast<Scalar>(j) - static_cast<Scalar>(ngy) + Scalar(0.5)) * dy;
  }
  Scalar z_center(std::size_t k) const
  {
    return origin_z + (static_cast<Scalar>(k) - static_cast<Scalar>(ngz) + Scalar(0.5)) * dz;
  }
};

}  // namespace cfe
