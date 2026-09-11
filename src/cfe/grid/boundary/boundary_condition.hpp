// Boundary conditions: static (fixed value) and periodic only (task spec
// item 3). Each fills the ghost layer of one axis of a CartesianGrid,
// dispatched via cfe::parallel_for so the same code works whether the
// field lives in host or device memory (see field/field.hpp's
// backend-agnostic FieldView).
//
// Deliberately not a polymorphic base class (AGENTS.md #12: no virtual
// functions inside kernels) -- StaticBoundary and PeriodicBoundary are
// unrelated types with the same duck-typed fill_x/fill_y/fill_z shape,
// exactly like AoSLayout/SoALayout in field/layout.hpp. grid/ghost/
// ghost_fill.hpp provides the single call-site dispatch
// (fill_ghost_cells(field, grid, axis, boundary)) that is this project's
// actual "swappable neighbor provider" seam -- see that file for the
// AMR/MPI-readiness reasoning.
#pragma once

#include <cstddef>

#include "cfe/backend/parallel_for.hpp"
#include "cfe/field/field.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/math/fixed_array.hpp"

namespace cfe {

// Wraps ghost cells from the opposite real boundary of the same axis.
struct PeriodicBoundary
{
  template <class Scalar, std::size_t N, class Layout>
  void fill_x(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngx == 0) return;
    const std::size_t py = grid.padded_ny();
    const std::size_t pz = grid.padded_nz();
    cfe::parallel_for(grid.ngx * py * pz, [=](std::size_t idx) mutable {
      const std::size_t g = idx % grid.ngx;
      const std::size_t rem = idx / grid.ngx;
      const std::size_t j = rem % py;
      const std::size_t k = rem / py;

      const std::size_t low_ghost = grid.flat_index(grid.ngx - 1 - g, j, k);
      const std::size_t low_source = grid.flat_index(grid.ngx + grid.nx - 1 - g, j, k);
      const std::size_t high_ghost = grid.flat_index(grid.ngx + grid.nx + g, j, k);
      const std::size_t high_source = grid.flat_index(grid.ngx + g, j, k);

      for (std::size_t c = 0; c < N; ++c) {
        field(low_ghost, c) = field(low_source, c);
        field(high_ghost, c) = field(high_source, c);
      }
    });
  }

  template <class Scalar, std::size_t N, class Layout>
  void fill_y(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngy == 0) return;
    const std::size_t px = grid.padded_nx();
    const std::size_t pz = grid.padded_nz();
    cfe::parallel_for(grid.ngy * px * pz, [=](std::size_t idx) mutable {
      const std::size_t g = idx % grid.ngy;
      const std::size_t rem = idx / grid.ngy;
      const std::size_t i = rem % px;
      const std::size_t k = rem / px;

      const std::size_t low_ghost = grid.flat_index(i, grid.ngy - 1 - g, k);
      const std::size_t low_source = grid.flat_index(i, grid.ngy + grid.ny - 1 - g, k);
      const std::size_t high_ghost = grid.flat_index(i, grid.ngy + grid.ny + g, k);
      const std::size_t high_source = grid.flat_index(i, grid.ngy + g, k);

      for (std::size_t c = 0; c < N; ++c) {
        field(low_ghost, c) = field(low_source, c);
        field(high_ghost, c) = field(high_source, c);
      }
    });
  }

  template <class Scalar, std::size_t N, class Layout>
  void fill_z(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngz == 0) return;
    const std::size_t px = grid.padded_nx();
    const std::size_t py = grid.padded_ny();
    cfe::parallel_for(grid.ngz * px * py, [=](std::size_t idx) mutable {
      const std::size_t g = idx % grid.ngz;
      const std::size_t rem = idx / grid.ngz;
      const std::size_t i = rem % px;
      const std::size_t j = rem / px;

      const std::size_t low_ghost = grid.flat_index(i, j, grid.ngz - 1 - g);
      const std::size_t low_source = grid.flat_index(i, j, grid.ngz + grid.nz - 1 - g);
      const std::size_t high_ghost = grid.flat_index(i, j, grid.ngz + grid.nz + g);
      const std::size_t high_source = grid.flat_index(i, j, grid.ngz + g);

      for (std::size_t c = 0; c < N; ++c) {
        field(low_ghost, c) = field(low_source, c);
        field(high_ghost, c) = field(high_source, c);
      }
    });
  }
};

// Writes a fixed state into every ghost cell on each side of the axis.
// The two sides may hold different values (e.g. different fixed
// concentrations at each end of a 1D domain).
template <class Scalar, std::size_t N>
struct StaticBoundary
{
  State<Scalar, N> low_value;
  State<Scalar, N> high_value;

  StaticBoundary() = default;
  StaticBoundary(const State<Scalar, N>& low, const State<Scalar, N>& high)
      : low_value(low), high_value(high)
  {
  }

  template <class Layout>
  void fill_x(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngx == 0) return;
    const std::size_t py = grid.padded_ny();
    const std::size_t pz = grid.padded_nz();
    const State<Scalar, N> low = low_value;
    const State<Scalar, N> high = high_value;
    cfe::parallel_for(grid.ngx * py * pz, [=](std::size_t idx) mutable {
      const std::size_t g = idx % grid.ngx;
      const std::size_t rem = idx / grid.ngx;
      const std::size_t j = rem % py;
      const std::size_t k = rem / py;

      const std::size_t low_ghost = grid.flat_index(grid.ngx - 1 - g, j, k);
      const std::size_t high_ghost = grid.flat_index(grid.ngx + grid.nx + g, j, k);

      for (std::size_t c = 0; c < N; ++c) {
        field(low_ghost, c) = low[c];
        field(high_ghost, c) = high[c];
      }
    });
  }

  template <class Layout>
  void fill_y(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngy == 0) return;
    const std::size_t px = grid.padded_nx();
    const std::size_t pz = grid.padded_nz();
    const State<Scalar, N> low = low_value;
    const State<Scalar, N> high = high_value;
    cfe::parallel_for(grid.ngy * px * pz, [=](std::size_t idx) mutable {
      const std::size_t g = idx % grid.ngy;
      const std::size_t rem = idx / grid.ngy;
      const std::size_t i = rem % px;
      const std::size_t k = rem / px;

      const std::size_t low_ghost = grid.flat_index(i, grid.ngy - 1 - g, k);
      const std::size_t high_ghost = grid.flat_index(i, grid.ngy + grid.ny + g, k);

      for (std::size_t c = 0; c < N; ++c) {
        field(low_ghost, c) = low[c];
        field(high_ghost, c) = high[c];
      }
    });
  }

  template <class Layout>
  void fill_z(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngz == 0) return;
    const std::size_t px = grid.padded_nx();
    const std::size_t py = grid.padded_ny();
    const State<Scalar, N> low = low_value;
    const State<Scalar, N> high = high_value;
    cfe::parallel_for(grid.ngz * px * py, [=](std::size_t idx) mutable {
      const std::size_t g = idx % grid.ngz;
      const std::size_t rem = idx / grid.ngz;
      const std::size_t i = rem % px;
      const std::size_t j = rem / px;

      const std::size_t low_ghost = grid.flat_index(i, j, grid.ngz - 1 - g);
      const std::size_t high_ghost = grid.flat_index(i, j, grid.ngz + grid.nz + g);

      for (std::size_t c = 0; c < N; ++c) {
        field(low_ghost, c) = low[c];
        field(high_ghost, c) = high[c];
      }
    });
  }
};

}  // namespace cfe
