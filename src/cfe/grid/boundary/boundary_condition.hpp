// Boundary conditions: static (fixed value, both ends), periodic, and
// inflow/outflow (fixed low end, zero-order-extrapolated high end --
// added for the Burgers moving-shock tutorial, task spec item 3's
// static/periodic-only scope having been for Phase 1 specifically, not
// a permanent ceiling). Each fills the ghost layer of one axis of a
// CartesianGrid, dispatched through `Backend::run` (default
// `CpuParallelFor`; pass `CudaParallelFor` from a `.cu` translation
// unit) so the same code works whether the field lives in host or
// device memory (see field/field.hpp's backend-agnostic FieldView).
//
// Deliberately not a polymorphic base class (AGENTS.md #12: no virtual
// functions inside kernels) -- StaticBoundary, PeriodicBoundary, and
// InflowOutflowBoundary are unrelated types with the same duck-typed
// fill_x/fill_y/fill_z shape, exactly like AoSLayout/SoALayout in
// field/layout.hpp. grid/ghost/ghost_fill.hpp provides the single
// call-site dispatch (fill_ghost_cells(field, grid, axis, boundary))
// that is this project's actual "swappable neighbor provider" seam --
// see that file for the AMR/MPI-readiness reasoning.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include "cfe/backend/parallel_for.hpp"
#include "cfe/core/macros.hpp"
#include "cfe/field/field.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/math/fixed_array.hpp"

namespace cfe {

namespace detail {

// A plain, always-on runtime check (never an `assert`): the invariant
// below must hold in every build configuration, including Release
// (`NDEBUG`), because this project's own default and benchmarked build
// type *is* Release -- an `assert`-based guard would silently compile
// out exactly where it matters most (caught in code review; see
// agent_history.md's 2026-09-30 follow-up entry). Host-only; called
// once per `fill_x/y/z` invocation, not per cell, so this costs nothing
// measurable even though it runs unconditionally.
inline void require_periodic_extent_covers_ghost_depth(std::size_t extent, std::size_t ghost_depth,
                                                        const char* axis_name)
{
  if (extent < ghost_depth) {
    throw std::invalid_argument(
        std::string("PeriodicBoundary needs n") + axis_name + " >= ng" + axis_name +
        ": a narrower real extent would make the opposite-boundary source index read a "
        "ghost cell instead of a real one (got n" + axis_name + "=" + std::to_string(extent) +
        ", ng" + axis_name + "=" + std::to_string(ghost_depth) + ")");
  }
}

}  // namespace detail

// Wraps ghost cells from the opposite real boundary of the same axis.
//
// Requires nx (ny/nz) >= ngx (ngy/ngz) on the axis being filled: the
// "opposite real boundary" index arithmetic below reads real-cell
// indices [ngx, ngx+nx); if the real extent is *narrower* than the
// ghost depth, that arithmetic underflows into the ghost region itself
// -- one ghost cell's "source" becomes another ghost cell being written
// by this same parallel fill, at a different thread/iteration. Some
// backends/execution orders then read that other ghost cell before it
// has been written, a genuine cross-thread race (caught in code review
// with a forced-reverse-order repro at nx=1, ngx=2; see
// agent_history.md) -- not merely a "looks wrong" result. Rejected via
// an always-on runtime check (not handled with modulo-wrapped source
// indices) because no current grid configuration needs nx < ngx; lift
// this restriction with real modulo-wrapping if a future AMR block
// genuinely needs it.
struct PeriodicBoundary
{
  template <class Backend = CpuParallelFor, class Scalar, std::size_t N, class Layout>
  void fill_x(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngx == 0) return;
    detail::require_periodic_extent_covers_ghost_depth(grid.nx, grid.ngx, "x");
    const std::size_t py = grid.padded_ny();
    const std::size_t pz = grid.padded_nz();
    Backend::run(grid.ngx * py * pz, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
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

  template <class Backend = CpuParallelFor, class Scalar, std::size_t N, class Layout>
  void fill_y(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngy == 0) return;
    detail::require_periodic_extent_covers_ghost_depth(grid.ny, grid.ngy, "y");
    const std::size_t px = grid.padded_nx();
    const std::size_t pz = grid.padded_nz();
    Backend::run(grid.ngy * px * pz, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
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

  template <class Backend = CpuParallelFor, class Scalar, std::size_t N, class Layout>
  void fill_z(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngz == 0) return;
    detail::require_periodic_extent_covers_ghost_depth(grid.nz, grid.ngz, "z");
    const std::size_t px = grid.padded_nx();
    const std::size_t py = grid.padded_ny();
    Backend::run(grid.ngz * px * py, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
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

  template <class Backend = CpuParallelFor, class Layout>
  void fill_x(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngx == 0) return;
    const std::size_t py = grid.padded_ny();
    const std::size_t pz = grid.padded_nz();
    const State<Scalar, N> low = low_value;
    const State<Scalar, N> high = high_value;
    Backend::run(grid.ngx * py * pz, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
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

  template <class Backend = CpuParallelFor, class Layout>
  void fill_y(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngy == 0) return;
    const std::size_t px = grid.padded_nx();
    const std::size_t pz = grid.padded_nz();
    const State<Scalar, N> low = low_value;
    const State<Scalar, N> high = high_value;
    Backend::run(grid.ngy * px * pz, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
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

  template <class Backend = CpuParallelFor, class Layout>
  void fill_z(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngz == 0) return;
    const std::size_t px = grid.padded_nx();
    const std::size_t py = grid.padded_ny();
    const State<Scalar, N> low = low_value;
    const State<Scalar, N> high = high_value;
    Backend::run(grid.ngz * px * py, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
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

// Fixed Dirichlet ("inflow") value on the low (-axis) end, zero-order
// ("outflow") extrapolation on the high (+axis) end: the low-end ghost
// layer is held at a constant `inflow_value`, identical to
// `StaticBoundary`'s own low-end behavior, while every high-end ghost
// cell -- all `ngx`/`ngy`/`ngz` layers of it -- copies the single
// nearest real cell's value (a zero gradient at the boundary, so a wave
// exits without reflecting back into the domain).
//
// This is the first real implementation of AGENTS.md #17's named
// "extrapolation/outflow" boundary-condition category. Deliberately the
// *simple*, zero-order kind: AGENTS.md #17 also describes a
// characteristic-based, partial-specification outflow (e.g. a subsonic
// inlet where pressure is extrapolated but temperature is fixed) that
// needs an equation of state and is reserved for Phase 3's compressible
// Euler work -- this type does not attempt that, and should not be
// mistaken for it.
template <class Scalar, std::size_t N>
struct InflowOutflowBoundary
{
  State<Scalar, N> inflow_value;

  InflowOutflowBoundary() = default;
  explicit InflowOutflowBoundary(const State<Scalar, N>& inflow) : inflow_value(inflow) {}

  template <class Backend = CpuParallelFor, class Layout>
  void fill_x(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngx == 0) return;
    const std::size_t py = grid.padded_ny();
    const std::size_t pz = grid.padded_nz();
    const State<Scalar, N> inflow = inflow_value;
    const std::size_t nearest_real = grid.ngx + grid.nx - 1;
    Backend::run(grid.ngx * py * pz, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
      const std::size_t g = idx % grid.ngx;
      const std::size_t rem = idx / grid.ngx;
      const std::size_t j = rem % py;
      const std::size_t k = rem / py;

      const std::size_t low_ghost = grid.flat_index(grid.ngx - 1 - g, j, k);
      const std::size_t high_ghost = grid.flat_index(grid.ngx + grid.nx + g, j, k);
      const std::size_t high_source = grid.flat_index(nearest_real, j, k);

      for (std::size_t c = 0; c < N; ++c) {
        field(low_ghost, c) = inflow[c];
        field(high_ghost, c) = field(high_source, c);
      }
    });
  }

  template <class Backend = CpuParallelFor, class Layout>
  void fill_y(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngy == 0) return;
    const std::size_t px = grid.padded_nx();
    const std::size_t pz = grid.padded_nz();
    const State<Scalar, N> inflow = inflow_value;
    const std::size_t nearest_real = grid.ngy + grid.ny - 1;
    Backend::run(grid.ngy * px * pz, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
      const std::size_t g = idx % grid.ngy;
      const std::size_t rem = idx / grid.ngy;
      const std::size_t i = rem % px;
      const std::size_t k = rem / px;

      const std::size_t low_ghost = grid.flat_index(i, grid.ngy - 1 - g, k);
      const std::size_t high_ghost = grid.flat_index(i, grid.ngy + grid.ny + g, k);
      const std::size_t high_source = grid.flat_index(i, nearest_real, k);

      for (std::size_t c = 0; c < N; ++c) {
        field(low_ghost, c) = inflow[c];
        field(high_ghost, c) = field(high_source, c);
      }
    });
  }

  template <class Backend = CpuParallelFor, class Layout>
  void fill_z(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngz == 0) return;
    const std::size_t px = grid.padded_nx();
    const std::size_t py = grid.padded_ny();
    const State<Scalar, N> inflow = inflow_value;
    const std::size_t nearest_real = grid.ngz + grid.nz - 1;
    Backend::run(grid.ngz * px * py, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
      const std::size_t g = idx % grid.ngz;
      const std::size_t rem = idx / grid.ngz;
      const std::size_t i = rem % px;
      const std::size_t j = rem / px;

      const std::size_t low_ghost = grid.flat_index(i, j, grid.ngz - 1 - g);
      const std::size_t high_ghost = grid.flat_index(i, j, grid.ngz + grid.nz + g);
      const std::size_t high_source = grid.flat_index(i, j, nearest_real);

      for (std::size_t c = 0; c < N; ++c) {
        field(low_ghost, c) = inflow[c];
        field(high_ghost, c) = field(high_source, c);
      }
    });
  }
};

}  // namespace cfe
