// Same-process cross-block ghost-cell boundary condition (task spec
// tasks/0010-phase4-forward-step-multiblock.md, design direction agreed
// with Amelia before implementation -- see that file and
// docs/adr/0011-forward-step-block-interface.md for the full reasoning,
// not repeated here).
//
// This is the mechanism that lets an L-shaped domain (the Woodward &
// Colella forward-facing-step tunnel: a full-height block upstream of
// the step joined to a shorter, post-step-height block downstream of
// it) be assembled from TWO ordinary `CartesianGrid` blocks, each with
// its own storage and its own `FvmSolver`, instead of one single grid
// with step cells masked out (rejected -- see the ADR). It is the
// same-process analogue of `grid/boundary/mpi_halo_boundary.hpp`'s
// cross-partition read: instead of packing into an MPI send buffer and
// calling `MPI_Sendrecv`, this just reads the neighbor block's
// `FieldView` directly, since both blocks live in the same process.
//
// One instance fills ONE axis-end of ONE block's ghost layer, where
// that end is this block's "interface face" -- the X face it shares
// with the other block -- and the OPPOSITE end of that same axis is a
// genuine physical domain edge (this is a two-block *channel*: flow
// enters one physical end, crosses the interface partway through,
// exits the other physical end). `InterfaceSide` (`cfe::Side`,
// core/types.hpp) picks which end is which:
//
//   - `Side::High`: the interface is this block's HIGH-x end (it is the
//     upstream block); its LOW end is held at a fixed value
//     (`domain_edge_value`, e.g. the tunnel's supersonic inlet state --
//     the same role `StaticBoundary`'s low end already plays).
//   - `Side::Low`: the interface is this block's LOW-x end (it is the
//     downstream block); its HIGH end is zero-order-extrapolated (the
//     tunnel's supersonic outlet -- the same role
//     `InflowOutflowBoundary`'s high end already plays).
//
// This pairing (interface-on-high <-> fixed-low, interface-on-low <->
// extrapolated-high) is not an arbitrary restriction: for any two-block
// channel assembled this way, the block whose OWN edge is upstream of
// the interface is, by construction, also the block where flow enters
// the whole domain, and symmetrically for the downstream block/exit --
// so the two valid combinations above are the only ones this shape of
// problem ever needs. A future channel wanting the mirror image (e.g. a
// backward-facing step, interface-on-low paired with a FIXED exit
// state) is a different physical setup and would need its own small
// variant, not a reason to generalize this type further ahead of that
// actual need (AGENTS.md #6).
//
// Per-cell branching inside the interface end's own fill, not a second
// boundary type or a change to grid/ghost/ghost_fill.hpp's dispatch
// shape (exactly `PeriodicBoundary::fill_x`'s existing per-cell
// (j,k)-inside-the-kernel-body pattern, generalized to branch on j
// too): wherever this block and its neighbor share the same physical
// height (`j_valid_lo <= j < j_valid_hi`, in THIS block's own padded-j
// space), the ghost cell is read directly from the neighbor's real
// cell at the corresponding height. Outside that band -- the forward
// step's own riser face, where the downstream block simply does not
// exist at that height -- there is no neighbor real cell to read, so
// this falls back to a LOCAL slip-wall-style mirror instead (identical
// formula to `SlipWallBoundary::fill_x`'s own high/low branch,
// restricted to this one side): density/energy/tangential-momentum
// copy unchanged, the single momentum component normal to this axis
// (`Field::momentum_index(Axis::X)`) is negated. This is exactly the
// physical boundary condition at that face anyway (a solid, reflecting
// wall -- the step's vertical riser), so the fallback is not an
// approximation of convenience, it is the correct physical BC for that
// part of the ghost layer.
//
// Field-aware (like `SlipWallBoundary`, not like
// `PeriodicBoundary`/`StaticBoundary`/`InflowOutflowBoundary`): the
// riser-face fallback needs `Field::momentum_index(Axis::X)` to know
// which raw component to negate. Works for any Field exposing that
// static method, not Euler-specific by name.
//
// Requires the two blocks to share identical `dy`/`dz` (and, for a 3D
// extrusion, `nz`/`ngz`) so that one cell on one side of the interface
// lines up with EXACTLY one cell on the other -- no interpolation is
// performed. Checked once per `fill_x` call (host-only, always-on, not
// a per-cell cost) via `detail::require_matching_block_resolution`
// below, same "always-on invariant check, not `assert`" shape
// `detail::require_periodic_extent_covers_ghost_depth`
// (boundary_condition.hpp) already established and the same reasoning
// (Release is this project's default and benchmarked build type, so an
// `NDEBUG`-compiled-out `assert` would silently skip exactly the build
// that matters).
//
// Conservation note (see the ADR and
// tests/verification/test_forward_step_interface_conservation.cpp):
// because the neighbor read copies the neighbor's real-cell values
// VERBATIM (no interpolation, no recomputation), the reconstructed face
// values `FvmSolver`'s own stencil builds on either side of the shared
// interface from this ghost data are bit-identical to what the
// neighbor's own stencil builds from ITS real cells -- so the Rusanov
// (or any other) numerical flux evaluated at that face from either
// block's own residual computation is bit-identical too, not merely
// "close." That bit-identical agreement is the actual conservation
// guarantee this type provides, and is exactly what that test checks
// directly, not just "the simulation ran and looked reasonable."
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include "cfe/backend/parallel_for.hpp"
#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/grid/boundary/cartesian_fill.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/math/fixed_array.hpp"
#include "cfe/math/scalar_ops.hpp"

namespace cfe {

namespace detail {

// See this file's header comment. Host-only, called once per `fill_x`
// invocation (negligible cost), not per cell.
template <class Scalar>
inline void require_matching_block_resolution(const CartesianGrid<Scalar>& a, const CartesianGrid<Scalar>& b)
{
  constexpr Scalar kRelTol = Scalar(1e-9);
  auto close = [&](Scalar x, Scalar y) {
    const Scalar scale = abs_value(x) + abs_value(y) + Scalar(1);
    const Scalar diff = x - y;
    return abs_value(diff) <= kRelTol * scale;
  };
  if (!close(a.dy, b.dy)) {
    throw std::invalid_argument(
        "BlockInterfaceBoundary requires both blocks to share the same dy (got " + std::to_string(a.dy) +
        " and " + std::to_string(b.dy) + ") -- cells on either side of the interface must line up 1:1");
  }
  if (!close(a.dz, b.dz) || a.nz != b.nz || a.ngz != b.ngz) {
    throw std::invalid_argument(
        "BlockInterfaceBoundary requires both blocks to share the same dz/nz/ngz (a 3D extrusion must use "
        "identical Z resolution on both blocks) -- cells on either side of the interface must line up 1:1");
  }
  // `fill_x` reads the neighbor's g-th real cell from its own shared
  // face, for g in [0, a.ngx) -- the neighbor must therefore have at
  // least `a.ngx` real cells along X, or that read would reach past the
  // neighbor's real-cell range (into ITS OWN ghost layer, or out of
  // bounds entirely for a narrow enough neighbor). Mirrors the same
  // hazard `detail::require_periodic_extent_covers_ghost_depth`
  // (boundary_condition.hpp) already guards against for
  // `PeriodicBoundary`'s own opposite-boundary read.
  if (b.nx < a.ngx) {
    throw std::invalid_argument(
        "BlockInterfaceBoundary requires the neighbor block to have at least as many real X cells as this "
        "block's own ghost depth (got neighbor nx=" + std::to_string(b.nx) + ", ngx=" + std::to_string(a.ngx) +
        ") -- otherwise the g-th-cell-from-the-shared-face read reaches past the neighbor's real cells");
  }
}

}  // namespace detail

template <class Scalar, class Field, Side InterfaceSide, class Layout = AoSLayout>
struct BlockInterfaceBoundary
{
  static constexpr std::size_t N = Field::n_components;

  // The physical domain edge opposite the interface (see this file's
  // header comment for which end that is). Only read when
  // `InterfaceSide == Side::High` (this block's low end is then the
  // fixed inlet) -- ignored, but harmless to leave default-constructed
  // (zero), when `InterfaceSide == Side::Low`.
  State<Scalar, N> domain_edge_value;

  // The neighbor block's own storage and grid, read directly (no MPI,
  // no copy into a buffer -- same process, same rank).
  FieldView<Scalar, N, Layout> neighbor_field;
  CartesianGrid<Scalar> neighbor_grid;

  // This block's padded j maps to the neighbor's padded j via
  // `j + j_offset_to_neighbor`, evaluated only for
  // `j_valid_lo <= j < j_valid_hi` (this block's own padded-j indices
  // spanning the shared-height band). Outside that band, the riser-face
  // fallback applies instead -- see this file's header comment.
  long j_offset_to_neighbor = 0;
  std::size_t j_valid_lo = 0;
  std::size_t j_valid_hi = 0;

  template <class Backend = CpuParallelFor>
  void fill_x(FieldView<Scalar, N, Layout> field, const CartesianGrid<Scalar> grid) const
  {
    if (grid.ngx == 0) return;
    detail::require_matching_block_resolution(grid, neighbor_grid);

    constexpr std::size_t normal_component = Field::momentum_index(Axis::X);

    const State<Scalar, N> domain_edge = domain_edge_value;
    const FieldView<Scalar, N, Layout> neighbor = neighbor_field;
    const CartesianGrid<Scalar> ngrid = neighbor_grid;
    const long j_offset = j_offset_to_neighbor;
    const std::size_t j_lo = j_valid_lo;
    const std::size_t j_hi = j_valid_hi;
    // Only meaningful when InterfaceSide == Side::Low (the extrapolated
    // high/outlet end); computed unconditionally since it is cheap and
    // this lambda must be the same for both `if constexpr` branches.
    const std::size_t nearest_real_high = grid.ngx + grid.nx - 1;

    // Only the shared decode/dispatch skeleton is reused here
    // (grid/boundary/cartesian_fill.hpp, PR #6 cleanup task) -- the
    // neighbor-validity branching and cross-grid read below are genuinely
    // different control flow from PeriodicBoundary/StaticBoundary/
    // InflowOutflowBoundary/SlipWallBoundary's shared callback shape, so
    // they are NOT forced through a common transform signature; this
    // type just supplies its own (much larger) callback body.
    cartesian_fill<Axis::X, Backend>(grid, [=] CFE_HOST_DEVICE(std::size_t g, std::size_t j,
                                                                std::size_t k) mutable {
      const bool neighbor_valid = (j >= j_lo && j < j_hi);
      const std::size_t neighbor_j = static_cast<std::size_t>(static_cast<long>(j) + j_offset);

      if constexpr (InterfaceSide == Side::High) {
        // Low end: fixed domain value (e.g. the tunnel's supersonic inlet).
        const std::size_t low_ghost = grid.flat_index(grid.ngx - 1 - g, j, k);
        for (std::size_t c = 0; c < N; ++c) field(low_ghost, c) = domain_edge[c];

        // High end: the interface. Ghost layer g (0 = nearest the
        // interface) reads the neighbor's g-th real cell from ITS own
        // low-x real boundary, when valid; otherwise mirrors this
        // block's own g-th real cell from ITS high-x real boundary,
        // negating the normal (X) momentum component -- the riser-face
        // reflecting wall.
        const std::size_t high_ghost = grid.flat_index(grid.ngx + grid.nx + g, j, k);
        if (neighbor_valid) {
          const std::size_t neighbor_cell = ngrid.flat_index(ngrid.ngx + g, neighbor_j, k);
          for (std::size_t c = 0; c < N; ++c) field(high_ghost, c) = neighbor(neighbor_cell, c);
        } else {
          const std::size_t source = grid.flat_index(grid.ngx + grid.nx - 1 - g, j, k);
          for (std::size_t c = 0; c < N; ++c) {
            const Scalar sign = (c == normal_component) ? Scalar(-1) : Scalar(1);
            field(high_ghost, c) = sign * field(source, c);
          }
        }
      } else {
        // Low end: the interface. Ghost layer g (0 = nearest the
        // interface) reads the neighbor's g-th real cell from ITS own
        // high-x real boundary, when valid; otherwise mirrors this
        // block's own g-th real cell from ITS low-x real boundary,
        // negating the normal (X) momentum component.
        const std::size_t low_ghost = grid.flat_index(grid.ngx - 1 - g, j, k);
        if (neighbor_valid) {
          const std::size_t neighbor_cell = ngrid.flat_index(ngrid.ngx + ngrid.nx - 1 - g, neighbor_j, k);
          for (std::size_t c = 0; c < N; ++c) field(low_ghost, c) = neighbor(neighbor_cell, c);
        } else {
          const std::size_t source = grid.flat_index(grid.ngx + g, j, k);
          for (std::size_t c = 0; c < N; ++c) {
            const Scalar sign = (c == normal_component) ? Scalar(-1) : Scalar(1);
            field(low_ghost, c) = sign * field(source, c);
          }
        }

        // High end: zero-order extrapolation (e.g. the tunnel's
        // supersonic outlet) -- identical formula to
        // `InflowOutflowBoundary`'s own high end.
        const std::size_t high_ghost = grid.flat_index(grid.ngx + grid.nx + g, j, k);
        const std::size_t high_source = grid.flat_index(nearest_real_high, j, k);
        for (std::size_t c = 0; c < N; ++c) field(high_ghost, c) = field(high_source, c);
      }
    });
  }

  // `grid/ghost/ghost_fill.hpp`'s single `fill_ghost_cells` function
  // template is instantiated once per `Boundary` type, with all three
  // `fill_x`/`fill_y`/`fill_z` branches compiled into that one
  // instantiation regardless of which axis is actually passed at
  // runtime -- so any type used as a `BoundaryX`/`BoundaryY`/`BoundaryZ`
  // at all must provide all three, even though this type is only ever
  // plugged in as `BoundaryX` (the forward step's X-direction interface)
  // and these two are never actually reached (same documented gotcha
  // `tutorials/burgers_2d_diagonal_shock/burgers_2d.cpp`'s
  // `DiagonalShockExactBoundary::fill_z` already works around). Unlike
  // that example (a real no-op, safe because it is genuinely inert for
  // its own 2D-only use), these throw instead of silently doing
  // nothing: silently no-op'ing here would hide a real misconfiguration
  // (this type accidentally wired up as `BoundaryY`/`BoundaryZ`) behind
  // output that looks plausible but is physically wrong.
  template <class Backend = CpuParallelFor>
  [[noreturn]] void fill_y(FieldView<Scalar, N, Layout>, const CartesianGrid<Scalar>) const
  {
    throw std::logic_error(
        "BlockInterfaceBoundary is only ever intended for the X axis (the forward step's cross-block "
        "interface) -- it should never be constructed as a BoundaryY");
  }

  template <class Backend = CpuParallelFor>
  [[noreturn]] void fill_z(FieldView<Scalar, N, Layout>, const CartesianGrid<Scalar>) const
  {
    throw std::logic_error(
        "BlockInterfaceBoundary is only ever intended for the X axis (the forward step's cross-block "
        "interface) -- it should never be constructed as a BoundaryZ");
  }
};

}  // namespace cfe
