// Shared ghost-fill iteration skeleton for axis-aligned, single-grid
// `BoundaryCondition` types (PR #6 cleanup task -- see
// tasks/0014-pr6-cleanup.md and agent_history.md's entry for this task).
// `PeriodicBoundary`/`StaticBoundary`/`InflowOutflowBoundary`
// (boundary_condition.hpp) and `SlipWallBoundary` (slip_wall_boundary.hpp)
// each independently hand-wrote the same "decode a flat index into
// (g, o1, o2), look up the other two axes' padded extents, dispatch one
// `Backend::run` call" skeleton three times each (once per axis) --
// twelve near-identical copies differing only in which `CartesianGrid`
// members feed the decode. What actually differs between these types
// (how many source indices each computes, what it does with them per
// component: verbatim copy, wrap-around, fixed value, mirror-with-
// sign-flip, ...) stays in each type's own file; `cartesian_fill<Axis>`
// factors out only the decode/dispatch, handing everything else to a
// caller-supplied callback.
//
// `(g, o1, o2)`'s meaning -- which of `(i, j, k)` each of `o1`/`o2`
// actually is -- matches `MpiHaloBoundary::exchange<Axis>`'s own
// existing convention exactly (X: (o1,o2)=(j,k); Y: (i,k); Z: (i,j)) --
// that type is this file's in-house precedent for unifying an
// axis-dispatched kernel via `if constexpr`. This header does not touch
// `MpiHaloBoundary` itself (bulk pack/`MPI_Sendrecv`/unpack -- a
// different problem already unified on its own terms, out of scope
// here), and deliberately does NOT also unify the per-component
// transform across types (AGENTS.md #6; ADR 0010/0011's "duck-typed
// `BoundaryCondition`, no common base class" decision is unaffected) --
// see `block_interface_boundary.hpp`, which reuses only this decode/
// dispatch skeleton and keeps its neighbor-validity branching local.
#pragma once

#include <cstddef>

#include "cfe/backend/parallel_for.hpp"
#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"

namespace cfe {

// Runs `transform(g, o1, o2)` once for every ghost-layer coordinate on
// axis `A` of `grid` -- `g` in `[0, ghost_depth)` (0 = nearest the real
// domain), `o1`/`o2` the padded indices of the other two axes -- through
// `Backend::run`. A no-op when axis `A` has no ghost layer (same early
// return every existing `fill_x/y/z` already had).
//
// `transform` must be host/device-callable (ordinarily a `[=] CFE_HOST_DEVICE`
// lambda, `mutable` if it needs to call a non-`const` operation such as
// `FieldView::operator()`) taking `(Index g, Index o1, Index o2)`.
// Captured by value into this function's own dispatch lambda, so it is
// invoked from inside a `CFE_HOST_DEVICE` context -- no new host/device
// boundary is introduced.
template <Axis A, class Backend = CpuParallelFor, class Scalar, class Transform>
void cartesian_fill(const CartesianGrid<Scalar>& grid, Transform transform)
{
  Index ghost_depth, p_o1, p_o2;
  if constexpr (A == Axis::X) {
    ghost_depth = grid.ngx;
    p_o1 = grid.padded_ny();
    p_o2 = grid.padded_nz();
  } else if constexpr (A == Axis::Y) {
    ghost_depth = grid.ngy;
    p_o1 = grid.padded_nx();
    p_o2 = grid.padded_nz();
  } else {
    ghost_depth = grid.ngz;
    p_o1 = grid.padded_nx();
    p_o2 = grid.padded_ny();
  }

  if (ghost_depth == 0) return;

  Backend::run(ghost_depth * p_o1 * p_o2, [=] CFE_HOST_DEVICE(Index idx) mutable {
    const Index g = idx % ghost_depth;
    const Index rem = idx / ghost_depth;
    const Index o1 = rem % p_o1;
    const Index o2 = rem / p_o1;
    transform(g, o1, o2);
  });
}

}  // namespace cfe
