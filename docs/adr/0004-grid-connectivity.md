# ADR 0004: Grid connectivity architecture

**Status:** Accepted for the Cartesian/FVM scope; unstructured connectivity remains future work
**Date:** 2026-08-30 (evidence added 2026-09-29, Phase 1)

## Context

Initial development uses Cartesian box grids, but CMU-CFE must later support unstructured hybrid grids on GPUs.

A purely `(i,j,k)`-centric API could make later unstructured support difficult.

Conversely, implementing general unstructured connectivity immediately would slow initial development.

## Proposed approach

Start with specialized Cartesian indexing and neighbor access.

Keep solver/numerics interfaces from assuming that connectivity must always be analytically reconstructed.

Future unstructured grids should use compact contiguous adjacency structures suitable for GPU execution, likely CSR-like or equivalent.

## Evidence (Phase 1, `tasks/0002-phase1-cartesian-grid-scalar-transport.md`)

Implemented as `CartesianGrid<Scalar>` (`src/cfe/grid/structured/cartesian_grid.hpp`):

- **`(i,j,k)` -> flat-cell-index conversion lives entirely in the grid,
  not in `Field`/`FieldView`.** `flat_index(i,j,k)` is the *only* place
  this conversion happens; `Field`/`FieldView` themselves needed zero
  changes to support a grid at all — they already indexed purely by a
  flat `cell` integer with no notion of dimensionality. This is direct
  evidence for the "solver/numerics interfaces must not assume
  connectivity is always `(i,j,k)`-reconstructible" constraint above:
  everything above the grid layer (ghost-fill, reconstruction, numerical
  flux, the solver's residual assembly) only ever calls `flat_index`,
  never computes an offset itself, so a future connectivity scheme only
  has to change what sits *behind* that one call, not any of its callers.
- **Ghost cells live in the same contiguous `Field` allocation as real
  cells**, sized `(nx + 2*ngx) x (ny + 2*ngy) x (nz + 2*ngz)`
  (`padded_nx/ny/nz()`), with `flat_index` mapping padded indices to the
  flat cell. No separate ghost-cell storage, no parallel scalar-only
  path — the ghost-cell/BC layer is generic over `NComponents` and
  `Layout` by construction, since it operates on the same `FieldView`
  physics code does.
- **Grid spacing (`dx`/`dy`/`dz`) and extents are scoped to one
  `CartesianGrid` instance** (one block), not a global constant — Phase 1
  only ever instantiates one, but this is exactly what the AMR-readiness
  constraint (task spec, PI direction 2026-09-10) required: multiple
  blocks at different resolutions can coexist as multiple `CartesianGrid`
  instances without redesigning this type.
- **Ghost-cell filling is a swappable interface**
  (`PeriodicBoundary`/`StaticBoundary<Scalar,N>`, dispatched through
  `fill_ghost_cells` in `src/cfe/grid/ghost/ghost_fill.hpp`), not
  hardwired same-array `(i,j,k) +/- 1` indexing baked into the solver.
  This is the concrete "neighbor provider" seam the AMR-readiness
  constraint asked for: a future coarse-fine boundary condition (needed
  once blocks can differ in resolution) or an MPI halo-exchange boundary
  condition both implement the exact same `fill_x/fill_y/fill_z` shape
  `PeriodicBoundary`/`StaticBoundary` already do, with no change to
  `FvmSolver` or `fill_ghost_cells` itself.
- **Correctness evidence**: `test_grid_indexing.cpp` (1D/2D/3D flat-index
  correctness, ghost cells resolve to the correct neighbor, no gaps or
  overlap) and `test_boundary_conditions.cpp` (periodic/static ghost
  values correct at domain edges) — both pass, all backends (CPU serial,
  CPU threaded, CUDA — see ADR 0001/0002 and
  `docs/performance/0003-phase1-scalar-advection-cuda-results.md`).

## Decision

**Accepted** for the Cartesian, block-structured, FVM scope described
above — this is no longer an unevidenced placeholder. The general
unstructured-connectivity question remains open and deliberately
unimplemented, per the original proposed approach: do not implement
general unstructured connectivity during the scalar-transport phase.

## Revisit criteria

Revisit when the first unstructured DG prototype begins.
