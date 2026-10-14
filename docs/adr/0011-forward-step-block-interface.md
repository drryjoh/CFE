# ADR 0011: Cross-block interface boundary condition for the forward-facing-step multi-block case

**Status:** Accepted
**Date:** 2026-10-12

## Context

Task 0010 (`tasks/0010-phase4-forward-step-multiblock.md`) implements the
Woodward & Colella (1984), JCP 54, 115-173, Sec. IV.b, Mach 3
forward-facing-step wind tunnel: a 1-unit-wide, 3-unit-long tunnel with a
0.2-unit step 0.6 units from the inlet. This is an L-shaped domain, CFE's
first case that is not a single rectangular `CartesianGrid`.

This is a different kind of decision from ADR 0010 (`EulerField`,
MUSCL/Rusanov, supersonic/slip-wall BCs, the Sod tube), which is scoped to
single-block Euler physics. ADR 0010's own Decision 4 explicitly deferred
this multi-block mechanism to its own slice/ADR, on Amelia's recommendation,
specifically so a brand-new architectural mechanism (never reviewed) would
not be bundled with brand-new physics in one change -- see that ADR for the
rationale, not repeated here.

The design direction below was agreed with Amelia **before** this task's
implementation (per CLAUDE.md's "consult before a consequential design
decision" rule) and recorded in `tasks/0010-...md` so it would not need to
be re-derived; this ADR records the reasoning that decision rests on, the
rejected alternative, and the mechanism's actual implementation and
verification evidence.

## Decision 1: two `CartesianGrid` blocks, not one masked grid

### Options considered

**Option A (selected): two independent `CartesianGrid` blocks, each with
its own `FieldView` storage and its own `FvmSolver` instance** -- one block
("upstream") spanning `x in [0, 0.6)` at the full tunnel height (1.0), one
("downstream") spanning `x in [0.6, 3.0)` at the post-step height (0.8).
Joined by a new, same-process `BoundaryCondition`
(`grid/boundary/block_interface_boundary.hpp`, `BlockInterfaceBoundary`)
that reads the other block's real cells directly wherever the two blocks
share height, and falls back to a local slip-wall-style reflection where
they don't (the step's riser face).

**Option B (rejected): a single full-domain `CartesianGrid` with the step's
cells masked out** (e.g. a per-cell "is this cell real" flag checked inside
the residual loop, or a very large negative/`NaN` sentinel state in the
masked cells that the numerics are expected to never let escape).

Disadvantages of Option B (why rejected):

- pushes geometry awareness into `FvmSolver::residual()`'s hot loop --
  every cell would need a masked/not-masked branch (or the reconstruction
  stencil would need to detect and route around masked neighbors),
  violating ARCHITECTURE.md #2 ("solver must not contain physics/geometry
  it shouldn't") and AGENTS.md #14 (solver orchestrates, does not embed
  geometry-specific logic);
- wastes storage and compute on dead cells: the step's masked region is a
  real fraction of the full bounding box (0.6 x 0.2 of a 3.0 x 1.0 domain,
  ~6.7% of cells at this problem's specific proportions, more at other step
  geometries) that Option A's two-block layout never allocates or computes
  at all;
- is not what `CartesianGrid`'s own design comment anticipated. That
  comment (`grid/structured/cartesian_grid.hpp`) explicitly scopes spacing/
  extents to one block specifically "so that fixed, block-based static
  refinement (multiple grids at different resolutions) can be added later
  without redesigning this type" -- Option A is a direct, intended use of
  that design, not a new capability bolted on; Option B works against it.

Option A costs a genuinely new mechanism (the cross-block boundary
condition, Decision 2 below) that Option B would not need -- that cost is
accepted because it is the smaller, more contained piece of new code, in
exactly the place (`BoundaryCondition`) ARCHITECTURE.md #11 already
anticipated could need Field-aware/non-trivial implementations, versus
spreading geometry logic through the solver's hot path.

## Decision 2: `BlockInterfaceBoundary` -- a same-process cross-block ghost read, not a second `fill_ghost_cells` dispatch shape

`grid/boundary/block_interface_boundary.hpp`'s `BlockInterfaceBoundary<Scalar,
Field, Side, Layout>` fills ONE block's X-axis ghost layer (both ends: the
genuine domain edge on one side, the cross-block interface on the other --
see Decision 3) by reading the neighbor block's `FieldView` directly,
wherever the two blocks share physical height, and falling back to a local
slip-wall-style mirror (reusing the exact formula
`SlipWallBoundary::fill_x` already established: mirror density/energy/
tangential-momentum, negate the one normal-momentum component) wherever
they don't -- the step's riser face, a genuine solid wall, not an
approximation of convenience.

This is deliberately the same-process analogue of
`grid/boundary/mpi_halo_boundary.hpp`'s cross-partition read: instead of
packing into a send buffer and calling `MPI_Sendrecv` with a neighbor
*rank*, it reads the neighbor *block*'s storage directly (same process,
same memory space, no copy into an intermediate buffer needed at all).
Per-cell branching (read-from-neighbor vs. local-mirror) happens **inside
the existing `fill_x` kernel body**, exactly the pattern
`PeriodicBoundary::fill_x` already uses for its own per-cell `(j,k)`
index arithmetic -- this is NOT a change to `grid/ghost/ghost_fill.hpp`'s
single-call-site dispatch shape (`fill_ghost_cells(field, grid, axis,
boundary)`); `FvmSolver`/`ghost_fill.hpp` needed zero changes to host this
type, the same way they needed none to host `SlipWallBoundary` or
`MpiHaloBoundary` before it.

### Why bit-identical neighbor reads, not interpolation

The two blocks are required (checked once per `fill_x` call, host-only,
always-on -- see `detail::require_matching_block_resolution`, same
"Release is this project's default/benchmarked build, so `assert` is the
wrong tool" reasoning `detail::require_periodic_extent_covers_ghost_depth`,
`boundary_condition.hpp`, already established) to share identical `dy`
(and, for a 3D extrusion, `dz`/`nz`/`ngz`) -- so one cell on one side of the
interface lines up with EXACTLY one cell on the other, and the ghost read
is a verbatim copy, never an interpolation. This is what makes the
required interface-conservation property exact rather than approximate:
see Decision 4/Evidence below.

## Decision 3: which end is the interface, which end is the domain edge

A two-block forward-facing-step channel is, by construction, also a
two-block *tunnel*: flow enters at the upstream block's OWN far edge and
exits at the downstream block's OWN far edge, with the interface always
sitting between them. `BlockInterfaceBoundary`'s `Side` template parameter
(`cfe::Side`, `core/types.hpp` -- reused rather than inventing a new enum)
picks which of a block's two X ends is the interface:

- `Side::High`: the interface is this block's high-x end (the "upstream"
  block); its low end is a FIXED value (the tunnel's supersonic inflow --
  the same role `StaticBoundary`'s low end already plays).
- `Side::Low`: the interface is this block's low-x end (the "downstream"
  block); its high end is zero-order EXTRAPOLATED (the tunnel's supersonic
  outflow -- the same role `InflowOutflowBoundary`'s high end already
  plays).

This pairing (interface-on-high <-> fixed-low, interface-on-low <->
extrapolated-high) is baked into the one type rather than exposed as a
fully independent 2x2 combination (fixed/extrapolated x low/high) because,
for any two-block channel assembled this way, the upstream block's own edge
is always where flow enters the whole domain and the downstream block's
own edge is always where it exits -- the other two combinations have no
use case in this problem family. A future case needing the mirror image
(e.g. a fixed exit state) would need its own small variant, not a reason to
generalize this type further ahead of that actual need (AGENTS.md #6: do
not over-generalize beyond the requested scope).

## Decision 4: the interleaved-stage time-stepping this mechanism requires

Unlike `MpiHaloBoundary` (where each MPI rank calls `ssp_rk2_step` once,
independently, and the two-sided blocking `MPI_Sendrecv` calls naturally
keep every rank's stage in lockstep with its neighbors'), a same-process,
single-threaded two-block solve has no second rank to synchronize with --
one process runs both blocks' time steps sequentially. `cfe::ssp_rk2_step`
bundles a stage's residual-then-combine together internally, which does
not fit this: block 1's STAGE-2 residual must read block 2's STAGE-1
buffer through its ghost layer, not block 2's stage-0 ("current") state.

`tests/verification/forward_step_setup.hpp`'s `ForwardStepBlocks::step()`
therefore interleaves the two blocks' stages BY HAND: compute both blocks'
stage-1 residuals (reading each other's stage-0 state, since
`BlockInterfaceBoundary::neighbor_field` already points there), combine
into both stage-1 buffers, re-point each block's interface boundary's
`neighbor_field` at the OTHER block's now-populated stage-1 buffer (a
plain field reassignment -- `FieldView` is a non-owning pointer+size, and
`BoundaryX` is a plain copyable member of each `FvmSolver`, so this is a
cheap, explicit, easy-to-read re-wiring, not a hidden side effect), then
compute both blocks' stage-2 residuals and combine.

`ssp_rk2_step`'s own signature does not allow anything to happen between
its two internal `residual()` calls, so it cannot be called directly
here -- but rather than duplicating its two internal combine formulas as
a second, local copy (an earlier draft of this file did exactly that,
flagged in Snorlax's review), `solver/time_integration/ssp_rk2.hpp` was
extended to expose those two combine steps as their own public,
`Backend`-generic, `CFE_HOST_DEVICE`-tagged functions
(`ssp_rk2_combine_stage1`/`ssp_rk2_combine_stage2`), with `ssp_rk2_step`
itself becoming a thin wrapper around them (identical behavior, no
change for any existing caller -- confirmed by the full existing test
suite still passing unchanged). `ForwardStepBlocks::step()` now calls
those two functions directly, once per block, interleaved with the
re-pointing above -- no duplicated formula, and the same CUDA-readiness
(`Backend`/`CFE_HOST_DEVICE`) as every other kernel-launching helper in
this codebase, which a second, ad hoc local copy would not have had for
free.

## Evidence

- **Ghost-fill correctness**
  (`tests/unit/test_block_interface_boundary.cpp`): both `Side::High` and
  `Side::Low` instances, checked cell-by-cell against hand-computed
  expected values, across the full shared-height band AND the riser-face
  fallback region, confirming the per-cell branch picks the right source
  in both cases (not just "some" cell).
- **Interface conservation** (the task's own required check --
  `tests/verification/test_forward_step_interface_conservation.cpp`): a
  smooth, fully-populated, non-uniform state (not a degenerate uniform-flow
  case) is filled into both blocks, ghosts filled via
  `BlockInterfaceBoundary`, and the Rusanov flux at the shared interface
  face is computed TWICE -- once from each block's own reconstruction --
  using the exact same `MusclMinmodReconstruction`/`RusanovFlux` pair
  `FvmSolver::residual()` uses internally. The two results agree to
  `1e-13` (effectively bit-exact; the loosest-possible floating-point
  reassociation noise, not a real discrepancy) at every shared row, not
  just one -- confirming the ghost read genuinely reproduces the
  neighbor's own values, not an approximation that happens to look close.
- **End-to-end sanity** (`tests/unit/test_forward_step_sanity.cpp`): the
  full interleaved-stage two-block solve, run at both `Dim=2` and `Dim=3`,
  stays finite and physical (positive density/pressure everywhere) over a
  short run -- confirms `Decision 4`'s stage interleaving is wired
  correctly end-to-end, not just that each piece type-checks.
- **3D extrusion stays exactly 2D**
  (`tutorials/euler_3d_forward_step_extrusion/`): a mid-span Z-slice of the
  3D run's final-time density field is bit-identical (max abs difference
  `0.0`) to the 2D run at the same resolution, and the solver's own
  printed max `|momentum_z|` is exactly `0.0` throughout -- confirms the
  periodic-Z extrusion introduces no spurious 3D behavior, as expected for
  a Z-uniform IC/BC set.
- **Qualitative benchmark** (`tutorials/euler_2d_forward_step/`): at the
  paper's three reference resolutions (20x60/40x120/80x240-equivalent),
  the expected bow shock off the step corner, a Mach-stem/triple-point
  reflection off the top wall that moves downstream and grows over time,
  and a trailing contact/slip line are all visible and sharpen
  recognizably with resolution -- see that tutorial's README.md for the
  figures and full qualitative writeup, including the documented (not
  fixed) step-corner artifact.
- Full CPU unit/verification suite: 123/123 passing, zero regressions to
  any pre-existing test (121 pre-existing plus this task's own 2
  additional tests added during review: the dynamic mass-balance check
  and the synthetic `Side::Low` riser-face-fallback test).

## Consequences

**Easier:** CFE now has a proven, reviewed pattern for assembling an
L-shaped (or more generally, differently-shaped-per-block) domain out of
multiple same-resolution-compatible `CartesianGrid` blocks with zero
changes to `FvmSolver`, `ghost_fill.hpp`, or any existing `Field`/
`BoundaryCondition` -- a template for a future case needing a similar
multi-block assembly.

**Harder:** a same-process multi-block solve still cannot use
`ssp_rk2_step` directly per block -- any future multi-block case needs
its own interleaved-stage driver (Decision 4), not just a new
`BoundaryCondition` plugged into the existing single-block helper. This
is a real, documented limitation of the time-integration helper's API,
not of the boundary-condition mechanism itself -- though it is a smaller
limitation than it was before this task: `ssp_rk2_combine_stage1`/
`ssp_rk2_combine_stage2` (Decision 4) now give a future interleaved-stage
driver a tested, reusable, CUDA-ready building block to call instead of
re-deriving/duplicating the combine formula itself.

**Known limitations (named, not silently dropped):**

- The step corner's own singular-point (rarefaction-fan center) artifact
  is observed and documented (see the 2D tutorial's README.md), not fixed
  -- matching the task's explicit scope (replicating the paper's exact
  corner treatment was named out of scope).
- This mechanism assumes exactly two blocks sharing one axis's worth of
  overlap; a domain needing three or more blocks, or blocks joined along
  more than one axis simultaneously, is not covered and would need its own
  design work.
- CPU-only; the 3D run here is a "see it work" extrusion demonstration,
  explicitly NOT the standing separate requirement (every PR adding a
  Field/scheme needs a 3D CUDA correctness test + at-scale benchmark on
  real hardware, as task 0009 delivered for `EulerField` itself on PSC
  Bridges-2) -- that CUDA work remains outstanding for this multi-block
  mechanism specifically and is tracked in `agent_history.md`'s completion
  entry for this task, not silently dropped.

## Revisit criteria

- A future case needing three or more blocks, or blocks joined along more
  than one shared axis at once, would need to generalize this mechanism
  (or design a different one) rather than reuse it as-is.
- A future multi-block case wanting to reuse `ssp_rk2_step` directly
  (rather than a hand-interleaved driver) would motivate either a
  multi-solver-aware time-integration helper, or accepting the
  hand-interleaved pattern as the permanent convention for this class of
  problem.
- Real CUDA hardware access would let this mechanism's 3D path get the
  same CUDA-correctness/at-scale-benchmark treatment task 0009 already
  gave single-block `EulerField`.
