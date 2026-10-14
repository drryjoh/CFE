# Task 0010: Phase 4 (pulled forward) — Forward-Facing-Step Multi-Block Benchmark

**Status: done** (2026-10-12) -- see `docs/adr/0011-forward-step-block-interface.md`
for the implemented mechanism's full reasoning and `agent_history.md`'s
"Task 0010" entry (same date) for the complete completion report,
review trail, and known limitations/next steps. Originally split out of
the user's original request for task 0009 (see that task file and
`docs/adr/0010-euler-field-and-shock-capturing.md`'s Decision 4) on
Amelia's explicit recommendation, to avoid bundling a brand-new
architectural mechanism with brand-new physics in one slice. This file's
own scope/design-direction record below is unchanged from how it was
originally written and should still be read as-is for that history.

Read and follow `AGENTS.md`, `ARCHITECTURE.md`, `ROADMAP.md` (Phase 4),
`agent_history.md`, and `docs/adr/0010-euler-field-and-shock-capturing.md`
before starting.

## Objective

Woodward & Colella (1984), "The numerical simulation of two-dimensional
fluid flow with strong shocks," JCP 54, 115-173, Sec. IV.b: the Mach 3
forward-facing-step wind tunnel. Domain: 1 unit wide x 3 units long
tunnel; step 0.2 units high, 0.6 units from the inlet. Inlet/IC:
gamma=1.4, density=1.4, pressure=1.0, velocity=3.0 (M=3) everywhere,
continually fed from the left. Left: supersonic inflow. Right:
supersonic outflow. Tunnel walls + step faces: reflecting slip wall
(`cfe::SlipWallBoundary<EulerField<Scalar,Dim>>`, already implemented,
task 0009).

Known artifact to document, not fix: the step's corner is a singular
point (rarefaction-fan center); a naive corner treatment produces a
visible spurious boundary layer/extra Mach reflection off the step --
replicating the paper's exact corner fix is not required, just document
what is observed.

Qualitative checks (not a convergence table): bow shock off the step
corner, Mach-stem reflection off the top wall growing in length/
position over time, trailing contact discontinuity (can show mild KH
waviness at finer resolution), unsteady up to t~4 before settling.
Reference grids from the paper: 80x240 ("essentially converged" for
PPM), 20x60 and 40x120 for convergence illustration -- pick comparable
resolutions for MUSCL+Rusanov/HLLC (PPM itself is not required).

**Primary run: 2D.** Additionally run the full case in 3D (simple
extrusion, symmetry/periodic on the new faces) on the CPU backend (this
machine is Apple Silicon, no CUDA) as a "see it work" deliverable --
this is NOT a substitute for this project's standing separate
requirement that every PR adding a Field/scheme include a 3D CUDA
correctness test + at-scale benchmark on real CUDA hardware (e.g. PSC
Bridges-2, as prior phases did); keep these two 3D requirements
distinct in the completion report.

## Design direction already agreed (Amelia, consulted before task 0009's
implementation -- do not re-derive, use this)

**Multi-block geometry: two `CartesianGrid` blocks, not a single masked
grid.** One block for `x in [0, 0.6)` at full tunnel height (1.0), one
for `x in [0.6, 3.0)` at the post-step height (0.8), each with its own
`FieldView` storage and its own `FvmSolver` instance. A new, same-
process cross-block `BoundaryCondition` (analogous to
`grid/boundary/mpi_halo_boundary.hpp`'s cross-partition read, but
without MPI -- same process, same rank, just reading the other block's
storage directly) fills block 1's `+x` ghost layer from block 2's real
cells where the two blocks share height (`y >= 0.2`), and from a local
slip-wall reflection where they don't (`y < 0.2`, the step's riser
face) -- i.e., the existing per-cell-branching-inside-a-`fill_x`-kernel
pattern already used elsewhere in this codebase (e.g.
`PeriodicBoundary::fill_x` already computes per-cell `(j,k)` inside its
kernel body), not a change to `grid/ghost/ghost_fill.hpp`'s dispatch
shape. Block 2's `-x` face matches block 1's upper portion 1:1 (no
per-cell branching needed there). Block 2's `-y` face (step top) and
every tunnel-wall face on both blocks are ordinary
`SlipWallBoundary<EulerField<Scalar,Dim>>`, already implemented.

Rejected: masking step cells inside one single full-domain grid --
pushes geometry awareness into `FvmSolver::residual()`'s hot loop
(violates "Solver must not contain physics/geometry it shouldn't,"
ARCHITECTURE.md #2/#14), wastes storage/compute on dead cells, and is
not what `CartesianGrid`'s own design comment anticipated ("block-based
static refinement... without redesigning this type").

**Required test:** flux computed from each side of the shared block
interface must agree (a direct conservation check at the new
cross-block boundary, not just "the simulation ran and looked
reasonable").

## Required functionality (not yet built)

- A two-`CartesianGrid` L-shaped domain assembly for the forward step.
- A new cross-block `BoundaryCondition` type (name/location TBD by
  whoever picks this up; natural home is
  `src/cfe/grid/boundary/block_interface_boundary.hpp` or similar).
- Its own ADR (the cross-block mechanism is itself a "major
  architectural decision" per AGENTS.md #6) -- do not fold it into ADR
  0010, which is already scoped to task 0009's single-block work.
- 2D run at the paper's reference resolutions (or comparable), with the
  qualitative checks above evaluated and documented (figures/video,
  plus honest description of the corner-artifact behavior).
- 3D CPU extrusion run (this machine), documented as the "see it work"
  deliverable distinct from the standing CUDA 3D-correctness/at-scale
  requirement (which stays deferred to real CUDA hardware, as in prior
  phases, and should be called out as still outstanding in the
  completion report, not silently dropped).
- Violet/Snorlax/Sally/Enzo review of the new cross-block mechanism and
  the benchmark code, then Ilia final review + Amelia alignment check,
  per CLAUDE.md's review-effort-proportional-to-change rule (this is a
  new grid/memory-layout mechanism -- all four apply).

## Architecture decisions

A new ADR for the cross-block `BoundaryCondition` mechanism (see
"Required functionality" above) -- not yet written.

## Completion report

At the end report:

1. files added/changed;
2. the new ADR's reasoning;
3. 2D qualitative-check evidence (bow shock, Mach stem growth, contact/
   KH waviness, corner-artifact description) at the chosen resolutions;
4. the cross-block interface conservation check's result;
5. 3D CPU run evidence, explicitly distinguished from the still-
   outstanding CUDA 3D-correctness/at-scale requirement;
6. known limitations and recommended next task.
