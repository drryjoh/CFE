# Task 0005: Phase 2 — Full 3D MPI Block Decomposition + Strong-Scaling Tutorial

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `ROADMAP.md` (Phase 2)
4. `docs/adr/0004-grid-connectivity.md`
5. `docs/adr/0009-mpi-domain-decomposition.md` (including its
   2026-10-11 amendment -- this task IS that amendment's described work)
6. `VERIFICATION.md`
7. `agent_history.md`

before making changes.

## Objective

Generalize task 0004's 1D-slab-along-X MPI decomposition prototype to
full 3D block decomposition, and build the full-solver strong-scaling
tutorial task 0004 deferred (it only shipped a halo-exchange-only
benchmark, no tutorial exercising MPI at all).

**Key design simplification, discovered and verified while planning
this task**: this solver's reconstruction is strictly axis-split (see
`fvm_solver.hpp`'s `residual()`/`detail::axis_flux_difference`) -- no
code path ever reads a ghost cell that is simultaneously a ghost on two
axes at once (a "corner"). Full 3D block decomposition therefore needs
only face-neighbor exchange (6 directions), generalizing the existing
per-axis halo exchange to all three axes independently -- NOT the
diagonal/corner communication originally assumed necessary (see ADR
0009's amendment for the full reasoning).

**Do not implement in this task:**

- non-blocking (`Isend`/`Irecv`) communication/computation overlap --
  still "eventually," per AGENTS.md #16, unchanged from task 0004;
- `BurgersField`/any state-dependent-CFL field decomposition -- still
  needs an `MPI_Allreduce(MAX)` not yet built, unchanged from task 0004;
- GPU-aware MPI -- `MpiHaloBoundary` stays CPU-only;
- a non-periodic domain's true physical-boundary ranks getting a
  composed real value from `MpiHaloBoundary` alone -- unchanged
  limitation from task 0004, now applying independently per axis;
- auto-factoring a rank count into a process-grid shape inside the core
  library (`CartesianPartition` takes `px,py,pz` as explicit caller
  parameters) -- the tutorial's own small local factorization helper is
  tutorial-specific, not a new core type;
- a DG storage/communication prototype, state sizes through 100, or a
  memory-layout study (the rest of `ROADMAP.md` Phase 2 -- unchanged,
  each its own follow-up task);
- diagonal/corner-neighbor exchange -- explicitly a **documented
  non-requirement** for this solver's axis-split stencil shape, not an
  oversight (see Objective above and ADR 0009's amendment).

## Required functionality

- `src/cfe/grid/partition/cartesian_partition.hpp`: `CartesianPartition`
  + `make_cartesian_partition(global_nx, global_ny, global_nz, px, py,
  pz, rank, size, periodic_x, periodic_y, periodic_z)` -- reuses
  `make_slab_partition` once per axis for local extent/offset, computes
  6 face-neighbor ranks via a 3D rank-coordinate unravel/flatten.
- `src/cfe/grid/boundary/mpi_halo_boundary.hpp` (refactor): constructor
  takes `(MPI_Comm, int left_rank, int right_rank)` directly (axis-
  agnostic) instead of a whole `SlabPartition`; `fill_x`/`fill_y`/
  `fill_z` are all real implementations sharing one `if
  constexpr`-dispatched private `exchange<Axis>` helper.
- `tutorials/mpi_scalar_advection_3d_strong_scaling/`: full-solver
  strong-scaling demonstration (fixed 128^3 global problem, 1/2/4/8
  ranks), each rank writing its own local VTK block for visualization.

## Architecture constraints

- Zero changes to `FvmSolver`, `fill_ghost_cells`, `CartesianGrid`, or
  `ssp_rk2_step` -- same seam task 0004 already proved out, now used on
  all three axes at once via explicit aggregate construction
  (`solver{grid, field, boundary_x, boundary_y, boundary_z}`).
- Existing `tests/mpi/test_mpi_halo_exchange.cpp` (X-only, np=2/np=4)
  must still pass unmodified in behavior after the `MpiHaloBoundary`
  constructor refactor -- a pure regression check, not a new
  requirement.

## Tests

- `tests/unit/test_cartesian_partition.cpp` -- pure index arithmetic,
  no MPI toolchain needed (3D topology at several `px,py,pz` shapes
  including non-cubic ones, periodic/non-periodic edges, single-rank
  case).
- `tests/mpi/test_mpi_halo_exchange_3d.cpp` -- fixed 2x2x2 (np=8),
  decomposes `ScalarAdvectionField<Scalar,3>`, checks bit-identical
  (tolerance 0) against an independently-computed single-process
  reference, same oracle shape as the existing 1D test. Verified to
  have teeth via a deliberate Y-axis sabotage-then-revert check.

## Benchmarks

None new -- task 0004's `bench_mpi_halo_exchange` already parameterizes
the Y/Z cross-section size freely; a true 3-axis-decomposed
communication-only benchmark sweep is explicit future work, not blocking
this task's tutorial deliverable.

## Architecture decisions

Amendment to `docs/adr/0009-mpi-domain-decomposition.md` (not a new ADR
number -- same decision, extended with new evidence).

## Completion report

At the end report:

1. files added/changed;
2. test results (CPU-only build unaffected; existing 1D X-only test
   re-verified passing after the constructor refactor; new 3D topology
   unit tests; new 3D MPI correctness test, including the
   sabotage-then-revert oracle check) -- local (Homebrew OpenMPI) and
   PSC Bridges-2, explicitly distinguishing which is which;
3. the strong-scaling tutorial's actual numbers (local laptop and
   Bridges-2), with an honest interpretation of where scaling falls off
   and why (communication-to-computation ratio growing with rank count,
   not a bug);
4. the corner-cell-not-needed finding and why it was verified, not
   assumed;
5. unresolved items carried forward unchanged from task 0004 (Burgers
   `MPI_Allreduce(MAX)`, non-periodic edge composition, non-blocking
   overlap) plus any new ones;
6. recommended next task.
