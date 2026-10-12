# Task 0007: Phase 2 — 3D MPI Burgers Strong-Scaling Tutorial

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `ROADMAP.md` (Phase 2)
4. `docs/adr/0009-mpi-domain-decomposition.md` (including all amendments)
5. `VERIFICATION.md`
6. `agent_history.md`

before making changes.

## Objective

User asked, after task 0006 made `BurgersField` MPI-decomposition-safe:
"do you have scaling for burgers?" -- answer was no (only
`ScalarAdvectionField` had a strong-scaling tutorial). This task adds
the direct sibling of `tutorials/mpi_scalar_advection_3d_strong_scaling/`
for Burgers: both a visual demonstration (a shock forming and crossing
a rank boundary with no visible seam) and a quantitative strong-scaling
measurement, exercising task 0006's `MPI_Allreduce(MAX)` fix for real
(the bump's peak sits in only one rank's local block, so most ranks
see a meaningfully smaller local maximum than the true global one --
this is not a contrived edge case, it is what a localized feature in a
decomposed domain always looks like).

**Do not implement in this task:**

- a new correctness test -- `tests/mpi/test_mpi_burgers_steepening.cpp`
  (task 0006) already proved this numerically; this task is the visual/
  scaling complement, not a second correctness proof;
- 2D Burgers MPI decomposition or any other equation/dimensionality
  combination;
- non-blocking overlap, non-periodic domain composition, GPU-aware MPI
  -- all unchanged, out of scope, same as every prior task in this
  series.

## Required functionality

`tutorials/mpi_burgers_3d_strong_scaling/` -- same structure as the
sibling scalar-advection tutorial (`CartesianPartition` +
`MpiHaloBoundary` on all three axes, auto-factored `(px,py,pz)`,
per-rank VTK output, `MPI_Barrier`-synchronized wall-clock timing,
`plot_results.py` with the same expected-scaling reference figure),
with `BurgersField<Scalar,3>` + `MusclMinmodReconstruction` +
`RusanovFlux` in place of `ScalarAdvectionField<Scalar,3>`, the same
Gaussian-bump IC and constants `tutorials/burgers_3d_visualization/`
already uses, and `cfe::backend::mpi::allreduce_max` (task 0006)
synchronizing the CFL-driving `max|u|` before computing `dt`.

## Architecture constraints

Zero changes to any production code -- this task is purely an
application of already-built, already-verified machinery
(`CartesianPartition`, `MpiHaloBoundary`, `allreduce_max`) to a new
tutorial.

## Tests

None new (see "Do not implement" above). Verification is via actually
running the tutorial and checking its output, per this project's
standing "look at the rendered output for user-facing work" convention
(`agent_history.md`'s 2026-10-02 entry): confirmed TVD boundedness
(`state` never leaves `[1.0, 1.49921]`, the initial condition's own
bounds) across every committed frame and every rank, not just spot-
checked, and confirmed per-rank VTK tiling numerically (origin/
dimensions of all 8 pieces in a 2x2x2 run exactly cover the unit cube).

## Benchmarks

None new -- this tutorial itself is the scaling measurement; no
separate `benchmarks/` addition.

## Architecture decisions

None new -- no design decisions, purely an application of tasks
0004/0005/0006's already-decided machinery.

## Completion report

At the end report:

1. files added/changed;
2. the strong-scaling numbers (local laptop and Bridges-2), with the
   same honest local-falloff-vs-cluster-near-ideal interpretation the
   sibling tutorial's README already established;
3. confirmation that task 0006's fix is genuinely exercised here (not
   just invoked for show) -- which ranks see a smaller local maximum
   and by how much;
4. the TVD boundedness and VTK-tiling verification results;
5. recommended next task.
