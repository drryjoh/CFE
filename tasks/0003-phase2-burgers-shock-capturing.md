# Task 0003: Phase 2 (first slice) — Burgers Equation and Shock-Capturing Numerics

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `ROADMAP.md` (Phase 2)
4. `docs/adr/0002-state-memory-layout.md`
5. `docs/adr/0004-grid-connectivity.md`
6. `docs/adr/0007-interface-flux-and-time-integration.md`
7. `VERIFICATION.md`
8. `agent_history.md`

before making changes.

## Objective

Add the inviscid Burgers equation and a shock-capturing (TVD) FVM
reconstruction/flux pair, plugging both into the existing generic
`FvmSolver`/`ssp_rk2_step` machinery from Phase 1 with zero changes to
that machinery -- closing the two canonical problems `VERIFICATION.md`
already names but nothing yet implements: "Burgers smooth convergence"
and "Burgers shock formation."

This task is deliberately a narrower slice of `ROADMAP.md`'s full Phase 2
("Large-state Burgers and communication"), not all of it -- see
"Do not implement in this task" below for what is explicitly deferred.

**Do not implement in this task:**

- CUDA port of Burgers (Phase 1's own precedent: CPU correctness before
  GPU port -- a separate follow-up task once this one is merged);
- a benchmark sweep or visualization tutorial for Burgers (follow-up,
  same reasoning);
- MPI or domain decomposition, a DG storage/communication prototype,
  state sizes through 100, or a memory-layout study (the rest of
  `ROADMAP.md` Phase 2 -- each is its own follow-up task);
- compressible Euler, Navier-Stokes, Sod shock tube, or any multi-
  component system (Phase 3/5 -- `FvmSolver` hardcodes
  `FieldView<Scalar, 1, Layout>`/component `0` throughout and needs
  generalizing over `NComponents` first; see
  `fields/scalar_advection/field.hpp`'s own header comment);
- diffusive/viscous fluxes (Phase 5);
- any new boundary-condition type (reuse `StaticBoundary`/
  `PeriodicBoundary` unchanged);
- unstructured grids, AMR, or chemistry.

## Required functionality

1. `BurgersField<Scalar, Dim>`: the inviscid Burgers equation
   `dQ/dt + div(F(Q)) = 0`, `F(Q) = Q^2/2`, same per-axis
   `physical_flux(state, axis)` / `wave_speed(left, right, axis)`
   Calculator shape `ScalarAdvectionField` already established, so
   `FvmSolver`/`fvm_solver.hpp` requires zero interface changes.
2. `RusanovFlux`: a numerical flux valid for a genuinely nonlinear scalar
   flux, including at a sonic/transonic point -- `UpwindFlux`'s own
   header comment documents why it is not entropy-correct there. Same
   `NumericalFlux::operator()(left, right, axis, field)` shape as
   `UpwindFlux`; solver residual code does not change.
3. A shock-capturing (TVD, slope-limited) FVM reconstruction --
   `MusclMinmodReconstruction` -- producing face values that do not
   oscillate at a discontinuity, unlike `CentralDifferenceReconstruction`.
   Same `Reconstruction::right(...)/left(...)` shape, same 3-point
   stencil `fvm_solver.hpp` already calls it with; solver residual code
   does not change.
4. Both canonical problems `VERIFICATION.md` names:
   - **Burgers smooth convergence**: a smooth periodic initial condition,
     run strictly before the analytically-computed characteristics
     breaking time, checked against an analytic (implicit-
     characteristics) reference solution under grid refinement.
   - **Burgers shock formation**: a Riemann-type step run forward,
     checked against the exact Rankine-Hugoniot shock speed, verified to
     stay within the bounding states (no overshoot/undershoot), verified
     non-increasing total variation, and verified conservation.

## Architecture constraints

- `BurgersField`, `RusanovFlux`, and `MusclMinmodReconstruction` must be
  new, additive types substituted as template arguments into the
  existing `FvmSolver<Scalar, Layout, Field, BoundaryX, BoundaryY,
  BoundaryZ, Reconstruction, NumericalFlux, Backend>` -- if this task
  finds itself needing to change `fvm_solver.hpp`, `ssp_rk2.hpp`,
  `ghost_fill.hpp`, or `cartesian_grid.hpp`, stop and reassess; none of
  these were expected to need changes.
- `CentralDifferenceReconstruction`/`UpwindFlux` must keep serving linear
  scalar advection unchanged -- this task adds a second reconstruction/
  flux pair alongside the first, it does not replace it.
- Per AGENTS.md #18, numerical schemes must be documented with exact
  formulations and references (Rusanov/local Lax-Friedrichs; MUSCL with
  a minmod limiter) directly in the relevant header doc comments.
- Do not allocate inside parallel loops.

## Tests

At minimum:

- `RusanovFlux` matches hand-computed reference values for a shock case
  (`uL > uR`), a rarefaction case (`uL < uR`), and a degenerate
  equal-state case;
- `MusclMinmodReconstruction` matches hand-computed reference values for
  monotone data, a local extremum (verifying the TVD clip fires), and
  genuinely linear data (verifying the limiter is non-dissipative there);
- **Burgers smooth convergence**, checked against an analytic
  (implicit-characteristics) reference solution, not a numerical proxy;
- **Burgers shock formation**: exact Rankine-Hugoniot shock speed, no
  overshoot/undershoot, non-increasing total variation, and conservation
  -- "it ran and looked reasonable" does not satisfy this, matching
  Phase 1's own convergence-test precedent;
- CPU serial and threaded backends agree (same pattern as Phase 1's
  existing scalar-advection tests).

## Benchmarks

Not required this task -- explicitly deferred (see "Do not implement in
this task"). Document that fact in the completion report rather than
silently omitting it.

## Architecture decisions

- Open a new ADR (`docs/adr/0008-...md`) recording the scheme choice:
  minmod-limited MUSCL and Rusanov as this task's first shock-capturing
  reconstruction/flux pair, versus named alternatives (superbee/van Leer/
  MC limiters, WENO; HLLC/AUSM/exact Godunov) left for later phases.

## Completion report

At the end report:

1. files added/changed;
2. tests performed;
3. convergence study results (observed order vs. expected order, and the
   documented reason for any loosened acceptance band);
4. shock-formation results (measured shock speed vs. Rankine-Hugoniot,
   overshoot/TVD/conservation checks);
5. benchmark status (deferred, not run this task);
6. unresolved design questions;
7. ADR changes;
8. recommended next task.

Append the same work to `agent_history.md`.
