# Task 0009: Phase 3 — Compressible Euler, MUSCL/Rusanov, Supersonic/Slip-Wall BCs, Sod Shock Tube

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `ROADMAP.md` (Phase 3)
4. `VERIFICATION.md`
5. `agent_history.md`
6. `docs/adr/0010-euler-field-and-shock-capturing.md`

before making changes.

## Objective

User asked for the next development slice combining ROADMAP.md Phase 3
("Shocked compressible Euler") with two Phase-4 items pulled forward
(a forward-facing-step wind tunnel benchmark and its 3D extrusion),
reasoning that the benchmark only makes sense as a package. Before
implementation, Amelia (project direction) was consulted per CLAUDE.md
on whether bundling all of this into one slice is coherent -- her
recommendation (full reasoning in agent_history.md's entry for this
task) was to split it: land Euler + MUSCL + Rusanov + supersonic/slip-
wall boundary conditions + the Sod shock tube now (this task), and treat
the forward-step benchmark (which needs a brand-new, never-reviewed
multi-block grid-joining mechanism) as a separate follow-on slice.
**This task covers only the first half.** The second half is tracked at
`tasks/0010-phase4-forward-step-multiblock.md` (not yet implemented).

## Required functionality

- `cfe::EulerField<Scalar, Dim>` (`src/cfe/fields/euler/field.hpp`):
  gamma-law gas (gamma=1.4 default, runtime member), conserved-state
  Calculators (`physical_flux`, `wave_speed`, `pressure`,
  `sound_speed`, `conserved_from_primitive`), `n_components = Dim+2`.
- `FvmSolver` (`src/cfe/solver/explicit/fvm_solver.hpp`) generalized
  from a hardcoded single-component assumption to generic
  `Field::n_components`, `if constexpr (N==1)` preserving the exact
  prior behavior for every existing N=1 Field with zero change.
- `rusanov_flux`/`RusanovFlux` (`src/cfe/numerics/numerical_flux/rusanov.hpp`):
  new `State<Scalar,N>`-in/-out overload for multi-component Fields,
  alongside the unchanged scalar overload.
- MUSCL reconstruction: reused unchanged (`MusclMinmodReconstruction`
  is already scalar-in/-out, applied componentwise by the generalized
  `FvmSolver` -- no Euler-specific reconstruction code needed).
- Boundary conditions: supersonic inflow and supersonic outflow reuse
  existing, physics-agnostic types (`StaticBoundary`/
  `InflowOutflowBoundary`) -- valid because supersonic flow has every
  characteristic pointing one direction, so no Field-awareness is
  needed. `cfe::SlipWallBoundary<Field>`
  (`src/cfe/grid/boundary/slip_wall_boundary.hpp`) is new: CFE's first
  Field-aware `BoundaryCondition` (AGENTS.md #17, ARCHITECTURE.md #11),
  mirroring real-cell values but negating the momentum component normal
  to the wall via `Field::momentum_index(axis)`.
- Sod shock tube (1978): verification test
  (`tests/verification/test_euler_sod_shock_tube.cpp`) and tutorial
  (`tutorials/euler_1d_sod_shock_tube/`), both against an
  independently-implemented exact Riemann solver
  (`tests/verification/sod_shock_tube_exact.hpp`). Fixed grid, no
  moving-frame/grid-recycling machinery (explicitly out of scope --
  that is Phase 4's own separate capability).

## Explicitly not in this task

- The Woodward & Colella (1984) forward-facing-step benchmark, its
  L-shaped two-block geometry, and the new cross-block
  `BoundaryCondition` it needs -- see
  `tasks/0010-phase4-forward-step-multiblock.md`.
- HLLC / AUSM-family numerical flux (tracked open against ROADMAP.md
  Phase 3's "Rusanov and/or HLLC" acceptance text; the user's own task
  description used the same "and/or" phrasing, so Rusanov alone
  satisfies this slice).
- The characteristic-based partial-specification subsonic
  inflow/outflow BC (ARCHITECTURE.md #11's other named Field-aware BC
  shape) -- no case in this slice needs a subsonic boundary.
- Characteristic (eigenvector-projected) reconstruction -- componentwise
  reconstruction is used instead; see ADR 0010's Decision 1 for the
  justification and revisit criterion.
- Shu-Osher problem (ROADMAP.md Phase 3 also names this; not requested
  by the user for this slice).
- DG `p=1` and `p=2` prototype "where feasible" (ROADMAP.md Phase 3
  names this with soft, non-mandatory language; never requested by the
  user for this slice either -- listed here for the same completeness
  this section gives every other deferred Phase-3 roadmap item, not
  because it was close to in scope).

## Architecture constraints

- No change to `ssp_rk2_step`, `fill_ghost_cells`, `CartesianGrid`, or
  any existing N=1 Field's physics.
- `SlipWallBoundary` stays in the same duck-typed
  `fill_x/fill_y/fill_z(FieldView, grid)` shape every other
  `BoundaryCondition` uses.
- See `docs/adr/0010-euler-field-and-shock-capturing.md` for the full
  design rationale and options considered.

## Tests

- `tests/unit/test_euler_flux.cpp`: `EulerField` Calculator hand-computed
  values (conserved/primitive round trip, pressure, sound speed,
  physical flux, wave speed), vector `RusanovFlux` degenerate case and
  functor/free-function agreement.
- `tests/unit/test_slip_wall_boundary.cpp`: mirrors non-normal
  components unchanged, negates the normal momentum component, on both
  the X and Y axis (`EulerField<Scalar,2>`).
- `tests/verification/test_euler_sod_shock_tube.cpp`: exact mass/energy
  conservation (`1e-10`), momentum change matches the exact boundary-
  impulse prediction (`1e-8`), density matches the exact solution in
  the far field/rarefaction fan/star plateaus (tolerances documented
  per region, loosest nearest the Rusanov-smeared contact), L1 error
  decreases under grid refinement.
- `tests/unit/test_euler_3d_sanity.cpp` (added as a same-PR follow-up,
  see below): 3D CPU correctness, the Sod tube repeated column-for-
  column across Y/Z, matching the shared 1D reference to `1e-13`
  (tightened from an initial `1e-10` after review found the stricter
  bound holds -- and is the portable choice across this project's
  toolchains, an exact `0.0` check passes on arm64/clang but not on
  Bridges-2's x86-64/gcc).
- `tests/unit/test_euler_cuda.cu` / `test_euler_3d_cuda.cu` (added as a
  same-PR follow-up): 1D and 3D CPU-vs-GPU correctness on real V100
  hardware, `1e-9` cell-by-cell, plus a mass/total-energy conservation
  check (`1e-10`); both run long enough (t=0.5) to genuinely exercise a
  wave reflecting off `SlipWallBoundary`, corrected during review after
  the original, shorter run was found to never reach either wall.
- Full suite: 116/116 passing (CPU, `CFE_ENABLE_CUDA=OFF`); 127/127
  passing on PSC Bridges-2 V100 hardware (`CFE_ENABLE_CUDA=ON`,
  including both new CUDA tests); `compute-sanitizer --tool initcheck`:
  0 errors. Zero regressions to any pre-existing test. See
  `docs/performance/0007-phase3-euler-cuda-results.md` for full
  methodology and environment.

## Benchmarks

Per this project's standing per-PR policy (every PR adding a
Field/scheme needs a 3D CUDA correctness test and an at-scale GPU
benchmark in the same PR, not deferred -- raised during this task's own
review before it was committed): `benchmarks/euler/bench_euler_cuda.cu`
/ `bench_euler_3d_cuda.cu`, run for real on PSC Bridges-2 (V100-32,
`sm_70`). 1D up to 10^8 cells (1.648e9 cell-updates/s at 10^8); 3D cube
up to 512^3 / ~134M cells (3.512e8 cell-updates/s at 512^3, the
flagged memory-ceiling risk at that size measured and found not to
bind). Real numbers, methodology, and register/spill data in
`docs/performance/0007-phase3-euler-cuda-results.md`.

## Architecture decisions

`docs/adr/0010-euler-field-and-shock-capturing.md` -- covers the
`FvmSolver` N=1 -> N-component generalization, the componentwise-vs-
characteristic reconstruction choice, `SlipWallBoundary`'s Field-aware
shape (and why it is NOT ARCHITECTURE.md #11's characteristic-based
subsonic case), and the decision to split the forward-step benchmark
into a separate follow-on slice.

## Completion report

At the end report:

1. files added/changed;
2. Amelia's consultation outcome (multi-block design recommendation +
   the bundling-split recommendation) and why it was followed;
3. full test/build evidence (actually run, not assumed);
4. known limitations (Rusanov's contact smearing, componentwise
   reconstruction, no HLLC yet, no subsonic BC yet);
5. recommended next task (`tasks/0010-phase4-forward-step-multiblock.md`).
