# ADR 0010: Compressible Euler field, generalizing FvmSolver to N components, and the first Field-aware BoundaryCondition

**Status:** Accepted
**Date:** 2026-10-12

## Context

Phase 3 (ROADMAP.md) adds the compressible Euler equations -- CFE's
first genuinely **multi-component** Field (`n_components = Dim + 2`),
unlike every Field so far (`ScalarAdvectionField`, `BurgersField`, both
`n_components = 1`). Three things this forces a decision on:

1. `FvmSolver::residual()` (`solver/explicit/fvm_solver.hpp`) and its
   helper `detail::axis_flux_difference` were written assuming a single
   scalar component (`FieldView<Scalar, 1, Layout>`, component `0`
   hardcoded throughout). Euler needs the same reconstruction -> flux ->
   flux-difference pipeline applied to a 3/4/5-component conserved-state
   vector per cell, with the flux Calculator depending on the WHOLE
   state (e.g. the mass flux `rho*u` needs `u = momentum/density`, not
   just `rho`).
2. `rusanov_flux`/`RusanovFlux` (`numerics/numerical_flux/rusanov.hpp`)
   likewise only had a `Scalar`-in/`Scalar`-out form.
3. A reflecting ("slip wall") boundary condition for Euler needs to know
   which raw conserved-component index is "momentum normal to this
   axis" to negate it correctly -- unlike every existing
   `BoundaryCondition` (`PeriodicBoundary`/`StaticBoundary`/
   `InflowOutflowBoundary`), which only ever copies or writes raw
   component values with no interpretation at all. This is exactly the
   "not every `BoundaryCondition` can stay physics-agnostic" case
   ARCHITECTURE.md #11 anticipated and reserved for Phase 3.

This decision was reached after consulting Amelia (project-direction)
and Violet (numerical correctness) before implementation, specifically
on whether the Field-aware boundary condition needed here is the same
kind ARCHITECTURE.md #11 describes for a subsonic characteristic
inflow/outflow BC (it is not -- see Decision 3 below) and on how far to
generalize `FvmSolver` (componentwise vs. characteristic
reconstruction -- see Decision 2).

## Decision 1: generalize FvmSolver over `Field::n_components`, not fork it

### Options considered

**Option A (selected): generalize the existing `FvmSolver`/
`axis_flux_difference` over a `constexpr std::size_t N =
Field::n_components`, with `if constexpr (N == 1)` branching to the
exact pre-existing scalar call sequence.**

Advantages:
- Zero behavior change, zero extra indirection, for every existing N=1
  Field -- the `if constexpr (N==1)` branch is bit-for-bit the old code
  path, just reached through a length-1 `State` on the way in/out.
- One solver type keeps serving every Field, matching ARCHITECTURE.md
  #2's "Solver ... Field(s)" decomposition (the solver should not need
  to know how many components a Field has beyond this one compile-time
  number).
- `ssp_rk2_step`, `active_cell_index_map()`, ghost-cell filling
  (`fill_ghost_cells`) already operate on `FieldView<Scalar, N, ...>`
  generically -- they needed zero changes.

Disadvantages:
- `axis_flux_difference`'s body is now two branches instead of one;
  slightly more to read in one place (mitigated: the branches are
  clearly separated and each is a direct, undiluted copy of a single
  prior behavior).

**Option B: a separate `VectorFvmSolver` type for multi-component
Fields, left entirely independent of `FvmSolver`.**

Disadvantages (why rejected): duplicates the entire ghost-fill /
reconstruction-loop / residual-assembly orchestration Phase 1/2 already
wrote and tested, for a difference that is really just "how many
components does the per-cell state have" -- violates AGENTS.md #14 ("do
not duplicate physics across numerical schemes") in spirit, and would
require every future BoundaryCondition/Reconstruction/NumericalFlux to
be written or wrapped twice.

**Option C: force every Field's Calculators onto `State<Scalar, N>`
uniformly -- rewrite `ScalarAdvectionField`/`BurgersField` to take/
return a length-1 `State` always, instead of a bare `Scalar`, so
`axis_flux_difference` never needs an `if constexpr` branch at all.**

Disadvantages (why rejected): this is a real simplification of
`fvm_solver.hpp` internally, but it changes EVERY existing Field's
public Calculator signature (`physical_flux(Scalar, Axis) -> Scalar`
becomes `physical_flux(State<Scalar,1>, Axis) -> State<Scalar,1>`) for
no benefit to those Fields themselves -- their physics is genuinely
scalar, and forcing a length-1 container through every call site (every
existing test's hand-computed values, every existing tutorial) is
churn across unrelated, already-shipped, already-verified code
specifically to avoid one `if constexpr` branch in one function. This
would also be a silent interface-scope expansion beyond what this task
asked for (AGENTS.md #6: "implement only the requested scope... do not
redesign unrelated interfaces"). Rejected in favor of Option A's
narrower, purely-additive branch.

### Reconstruction: componentwise, not characteristic

Each conserved component (density, each momentum component, energy) is
reconstructed independently by `Reconstruction::right/left`
(`MusclMinmodReconstruction`, unchanged) -- NOT projected onto local
characteristic (eigenvector) variables first. This is a real,
documented simplification: characteristic reconstruction is generally
less prone to spurious oscillation right at a shock for a genuine
system, and is the more common choice in production Euler codes.
Componentwise reconstruction is accepted for this slice because:

- it requires zero new math-layer infrastructure (no Jacobian
  eigendecomposition);
- `RusanovFlux`'s dissipation already uses a single scalar `alpha`
  (the system's largest wave speed) shared across every component,
  which is itself a system-level (not per-characteristic) simplification
  -- pairing it with characteristic reconstruction would be an
  inconsistent half-measure;
- the Sod shock tube (this slice's verification case,
  `tests/verification/test_euler_sod_shock_tube.cpp`) does not expose
  componentwise reconstruction's known weak point (spurious
  oscillation at a strong, genuinely multi-wave interaction) -- it
  passes conservation and exact-solution-agreement checks cleanly.

**Revisit criterion:** a future canonical case (e.g. Shu-Osher,
ROADMAP.md Phase 3's other named benchmark, or the forward-facing-step
benchmark's bow shock/Mach-stem interaction) that shows componentwise
reconstruction producing visible spurious oscillation at a strong
interaction would justify adding characteristic reconstruction as an
alternative -- this would need `contract`/eigenvector math-layer support
(AGENTS.md #13) not yet built.

## Decision 2: EulerField's Calculator shape

`EulerField<Scalar, Dim>::physical_flux`/`wave_speed` take/return full
`State<Scalar, n_components>` vectors (not per-component scalars),
because Euler's flux for any one component genuinely depends on the
whole state (`rho*u` needs `u = momentum/density`). `wave_speed`
returns a single non-negative `|u_axis| + c` (the larger of the two
one-sided states) -- the one scalar dissipation coefficient Rusanov
needs, matching the existing contract `BurgersField::wave_speed`
established (a non-negative magnitude, not a signed velocity like
`ScalarAdvectionField::wave_speed`; `rusanov_flux`'s own `abs()` call is
unaffected either way). `gamma` is a runtime data member (like
`ScalarAdvectionField::velocity`), not a template parameter, since
nothing here depends on it at compile time and a case should be able to
pick a different gas-law gamma without recompiling.

## Decision 3: SlipWallBoundary is the "lighter-weight" Field-aware BC, not the characteristic subsonic one

ARCHITECTURE.md #11 names two different reasons a `BoundaryCondition`
might need to stop being physics-agnostic:

(a) a characteristic-based, partial-specification subsonic
inflow/outflow (e.g. specify temperature, extrapolate pressure, split
decided by which way each characteristic points) -- needs an equation
of state AND wave-speed information, reads the adjacent interior cell's
full state through the Field's Calculator;

(b) a reflecting/slip wall -- needs only a static lookup of which
raw component index is "momentum normal to this axis" (`EulerField::
momentum_index(Axis)`, `static constexpr`, no Field *instance* needed)
plus a sign flip. No equation of state or wave-speed evaluation at all.

This task's scope (the user's own task description, not ROADMAP.md's
Phase 3 acceptance text) calls for supersonic inflow (fixed full
state -- already exactly `StaticBoundary`'s per-side behavior, since all
characteristics point into the domain for supersonic flow, no
Field-awareness needed), supersonic outflow (zero-order extrapolation --
already exactly `InflowOutflowBoundary`'s high-side behavior, since all
characteristics point out of the domain, again no Field-awareness
needed), and reflecting/slip wall. Only the slip wall needed new code
(`grid/boundary/slip_wall_boundary.hpp`, `SlipWallBoundary<Field>`) --
it is case (b) above, confirmed with Amelia/Violet before
implementation.

Caveat on "supersonic inflow/outflow needs no new code," stated
precisely (Ilia review finding): this is argued correct by a
TYPE-LEVEL check -- `StaticBoundary<Scalar,N>`/
`InflowOutflowBoundary<Scalar,N>` are already generic over any
component count `N`, so they compile and type-check against
`EulerField`'s `State<Scalar,3>` with no changes needed -- not
demonstrated by an actual `EulerField` run exercising
`InflowOutflowBoundary` specifically. This slice's own verification
case (the Sod tube) only exercises `StaticBoundary` with `EulerField`
(both ends fixed, no outflow extrapolation needed since neither wave
reaches the domain edges). `InflowOutflowBoundary`'s zero-order
extrapolation logic is already covered by Burgers' own tests
(`tests/unit/test_burgers_*`) at `N=1`, and the logic itself does not
depend on what the components MEAN (it is physics-agnostic by
construction, per its own header comment) -- so the risk this leaves
open is low, but it is still argued, not demonstrated, for Euler
specifically. The forward-facing-step follow-on
(`tasks/0010-phase4-forward-step-multiblock.md`) exercises it for
real: that benchmark's own inlet is exactly a supersonic
`InflowOutflowBoundary`-shaped condition on `EulerField`.

The characteristic-based subsonic BC (case (a)) remains
unimplemented and is tracked as still-open against Phase 3's own
roadmap acceptance text, out of scope for this slice (this slice's only
benchmark is the Sod tube, which needs no subsonic boundary; the
forward-facing step -- also supersonic-only, see Decision 4 below --
is deferred to a separate follow-on slice, not delivered here, and
would not need one either).

`SlipWallBoundary<Field>` stays in the same duck-typed
`fill_x/fill_y/fill_z(FieldView, grid)` shape every other
`BoundaryCondition` uses -- `grid/ghost/ghost_fill.hpp`'s dispatch
needed zero changes. It is templated on `Field` (not hardcoded to
`EulerField`) and only requires a `static constexpr std::size_t
momentum_index(Axis)` -- any future multi-component Field with the same
"momentum is some contiguous set of components" shape can reuse it
without modification.

## Decision 4 (scope): forward-facing-step benchmark deferred to a separate slice

The user's task bundled this Euler/MUSCL/flux/BC/Sod-tube work with the
Woodward & Colella (1984) Mach-3 forward-facing-step wind tunnel
benchmark and its L-shaped two-block geometry. Per Amelia's explicit
recommendation (consulted before implementation, per CLAUDE.md), these
are split into two slices:

- **This slice:** `EulerField`, MUSCL (reused, unchanged), Rusanov
  (reused, unchanged formula, new vector overload), supersonic
  inflow/outflow (reused existing BCs), `SlipWallBoundary` (new), Sod
  shock tube verification.
- **Follow-on (tracked, not yet implemented):** the forward-step
  benchmark, needing a new same-process cross-block ghost-fill
  `BoundaryCondition` (two `CartesianGrid` blocks, each with its own
  `FvmSolver` instance, joined along the shared interface above the
  step -- the same shape `grid/boundary/mpi_halo_boundary.hpp` already
  uses for cross-partition reads, without MPI) plus its own ADR,
  conservation-at-the-interface test, and 2D/3D runs. See
  `tasks/0010-phase4-forward-step-multiblock.md`.

Rationale: a brand-new architectural mechanism (cross-block ghost
read, never reviewed) bundled with brand-new physics (Euler/MUSCL/
flux/BCs) in one slice makes it impossible to tell, if something
misbehaves, which of the two is at fault -- exactly the situation
AGENTS.md #25's compile -> unit test -> conservation -> convergence ->
canonical-verification sequence is designed to avoid conflating.

## Evidence

See `tests/unit/test_euler_flux.cpp` (Calculator hand-computed values),
`tests/unit/test_slip_wall_boundary.cpp` (mirrors non-normal components,
negates the normal one, on both the X and Y axis), and
`tests/verification/test_euler_sod_shock_tube.cpp` (Sod, 1978, classic
parameters, gamma=1.4, compared against an independently-implemented
exact Riemann solver, `tests/verification/sod_shock_tube_exact.hpp`):

- exact mass/energy conservation (flux-form telescoping, boundary flux
  is exactly zero at both ends throughout the run since neither wave
  reaches the domain edge by `t=0.2`) to `1e-10`;
- total momentum change matches the exact boundary-impulse prediction
  `(p_left - p_right) * t_final` to `1e-8`;
- density matches the exact self-similar solution to `1e-6` in the
  undisturbed far field, `2%` inside the smooth rarefaction fan and the
  star-left plateau, `8%` on the star-right plateau (looser: nearest the
  Rusanov-smeared contact discontinuity -- a documented, expected
  property of Rusanov's dissipation at a zero-eigenvalue wave, not a
  bug; see that flux's own header comment);
- density L1 error decreases under grid refinement (100 -> 400 cells) --
  a weak signal, not a formal-order claim, since the exact solution has
  genuine discontinuities no single scheme recovers full order across.

Full test suite (115/115, CPU, `CFE_ENABLE_CUDA=OFF`) passes with these
additions, with zero regressions to any pre-existing test.

## Consequences

**Easier:** a genuinely multi-component Field now runs through the
exact same `FvmSolver`/`ssp_rk2_step`/ghost-fill machinery every
previous Field used, with no new solver type. A future Field needing
its own count of components (e.g. a reacting-flow Field with species
mass fractions) has a proven path to follow.

**Harder:** `axis_flux_difference`'s body now carries both the N=1 and
N>1 branches -- a reader needs to know which applies to the Field they
care about (mitigated by the branches being clearly separated and
individually simple).

**Future constraint:** HLLC has since been added (task 0011, ADR 0012)
-- ROADMAP.md Phase 3's "Rusanov and/or HLLC" acceptance text is now
satisfied by both, not Rusanov alone. Characteristic reconstruction
remains open per Decision 1's revisit criterion. The characteristic-based
subsonic partial-specification inflow/outflow BC (ARCHITECTURE.md #11
case (a)) remains unimplemented.

## Revisit criteria

- A canonical case showing componentwise reconstruction producing
  visible spurious oscillation at a strong multi-wave interaction.
- A case needing a genuinely subsonic boundary (motivates the
  characteristic-based partial-specification BC).
