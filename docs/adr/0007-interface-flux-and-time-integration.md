# ADR 0007: Method-agnostic interface-flux abstraction and SSP-RK2 time integration

**Status:** Accepted
**Date:** 2026-09-29

## Context

Phase 1 (`tasks/0002-phase1-cartesian-grid-scalar-transport.md`) needed
two design decisions the task spec explicitly asked to be recorded,
not just implemented:

1. **Item 4**: "A method-agnostic 'interface value' contract... the
   shape of this interface must not bake in 'FVM cell-average' as the
   only thing that could ever sit on either side of it." Per AGENTS.md
   #19, DG and FVM must remain hybridizable, with hybridization logic
   living outside both — this task only implements the FVM side, but the
   *shape* of the contract must not need to change when a DG side
   eventually exists.
2. **Item 6**: "Pick SSP-RK2 or SSP-RK3 and document which and why (e.g.
   stability region vs. cost)."

## Interface-flux design

### Options considered

**Option A — a single combined "compute the flux at this face" function**,
taking both neighboring cells' full stencils directly and returning one
number. Simplest to write for FVM alone.

Disadvantages: bakes in exactly the assumption item 4 forbids — the
function's own signature assumes there is one cell average on each side
of the face, reachable by indexing a shared array. A DG element (which
represents a face value as an evaluation of internal nodal/modal
degrees of freedom, not a single stored cell-average) cannot implement
this signature without CMU-CFE inventing a fake "cell average" for it
first.

**Option B (chosen) — two independently-callable stages**: each side of
a face produces *its own* face value from *its own* local representation
(`Reconstruction::right(...)`/`Reconstruction::left(...)` in
`src/cfe/numerics/fvm/interface_value.hpp`), then a separate, tiny
combinator (`NumericalFlux`, `src/cfe/numerics/numerical_flux/upwind.hpp`)
takes only the two resulting face values (plus the `Field`'s physics
Calculators) and produces one flux.

Advantages: the *reconstruction* stage is the only place that needs to
know how a side's representation works (cell-average one-ring stencil,
for FVM's `CentralDifferenceReconstruction`; a DG element's own
nodal/modal evaluation, for a hypothetical future DG side) — the
*combinator* stage (upwind selection, or eventually a Riemann solver)
needs to know nothing about either side's internal representation, only
the two resulting numbers. This is exactly the FVM/DG hybridization
seam AGENTS.md #19 asks for: hybridization logic (the combinator) lives
outside both individual implementations (the two reconstructions).

Disadvantages: two function calls and two intermediate values per face
instead of one, a small amount of extra indirection for a scheme (plain
upwind FVM) that does not itself need the separation. Accepted as the
cost of not foreclosing DG later.

### Genericity of both stages

Both `Reconstruction` and `NumericalFlux` are **template parameters** of
`FvmSolver` (`src/cfe/solver/explicit/fvm_solver.hpp`), not hardcoded
function calls by name — matching the same pattern `Layout` and
`BoundaryX/Y/Z` already use. This was a correction caught mid-implementation
(see `agent_history.md`): hardcoding the call by name inside
`residual()` would have made "eventually swap out MUSCL/WENO/Rusanov/
HLLC reconstructions or fluxes without touching `residual()`" (explicitly
asked for by whoever reviewed the initial design) impossible despite the
two-stage split existing. The template-parameter design makes swapping
either stage a type-level change at the `FvmSolver<...>` instantiation,
not a code change inside `residual()`.

`NumericalFlux` also does not hardcode a physical-flux formula: it takes
the `Field` object itself and calls `field.physical_flux(...)`/
`field.wave_speed(...)` (`src/cfe/numerics/numerical_flux/upwind.hpp`),
so a future Burgers or Euler `Field` supplies its own physics with zero
changes to the numerical-flux combinator — this is the Field/Calculator
split described in ARCHITECTURE.md #2, and is the other half of what
keeps this abstraction genuinely physics-agnostic, not just
DG/FVM-agnostic.

## Time integration: SSP-RK2 vs. SSP-RK3

### Options considered

**Option A — SSP-RK3**: 3rd-order accurate in time, larger CFL-stable
region, standard choice for many explicit FVM codes.

**Option B (chosen) — SSP-RK2 (Heun's method / improved Euler)**:
2nd-order accurate in time, smaller stability region, one fewer stage.

### Evidence / reasoning

The spatial scheme this phase pairs the time integrator with
(`CentralDifferenceReconstruction`, a central-difference-style 1-ring
reconstruction) is itself 2nd-order accurate in space. For an explicit
method-of-lines scheme, the overall observed convergence order is
bounded by the *lower* of the spatial and temporal orders — a 3rd-order
time integrator paired with a 2nd-order spatial scheme cannot produce
better than 2nd-order overall convergence, since the spatial truncation
error dominates asymptotically. SSP-RK3's third stage would therefore
add computational cost without improving the actual quantity Phase 1's
acceptance bar measures (observed convergence order under grid
refinement), for this specific pairing.

This was verified directly, not just argued: `test_scalar_advection_second_order_convergence`
(`tests/unit/test_scalar_advection_convergence.cpp`) measures the observed
order at four grid refinements (`nx = 20, 40, 80, 160`, later confirmed
at 320 too) and finds it converging to almost exactly 2:

| `nx` | L2 error | ratio vs. previous | observed order |
|---|---|---|---|
| 20 | 3.390e-2 | — | — |
| 40 | 8.458e-3 | 4.0085 | 2.0030 |
| 80 | 2.112e-3 | 4.0056 | 2.0020 |
| 160 | 5.277e-4 | 4.0018 | 2.0007 |
| 320 | 1.319e-4 | 4.0005 | 2.0002 |

The observed order converges toward 2.0000 as resolution increases (the
small excess above 2.0 at coarser resolutions is the expected higher-order
correction term, shrinking as `dx` shrinks) — direct confirmation that
SSP-RK2 is not leaving accuracy on the table relative to what the spatial
scheme can deliver, and that a 3rd-stage time integrator would not have
changed this result.

## Decision

Adopt the two-stage (`Reconstruction` + `NumericalFlux`, both template
parameters of `FvmSolver`) interface-flux design, and SSP-RK2 as the time
integrator, for the reasons above.

## Consequences

- **Easier**: swapping reconstruction or numerical-flux schemes (MUSCL,
  WENO, Rusanov, HLLC, ...) is a type-level change at the `FvmSolver<...>`
  instantiation site, no `residual()` edits. A future DG element can
  implement the `Reconstruction`-shaped contract (produce a face value
  from its own representation) without CMU-CFE needing to retrofit this
  abstraction.
- **Harder**: nothing structurally harder is introduced by this choice;
  the two-stage split's only cost is the extra indirection noted above,
  which measured benchmarks (`docs/performance/0003-...`,
  `docs/performance/0004-...`) show is not a bottleneck at any tested
  scale.
- **Future constraint**: if a future spatial scheme is 3rd-order or
  higher accurate (e.g. MUSCL, WENO), SSP-RK2 would then become the
  accuracy-limiting factor and this decision would need revisiting (see
  Revisit criteria).

## Revisit criteria

Revisit the SSP-RK2 choice specifically once a spatial reconstruction
scheme with formal order > 2 is introduced (MUSCL, WENO, or similar) —
at that point SSP-RK2 would become the limiting term and SSP-RK3 (or
higher) would be needed to realize the spatial scheme's full accuracy.
Revisit the interface-flux design if implementing an actual DG side
reveals the `Reconstruction` contract's assumed shape (two scalar
one-sided face values in, one flux out) does not fit DG's actual needs
(e.g. if DG needs to exchange more than a single scalar per face, such
as a full nodal state vector).
