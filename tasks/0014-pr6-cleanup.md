# Task 0014: PR #6 cleanup — math-utility dedup, boundary-condition fill_x/y/z unification, tutorial-data commit policy

**Status: done** (2026-10-14) -- see `agent_history.md`'s "Task 0014"
entry for the complete completion report, review trail, and test
evidence; `docs/adr/0013-tutorial-data-commit-policy.md` for the data
policy decision; `docs/adr/0010-euler-field-and-shock-capturing.md`'s
addendum for the transcendental-function convention decision.

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `ROADMAP.md`
4. `agent_history.md` (tasks 0009-0013's entries, PR #6's full scope)
5. `docs/adr/0010-euler-field-and-shock-capturing.md`
6. `docs/adr/0011-forward-step-block-interface.md`
7. `docs/adr/0012-hllc-numerical-flux.md`

before making changes.

## Objective

PR #6 (Euler core, HLLC, forward-step multiblock, Shu-Osher,
propagating shock, `BlockSet`) squash-merged into `main` as `f2f81d2`.
This task is the repo owner's own follow-on cleanup from that PR's
review comments, already scoped in-session by Amelia/Violet/Snorlax
before implementation began (not re-scoped here) -- three independent
pieces of work, done on a fresh branch off current `main`:

1. A small math abstraction gap: `EulerField::kinetic_energy` hand-rolled
   a sum-of-squares loop instead of calling the already-existing
   `contract(a, b)` (`math/operations.hpp`); several files independently
   hand-rolled the same `x < 0 ? -x : x` absolute-value ternary; and
   `EulerField::sound_speed`'s unqualified `sqrt()` call was flagged
   during an earlier review as needing a convention decision that was
   never actually made.
2. The real work: `PeriodicBoundary`/`StaticBoundary`/
   `InflowOutflowBoundary` (`grid/boundary/boundary_condition.hpp`) and
   `SlipWallBoundary` (`grid/boundary/slip_wall_boundary.hpp`) each
   independently hand-wrote the same "decode a flat index into
   `(g, o1, o2)`, dispatch through `Backend::run`" skeleton three times
   each (once per axis) -- extract it into one shared, reusable helper
   and have all four (plus `BlockInterfaceBoundary`, partially) call
   into it instead.
3. A data-policy decision Amelia flagged as needing to be explicit
   rather than an incidental `.gitignore` comment: going forward,
   tutorials/verification cases commit `figures/*.png` and
   `data/**/summary.csv` only, never raw per-snapshot field data.

## Required functionality

### 1. Math abstraction

- `EulerField::kinetic_energy` (`fields/euler/field.hpp`) calls
  `contract(momentum, momentum)` instead of hand-rolling the same
  sum-of-squares; `EulerField::conserved_from_primitive` computes its
  own kinetic-energy term directly from the already-available
  `velocity` vector via `contract`, eliminating its hand-rolled sum
  entirely (not just delegating it). `hllc_floored_primitives`
  (`numerics/numerical_flux/hllc.hpp`) gets the same treatment, added
  during review on Violet's recommendation (its sum-of-squares is
  independent, not fused into another loop, so converting it is the
  same low-risk pattern as `kinetic_energy` -- see "Explicitly not in
  this task" below for why `hllc_star_state`'s own, FUSED sum-of-squares
  was left alone instead).
- New `src/cfe/math/scalar_ops.hpp`: `abs_value<Scalar>(x)`, collecting
  the `x < Scalar(0) ? -x : x` ternary that was independently
  hand-rolled in `BurgersField::wave_speed`, `EulerField::wave_speed`,
  both `rusanov_flux` overloads (`numerics/numerical_flux/rusanov.hpp`),
  and `BlockInterfaceBoundary::detail::close`
  (`grid/boundary/block_interface_boundary.hpp`) -- five independent
  copies across four files, found by grepping for the exact pattern
  repo-wide, not just the two sites named in the original review.
- No new `mag_sqr` function, no new `common_math.hpp` dumping ground,
  no parameter-pack/fold-expression/`index_sequence` unrolling of the
  `Dim`-sized loops in `kinetic_energy`/`physical_flux` (Snorlax/Violet
  both independently concluded this has no correctness or performance
  justification for this codebase's student audience -- recorded as a
  one-sentence gated backlog note in `fields/euler/field.hpp`, not
  built).
- The `sqrt`/`pow` transcendental-function calling-convention question
  (flagged, not acted on, during an earlier review of this same file) is
  now settled: call them unqualified, not `std::`-qualified -- already
  the de facto convention (`hllc.hpp`), now written down explicitly as
  an addendum to `docs/adr/0010-euler-field-and-shock-capturing.md`
  rather than left open. No code changes needed (every call site already
  followed it).
- Two `std::min` call sites (host-side partition/chunking code) left
  untouched, per the task's explicit scope.

### 2. `fill_x`/`fill_y`/`fill_z` deduplication

- New `src/cfe/grid/boundary/cartesian_fill.hpp`: `cartesian_fill<Axis,
  Backend=CpuParallelFor>(grid, transform)`, modeled on
  `MpiHaloBoundary::exchange<Axis>`'s existing `if constexpr`
  axis-dispatch precedent. Owns exactly the shared skeleton (decode a
  flat index into `(g, o1, o2)`, compute the other two axes' padded
  extents, dispatch one `Backend::run` call) and nothing else --
  `transform(g, o1, o2)` is a caller-supplied callback holding
  everything specific to one `BoundaryCondition` type.
- `PeriodicBoundary`, `StaticBoundary`, `InflowOutflowBoundary`
  (`grid/boundary/boundary_condition.hpp`) and `SlipWallBoundary`
  (`grid/boundary/slip_wall_boundary.hpp`): every `fill_x`/`fill_y`/
  `fill_z` becomes a thin call into `cartesian_fill<Axis>` passing its
  own per-component transform as a lambda.
- `BlockInterfaceBoundary` (`grid/boundary/block_interface_boundary.hpp`):
  partial reuse only, per the task's own explicit scoping --
  `fill_x` reuses `cartesian_fill<Axis::X>` for its decode/dispatch, but
  its neighbor-validity branching and cross-grid read stay local, not
  forced into the shared four types' callback shape. `fill_y`/`fill_z`
  (already throwing stubs) are untouched.
- `MpiHaloBoundary` is untouched, per the task's explicit scope -- a
  different problem (bulk pack/`MPI_Sendrecv`/unpack) already unified on
  its own terms.
- No common `BoundaryCondition` base class/interface; no unifying the
  actual index arithmetic across fundamentally different physics
  (periodic wrap vs. fixed value vs. extrapolation vs. mirror vs.
  cross-block read stay separate per type, per ADR 0010/0011's existing
  duck-typing decision -- not reopened here).
- Behavior-preserving, verified directly (see Evidence in
  `agent_history.md`'s entry): full existing suite re-run before/after
  with identical pass count and identical reported values; the Sod,
  Shu-Osher, and propagating-shock tutorials' `summary.csv` output
  diffed byte-for-byte against the pre-refactor committed baseline; the
  forward-step tutorial's `res20` run compared field-for-field against
  its committed baseline (only the wall-clock-timing column differs, as
  expected). `CFE_HOST_DEVICE` tagging on every shared lambda/callback
  confirmed to survive the refactor (same per-cell lambda shape as
  before, now nested one level deeper through `cartesian_fill`'s own
  `CFE_HOST_DEVICE`-tagged dispatch lambda).

### 3. Tutorial/verification data commit policy

- `docs/adr/0013-tutorial-data-commit-policy.md` (new): the explicit,
  repo-wide decision -- commit `figures/*.png` and `data/**/summary.csv`
  (plus other small derived-metric CSVs) only, never raw per-snapshot
  field data (VTK frames or any future equivalent). No external
  data-storage infrastructure needed at this repository's current scale
  (committed-size discipline plus the derived-metric-oracle pattern
  every verification test already uses is sufficient); revisit criteria
  stated explicitly rather than left implicit.
- `.gitignore` generalized from the two forward-step-specific
  `*.vtk`-ignore lines to one repo-wide `tutorials/*/data/**/*.vtk`
  pattern, so a future tutorial does not need its own addition.
- `VERIFICATION.md` gains a one-paragraph cross-reference tying the new
  ADR's reasoning back to the derived-metric-oracle pattern its
  "Canonical problems with no closed-form exact solution" section
  already established.
- Does not retroactively untrack already-committed VTK frames (the
  3D forward-step extrusion tutorial's ~34 committed frames) -- stated
  explicitly in the ADR as a known, accepted inconsistency, not silently
  left unexplained.

## Explicitly not in this task

- `hllc_star_state`'s own `momentum_squared_sum` loop
  (`numerics/numerical_flux/hllc.hpp`) is left untouched: unlike
  `hllc_floored_primitives`'s independent sum-of-squares (converted to
  `contract`, see below -- Violet's review recommendation, taken), this
  one is FUSED into the same loop that builds `q_star`'s own components,
  so converting it would need an extra array-copy pass over an
  already-hardened, floor-safety-critical function for no behavior
  change. Flagged to Violet/Snorlax during review rather than decided
  unilaterally -- both agreed converting the fused loop was not worth
  it, while converting the independent one in `hllc_floored_primitives`
  was low-risk and taken. See `agent_history.md`'s entry for the full
  review account.
- `std::min` call sites (host-side partition/chunking code).
- Any parameter-pack/fold-expression/`index_sequence` unrolling of
  `Dim`-sized loops.
- A common `BoundaryCondition` base class/interface, or unifying index
  arithmetic across fundamentally different boundary physics.
- `MpiHaloBoundary`.
- External tutorial-data storage infrastructure.
- Retroactively untracking already-committed VTK frames.

## Review

Full review per CLAUDE.md's proportional-effort rule: this touches
boundary-condition production code (§2, used by every Euler tutorial
and test in the repository) and a math utility (§1), so Violet
(correctness/behavior-preservation), Snorlax (did the dedup actually
improve things, not just move the duplication), and Enzo (confirm the
shared helper adds no CPU/GPU overhead -- no new allocations, no new
indirection cost in the hot path) all apply; Sally's review judged not
needed (internal plumbing, not student-facing tutorial code). Ilia's
independent final review and Amelia's alignment check close out the
task, same discipline as every other piece of work this session.
