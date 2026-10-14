# Task 0013: Phase 4 — 1D propagating shock, feature tracking, propagation-speed estimate

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `ROADMAP.md` (Phase 4's "1D propagating shock demonstration and
   periodic propagation-speed estimate")
4. `VERIFICATION.md`
5. `agent_history.md`
6. `docs/adr/0010-euler-field-and-shock-capturing.md`
7. `docs/adr/0012-hllc-numerical-flux.md`

before making changes.

## Objective

ROADMAP.md Phase 4 names "a 1D propagating shock demonstration and
periodic propagation-speed estimate" as a deliverable, distinct from (and
a building block toward) the later moving-frame/grid-recycling prototype
(`ARCHITECTURE.md` #12), which needs exactly this capability — track one
feature, estimate its speed periodically — as its own first step. This
task builds and verifies that capability in isolation, with no recycling
machinery yet.

Per the user's explicit instruction, run with both `RusanovFlux` and
`HllcFlux` and report the measured vs. analytic shock speed for each,
checking (not assuming) that both converge to the same physical speed.

Amelia and Violet were consulted before implementation (see
`agent_history.md`'s entry for this task). Key decisions carried into
this task:

- **No new ADR** — "run existing, already-reviewed machinery, measure a
  number" needs no new architectural decision (confirmed by both
  consults).
- **No new production Field/BoundaryCondition code** — reuses
  `EulerField`, `MusclMinmodReconstruction`, `RusanovFlux`/`HllcFlux`,
  `FvmSolver`, `StaticBoundary`.
- **A standalone single right-moving shock into quiescent gas, NOT a
  reuse of Sod's own right shock** (both Amelia's and Violet's
  independent recommendation, overriding this task's own original
  draft plan to reuse Sod). Reasons:
  - Sod's right shock sits adjacent to a contact discontinuity whose
    own smearing differs materially between Rusanov and HLLC (ADR
    0012's headline 17-vs-13-cell result) — risking that difference
    bleeding into the shock-position estimate as noise, confounding the
    one thing this task measures.
  - A standalone shock's state is pure closed-form Rankine-Hugoniot
    algebra (no Newton iteration at all) — actually lower-complexity
    than reusing Sod, which would still need the exact Riemann solver's
    iteration to locate the contact/rarefaction if any ambiguity check
    were ever needed.
  - Matches this codebase's own existing precedent
    (`tests/verification/test_euler_isolated_contact_hllc.cpp` isolates
    one wave type deliberately, rather than reusing Sod's combined
    problem) and stays reusable, unmodified, for the future moving-frame
    prototype, which needs an isolated single feature.
- Kept in its own task file, separate from task 0012's Shu-Osher case
  (Amelia's finding — one canonical case per task file, matching this
  repo's existing 0009/0010/0011 granularity, for unambiguous
  `agent_history.md`/test-failure attribution).
- Stays on `cfe/development/phase_0005-euler-forward-step` (PR #6), same
  reasoning as task 0012.
- Same "fluxes that exist today" scope boundary as task 0012 applies to
  "test all flux approaches."

## Verified problem parameters

Standalone right-moving shock, Mach `M=2` relative to the ahead state's
sound speed, `gamma=1.4`:

- Ahead (right, at rest): `rho=1, u=0, p=1`, `c=sqrt(1.4)=1.183216`.
- Rankine-Hugoniot jump (shock frame, standard normal-shock relations,
  same formulas as task 0012's Shu-Osher left state, applied at a
  different Mach number):
  - `rho2/rho1 = (gamma+1)*M^2/((gamma-1)*M^2+2) = 9.6/3.6 = 2.666667`
  - `p2/p1 = (2*gamma*M^2-(gamma-1))/(gamma+1) = 10.8/2.4 = 4.5`
  - `u2 = (2/(gamma+1))*(M-1/M)*c = 1.479020` (lab-frame velocity behind
    the shock, in the direction of propagation)
  - Shock speed (lab frame): `S = M*c = 2.366432`.
  - Behind (left) state: `rho=2.666667, u=1.479020, p=4.5`.
- Domain `x in [0,1]`, shock initially at `x0=0.2` (leaves room for it to
  propagate while staying inside the domain for the run duration).
  `t_final` chosen so the shock travels a well-resolved distance (tens
  of dx) while staying inside the domain with margin (e.g. `t_final=0.2`
  puts the shock around `x=0.2+2.366*0.2=0.673`, comfortably inside
  `[0,1]`).
- Boundary conditions: `StaticBoundary` at both ends (left = behind
  state, right = ahead state) — both ends remain untouched/at their own
  far-field state for the whole run, same reasoning Sod's own
  `StaticBoundary` choice uses.

## Feature tracking method (Violet's consult)

**Interpolated threshold-crossing at the midpoint density** (`0.5*(
rho_ahead + rho_behind)`), linear interpolation between the two
bracketing cells for sub-dx precision — not steepest-gradient
(`argmax|drho/dx|`). Rationale: a midpoint-density crossing is, to first
order, insensitive to the width/shape of the numerically smeared
transition as long as the profile is monotonic through it (true for a
clean shock with no overshoot); steepest-gradient depends directly on
the smeared profile's shape, so a flux with a differently-shaped
smearing kernel could shift the measured location even when the
physical shock has not moved — exactly the "spuriously different
speeds between fluxes" risk this task must avoid manufacturing itself.

Implemented as a small, reusable utility
(`tests/verification/shock_feature_tracking.hpp`), not inlined
one-off in the test — intended to be reused by the future moving-frame
prototype's own feature-tracking step.

## Acceptance criteria (Violet's consult)

- **Resolution sweep, not a single fixed tolerance**: run at several
  resolutions (e.g. `nx=100/200/400/800`), and require the measured
  shock-speed error `|v_measured - 2.366432|` to decrease (or at least
  not increase) with resolution — matching this codebase's own
  established convergence-check convention
  (`test_sod_shock_tube_density_l1_error_decreases_with_resolution`).
- At the finest resolution tested, an absolute tolerance chosen from the
  actually-measured error (documented with margin), not guessed in
  advance.
- **Record actual simulation time at each snapshot, not an assumed
  uniform spacing** — the existing CFL-adaptive timestep (`sod_shock_
  tube_runner.hpp`'s own pattern: recompute `dt` from the current max
  wave speed every step) makes snapshot spacing non-uniform in general.
  Clamp `dt` against the next requested output time (generalizing the
  existing `if (t + dt > t_final) dt = t_final - t;` pattern), and fit
  position against the real recorded `t`, not `index * dt_avg`.
- Use >= 8 snapshots spread over the run, least-squares linear fit
  (not a two-point endpoint difference) of position vs. time to estimate
  speed.
- Report both fluxes' measured speed and whether they differ
  measurably from each other — expected not to (same physical shock),
  verified rather than assumed.

## Required functionality

- `tests/verification/propagating_shock_setup.hpp`: RH jump-condition
  constants/formulas (pure algebra, generalized over Mach number — not
  copied from `sod_shock_tube_exact.hpp`, which is a different thing,
  the general iterative Riemann solver; documented cross-reference
  between the two so they are not confused), IC/grid/BC setup, and a
  snapshot-capable runner (list of requested output times -> list of
  profiles with real recorded `t`), templated on `Scalar` and
  `NumericalFlux` (default `cfe::RusanovFlux`).
- `tests/verification/shock_feature_tracking.hpp`: the interpolated
  threshold-crossing utility (above).
- `tests/verification/test_euler_propagating_shock.cpp`: the resolution
  sweep and acceptance checks above, for both `RusanovFlux` and
  `HllcFlux`.
- `tutorials/euler_1d_propagating_shock/`: runs both fluxes, plots
  tracked position vs. time with the linear fit overlaid, and reports
  measured-vs-analytic speed per flux/resolution in its README.
- `agent_history.md` entry (written before declaring this task done).

## Explicitly not in this task

- Moving-frame/grid-recycling itself (window shifting, cell
  insertion/removal) — this task is only the tracking/speed-estimate
  building block `ARCHITECTURE.md` #12 lists as step 1-2 of that future
  mechanism.
- Reuse of Sod's own shock — explicitly rejected above, see rationale.
- A third numerical flux — same scope boundary as task 0012.

## Review

Full review per CLAUDE.md's proportional-effort rule: Violet, Snorlax,
Sally, Enzo as applicable, then Ilia's independent final review and
Amelia's alignment check.
