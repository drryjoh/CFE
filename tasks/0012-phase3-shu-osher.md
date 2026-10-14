# Task 0012: Phase 3 completion — Shu-Osher shock/density-wave interaction, Rusanov vs. HLLC

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `ROADMAP.md` (Phase 3's "Shu-Osher problem")
4. `VERIFICATION.md` ("Canonical problems", this task's own addendum below)
5. `agent_history.md`
6. `docs/adr/0010-euler-field-and-shock-capturing.md`
7. `docs/adr/0012-hllc-numerical-flux.md`

before making changes.

## Objective

ROADMAP.md Phase 3 names both "Sod shock tube" and "Shu-Osher problem"
as acceptance items; only Sod has been implemented so far (task 0009).
This task adds Shu-Osher (Shu & Osher, "Efficient Implementation of
Essentially Non-oscillatory Shock-Capturing Schemes, II," JCP 83, 1989):
a Mach-3 shock propagating into a sinusoidally-perturbed density field,
producing fine post-shock oscillatory structure that a scheme's own
numerical dissipation can smear out if too diffusive.

Per the user's explicit instruction ("we should test all flux
approaches"), run with **both** `RusanovFlux` and `HllcFlux` and report
honestly which preserves more fine post-shock structure. This is a
second, independent data point on the open question task 0011 /
`docs/adr/0012-...md` first surfaced (HLLC showing real Kelvin-Helmholtz
waviness on the forward-step case where Rusanov showed none) — Shu-Osher
is the standard, literature-established test for exactly this kind of
resolution-of-fine-structure question, not a redundant re-ask of the
same question.

Amelia and Violet were consulted before implementation (see
`agent_history.md`'s entry for this task). Key decisions from that
consult, carried into this task:

- **No new ADR.** Choosing a reference-resolution/error-metric for a
  canonical case with no exact solution is a verification-methodology
  decision (AGENTS.md #4.10 reserves ADRs for architectural decisions),
  not an architectural one. `VERIFICATION.md`'s "Canonical problems"
  section gains a short addendum instead (see below).
- **No new production Field/BoundaryCondition/Reconstruction/
  NumericalFlux code is needed** — this reuses `EulerField`,
  `MusclMinmodReconstruction`, `RusanovFlux`/`HllcFlux`, `FvmSolver`,
  `StaticBoundary`/`InflowOutflowBoundary` exactly as they exist today.
- **Scope boundary on "test all flux approaches" (recorded per Amelia's
  consult, applies to this and future tasks):** this phrase is read as
  "the fluxes that exist today" (`RusanovFlux`, `HllcFlux`), not a
  standing requirement to re-run every future canonical case against
  every future flux automatically. Re-evaluate per case/per flux when
  either grows; do not treat it as a combinatorial product rule.
- Stays on `cfe/development/phase_0005-euler-forward-step` (PR #6) per
  the user's own instruction and Amelia's finding that the Phase
  3/Phase 4 ROADMAP boundary is a planning label, not a release gate
  (precedent: task 0010's forward-step work, itself Phase-4-named
  content, already shipped inside this same, Phase-3-labeled PR).
- Kept in its own task file (this one), separate from task 0013's
  propagating-shock case, so a finding in one is never ambiguous about
  which case it belongs to (Amelia's finding, extending ADR 0010
  Decision 4's attribution-clarity reasoning down to task-file
  granularity, not just architecture/branch granularity).

## `VERIFICATION.md` addendum (apply in this task)

Add a short note under "Canonical problems" (or as new text near the
verification hierarchy's "canonical numerical benchmark" entry):
canonical problems with no closed-form exact solution (Shu-Osher named
explicitly) are verified against (a) a converged fine-grid reference
solution run with the same code, not an analytic oracle, consistent
with standard practice in the shock-capturing literature; (b) an exact
solution in any subregion that genuinely has one (e.g. the
still-undisturbed region ahead of a shock, where the initial condition
itself remains an exact solution until the shock arrives); and (c) a
resolution-converging error metric (error shrinks as resolution
increases) rather than a single fixed-resolution pass/fail. This is a
reusable pattern (TGV/flat-plate will likely need the same treatment
later per Amelia's consult), recorded once here rather than
re-litigated per task.

## Verified problem parameters (Violet's consult; cross-checked against
Rankine-Hugoniot + literature, not reconstructed from memory alone)

- Domain `x in [-5, 5]`, `gamma=1.4`.
- Ahead (right) state: `rho(x) = 1 + 0.2*sin(5x)`, `u=0`, `p=1`.
- Shock initially at `x=-4`, Mach 3 relative to the ahead state's sound
  speed `c=sqrt(gamma*p/rho)=sqrt(1.4*1/1)=sqrt(1.4)=1.183216`:
  - `rho2/rho1 = (gamma+1)*M^2 / ((gamma-1)*M^2+2) = 21.6/5.6 = 3.857143`
  - `p2/p1 = (2*gamma*M^2-(gamma-1))/(gamma+1) = 24.8/2.4 = 10.333333`
  - `u2 = (2/(gamma+1))*(M-1/M)*c = 2.629369`
  - Left (post-shock) state: `rho=3.857143, u=2.629369, p=10.333333`.
- Known, documented IC subtlety (do not "fix"): the right state's own
  formula evaluated at `x=-4` gives `rho(-4) = 1+0.2*sin(-20) ~= 0.817`,
  not exactly `1` — the RH jump above is only exact against the uniform
  background `rho=1`; the standard IC (used identically in the original
  paper and every subsequent reproduction) superimposes the sine
  perturbation onto the background without locally re-balancing the
  jump. A small, expected inconsistency right at `x=-4`, not a bug.
- Run to `t_final ~= 1.8`.
- Boundary conditions: `StaticBoundary` at the left (confirmed: all
  three post-shock characteristics point rightward, `u-c = 0.692 > 0`,
  the same "all characteristics point into the domain" case ADR 0010
  Decision 3 already covers), `InflowOutflowBoundary` (zero-order
  extrapolation) at the right (confirmed: any rightward-propagating
  acoustic precursor travels at the local ahead-state sound speed
  `~1.18`, strictly slower than the shock's own `~3.5`, so the shock is
  always the rightmost disturbed point and the right edge stays exactly
  on the undisturbed initial condition through `t=1.8` — verified
  directly by this task's own "ahead-of-shock exact match" test, not
  merely assumed from the speed estimate).

## Acceptance criteria (Violet's consult)

1. **Primary, blocking: ahead-of-shock region matches the exact
   stationary initial condition.** With `u=0`, `p=1` uniform ahead of
   the shock, the ahead region is an exact, time-independent solution
   of Euler until the shock reaches it — check numerical density in a
   region known to remain ahead of the shock at `t=1.8` against
   `1+0.2*sin(5x)` directly, tight tolerance. This is a genuine
   oracle-based check, not merely two runs agreeing with each other.
2. **Fine-grid reference comparison.** Run a coarse (test) resolution
   (e.g. `nx=200`, standard in the literature) and a fine reference
   (e.g. `nx=2000`), same flux, same code. Compare L1/L2 error in the
   **post-shock oscillatory wake only**, explicitly excluding a margin
   around the shock's own current location (shock position/width
   differences are pure O(dx) smearing, unrelated to oscillation
   fidelity, and would contaminate the metric).
3. **Extrema-count stability** as an automatable proxy for wavelength
   fidelity: count sign changes of a discrete density gradient in a
   fixed x-window behind the shock at `t=1.8`; check this count is
   stable (e.g. within +/-1) between the coarse test resolution and the
   fine reference. Do not attempt to script "matches a digitized
   literature figure" — produce the plot for human review (tutorial),
   keep only the scriptable proxy in `CFE_CHECK`.
4. **Cross-flux agreement at fine resolution** is a weak, non-independent
   signal (label it as such in the test's own comment, same convention
   ADR 0010 uses for its own L1-convergence check) — not proof of
   correctness by itself.
5. **The headline comparison**: at the SAME coarse resolution, measure
   each flux's error against the fine reference in the oscillatory wake
   region, and report which is smaller — honestly, without assuming the
   forward-step/Sod-contact direction transfers here without checking.//

Expect mild, bounded overshoot/undershoot at the shock front from
componentwise (not characteristic) MUSCL reconstruction (ADR 0010/0012's
documented limitation) and amplitude-damped oscillations relative to
high-order (WENO) literature results (this scheme is 2nd-order TVD, not
5th-order WENO — Shu-Osher is literature's own standard illustration of
why WENO was built; do not tune against a WENO profile expecting a
close match). Do **not** expect the forward-step triple-point's
near-vacuum/NaN failure mode — this is a 1D single-shock interaction
with no analogous multi-dimensional geometric singularity, and the
ahead-state minimum density (`1-0.2=0.8`) is comfortably positive.

## Required functionality

- `tests/verification/shu_osher_setup.hpp`: shared driver (same shape as
  `sod_shock_tube_runner.hpp`), templated on `Scalar` and
  `NumericalFlux` (default `cfe::RusanovFlux`), builds the grid/IC/BCs
  above and runs to `t_final`, returning a profile (x, rho, u, p) plus
  conserved-integral diagnostics. Includes the RH left-state constants
  and the ahead-state exact-IC formula as reusable free functions.
- `tests/verification/test_euler_shu_osher.cpp`: the five acceptance
  checks above, run for both `RusanovFlux` and `HllcFlux`.
- `tutorials/euler_1d_shu_osher/`: runs both fluxes at a few resolutions,
  writes CSVs, plots density profiles against the fine reference
  (Python, matching `tutorials/euler_1d_sod_shock_tube/`'s convention),
  and reports the measured Rusanov-vs-HLLC wake-error comparison and
  extrema counts in its README as the actual takeaway, not a raw dump.
- `VERIFICATION.md` addendum (above).
- `agent_history.md` entry (written before declaring this task done,
  not after — Ilia caught this gap on task 0011).

## Explicitly not in this task

- Characteristic (eigenvector-projected) reconstruction or PPM — out of
  scope regardless of what this task's overshoot/damping findings show;
  any revisit stays tracked against ADR 0010 Decision 1's existing
  criterion, not re-opened here.
- A third numerical flux (AUSM-family) — ROADMAP's "and/or" is already
  satisfied; this task's own "fluxes that exist today" scope boundary
  applies.
- The propagating-shock/feature-tracking case — task 0013, kept
  separate per Amelia's consult.

## Review

Full review per CLAUDE.md's proportional-effort rule (new canonical
verification case, reused numerics): Violet, Snorlax, Sally, Enzo as
applicable, then Ilia's independent final review and Amelia's alignment
check.
