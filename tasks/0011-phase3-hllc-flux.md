# Task 0011: Phase 3 completion — HLLC numerical flux, Sod contact-sharpening evidence, forward-step Kelvin-Helmholtz re-check

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `ROADMAP.md` (Phase 3's "HLLC and/or AUSM-family implementation", Phase 5's PPM item)
4. `VERIFICATION.md`
5. `agent_history.md`
6. `docs/adr/0010-euler-field-and-shock-capturing.md` (Decision 1, "Future constraint"/"Revisit criteria")
7. `docs/adr/0011-forward-step-block-interface.md`
8. `tutorials/euler_2d_forward_step/README.md`'s "Looking for Kelvin-Helmholtz rollup" section

before making changes.

## Objective

The user asked for both HLLC and PPM as swappable numerical-method
options for `EulerField`, specifically to re-check the forward-facing-
step benchmark for the Kelvin-Helmholtz (KH) waviness Woodward & Colella
(1984) describe on the trailing contact discontinuity — a feature
MUSCL+Rusanov shows *no* trace of at any resolution tested up to
res288 (`tutorials/euler_2d_forward_step/README.md`).

Per CLAUDE.md, Amelia (project direction) and Violet (numerical
correctness) were consulted **before** implementation. Both independently
recommended the same split:

- **This task (0011): HLLC only.** It is a drop-in `NumericalFlux`
  swap (same slot `RusanovFlux` occupies; `MusclMinmodReconstruction`
  untouched), already a named Phase 3 deliverable
  (`ROADMAP.md`: "Rusanov and/or HLLC"; task 0009 deferred it as
  optional for that slice), and the more parsimonious first thing to
  try: Rusanov's own documented excess dissipation at a contact
  discontinuity (`numerics/numerical_flux/rusanov.hpp`) is a more
  likely suspect for "no KH waviness at any resolution" than "MUSCL
  reconstruction isn't PPM." Land HLLC, verify it the way Rusanov was
  verified, then re-run the forward-step case and report honestly
  whatever is observed.
- **PPM is explicitly deferred, not in this task.** It is a
  genuinely new `Reconstruction` mechanism (its own monotonicity
  constraint and discontinuity-flattening logic, Colella & Woodward
  1984, JCP 54, 174-201 — a *different* 1984 JCP paper from the
  forward-step benchmark paper, same volume/year, easy to conflate),
  roadmapped for Phase 5 for a reason, and comparable in scope to
  standing up MUSCL itself. Per Amelia's recommendation: only pull it
  forward (its own new task/ADR, its own branch decision) if this
  task's HLLC-only re-run of the forward-step case does **not**
  reproduce anything resembling the paper's "mild waviness... visible
  at times 2.5 and 3." If it does, Phase 3's own target is already met
  and Phase 5's PPM schedule is undisturbed.

Continues on `cfe/development/phase_0005-euler-forward-step` (PR #6,
not yet merged) — Amelia's recommendation, since this is squarely the
Phase 3 scope that PR already carries, and the forward-step tutorial
this re-run uses is already owned by that PR.

## Required functionality

- `cfe::hllc_flux` / `cfe::HllcFlux`
  (`src/cfe/numerics/numerical_flux/hllc.hpp`): the Toro/Spruce/Speares
  (1994) HLLC approximate Riemann solver (Toro, *Riemann Solvers...*,
  Ch. 10), pressure-based (PVRS) wave-speed estimate per Violet's
  pre-implementation consult:
  - `p*` from the PVRS estimate (`0.5*(p_L+p_R) -
    0.5*(u_R-u_L)*rho_bar*c_bar`), floored at zero — an explicitly
    documented simplification of Toro's full adaptive (two-shock/
    two-rarefaction) estimate, same "accepted, documented
    simplification" precedent ADR 0010 set for componentwise
    reconstruction. Robustness at this simplification's weak point
    (strong rarefaction) is specifically stress-tested (see Tests).
  - `q_K`/`S_L`/`S_R` (Toro Sec. 10.6), `S_star` (Toro eq. 10.37), star
    states (Toro eq. 10.38/10.39) generalized over `Dim` via
    `Field::momentum_index(axis)` (only the normal-momentum component
    gets the `S_star` treatment; every tangential component is
    carried through density-scaled only), four-region flux selection.
  - Euler/EOS-specific, **not** a generic drop-in the way `RusanovFlux`
    is: needs `field.density/pressure/velocity_component/sound_speed/
    gamma` plus `Field::momentum_index/energy_index`, not just
    `field.wave_speed`/`field.physical_flux`. Document this distinction
    explicitly in the header (Violet's consult finding) so nobody
    assumes `HllcFlux` works with `BurgersField`/`ScalarAdvectionField`
    the way `RusanovFlux` does. No scalar (N=1) overload is provided.
- `tests/verification/sod_shock_tube_runner.hpp`'s
  `run_sod_shock_tube` and `tests/verification/forward_step_setup.hpp`'s
  `ForwardStepBlocks<Scalar, Dim>` both gain a `class NumericalFlux =
  cfe::RusanovFlux` template parameter (default preserves every
  existing call site unchanged) so both the Sod tube and the
  forward-step case can be re-run with `cfe::HllcFlux` through the
  SAME shared driver code, not a duplicated copy.
- Sod shock tube re-verified with HLLC
  (`tests/verification/test_euler_sod_shock_tube_hllc.cpp`): same
  conservation/momentum-impulse/exact-solution checks as the existing
  Rusanov test, PLUS a direct, quantitative contact-sharpening
  comparison against Rusanov at the same resolution (not just "also
  passes a Rusanov-calibrated tolerance").
- Isolated pure-contact Riemann problem
  (`tests/verification/test_euler_isolated_contact_hllc.cpp`): the one
  decoupled, textbook demonstration that HLLC resolves a contact
  discontinuity exactly (no shock/rarefaction in the problem at all) —
  compares HLLC's and Rusanov's numerical diffusion of the same initial
  density jump through the same harness.
- Unit tests (`tests/unit/test_hllc_flux.cpp`): consistency invariant
  (`HllcFlux(q,q,axis,field) == physical_flux(q,axis)` exactly),
  left/right-swap symmetry invariant, an oblique 2D/3D test confirming
  only the normal momentum component is treated specially (guards the
  "applied S_star to every component" bug class Violet flagged), `
  S_star` agreement with the existing exact Riemann solver's `u_star`
  for the classic Sod parameters, and a strong double-rarefaction
  stress test (Toro's Test 2 parameters) confirming the wave-speed
  estimate stays robust (finite, positive density/pressure in the star
  region) at its documented weak point. **Note (stale scope bullet,
  caught in Ilia/Amelia's final review): the estimate actually shipped
  is ANRS seeded + Newton-Raphson refined, not plain floored-PVRS as
  originally scoped here -- plain PVRS and ANRS-alone were both tried
  and rejected after real bugs were found; see
  `docs/adr/0012-hllc-numerical-flux.md` Decision 2/2b/2c for the full,
  final account.**
- `tutorials/euler_1d_sod_shock_tube/`: extended to also run MUSCL+HLLC
  at the same resolutions as MUSCL+Rusanov, reporting a measured
  contact-width comparison for a reader (not just a second set of CSVs).
- `tutorials/euler_2d_forward_step/`: re-run with MUSCL+HLLC (reusing
  the generalized `ForwardStepBlocks`) at the resolutions already
  explored for Rusanov, generating the same numerical-schlieren
  diagnostic. README's "Looking for Kelvin-Helmholtz rollup" section
  updated with whatever is actually observed — sharper/thinner trailing
  contact line is the expected, defensible claim regardless of outcome;
  visible rollup/billowing is a bonus to report if seen, not something
  to imply if absent (see Violet's calibration note below).
- `docs/adr/0012-hllc-numerical-flux.md`: records the PVRS-vs-full-
  adaptive-estimate decision, the Euler-specific-not-generic-
  `NumericalFlux` decision, test evidence, and revisit criteria.
  `docs/adr/0010-...md`'s "Future constraint"/"Revisit criteria" entries
  about HLLC are updated to point at this ADR (replace, not duplicate,
  per CLAUDE.md).

## Explicitly not in this task

- PPM (`Reconstruction`) — deferred pending this task's forward-step
  result, per Amelia's recommendation above. If pulled forward, it
  needs its own task file, its own branch/PR decision, and (per
  Violet's consult) likely its own ADR for widening the `Reconstruction`
  concept to a 5-point, `Field`-aware shape if discontinuity-flattening
  is to be included at all — none of that is decided here.
- Characteristic (eigenvector-projected) reconstruction — unchanged
  from ADR 0010's Decision 1.
- The characteristic-based partial-specification subsonic
  inflow/outflow BC — unchanged from ADR 0010's Decision 3.
- AUSM-family flux — `ROADMAP.md`'s "and/or" is satisfied by HLLC alone,
  same framing task 0009 used for Rusanov.
- Toro's fully-iterative Newton-Raphson exact solve as the EXACT solver
  (`tests/verification/sod_shock_tube_exact.hpp`) implements it, seeded
  from its own crude guess, was never the plan — what shipped (see ADR
  0012) is ANRS as a SEED, refined by a fixed 5-iteration Newton-Raphson
  pass of the same underlying formulas, which is both more accurate
  than a bare non-iterative estimate and cheaper/more GPU-friendly than
  the exact solver's own convergence-gated loop.

## Architecture constraints

- No change to `FvmSolver`, `ghost_fill.hpp`, `CartesianGrid`, or any
  existing Field's physics.
- `sod_shock_tube_runner.hpp`/`forward_step_setup.hpp`'s new
  `NumericalFlux` template parameter must default to `cfe::RusanovFlux`
  so every existing call site (the Rusanov Sod test/tutorial, the
  existing forward-step tests/tutorials) is unaffected.
- `HllcFlux` stays in the same `operator()(left_state, right_state,
  axis, field)` functor shape `RusanovFlux` uses, so `FvmSolver`'s
  `NumericalFlux` template slot accepts it with zero solver changes —
  but see the "Euler/EOS-specific, not generic" note above: this is a
  narrower contract than `RusanovFlux`'s, not a drop-in for every Field.
- Per the user's standing requirement (MEMORY.md): this PR adds a new
  scheme (`HllcFlux`), so a 3D CUDA correctness test + at-scale
  benchmark on real hardware (PSC Bridges-2, as task 0009 delivered for
  `EulerField` itself) is required, not deferred — see
  `docs/performance/0007-phase3-euler-cuda-results.md` for the
  precedent to extend, not duplicate.
- Do not commit VTK snapshot data from the forward-step re-run — only
  `figures/*.png` and `summary.csv` (existing `.gitignore` rule already
  covers this).

## Tests

- `tests/unit/test_hllc_flux.cpp` (new, see above).
- `tests/verification/test_euler_sod_shock_tube_hllc.cpp` (new, see
  above).
- `tests/verification/test_euler_isolated_contact_hllc.cpp` (new, see
  above).
- `tests/unit/test_euler_hllc_3d_cuda.cu` (new, CUDA correctness,
  mirroring `test_euler_3d_cuda.cu`'s existing pattern but with
  `HllcFlux`) — gated behind `CFE_ENABLE_CUDA`, run on PSC Bridges-2.
- Full existing CPU suite re-run after every change, zero regressions.

## Review

Full review per CLAUDE.md's proportional-effort rule (new numerical
method): Violet, Snorlax, Sally, Enzo, then Ilia's independent final
review and Amelia's alignment check.
