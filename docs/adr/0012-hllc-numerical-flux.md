# ADR 0012: HLLC numerical flux, and two robustness bugs the forward-step benchmark surfaced

**Status:** Accepted
**Date:** 2026-10-13

## Context

ROADMAP.md Phase 3 names "HLLC and/or AUSM-family implementation" as a
deliverable; task 0009 satisfied the "and/or" with Rusanov alone and
left HLLC tracked as still-open (ADR 0010's "Future constraint"/"Revisit
criteria": "a case needing a sharper contact discontinuity than Rusanov
can provide"). The user has now hit exactly that case: the
forward-facing-step wind tunnel benchmark
(`tutorials/euler_2d_forward_step/`) showed **zero** Kelvin-Helmholtz
(KH) rollup on the trailing contact discontinuity at every resolution
tested up to 288 cells-per-unit-height with MUSCL+Rusanov, and the user
asked for HLLC (and, pending this task's result, possibly PPM) to check
whether a sharper-contact flux reveals the "mild waviness... visible at
times 2.5 and 3" Woodward & Colella's own paper describes.

Before implementation, Amelia (project direction) and Violet (numerical
correctness) were consulted per CLAUDE.md. Both independently
recommended: land HLLC first (smaller, already-roadmapped, drop-in
`NumericalFlux` swap), re-run the forward-step case, and only pursue
PPM (a much larger, Phase-5-scoped `Reconstruction` mechanism) if HLLC
alone does not reproduce the paper's mild waviness. This ADR covers
HLLC only; see `tasks/0011-phase3-hllc-flux.md` for the full scope
decision and `agent_history.md` for the consult transcripts.

## Decision 1: HLLC formulation -- Toro's star-region construction, generalized over `Dim`

`src/cfe/numerics/numerical_flux/hllc.hpp` implements the Toro/Spruce/
Speares (1994) HLLC approximate Riemann solver (reproduced as Toro,
*Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed.,
Ch. 10): four regions separated by three waves (`S_L`, `S_star`,
`S_R`), with the contact resolved as a genuine jump at its own speed
`S_star` rather than smeared by a single dissipative wave the way
Rusanov is. The star-region conserved state (eq. 10.38/10.39) is
generalized over `Dim` via `Field::momentum_index(axis)`: only the
NORMAL momentum component is set to `S_star`; every tangential
component is carried through density-rescaled at its ORIGINAL
velocity -- a genuinely different formula from the normal component,
not the same one applied componentwise (Violet's pre-implementation
consult specifically flagged this as the likeliest implementation
pitfall; `tests/unit/test_hllc_flux.cpp`'s oblique X/Y-axis tests target
it directly).

`HllcFlux` is **not** a generic drop-in `NumericalFlux` the way
`RusanovFlux` is: it needs `field.density/pressure/velocity_component/
sound_speed/gamma` and `Field::momentum_index/energy_index` (a full
ideal-gas equation of state), not just `field.wave_speed`/
`field.physical_flux`. It is implicitly Euler-specific (or any future
Field exposing the same Calculator surface); instantiating it against
`BurgersField`/`ScalarAdvectionField` simply fails to compile. Documented
explicitly in the header so this scoping is never assumed away.

## Decision 2: wave-speed estimate -- a two-stage design, the second stage forced by a real bug

### Stage 1 (rejected): plain PVRS

Toro's PVRS (primitive-variable Riemann solver, eq. 9.20) estimate used
directly as `p*`. Measured against the independent exact solver
(`tests/verification/sod_shock_tube_exact.hpp`) for the classic Sod
parameters: `p* = 0.55` vs. exact `p* = 0.30313` -- an 82% error, because
Sod's 10:1 pressure ratio is outside PVRS's own stated "weak wave"
validity range. Caught in review before being accepted.

### Stage 2 (also rejected, after a severe bug was found): Toro's ANRS

Toro's Adaptive Noniterative Riemann Solver (Sec. 9.5.2): adaptively
switches among PVRS/TRRS (two-rarefaction, exact for genuine double
rarefactions)/TSRS (two-shock, one-shot) depending on how strong the
jump is. Measurably better for Sod (`p* = 0.315`, ~4% error) -- a
genuine, documented improvement, matching the kind of "documented
simplification" precedent ADR 0010 set for componentwise reconstruction.

**This was initially accepted, then rejected after end-to-end testing
against the forward-step benchmark surfaced a severe, silent failure
ANRS does not protect against.** At the step's riser face (a solid wall
reflecting the Mach-3 inflow -- `rho=1.4`, `u=+3` meeting its mirror
`u=-3`, `p=1`, `gamma=1.4`), ANRS's TSRS branch gives `p0 = 10.0`
against the true (Newton-solved) `p* = 17.2`: a 42%-low estimate that
happens to land on the WRONG SIDE of the threshold (`p*/p_L > ~10.33`)
needed to make `S_L = u_L - c_L*q_L` negative. The practical
consequence was total, not partial: `S_L` came out `+0.047` (positive),
`hllc_flux` took the `S_L >= 0` branch, and returned `F(qL)` **completely
unchanged** -- zero numerical dissipation at exactly the interface that
most needs it. Reproduced directly (`/tmp/debug_hllc_step.cpp`,
`/tmp/debug_hllc_direct.cpp`, not committed -- throwaway diagnostics):
the entire forward-step domain stayed frozen at the exact uniform
inflow state (density bit-identical to `1.4` in every cell) for the
whole `t=0` to `t=4` run, at **every** resolution tested, with no
crash, NaN, or visible error -- a silent correctness failure, not a
performance or robustness nitpick. `final_mass` for every HLLC
resolution converged to ~3.52-3.54 (consistent with a static, unperturbed
domain) against Rusanov's own, correctly-evolving ~6.29-6.41.

This is NOT the same class of "documented, accepted simplification"
ADR 0010 established for componentwise reconstruction: a wrong-signed
wave speed is a correctness defect (it silently disables the scheme's
own upwinding/dissipation logic), not a lower-order-but-still-consistent
approximation.

### Stage 3 (accepted): ANRS seed + fixed-iteration Newton-Raphson refinement

`hllc.hpp` now uses the ANRS estimate only as a starting guess, then
runs a FIXED, unconditional 5 iterations of Newton-Raphson (Toro
eq. 4.5-4.8's pressure function/derivative, reimplemented in `src/`
rather than included from the `tests/`-only exact solver) refining `p*`
toward the exact two-shock/two-rarefaction solution. Fixed iteration
count (no convergence-gated early exit) specifically for GPU
warp-uniformity (every thread does identical work regardless of how
quickly its own Riemann problem happens to converge) -- Newton's
quadratic convergence means 5 iterations, seeded from ANRS, converges to
full `double` precision in <=4 iterations even for this file's own
worst discovered case (`10.0 -> 17.2` in 4 iterations; see the iteration
trace below). `p` is floored at a tiny positive value after EVERY
iteration (not just the final result), guarding the next iteration's own
`sqrt`/`pow` evaluations against a bad intermediate step.

```
it=0 p=10.00799600  -> p=16.09999485
it=1 p=16.09999485  -> p=17.18770213
it=2 p=17.18770213  -> p=17.20832242
it=3 p=17.20832242  -> p=17.20832932   (converged to double precision)
```

Verified fix (`/tmp/debug_hllc_direct.cpp` rerun after the fix): `S_L`
now `-0.859126` (correctly negative), `S_star=0.0` (correct by
symmetry), flux = `(0, p*, 0, 0)` -- the physically correct stagnation
flux for a symmetric wall reflection (zero mass/energy flux, pure
pressure force). Permanent regression coverage added:
`tests/unit/test_hllc_flux.cpp`'s
`test_hllc_wave_speeds_correctly_signed_for_strong_symmetric_wall_
reflection`, reproducing this exact case with tight tolerance (the
Newton-converged values are now exact, not estimated) so a future change
weakening the refinement is caught immediately rather than silently
reappearing as a multi-minute tutorial re-run producing a suspiciously
frozen flow field.

An incidental, stronger piece of evidence this fix is correct, not just
"less wrong": with `p*`/`S_L`/`S_R` now Newton-converged for the
classic Sod problem too, `S_L=-1.183216` and `S_R=1.752156` match the
EXACT solver's own rarefaction-head and shock speeds exactly -- yet
`S_star=0.641832` still differs substantially from the exact
`u*=0.92745`. This proves the previously-documented "HLLC under-tracks
`u*` for Sod" gap is intrinsic to the star-speed FORMULA itself (it
implicitly treats both nonlinear waves via a Rankine-Hugoniot-style
relation, inexact whenever one side is a genuine rarefaction), not an
artifact of wave-speed-estimation accuracy as the Stage-1/Stage-2
writeup had left ambiguous.

### Cost

5 extra Newton iterations per face (each a few `sqrt`/`pow`
evaluations) vs. Stage 2's single-shot ANRS. Not yet benchmarked in
isolation against Rusanov's own per-face cost (tracked as an open
follow-up below) -- accepted without that measurement because AGENTS.md
#2's priority order places correctness above measured performance, and
this fix is closing a complete, silent correctness failure, not a minor
accuracy gap a benchmark should arbitrate.

## Decision 2b: a SECOND, separate robustness bug -- near-vacuum reconstruction artifact

After Decision 2's wave-speed-sign fix, re-running the full forward-step
resolution sweep (20/40/80/96/144/216/288) surfaced a second, different
failure: `res20` completed cleanly (`final_mass=6.356`, physically sane,
close to Rusanov's `6.407`), but `res40/80/96` all went to `NaN`
partway through the run. Direct reproduction (standalone debug driver,
not committed) isolated the FIRST NaN at `res40`, step 636 (`t=1.49`),
cell `(i=14,j=12)` -- in the bow-shock/Mach-stem multi-wave-interaction
region. Manually reconstructing (via `MusclMinmodReconstruction`, the
same formula `axis_flux_difference` uses) the face states feeding that
cell found one Y-direction face with `density=1.704418`, `pressure=
+0.0094` -- a near-vacuum but still-technically-positive implied
pressure. MUSCL's componentwise (not characteristic) reconstruction
limits density/momentum/energy independently; the IMPLIED pressure
`p=(gamma-1)*(E-KE)` can come out as a tiny near-cancellation residual
even when every individual component's own reconstruction looks
reasonable in isolation -- **this is exactly the failure mode ADR
0010's Decision 1 named as a revisit criterion** ("a future canonical
case... showing componentwise reconstruction producing visible spurious
oscillation at a strong multi-wave interaction would justify adding
characteristic reconstruction").

Consulted Violet (full diagnostic trace handed over, not just a
symptom report) before fixing, per CLAUDE.md. Her assessment (full
detail in agent_history.md's consult transcript for this task):
evaluating HLLC directly on the one identified near-vacuum face pair
does not itself produce non-finite output, but `hllc_star_state`'s
`density_star = rho_K*(S_K-u_K)/(S_K-S_star)` formula is only
guaranteed non-negative when the wave ordering `S_L <= S_star <= S_R`
holds EXACTLY, which is a property of the true `p*` -- a near-vacuum
input can perturb the Newton-refined `p*` (Decision 2) just enough that
this ordering guarantee slips, producing a negative `density_star` that
poisons the NEXT RK stage's own reconstruction (eventually a genuinely
negative implied pressure, then `sqrt` of it inside `sound_speed`, then
`NaN`). Rusanov is not immune to the same `sqrt(negative)` failure mode
in principle (`RusanovFlux::wave_speed` calls the same `sound_speed()`
unconditionally) -- it simply hasn't produced a negative-pressure
reconstructed state on ITS OWN trajectory through state space for this
run, consistent with Rusanov's higher dissipation making this near-
vacuum artifact less likely to arise or survive, not with Rusanov being
structurally protected against it.

**Fix (narrowly scoped to `hllc.hpp`, per Violet's explicit
recommendation NOT to fall back to Rusanov -- Rusanov would hit the same
`sqrt(negative)` if the input were already unphysical, so a flux-level
fallback does not address the root cause)**: floor density and pressure
at their two entry points, both local to this file, both no-ops for any
well-separated/physically-reasonable input:

1. In `hllc_wave_speeds`: floor the raw `rho_l/rho_r/p_l/p_r` BEFORE
   `sound_speed` is computed (computed manually as `sqrt(gamma*p/rho)`
   from the FLOORED values, not via `field.sound_speed(...)`, which
   would evaluate `sqrt` on the unfloored, possibly-negative inputs
   first).
2. In `hllc_star_state`: floor `density_star` itself, AND floor the
   returned star state's IMPLIED pressure (not just its raw energy
   component) -- the same "components individually reasonable, implied
   pressure not" risk applies to the star state's own construction, not
   only to reconstruction's output.

Verified: the `res40` reproduction that previously went `NaN` at step
636 now completes all steps (1652 steps to `t=4`) with no non-finite or
non-positive state anywhere in either block. Two new regression tests
added (`tests/unit/test_hllc_flux.cpp`): the exact near-vacuum face pair
from the real failure (asserts finite flux output), and a direct,
engineered exercise of `hllc_star_state`'s floor itself (asserts
`density_star` is floored, not negative, for `S_K`/`S_star` values
chosen to force the raw formula negative). Full suite: 143/143 passing.

**Scope note (explicitly NOT done here):** this floors the SYMPTOM
(HLLC's own formulas breaking on a near-vacuum/unphysical input), not
the root cause (componentwise MUSCL reconstruction producing that input
in the first place). ADR 0010 Decision 1's own revisit criterion is now
satisfied by direct, reproducible evidence -- characteristic
reconstruction (or a lighter-weight positivity-preserving reconstruction
fallback, which would need `Reconstruction` to become `Field`-aware, the
same architecture gap Violet's PPM pre-implementation consult separately
flagged for discontinuity-flattening) remains tracked as a real,
evidenced future need, not a hypothetical one -- see Revisit criteria.

## Decision 2c: the floor fix itself had a gap -- caught by Enzo's review, not by testing

Decision 2b's floors protect `hllc_wave_speeds`'s `rho_l/rho_r/p_l/p_r`
and `hllc_star_state`'s `rho_k/p_k` -- but the ORIGINAL implementation of
`hllc_wave_speeds` still computed `u_l`/`u_r` via
`field.velocity_component(...)` and `p_l_raw`/`p_r_raw` via
`field.pressure(...)` directly, both of which internally divide by the
RAW (unfloored) density, not the floored one computed alongside them.
Enzo's performance review (reading the code, not running a new
reproduction) traced the consequence: a reconstructed face with a
slightly NEGATIVE raw density makes `kinetic_energy = momentum^2 /
(small negative rho)` swing to a large NEGATIVE number, which makes
`pressure = (gamma-1)*(E - kinetic_energy)` swing to a large POSITIVE,
FINITE number -- which silently PASSES the `p > kTinyPressureFloor`
check (it is not negative, zero, or non-finite) even though it is
wildly wrong. This reintroduces Decision 2's own "complete, silent
failure, no crash/NaN" bug class at a different entry point, inside the
very fix meant to close it.

Fixed by factoring a single shared `hllc_floored_primitives(q, axis,
field) -> {rho, u, p}` helper that derives velocity and pressure from
the SAME floored density it returns (mirroring the pattern
`hllc_star_state` already used, just not yet applied to
`hllc_wave_speeds`), used by both functions -- closing the gap at both
call sites at once and incidentally resolving Snorlax's and Enzo's
overlapping "redundant per-side recomputation" findings (one formula,
one evaluation per side, shared by both functions that need it).

**Lesson for future floor/guard additions to this file (and recorded in
Violet's/Enzo's own accumulated knowledge, see `.claude/agents/`):** a
floor on a derived quantity (density) is not transitively a floor on
every OTHER quantity computed from the unfloored original -- every
accessor that reads the same underlying state must be re-derived from
the floored value explicitly, or checked to confirm it already is.
Testing alone did not catch this (the existing near-vacuum regression
test's one example state happened to have a positive, not negative, raw
density); Enzo's review did, by reading the dataflow rather than relying
on test outcomes alone.

New regression coverage: float-precision versions of both the
wall-reflection and near-vacuum regression tests (Enzo's Finding 6 --
`kTinyDensityFloor`/`kTinyPressureFloor` sit far closer to `float`'s own
epsilon than to `double`'s, and no Euler numerical flux in this
repository had ever been tested at `Scalar=float` before, a pre-existing
gap named here rather than silently left HLLC-only). Full suite:
**145/145** passing (143 plus 2 new float-precision tests).

## Decision 3: `sod_shock_tube_runner.hpp` / `forward_step_setup.hpp` generalized over `NumericalFlux`

Both gained a `class NumericalFlux = cfe::RusanovFlux` template
parameter (default preserves every existing call site bit-for-bit --
confirmed: the full pre-existing CPU test suite reproduces identical
numbers after this change) so the Sod tube and forward-step cases can
be re-run with `cfe::HllcFlux` through the SAME shared driver code,
rather than a second, duplicated simulation setup for each.

## Evidence

- **Unit tests** (`tests/unit/test_hllc_flux.cpp`, 14 tests): exact
  consistency (`hllc_flux(q,q,...) == physical_flux(q,...)`) in 1D/2D/3D;
  mirror-reflection odd/even-component symmetry; the normal-vs-tangential
  star-state distinction (both X- and Y-axis); `S_star` cross-checked
  against the independent exact solver (loose for Sod's mixed
  shock+rarefaction case, documented why; tight for a symmetric
  double-shock case where the formula's own assumptions hold); the
  strong-double-rarefaction stress test (Toro's Test 2); the
  wall-reflection regression test (Decision 2); the near-vacuum
  finite-output and floor-exercise regression tests (Decision 2b); and
  float-precision reruns of the wall-reflection and near-vacuum cases
  (Decision 2c).
- **Sod shock tube re-verification**
  (`tests/verification/test_euler_sod_shock_tube_hllc.cpp`): same
  conservation/momentum-impulse/exact-solution checks as the Rusanov
  test, plus a direct contact-width comparison at `nx=400`: **HLLC=13
  cells vs. Rusanov=17 cells** (5%-95%-crossing definition), and a
  near-contact density-error check with a negative control proving the
  comparison is genuinely discriminating (HLLC error `1.52e-3` passes a
  `2.5e-3` tolerance; Rusanov's `3.79e-3` fails the same tolerance).
- **Isolated pure-contact test**
  (`tests/verification/test_euler_isolated_contact_hllc.cpp`): the
  single cleanest, decoupled demonstration -- a stationary density step
  (equal pressure/velocity, differing density only) stays EXACTLY in
  place under HLLC (`0.000e+00` max deviation after ~150 steps) while
  Rusanov visibly diffuses it (`0.224` max deviation).
- **1D tutorial** (`tutorials/euler_1d_sod_shock_tube/`): both fluxes
  run at nx=100/200/400; HLLC gives a consistently narrower contact
  (~20-25% fewer cells) and lower L1 error on every reported quantity at
  every resolution -- see that tutorial's README.md and
  `figures/sod_contact_width.png`/`sod_contact_closeup.png`.
- **Forward-step re-run** (`tutorials/euler_2d_forward_step/`): see that
  tutorial's own README.md "Looking for Kelvin-Helmholtz rollup" section
  for the full, honest qualitative result after the Decision-2 fix.
- Full CPU unit/verification suite: 145/145 passing (131 pre-existing
  plus 14 new HLLC unit tests), zero regressions to any pre-existing
  test.

## Consequences

**Easier:** CFE now has a second, sharper-contact numerical flux
available as a pure template-parameter swap, with no change to
`FvmSolver`, `MusclMinmodReconstruction`, or any existing Field. The
Newton-refined wave-speed pattern (ANRS seed + fixed-iteration
correction) is a reusable template for any future approximate-Riemann-
solver work needing both O(1)-ish cost and strong-shock robustness.

**Harder:** `HllcFlux` is a narrower-contract `NumericalFlux` than
`RusanovFlux`/`UpwindFlux` (Euler/EOS-specific, not usable with
`BurgersField`/`ScalarAdvectionField`) -- a reader reaching for "the
generic flux interface" needs to know this distinction exists. The
fixed 5-iteration iteration count is itself unverified against a case
stronger than this ADR's own M=3 wall reflection; a sufficiently more
extreme case (not yet encountered) could in principle still need more
iterations, though Newton's quadratic convergence makes this unlikely
for any physically realistic gas-dynamics state.

**Future constraint:** HLLC's wave-speed robustness fix (Decision 2,
Stage 3) was discovered and fixed specifically because this task ran a
genuinely strong-shock 2D case (not just 1D Sod) end-to-end before
declaring the flux done -- a reminder that `tests/unit/`+`tests/
verification/`'s existing 1D-only coverage at the time this bug was
introduced was NOT sufficient to catch it; the new regression test
closes that specific gap but does not guarantee no OTHER 2D/3D-only
failure mode remains uncaught.

## Decision 4: the forward-step Kelvin-Helmholtz result, and a confirmed componentwise-reconstruction artifact

With Decisions 2/2b/2c's robustness fixes in place, the full
forward-step resolution sweep (20-288 cells-per-unit-height) was re-run
with HLLC and compared directly against the existing Rusanov dataset at
every resolution, same grid, same IC/BCs, same MUSCL reconstruction.
Full qualitative result and figures: `tutorials/euler_2d_forward_step/
README.md`'s "Looking for Kelvin-Helmholtz rollup" section. Summary:

- **HLLC's trailing contact line develops a wavy, repeatedly-crossing
  structure that Rusanov never shows at any resolution tested**,
  growing in complexity over time with onset timing (visible by
  t=2.5-3) matching Woodward & Colella's own reported timing for the
  "mild Kelvin-Helmholtz instability ('water-wave'/'flag-waving')"
  almost exactly. This is a genuine, reproduced positive result for the
  user's original motivating question -- the first task in this
  session's forward-step work to show ANY trace of this feature.
- **A second, separate, confirmed finding (not the KH result itself):**
  at the bow-shock/Mach-stem triple point specifically -- a strong 2D
  multi-wave interaction -- HLLC's combination of low dissipation with
  componentwise (not characteristic) MUSCL reconstruction produces a
  real, bounded density overshoot (measured at res80: HLLC peak density
  ~7.1 vs. Rusanov's own ~6.3 at the same place/time; HLLC's minimum
  also undershoots Rusanov's). Finite, not NaN, and not conservation-
  violating (`final_mass` stays close to Rusanov's own at every
  resolution) -- but a real, visible local artifact distinct from the
  physical shock structure. This is **ADR 0010 Decision 1's own named
  revisit criterion, now directly confirmed by reproducible evidence**
  ("a future canonical case... showing componentwise reconstruction
  producing visible spurious oscillation at a strong multi-wave
  interaction would justify adding characteristic reconstruction") --
  not fixed in this task (characteristic reconstruction is explicitly
  out of scope here; see that ADR's own Decision 1).

**These two findings are logically separate, not the same phenomenon
wearing two names** -- the overshoot sits at the triple point itself;
the KH waviness is observed on the trailing contact line downstream of
it, growing over time rather than being a static artifact. Both stem
from the same root enabler (HLLC's lower dissipation relative to
Rusanov), but one is a wanted result (the KH finding) and the other is
an unwanted, already-anticipated side effect (the overshoot) -- reported
together here for completeness, not conflated.

## Revisit criteria

- A future case whose strongest shock exceeds what 5 Newton iterations
  (seeded from ANRS) converges correctly for -- would need either more
  iterations or a provably convergent bound (e.g. bisection fallback).
- **Resolved on GPU, still open on CPU:** a clean, isolated per-face
  cost comparison against Rusanov (Enzo review finding) -- the
  forward-step CPU wall-clock numbers in this ADR/README come from one
  single, non-interleaved, non-repeated run (a real, confirmed
  methodology confound; the identical Rusanov res288 computation gave
  three different timings across this session's own repeated runs).
  Now closed for GPU specifically: `docs/performance/0008-phase3-hllc-
  cuda-results.md`'s at-scale benchmark (`bench_euler_hllc_3d_cuda.cu`
  vs. `bench_euler_3d_cuda.cu`, identical IC/cube-sweep, differing only
  in flux) gives a clean **7.5x-9.4x** slowdown across five problem
  sizes, with a register-pressure measurement (149 vs. 96
  registers/thread, ~12.5% vs. ~25% occupancy) quantifying a real,
  substantial piece of why. An isolated CPU-side microbenchmark (no
  GPU-specific confounds, no occupancy effects) remains open if a CPU-
  specific number is ever needed.
- Whether a convergence-gated (not fixed) Newton loop would be faster on
  CPU specifically, behind a compile-time backend policy (Enzo review
  finding) -- the fixed-iteration choice is justified for GPU
  warp-uniformity (Decision 2's own reasoning), but was applied
  unconditionally to the CPU/threaded backend too, without measuring
  whether most (weak-wave, few-iterations-to-converge) faces would
  finish faster with an early exit there. Speculative until benchmarked,
  per AGENTS.md #9's "do not assume a GPU-motivated choice helps
  CPU/Apple Silicon" (here applied in the direction GPU-motivated ->
  CPU, the same principle in spirit).
- The 5-iteration margin itself is characterized by exactly ONE measured
  case (the Mach-3 wall reflection, `p0=10.0 -> 17.2`) -- a swept test
  (pressure ratio 10/100/1000/10000, iterations-to-double-precision from
  the ANRS seed) would turn this into a characterized margin rather than
  "5 worked for the one case found so far."
- **Resolved, not open:** the forward-step Kelvin-Helmholtz question
  that motivated pulling PPM forward conditionally (Amelia's
  recommendation, `tasks/0011-...md`) is now answered -- HLLC alone DID
  reproduce the paper's mild waviness (Decision 4), so that specific
  trigger for pulling PPM forward does not fire. PPM remains properly
  scoped to Phase 5 unless a DIFFERENT, new need arises.
- **Resolved, not open:** real CUDA hardware access to complete the 3D
  CUDA-correctness/at-scale-benchmark requirement for `HllcFlux` --
  both halves are now done (`docs/performance/0008-phase3-hllc-cuda-
  results.md`: 157/157 correctness, 0 compute-sanitizer errors, and a
  clean 7.5x-9.4x at-scale throughput comparison against Rusanov with a
  register-pressure explanation). This line was itself stale in an
  earlier revision of this ADR (said "still-outstanding" after the
  correctness half had already landed, before the at-scale half also
  landed) -- caught in Ilia's final review.
- **ADR 0010 Decision 1's own revisit criterion is now directly, not
  hypothetically, satisfied** (Decision 2b): componentwise MUSCL
  reconstruction producing a near-vacuum implied pressure at a strong 2D
  multi-wave interaction is confirmed, reproducible behavior, not a
  theoretical risk. The floor fix in Decision 2b treats the SYMPTOM; a
  future case where the floors themselves prove insufficient (e.g. a
  case whose near-vacuum artifact is severe enough to still visibly
  distort the solution even once NaN is prevented) would motivate either
  characteristic reconstruction or a positivity-preserving `Reconstruction`
  fallback -- the latter needs `Reconstruction` to become `Field`-aware
  (read pressure/density, not just raw scalars), the same architecture
  gap Violet's PPM pre-implementation consult flagged for discontinuity
  flattening, suggesting both needs might eventually be served by the
  same interface widening.
