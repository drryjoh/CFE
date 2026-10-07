# ADR 0008: Burgers' first shock-capturing scheme -- minmod-limited MUSCL + Rusanov

**Status:** Accepted
**Date:** 2026-10-02

## Context

Task 0003 (Phase 2, first slice) adds the inviscid Burgers equation
(`F(Q) = Q^2/2`) -- the first genuinely nonlinear conservation law in
this codebase, and the first whose smooth solutions can form a genuine
shock in finite time. Phase 1's existing numerics
(`CentralDifferenceReconstruction` + `UpwindFlux`) are documented, in
their own header comments, as unsuitable here:

- `UpwindFlux` switches on the *sign* of a single scalar wave speed,
  which is not entropy-correct for a genuinely nonlinear flux at a
  sonic/transonic point (upwind.hpp's own caveat).
- `CentralDifferenceReconstruction`'s unlimited central-difference slope
  oscillates (Gibbs-type overshoot/undershoot) next to a genuine
  discontinuity -- the opposite of what "shock capturing" requires.

A reconstruction/flux pair is needed that (a) does not oscillate at a
discontinuity and (b) is entropy-satisfying for Burgers' convex flux,
while fitting the existing `Reconstruction::right(...)/left(...)` and
`NumericalFlux::operator()(left, right, axis, field)` shapes unchanged
(`docs/adr/0007-...md`).

## Options considered

### Reconstruction

#### Option A: minmod-limited MUSCL (selected)

The simplest provably total-variation-diminishing (TVD) slope limiter
(Harten, 1983): picks the smaller-magnitude one-sided difference when
both agree in sign, clips to exactly zero at a local extremum or
discontinuity.

Advantages:

- Simplest possible TVD limiter -- a single `minmod(a,b)` call, easy to
  verify by hand (this task's `test_muscl_reconstruction.cpp` does
  exactly that).
- Provably non-oscillatory by construction; no tuning parameters.
- Same 3-point stencil and `right(...)/left(...)` shape
  `CentralDifferenceReconstruction` already uses -- zero changes to
  `fvm_solver.hpp`.

Disadvantages:

- The most *diffusive* of the standard limiter family (superbee, van
  Leer, monotonized-central (MC) are all less diffusive, sharper near
  the shock, but more complex and not needed to establish correctness
  first).
- Reduces formal accuracy to 1st order at smooth extrema (the limiter
  clips even where the true solution is smooth) -- an accepted,
  documented property of TVD limiters (Sweby, 1984), not unique to
  minmod, but worth flagging since it directly shapes this task's
  convergence-test acceptance band (see Evidence below).

#### Option B: superbee / van Leer / MC limiters

Description: other standard members of the same TVD-limiter family,
each less diffusive than minmod (sharper shock capture, closer to
2nd-order behavior away from extrema).

Advantages:

- Sharper shock resolution (narrower smeared region) for the same
  resolution.

Disadvantages:

- More complex formulas, more to get wrong on a first implementation.
- Not needed to establish that the codebase's `Reconstruction`/
  `NumericalFlux` abstraction genuinely supports shock capturing at
  all -- that claim only needs *one* correct TVD scheme, not the
  sharpest one.

#### Option C: WENO

Description: a high-order (3rd+) weighted essentially non-oscillatory
reconstruction.

Advantages:

- Higher formal order away from discontinuities than any TVD limiter.

Disadvantages:

- Formal order > 2 would outrun SSP-RK2's own 2nd-order time accuracy
  (ADR 0007 already notes this constraint) -- SSP-RK3 or higher would be
  needed to actually realize the benefit, a bigger change than this
  task's scope.
- Substantially more complex than minmod for a first implementation;
  AGENTS.md #18 already lists it as later work, not immediate.

### Numerical flux

#### Option A: Rusanov / local Lax-Friedrichs (selected)

`F = 0.5*(F(uL)+F(uR)) - 0.5*alpha*(uR-uL)`, `alpha =
max(|uL|,|uR|)` for Burgers.

Advantages:

- Entropy-satisfying for any convex scalar flux -- correct at Burgers'
  sonic point, closing `upwind.hpp`'s documented gap.
- Simplest possible dissipative flux; same `operator()(left, right,
  axis, field)` shape `UpwindFlux` already uses.
- Named first in `ARCHITECTURE.md`'s own planned `numerical_flux/`
  directory order (`rusanov/` before `hllc/`/`ausm/`).

Disadvantages:

- More dissipative (smears the shock over slightly more cells) than an
  exact Godunov solver or HLLC/AUSM-family fluxes.

#### Option B: exact Godunov (Riemann) solver for Burgers

Description: Burgers' scalar Riemann problem has a simple closed-form
exact solution (shock or rarefaction depending on `uL` vs. `uR`); a
Godunov flux evaluates the exact solution at the cell interface
(`x/t=0`).

Advantages:

- Exact at the Riemann-problem level; typically sharper than Rusanov.

Disadvantages:

- Requires explicitly branching on shock-vs-rarefaction-vs-sonic-
  rarefaction cases -- more to implement and get wrong for a scalar-only
  equation that is not this project's main target (Euler, Phase 3, needs
  its own Riemann-solver investment regardless).

#### Option C: HLLC / AUSM-family

Description: approximate Riemann solvers designed for systems
(Euler-family equations) with multiple characteristic families.

Advantages:

- The eventual target for Phase 3's Euler work.

Disadvantages:

- Overkill for a scalar equation with one characteristic family; their
  real value (resolving contact discontinuities a scalar equation
  doesn't have) is moot here.

## Evidence

Both canonical problems `VERIFICATION.md` names are verified against
this scheme (`minmod` + `Rusanov`), not just "ran and looked
reasonable":

**Burgers shock formation** (`tests/unit/test_burgers_shock_formation.cpp`,
Riemann step `u_left=2`, `u_right=1`, exact Rankine-Hugoniot speed
`s=1.5`, run to `t=2.0`):

| nx | mean abs. error vs. exact | max overshoot | max undershoot | TV(initial) | TV(final) |
|---|---|---|---|---|---|
| 200 | 4.21e-3 | 0.0 | 0.0 | 1.000000 | 1.000000 |
| 400 | 2.11e-3 | 0.0 | 0.0 | 1.000000 | 1.000000 |
| 800 | 1.05e-3 | 0.0 | 0.0 | 1.000000 | 1.000000 |
| 1600 | 5.26e-4 | 0.0 | 0.0 | 1.000000 | 1.000000 |

Mean absolute error halves almost exactly with each doubling of
resolution -- the expected O(1/nx) behavior for a captured shock (not a
formal 2nd-order claim; see the test file's own header comment). Zero
measured overshoot/undershoot at every resolution tested (not just
"under some small tolerance" -- the printed values are exactly `0.0`).
Total variation is exactly `1.0` (`=u_left-u_right`) before and after --
no oscillation anywhere, at any resolution. Flux-balance conservation
(`integral_final - integral_initial` vs. `(F(u_left)-F(u_right))*T`)
matched to `1e-6` at `nx=400` (both sides equal `3.000000` to the
printed precision).

**Burgers smooth convergence**
(`tests/unit/test_burgers_convergence.cpp`, `u0(x) = 1.0 +
0.5*sin(2*pi*x)`, run to `t = 0.5 * t_break` where `t_break =
1/(2*pi*0.5) ≈ 0.3183`, against the exact method-of-characteristics
solution). **Nominal vs. measured, stated explicitly (review finding):**
`MusclMinmodReconstruction` is 2nd-order accurate IN SMOOTH REGIONS
LOCALLY -- that is its nominal, design order. It is NOT globally
2nd-order for this specific problem: the sine IC has two smooth extrema
where minmod clips the slope to exactly zero regardless of resolution
(an accepted, documented TVD-limiter property, not a bug -- see
`numerics/fvm/muscl_minmod.hpp`'s own header comment), which drags the
MEASURED global (L2-norm) order down from the nominal 2. The table
below reports the observed order directly (`log2(error_ratio)`), not
just the raw ratio, specifically so this distinction is never implicit:

| nx | L2 error | ratio | observed order (`log2(ratio)`) |
|---|---|---|---|
| 40 | 7.18e-3 | -- | -- |
| 80 | 2.22e-3 | 3.23 | 1.69 |
| 160 | 6.92e-4 | 3.21 | 1.68 |
| 320 | 2.15e-4 | 3.22 | 1.69 |
| 640 | 6.69e-5 | 3.21 | 1.68 |

The observed order stabilizes tightly around **~1.68**, not the nominal
2 a clean 2nd-order-globally scheme would show (compare
`test_scalar_advection_convergence.cpp`'s linear-advection case, which
genuinely does hit ~4.0/order~2 globally). That comparison is NOT
because the linear-advection IC lacks smooth extrema -- its sine profile
`sin(2*pi*(x-a*t))` has exactly the same max/min structure as this
file's `1+0.5*sin(2*pi*x)` does. The difference is the RECONSTRUCTION:
`test_scalar_advection_convergence.cpp` uses
`CentralDifferenceReconstruction`, which is unlimited (no TVD clip at
all, at an extremum or anywhere else), so its extrema cost it nothing.
This test uses `MusclMinmodReconstruction` specifically, whose minmod
limiter clips to 1st order at a smooth extremum regardless of whether a
discontinuity is nearby (LeVeque Sec. 9.3, Sweby 1984) -- the order
reduction measured here is a property of pairing THIS IC with THIS
(limited) scheme, not a property of the IC alone. Stable across
refinements (not drifting toward either the nominal 2 or 1), so this is
read as the scheme's genuine, resolution-independent signature for this
IC, not under-resolution noise. The production test is named
`test_burgers_smooth_convergence_order_reduced_from_nominal_by_minmod_clipping`
(not `..._second_order_convergence`, which would overclaim this
specific measurement) and its acceptance band is expressed directly in
observed-order terms (`[log2(3.0), log2(4.5)] ≈ [1.58, 2.17]`) -- the
same effective threshold as before, just computed rather than
transcribed, so it cannot have been quietly loosened to force a pass.

**GPU port and at-scale verification** (2026-10-02 follow-up, per
review feedback that every PR needs 3D GPU correctness + an at-scale
benchmark, not a deferred follow-up): `BurgersField`/`RusanovFlux`/
`MusclMinmodReconstruction` port to CUDA unchanged (all already
`CFE_HOST_DEVICE`) -- `test_burgers_cuda.cu` (1D) and
`test_burgers_3d_cuda.cu` (3D) both match the CPU reference to `1e-9`
cell-by-cell on a real V100, `compute-sanitizer --tool initcheck` finds
0 errors, and `test_burgers_3d_sanity.cpp` independently confirms the 3D
residual loop matches the 1D reference column-for-column on CPU first.
Benchmarked at the same scale Phase 1 established for scalar advection
(10^8 cells 1D, 512^3 cells 3D): ~7.7e9 cell-updates/s (1D, 10^8 cells)
and ~4.05e9 cell-updates/s (3D, 512^3 cells) on the V100 -- a modest,
expected reduction from scalar advection's own numbers at the same
scale, attributed to Burgers' extra per-cell work (a `minmod` branch per
face per axis, a nonlinear flux evaluation, the Rusanov dissipation
term), not a regression. Full methodology and tables in
`docs/performance/0006-phase2-burgers-cuda-results.md`.

## Decision

Adopt minmod-limited MUSCL (`fvm::MusclMinmodReconstruction`) and
Rusanov (`RusanovFlux`) as Burgers' first reconstruction/numerical-flux
pair, per the Evidence above. Sharper limiters (superbee/van Leer/MC)
and more accurate fluxes (exact Godunov, HLLC/AUSM) remain named future
work, not required to establish that this codebase's abstractions
genuinely support shock capturing.

## Consequences

**Easier:** a genuinely nonlinear, shock-forming conservation law is now
exercisable end-to-end through the exact same `FvmSolver`/`ssp_rk2_step`
machinery Phase 1 built, with zero changes to either -- directly
confirms the Phase 1 design's own stated genericity claim.
`CentralDifferenceReconstruction`/`UpwindFlux` continue serving linear
scalar advection unchanged; this is a second, additive
reconstruction/flux pair, not a replacement.

**Harder:** nothing new for this task's scope. The `~3.2x` (not `~4x`)
convergence ratio on smooth data is a property of minmod specifically,
worth remembering when a future task adds a less-diffusive limiter and
compares against this baseline.

**Future constraint:** Phase 3's Euler work will need its own
`NumericalFlux` (HLLC/AUSM, per `ARCHITECTURE.md`) since `RusanovFlux`'s
underlying formula generalizes to vectors but this header's concrete
free function/functor does not (see `rusanov.hpp`'s own header comment)
-- a genuinely new type, same pattern as this task added alongside
`UpwindFlux`, not a modification of it.

## Revisit criteria

- A future case where minmod's extra diffusion measurably degrades a
  result that matters (e.g. an under-resolved shock interaction) would
  justify adding superbee/van Leer/MC as an alternative
  `Reconstruction`, not replacing minmod (which should stay available as
  the simplest, most robust option).
- Moving to SSP-RK3+ (e.g. for a future WENO reconstruction) would
  justify revisiting the time-integration order constraint this ADR
  inherits from ADR 0007.
