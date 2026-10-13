# Tutorial: 2D forward-facing-step wind tunnel (Mach 3)

Task 0010 (`tasks/0010-phase4-forward-step-multiblock.md`): Woodward &
Colella (1984), JCP 54, 115-173, Sec. IV.b -- the canonical Mach 3
forward-facing-step wind tunnel, CFE's first multi-block case. Two
`CartesianGrid` blocks (upstream, full tunnel height; downstream,
post-step height) joined by the new `BlockInterfaceBoundary`
(`src/cfe/grid/boundary/block_interface_boundary.hpp`) -- see
`docs/adr/0011-forward-step-block-interface.md` for the full design
reasoning and `tests/verification/forward_step_setup.hpp` for the shared
two-block assembly/stepping driver this tutorial just runs and writes
output for.

## Setup

- Domain: 1-unit-wide, 3-unit-long tunnel; 0.2-unit step, 0.6 units from
  the inlet.
- gamma=1.4, inlet density=1.4, pressure=1.0, velocity=3.0 (Mach 3)
  everywhere initially, continually fed from the left (fixed supersonic
  inflow).
- Outlet: zero-order-extrapolated supersonic outflow.
- Every tunnel-wall/step-top face: reflecting slip wall
  (`SlipWallBoundary<EulerField<Scalar,2>>`).
- MUSCL (minmod) + **both** `RusanovFlux` and `HllcFlux` (task 0011,
  `docs/adr/0012-hllc-numerical-flux.md` -- every resolution is now run
  with each flux, through the same `ForwardStepBlocks<Scalar,Dim,
  NumericalFlux>` driver), SSP-RK2, adaptive CFL=0.4.
- Seven resolutions: cells-per-unit-height 20/40/80 (the paper's own
  reference grids, equivalently "20x60"/"40x120"/"80x240" for the full
  1x3 bounding box -- PPM itself is not required/used here, since the
  paper's own "essentially converged" grid is for PPM specifically and
  this tutorial uses MUSCL+Rusanov/MUSCL+HLLC, per this task's own
  scope), plus 96, 144, 216, and 288 (each 1.5x the last) extending the
  sweep further for a finer convergence illustration than the paper's
  three points alone give, and specifically to check whether finer
  resolution (or a sharper-contact flux) reveals Kelvin-Helmholtz
  rollup on the trailing contact discontinuity that isn't visible at
  the paper's own resolutions -- see "Looking for Kelvin-Helmholtz
  rollup" below for task 0011's own finding.
- Run to t=4 (unsteady until roughly this time, per the paper and this
  task's own spec).

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCFE_DEFAULT_BACKEND=threaded
cmake --build build --target cfe_euler_2d_forward_step -j
cd tutorials/euler_2d_forward_step && ../../build/tutorials/euler_2d_forward_step/cfe_euler_2d_forward_step
```

`ForwardStepBlocks` never hardcodes a backend -- its `Solver1`/`Solver2`
use `FvmSolver`'s default `Backend` template parameter, `CpuParallelFor`,
which itself resolves at compile time to whichever of
`backend::serial`/`backend::threaded` `CFE_DEFAULT_BACKEND` was set to.
So passing `-DCFE_DEFAULT_BACKEND=threaded` (a real, unit-tested
`std::thread`-based implementation, not a stub -- see
`src/cfe/backend/cpu/threaded.hpp` and `tests/unit/test_backend_execution.cpp`)
parallelizes every per-cell residual/ghost-fill kernel across this
machine's cores with zero code changes to this tutorial. Omit the flag
(or pass `=serial`) for the original single-threaded behavior.

Writes `data/res{20,40,80,96,144,216,288}_{rusanov,hllc}/frame_NNNN_
{density,pressure}_block{1,2}.vtk` (ASCII legacy VTK, one pair of files
per block per snapshot, every `dt=0.1`) plus `data/res<N>_<flux>/
summary.csv` (step count, wall-clock time, final total mass) -- one
directory per resolution PER FLUX since task 0011 (previously one per
resolution, Rusanov only).

Measured wall-clock time per resolution, both fluxes (threaded backend,
this development machine, Apple Silicon M5, 10 logical cores; from each
run's own `summary.csv` -- not estimated):

| cells-per-unit-height | steps (Rusanov / HLLC) | wall-clock Rusanov | wall-clock HLLC |
|---:|---:|---:|---:|
| 20  | 821 / 823    | 1.46s  | 1.83s   |
| 40  | 1,645 / 1,652 | 3.29s  | 5.60s   |
| 80  | 3,302 / 3,321 | 7.83s  | 25.73s  |
| 96  | 3,966 / 3,992 | 9.98s  | 43.79s  |
| 144 | 5,972 / 6,025 | 18.85s | 124.22s |
| 216 | 8,990 / 9,085 | 42.88s | 382.20s |
| 288 | 12,025 / 12,231 | 88.13s | 850.15s |

**Important caveat on the HLLC/Rusanov cost ratio (Enzo review
finding):** these are single, non-interleaved, non-repeated
measurements from one continuous run (all Rusanov resolutions, then all
HLLC resolutions) -- the ratio growing with resolution is CONSISTENT
with HLLC genuinely costing more per face (its wave-speed estimate runs
a fixed 5-iteration Newton-Raphson refinement plus several more
`sqrt`/`pow` evaluations than Rusanov's two, `docs/adr/0012-...md`), but
has not been isolated from possible thermal/OS-contention confounds a
single long run cannot rule out. Treat the DIRECTION (HLLC costs more)
as reliable and the specific multiplier as indicative, not an isolated
per-face benchmark -- see that ADR's Revisit criteria for the cleaner
microbenchmark this would take to pin down precisely.

(96/144/216/288 don't divide the step position/height evenly the way
20/40/80 do -- e.g. 0.6*96=57.6 -- so their per-block cell counts come
from rounding in `make_grid1`/`make_grid2`, not a clean closed form;
taken directly from each run's own VTK output above rather than
computed by hand. This rounding also used to leave block2's real-space
origin at a sub-cell-width offset from block1's actual extent at these
resolutions specifically -- visible as a thin stray line at the
interface in an earlier version of `resolution_comparison.png` -- fixed
by deriving block2's origin from block1's actual rounded extent instead
of independently re-rounding the nominal step position/height; see
`forward_step_setup.hpp`'s constructor for the full explanation. Purely
a visualization/metadata fix -- confirmed to change none of the
simulated physics, since `BlockInterfaceBoundary` matches the two
blocks by cell index, never by real-space coordinate; every resolution
reproduced the exact same `final_mass` before and after.)

Cost grows roughly like cells-per-unit-height cubed (cell count scales
as N^2, step count as N, for a fixed CFL and final time) for Rusanov;
the measured Rusanov-only cost ratio N=288/N=216 varied across this
session's own repeated runs (49.79s originally recorded when this
tutorial first shipped (task 0010); 71.39s and 88.13s on two later
re-runs today, same code, same machine, same resolution) -- a real,
visible run-to-run variance (plausibly thermal or OS-contention, per
Enzo's review of this task's own HLLC-vs-Rusanov comparison above) that
is itself a useful caution against over-reading any single wall-clock
measurement on this development machine, Rusanov's own numbers included.
Interestingly, at the smallest two resolutions the threaded backend is
slightly *slower* than serial would be (compare `docs/performance/`-style
single-core numbers from when this sweep was 20/40/80/96/144-only):
`std::thread` spawn/join overhead genuinely dominates when each parallel
region's actual work is tiny (res20 is 1,008 total cells). Not a bug --
the threaded backend already falls back toward a plain loop for very
small `n`, but the threshold where spawning becomes worthwhile is a real
machine-dependent crossover, not something this tutorial tunes.

## Regenerating the figures

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib
python3 plot_results.py
```

Writes `figures/density_time_series.png`, `figures/
schlieren_time_series.png`, `figures/resolution_comparison.png`, and
`figures/contact_kh_detail.png` (all four Rusanov-only, unchanged from
task 0010) plus two new ones from task 0011:
`figures/contact_kh_rusanov_vs_hllc.png` (the direct Rusanov-vs-HLLC
comparison at res80/res288) and
`figures/contact_kh_hllc_time_evolution.png` (HLLC's own trailing-
contact instability growing over time at res288, t=1.5 through t=4).

## What the figures show

![Density field at t=0, 1, 2, 4 (res80)](figures/density_time_series.png)

**Bow shock off the step corner:** visible from the very first frame
after t=0 -- a shock front curving away from the step's leading (upper
left) corner, sweeping up and back toward the inlet wall as it
propagates. By t=1 it has already reflected once off the tunnel's top
wall.

**Mach-stem reflection moving upstream:** the reflection point where the
bow shock meets the top wall is a classic three-shock (Mach reflection)
configuration -- visible as a near-vertical "stem" segment joining the
incident and reflected shocks at a triple point. Measured directly from
the res80 density field (the x-location of the steepest density jump
along the top-wall row, block 2), that triple point is at x~0.975 at
t=1, x~0.863 at t=1.5, x~0.787 at t=2, x~0.688 at t=3, and x~0.637 at
t=4 -- moving steadily **upstream** (back toward the step) as the run
progresses, not downstream. This matches the source paper's own
description exactly: Woodward & Colella report "a Mach stem forms at
about time 1.5 and moves very slowly upstream along the upper wall" --
the simulation reproduces that specific, named qualitative behavior, not
just "some reflection."

**Trailing contact discontinuity:** the numerical-schlieren view below
(`|grad(density)|`, which makes weak features far more visible than raw
density) shows a second, much fainter curve trailing downstream from the
triple point, distinct from both shocks -- the slip line/contact
discontinuity the three-shock Mach-reflection structure is expected to
shed. It is visibly less sharp than either shock (a contact has no
pressure jump to sharpen it the way a shock does, and Rusanov's own
dissipation is documented -- `numerics/numerical_flux/rusanov.hpp` -- as
most smearing exactly at a contact). **Correction (task 0011, caught by
a more careful side-by-side comparison against HLLC at the identical
resolution/time): an earlier version of this section described this
line as showing "faint waviness... consistent with a weakly
Kelvin-Helmholtz-unstable slip line" with Rusanov. A direct comparison
at matched resolution shows this was an overstatement -- Rusanov's
trailing line is a smooth, single, non-wavy curve at every resolution
tested, including res288; see "Looking for Kelvin-Helmholtz rollup"
below for the actual (HLLC-only) waviness finding and why this
correction matters.**

![Numerical schlieren at t=1, 2, 4 (res80)](figures/schlieren_time_series.png)

**Step-corner artifact (documented, not fixed):** zoom into the step
corner (x=0.6, y=0.2) in the schlieren view -- a small, localized dark
notch/extra feature sits right at the corner, distinct from the smooth
bow shock curve leaving it. This is the expected singular-point artifact
named in the task spec: the step's convex corner is a genuine
rarefaction-fan center in the exact solution, and a naive (non-special-
cased) corner treatment is known to produce a small spurious boundary-
layer-like feature / extra weak reflection there (Woodward & Colella's
own paper discusses a dedicated corner fix; replicating that fix exactly
was explicitly out of this task's scope). It stays localized -- it does
not visibly contaminate the bow shock or Mach-stem structure away from
the immediate corner neighborhood at any resolution tested.

![Resolution comparison at t=4](figures/resolution_comparison.png)

**Convergence illustration:** res20 shows a visibly smeared, blurry bow
shock/reflection structure with the triple point barely distinguishable
from the surrounding smear. res40 sharpens this considerably -- the
triple point and both shock legs are clearly separated, and a faint
trailing line is visible behind it. res80 sharpens further still -- a
distinctly thinner shock front, a cleanly separated triple point, and a
visibly longer/better-defined trailing contact line. res96, res144,
res216, and res288 (added when this sweep was extended past the
paper's own three reference grids) continue the same trend -- a
progressively thinner shock front and a trailing contact line that
extends further before losing definition -- with no new qualitative
behavior appearing beyond res80, consistent with the solution having
already visually settled onto a stable shape rather than still changing
character grid-to-grid. This is the expected qualitative refinement
behavior for MUSCL+Rusanov on a genuine shock-and-contact flow, not a
formal convergence-order claim (none is made here, matching the task's
own "qualitative checks, not a convergence table" framing).

![Trailing contact discontinuity, zoomed, res80 vs res288 (Rusanov)](figures/contact_kh_detail.png)

**Looking for Kelvin-Helmholtz rollup -- MUSCL+Rusanov (original
finding, unchanged):** pushing resolution from 80 to 288 (a 3.6x finer
grid) sharpens the trailing contact line substantially -- it goes from
a visibly smeared, fairly thick band at res80 to a thin, crisp line at
res288, with the same smooth near-horizontal shape. **No rollup/
billowing appears even at res288, with Rusanov.** Rusanov's numerical
dissipation is largest exactly at a contact discontinuity
(`numerics/numerical_flux/rusanov.hpp`'s own documented behavior), and
that dissipation can suppress a weak KH instability's growth regardless
of how finely the smeared region itself is resolved, since the
instability has to grow faster than the scheme damps it.

**Task 0011 follow-up: MUSCL+HLLC DOES show it.** HLLC
(`src/cfe/numerics/numerical_flux/hllc.hpp`, `docs/adr/0012-hllc-
numerical-flux.md`) was added specifically to test this hypothesis --
re-running the identical case (same grid, same IC/BCs, same MUSCL
reconstruction, only the numerical flux changed) at every resolution
already explored:

![Rusanov vs. HLLC, same zoom, same resolutions](figures/contact_kh_rusanov_vs_hllc.png)

- **At every resolution from res80 upward, HLLC's trailing contact line
  visibly splits into a wavy, crossing double-line structure that
  Rusanov never shows, at any resolution tested, including res288.**
  Not present at res20/res40 (too coarse to resolve at all -- both
  fluxes show only a single blurry band there); first clearly visible
  at res80 (the paper's own reference resolution) as a modest splitting
  of the line with a slight waviness; progressively more developed and
  complex at res96/144/216/288.
- **The feature GROWS over time, with onset timing matching the paper's
  own description almost exactly.** At res216 (and res288, see below),
  the trailing line is clean and single at t=1.5, shows a small kink
  right at the triple point by t=2.0, and has grown into a clearly
  wavy, repeatedly-crossing ("herringbone") pattern cascading downstream
  by t=2.5-3.0 -- Woodward & Colella's own words: "the mild Kelvin-
  Helmholtz instability ('water-wave' or 'flag-waving' instability) of
  the contact discontinuity along the upper wall... is visible at times
  2.5 and 3." This is the SAME qualitative character (a wavy, crossing
  pattern, not discrete rolled-up vortices) and the SAME onset timing
  this task's own re-run independently reproduces.

![HLLC trailing-contact instability over time, res288](figures/contact_kh_hllc_time_evolution.png)

**Why Rusanov never showed this and HLLC does:** both fluxes share the
exact same MUSCL reconstruction and the exact same upstream physics
(bow shock, Mach-stem/triple-point reflection), so the seed
perturbation Woodward & Colella describe ("triggered by very small
oscillations in the entropy behind the Mach shock") is plausibly
present in both runs -- but Rusanov's own documented excess dissipation
*at the contact specifically* damps that seed before it can grow into
anything visible, at any resolution, while HLLC's sharper contact
resolution (already demonstrated quantitatively on the Sod tube, see
`tutorials/euler_1d_sod_shock_tube/README.md`) lets it actually grow.
This matches the calibration Violet's pre-implementation consult gave
before any of this was run (`docs/adr/0012-...md`): "sharper/thinner
trailing contact line and/or measurably smaller numerical diffusion...
rather than 'visible KH rollup'" was the expected, defensible outcome
-- what was actually observed (a real, growing wavy instability matching
the paper's own timing) is a stronger result than that calibration
anticipated, not a weaker one.

**Honest calibration of what this is -- and is not.** This is "mild
waviness," matching the paper's own description, not the dramatic
rolled-up vortices sometimes associated with "Kelvin-Helmholtz rollup"
in other contexts -- Woodward & Colella's own paper never produced
dramatic rollup either (Section IV.b, quoted in this task's own scope
note) even with PPM, their highest-fidelity scheme for this problem, at
this same resolution range. No claim is made here that this feature is
fully grid-converged (a formal convergence study of an instability's
growth rate is a substantially bigger undertaking than this task's own
qualitative-comparison scope) or that it is free of any influence from
the step corner's own documented singular-point artifact (the seed
perturbation's exact source was not independently isolated here, only
observed to appear with HLLC and not Rusanov, at the same place and
time the paper itself reports). A genuinely higher-order reconstruction
scheme (PPM) was NOT needed to reproduce this specific qualitative
feature -- the flux alone was sufficient -- though PPM could plausibly
still sharpen or otherwise change its character further; per Amelia's
pre-implementation recommendation, pulling PPM forward is not
independently justified by this result and remains deferred.

## Known limitations

- Componentwise (not characteristic) reconstruction -- an already-
  documented, carried-over limitation from `docs/adr/0010-euler-field-
  and-shock-capturing.md`. Task 0011 found direct, reproducible
  evidence of this limitation's own named revisit criterion: at a
  strong 2D multi-wave interaction (the forward-step's bow-shock/Mach-
  stem region), componentwise MUSCL reconstruction combined with
  HLLC's lower dissipation produces a real, bounded density overshoot
  right at the triple point (measured: HLLC's peak density reaches
  ~7.1 at res80 vs. Rusanov's own ~6.3 at the same resolution/time, with
  HLLC's minimum also undershooting Rusanov's) -- finite, not NaN, and
  not conservation-violating (`final_mass` stays close to Rusanov's own
  at every resolution), but a genuine, visible local artifact distinct
  from the physical bow shock/Mach-stem structure itself. Characteristic
  reconstruction (or a lighter-weight positivity-preserving
  reconstruction fallback) remains the documented fix, not implemented
  in this task.
- HLLC (not PPM, per this task's scope) -- see "Looking for
  Kelvin-Helmholtz rollup" above for the real, reproduced mild-waviness
  finding this flux choice enabled. PPM was deliberately NOT pulled
  forward, per Amelia's pre-implementation recommendation: this
  result's own existence means Phase 3's target is met without it.
- The step-corner artifact above is observed, not fixed.
- CPU-only; see `tutorials/euler_3d_forward_step_extrusion/` for the 3D
  CPU "see it work" extrusion, and `docs/adr/0011-...md` / `agent_history.md`
  for why a CUDA 3D-correctness/at-scale benchmark remains a separate,
  still-outstanding requirement for this multi-block mechanism.
