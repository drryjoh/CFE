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
- MUSCL (minmod) + Rusanov, SSP-RK2, adaptive CFL=0.4.
- Seven resolutions: cells-per-unit-height 20/40/80 (the paper's own
  reference grids, equivalently "20x60"/"40x120"/"80x240" for the full
  1x3 bounding box -- PPM itself is not required/used here, since the
  paper's own "essentially converged" grid is for PPM specifically and
  this tutorial uses MUSCL+Rusanov throughout, per the task's own scope),
  plus 96, 144, 216, and 288 (each 1.5x the last) extending the sweep
  further for a finer convergence illustration than the paper's three
  points alone give, and specifically to check whether finer resolution
  reveals Kelvin-Helmholtz rollup on the trailing contact discontinuity
  that isn't visible at the paper's own resolutions -- see "Looking for
  Kelvin-Helmholtz rollup" below.
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

Writes `data/res{20,40,80,96,144,216,288}/frame_NNNN_{density,pressure}_block{1,2}.vtk`
(ASCII legacy VTK, one pair of files per block per snapshot, every
`dt=0.1`) plus `data/res{20,40,80,96,144,216,288}/summary.csv` (step
count, wall-clock time, final total mass).

Measured wall-clock time per resolution (threaded backend, this
development machine, Apple Silicon M5, 10 logical cores; from
`data/res<N>/summary.csv`, written by the run itself -- not estimated):

| cells-per-unit-height | block1 cells | block2 cells | steps  | wall-clock |
|---:|---:|---:|---:|---:|
| 20  | 240    | 768    | 821    | 0.78s |
| 40  | 960    | 3,072  | 1,645  | 1.65s |
| 80  | 3,840  | 12,288 | 3,302  | 4.14s |
| 96  | 5,568  | 17,710 | 3,966  | 5.37s |
| 144 | 12,384 | 39,790 | 5,972  | 11.68s |
| 216 | 27,690 | 89,614 | 8,990  | 25.35s |
| 288 | 49,824 | 158,930| 12,025 | 49.79s |

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
as N^2, step count as N, for a fixed CFL and final time): N=288 is 1.5x
N=216, and the measured cost ratio is 49.79/25.35 = 1.96x -- well below
the naive 1.5^3 = 3.375x prediction, because the threaded backend's
per-call overhead is amortized better at larger problem sizes (more
work per spawned thread). Interestingly, at the smallest two
resolutions the threaded backend is slightly *slower* than serial would
be (compare `docs/performance/`-style single-core numbers from when
this sweep was 20/40/80/96/144-only): `std::thread` spawn/join overhead
genuinely dominates when each parallel region's actual work is tiny
(res20 is 1,008 total cells). Not a bug -- the threaded backend already
falls back toward a plain loop for very small `n`, but the threshold
where spawning becomes worthwhile is a real machine-dependent
crossover, not something this tutorial tunes. The full seven-resolution
sweep (threaded) takes under two minutes total.

## Regenerating the figures

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib
python3 plot_results.py
```

Writes `figures/density_time_series.png`, `figures/
schlieren_time_series.png`, `figures/resolution_comparison.png`, and
`figures/contact_kh_detail.png`.

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
most smearing exactly at a contact), consistent with the task's own
expectation of "mild KH waviness at finer resolution": at res80 this
line shows faint waviness rather than a perfectly smooth curve,
consistent with a (weakly) Kelvin-Helmholtz-unstable slip line, though
not a dramatic rollup at this resolution/this final time.

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

![Trailing contact discontinuity, zoomed, res80 vs res288](figures/contact_kh_detail.png)

**Looking for Kelvin-Helmholtz rollup:** pushing resolution from 80 to
288 (a 3.6x finer grid) sharpens the trailing contact line substantially
-- it goes from a visibly smeared, fairly thick band at res80 to a
thin, crisp line at res288, with the same smooth near-horizontal shape.
**No rollup/billowing appears even at res288.** This is a real,
honestly-reported negative result, not necessarily a resolution
limitation alone: Rusanov's numerical dissipation is largest exactly at
a contact discontinuity (`numerics/numerical_flux/rusanov.hpp`'s own
documented behavior), and that dissipation can suppress a weak KH
instability's growth regardless of how finely the smeared region itself
is resolved, since the instability has to grow faster than the scheme
damps it. The natural next experiment is a less-dissipative-at-contacts
numerical flux (HLLC, already named as a "Rusanov and/or HLLC" option in
`ROADMAP.md`'s Phase 3 scope) before concluding rollup requires a
higher-order reconstruction scheme (PPM) -- see `agent_history.md` for
the fuller discussion of why HLLC is the right next step to try first.

## Known limitations

- MUSCL+Rusanov (not PPM, per this task's own scope) and componentwise
  (not characteristic) reconstruction -- both already-documented,
  carried-over limitations from `docs/adr/0010-euler-field-and-shock-
  capturing.md`, not new to this case. The resolution sweep up to 288
  shows this concretely: no KH rollup appears on the trailing contact
  even at the finest resolution tested, consistent with Rusanov's own
  documented contact-smearing dissipation (see "Looking for
  Kelvin-Helmholtz rollup" above) -- HLLC (a less-dissipative-at-
  contacts flux, already named as a Phase 3 option) is the next thing
  to try before concluding PPM's higher-order reconstruction is needed.
- The step-corner artifact above is observed, not fixed.
- CPU-only; see `tutorials/euler_3d_forward_step_extrusion/` for the 3D
  CPU "see it work" extrusion, and `docs/adr/0011-...md` / `agent_history.md`
  for why a CUDA 3D-correctness/at-scale benchmark remains a separate,
  still-outstanding requirement for this multi-block mechanism.
