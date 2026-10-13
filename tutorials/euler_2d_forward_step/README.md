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
- Three resolutions, matching the paper's own reference grids: cells-
  per-unit-height 20/40/80 (equivalently "20x60"/"40x120"/"80x240" for the
  full 1x3 bounding box) -- PPM itself is not required/used here (the
  paper's own "essentially converged" grid is for PPM specifically; this
  tutorial uses MUSCL+Rusanov throughout, per the task's own scope).
- Run to t=4 (unsteady until roughly this time, per the paper and this
  task's own spec).

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cfe_euler_2d_forward_step -j
cd tutorials/euler_2d_forward_step && ../../build/tutorials/euler_2d_forward_step/cfe_euler_2d_forward_step
```

Writes `data/res{20,40,80}/frame_NNNN_{density,pressure}_block{1,2}.vtk`
(ASCII legacy VTK, one pair of files per block per snapshot, every
`dt=0.1`) plus `data/res{20,40,80}/summary.csv` (step count, wall-clock
time, final total mass). All three resolutions together take well under
ten seconds on a single core of this development machine (Apple Silicon)
-- this case is cheap enough that there was no need to run it as a
background job, unlike the longer MPI strong-scaling tutorials.

## Regenerating the figures

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib
python3 plot_results.py
```

Writes `figures/density_time_series.png`, `figures/
schlieren_time_series.png`, `figures/resolution_comparison.png`.

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
visibly longer/better-defined trailing contact line. This is the
expected qualitative refinement behavior for MUSCL+Rusanov on a genuine
shock-and-contact flow, not a formal convergence-order claim (none is
made here, matching the task's own "qualitative checks, not a
convergence table" framing).

## Known limitations

- MUSCL+Rusanov (not PPM, per this task's own scope) and componentwise
  (not characteristic) reconstruction -- both already-documented,
  carried-over limitations from `docs/adr/0010-euler-field-and-shock-
  capturing.md`, not new to this case.
- The step-corner artifact above is observed, not fixed.
- CPU-only; see `tutorials/euler_3d_forward_step_extrusion/` for the 3D
  CPU "see it work" extrusion, and `docs/adr/0011-...md` / `agent_history.md`
  for why a CUDA 3D-correctness/at-scale benchmark remains a separate,
  still-outstanding requirement for this multi-block mechanism.
