# 1D compressible Euler: the Sod shock tube

Phase 3's companion 1D benchmark (ROADMAP.md) for CFE's first
genuinely multi-component Field, `EulerField` (`src/cfe/fields/euler/field.hpp`).

## What this demonstrates

A gamma-law (gamma=1.4) ideal gas, initially at rest, split by a
diaphragm at `x=0.5` in a `[0,1]` tube:

| | density | velocity | pressure |
|---|---|---|---|
| left (`x<0.5`) | 1.0 | 0.0 | 1.0 |
| right (`x>=0.5`) | 0.125 | 0.0 | 0.1 |

Removing the diaphragm produces, moving left to right: a rarefaction
fan, a contact discontinuity, and a shock. This is one of the most
widely used verification cases in compressible-flow CFD specifically
because its exact solution is known in closed form (Sod, 1978; Toro,
"Riemann Solvers and Numerical Methods for Fluid Dynamics," Ch. 4) --
this tutorial compares the numerical solution directly against that
exact solution (`tests/verification/sod_shock_tube_exact.hpp`, the same
oracle `tests/verification/test_euler_sod_shock_tube.cpp` verifies
against; this tutorial does not reimplement it).

Numerics: `EulerField<Scalar,1>` + `MusclMinmodReconstruction` (the
same minmod-limited MUSCL already used for Burgers, ADR 0008) +
`RusanovFlux`'s vector (system) overload + SSP-RK2, through the exact
same `FvmSolver`/`ssp_rk2_step` stack every other field in this
codebase uses (ADR 0010 covers how `FvmSolver` was generalized from a
single-component to an N-component Field to make this possible).
Boundary: both ends held at their own far-field state
(`StaticBoundary`) -- valid because, at `t=0.2`, neither wave reaches
the domain's edges (see the exact-solution header's comment for the
wave positions).

## Known limitation: Rusanov smears the contact discontinuity

A **contact discontinuity** is the middle wave of the three that form
here: density and temperature jump across it, but pressure and velocity
do not (physically, it is the boundary between the original left-hand
and right-hand gas -- like a line of dye marking which gas parcel came
from which side -- carried along at the flow speed `u*`, not a pressure
wave at all).

Rusanov (local Lax-Friedrichs) adds dissipation proportional to the
*largest* characteristic speed at a face, shared across every wave --
including the contact, whose own characteristic speed is `u*` (much
slower than the shock or rarefaction edges). Using the fast wave's
dissipation strength on the contact's slow wave over-smooths it, so the
contact ends up visibly more smeared than the shock. This is the scheme
behaving exactly as documented
(`src/cfe/numerics/numerical_flux/rusanov.hpp`'s own header comment),
not a bug -- a contact-resolving flux (HLLC) is tracked as still-open
future work (ADR 0010).

## Build and run

From the repo root (builds everything else too):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cfe_euler_1d_sod_shock_tube -j
cd tutorials/euler_1d_sod_shock_tube
../../build/tutorials/euler_1d_sod_shock_tube/cfe_euler_1d_sod_shock_tube
```

Or standalone, from this directory alone:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cfe_euler_1d_sod_shock_tube -j
./build/cfe-root-build/tutorials/euler_1d_sod_shock_tube/cfe_euler_1d_sod_shock_tube
```

(See `tutorials/burgers_1d_shock_and_steepening/README.md` for why both
paths exist -- the same `if(NOT TARGET cfe_core)` pattern every
tutorial's `CMakeLists.txt` uses, see `tutorials/CMakeLists.txt`'s own
comment.)

Writes `data/summary.csv` (L1 error vs. exact, at nx=100/200/400) and
`data/case_nx*.csv` (full density/velocity/pressure profiles vs. exact
at each resolution).

## Regenerating the figures

```bash
python3 -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install pandas matplotlib
python3 plot_results.py
```

Dependencies: `pandas`, `matplotlib` (same as the Burgers tutorials'
own plot scripts). Renders `figures/sod_profiles.png`
(finest-resolution profile vs. exact) and
`figures/sod_l1_convergence.png` (L1 error vs. resolution).
