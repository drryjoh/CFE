# Tutorial: 3D Burgers visualization

Task 0003's correctness tests and benchmarks
(`tests/unit/test_burgers_3d_cuda.cu`,
`benchmarks/burgers/bench_burgers_3d_cuda.cu`) prove the 3D Burgers
solver is correct and fast, but neither writes anything to disk to
actually look at. This tutorial runs the identical solver (`FvmSolver` +
`BurgersField<Scalar,3>` + `MusclMinmodReconstruction` + `RusanovFlux` +
SSP-RK2, periodic boundaries) and dumps a time series you can open in
ParaView or VisIt.

## What it does

A Gaussian bump, `state = 1.0 + 0.5*exp(-r^2 / (2*sigma^2))` centered in
a unit cube (`sigma = 0.12`), evolves under the inviscid Burgers equation
on a periodic `64^3` grid for 500 SSP-RK2 steps, writing a snapshot
every 10 steps (51 frames total, including the initial condition).
CPU-only and deliberately small — this is for visualization, not the
performance measurement (see the benchmark above for that).

**This is not the same phenomenon as the linear scalar-advection
tutorial** (`tutorials/scalar_advection_3d_visualization/`): there, the
bump translates through the domain unchanged. Here, Burgers'
characteristic speed is the local state value itself, so a cell inside
the bump moves faster than the background ahead of it. The result:
- the bump's leading faces (in whatever direction a cell's neighbor
  ahead has a *lower* state value) steepen into genuine shocks, the same
  phenomenon `tests/unit/test_burgers_shock_formation.cpp` verifies
  against the exact Rankine-Hugoniot solution in 1D — here you can
  *watch* it happen in 3D;
- the trailing faces spread into a smooth rarefaction fan instead,
  since those characteristics diverge rather than converge.

This happens independently along all three axes at once (Burgers' flux
formula is the same scalar function of the local state on every axis),
so watch for the bump becoming visibly asymmetric over the run — steep
on one side, smeared out on the other — rather than just drifting.

Output goes to `vtk_output/` (created next to wherever you run the
binary from):
- `frame_0000.vtk` .. `frame_0050.vtk` — legacy VTK `STRUCTURED_POINTS`,
  ASCII, one cell-centered scalar field `state` each (see
  `src/cfe/io/vtk_writer.hpp`).
- `series.pvd` — a manifest referencing every frame with its simulation
  time, so ParaView can load the whole run as one time series with a
  slider instead of opening 51 files by hand.

## Build and run

From the repo root (builds everything else too):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cfe_advect_burgers_gaussian_3d -j
./build/tutorials/burgers_3d_visualization/cfe_advect_burgers_gaussian_3d
```

Or standalone, from this directory alone:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cfe_advect_burgers_gaussian_3d -j
./build/cfe-root-build/tutorials/burgers_3d_visualization/cfe_advect_burgers_gaussian_3d
```

The `--target` is required even in the standalone case: pulling in the
repo root as a nested subdirectory (to obtain `cfe_core`) also makes the
*whole* project's targets available to build — every test, every
benchmark, every other tutorial — since their `CFE_BUILD_TESTS`/
`CFE_BUILD_BENCHMARKS`/`CFE_BUILD_TUTORIALS` options default `ON`
regardless of which directory you configured from. `cmake --build build
-j` with no `--target` builds CMake's default `all` target, which is all
of that, not just this one tutorial. `cmake -S . -B build` itself also
configures (but does not build) that whole tree either way — only the
`--target`-qualified *build* step is scoped to just this executable.

Takes well under a second on a laptop (64^3 = 262,144 cells, CPU serial
backend by default).

## Viewing it

Open `vtk_output/series.pvd` in ParaView (File > Open), apply a
`Threshold` or `Contour` filter on `state` (a raw volume render also
works), and press play. Watch the console output's printed state range
too — it should stay bounded within roughly `[1.0, 1.5]` (the
background and peak values) for the whole run: Burgers' shock-capturing
scheme is TVD (`numerics/fvm/muscl_minmod.hpp`), so no overshoot above
the initial peak or undershoot below the initial background should ever
appear, the same guarantee `test_burgers_shock_formation.cpp` verifies
numerically in 1D.

## Things to try

- **Change the background/bump amplitude** (`kBackground`,
  `kBumpAmplitude` in `advect_burgers_gaussian_3d.cpp`) — a larger bump
  relative to the background steepens into a shock faster (shorter
  effective "breaking time", the same concept
  `tests/unit/test_burgers_convergence.cpp` computes analytically for
  its own 1D sine case).
- **Change the bump width/position** (`kBumpSigma`, `kBumpCenter`) — a
  narrower bump steepens faster, relative to the grid, since the
  per-cell state gradient driving the steepening is larger to start
  with.
- **Increase `kN`** for a sharper-looking result (the CUDA benchmark
  already demonstrates this same solver scales to 512^3; this tutorial
  stays small on purpose so it runs instantly on a laptop with no GPU).

## Where to go next

- For the correctness proof this tutorial's solver setup is drawn from:
  `tests/unit/test_burgers_3d_cuda.cu` and
  `tests/unit/test_burgers_shock_formation.cpp` (the exact 1D analytic
  shock-speed check this 3D case is a qualitative, visual extension of).
- For the at-scale GPU performance numbers:
  `docs/performance/0006-phase2-burgers-cuda-results.md`.
- For the scheme-choice rationale (why minmod + Rusanov, not a sharper
  limiter): `docs/adr/0008-burgers-shock-capturing-scheme.md`.
- For the writer itself: `src/cfe/io/vtk_writer.hpp`.
