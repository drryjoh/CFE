# Tutorial: 3D scalar-advection visualization

Phase 1's correctness tests and benchmarks (`tests/unit/test_scalar_advection_3d_cuda.cu`,
`benchmarks/scalar_advection/bench_scalar_advection_3d_cuda.cu`) prove the
3D solver is correct and fast, but neither writes anything to disk to
actually look at. This tutorial runs the identical solver
(`FvmSolver` + `ScalarAdvectionField<Scalar,3>` + `CentralDifferenceReconstruction`
+ `UpwindFlux` + SSP-RK2, periodic boundaries) and dumps a time series you
can open in ParaView or VisIt.

## What it does

A Gaussian bump, `state = exp(-r^2 / (2*sigma^2))` centered near one corner
of a unit cube (`sigma = 0.08`), advects with velocity `(0.5, 0.3, 0.2)`
on a periodic `64^3` grid for 640 SSP-RK2 steps, writing a snapshot every
16 steps (41 frames total, including the initial condition). CPU-only
and deliberately small — this is for visualization, not the performance
measurement (see the CUDA benchmark above for that).

Output goes to `vtk_output/` (created next to wherever you run the
binary from):
- `frame_0000.vtk` .. `frame_0040.vtk` — legacy VTK `STRUCTURED_POINTS`,
  ASCII, one cell-centered scalar field `state` each (see `src/cfe/io/vtk_writer.hpp`).
- `series.pvd` — a manifest referencing every frame with its simulation
  time, so ParaView can load the whole run as one time series with a
  slider instead of opening 41 files by hand.

## Build and run

From the repo root (builds everything else too):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cfe_advect_gaussian_3d -j
./build/tutorials/scalar_advection_3d_visualization/cfe_advect_gaussian_3d
```

Or standalone, from this directory alone (nothing else gets built):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/cfe-root-build/tutorials/scalar_advection_3d_visualization/cfe_advect_gaussian_3d
```

Takes well under a second on a laptop (64^3 = 262,144 cells, CPU serial
backend by default).

## Viewing it

Open `vtk_output/series.pvd` in ParaView (File > Open), apply a
`Threshold` or `Contour` filter on `state` (a raw volume render also works),
and press play — the bump should visibly translate diagonally through
the domain and wrap around the periodic boundary. `series.pvd`'s
timesteps let ParaView's time controls work directly; each individual
`frame_*.vtk` also opens standalone if you just want one snapshot.

## Things to try

- **Change the velocity or bump position/width** in `advect_gaussian_3d.cpp`
  (`kUx/kUy/kUz`, `kBumpCenter`, `kBumpSigma`) and re-run — the bump's
  path and shape should change accordingly, with no other code changes
  needed (this is the same `Field`/`Reconstruction`/`NumericalFlux`
  genericity the unit tests exercise).
- **Increase `kN`** for a sharper-looking bump (the CUDA benchmark
  already demonstrates this same solver scales to 512^3; this tutorial
  stays small on purpose so it runs instantly on a laptop with no GPU).

## Where to go next

- For the correctness proof this tutorial's solver setup is drawn from:
  `tests/unit/test_scalar_advection_3d_cuda.cu`.
- For the at-scale GPU performance numbers: `docs/performance/0003-phase1-scalar-advection-cuda-results.md`.
- For the writer itself: `src/cfe/io/vtk_writer.hpp`.
