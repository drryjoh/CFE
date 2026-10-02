# Phase 2 Burgers CUDA benchmark results (and CPU baseline)

Raw CSV (1D, GPU): [`benchmarks/results/phase2_burgers_v100.csv`](../../benchmarks/results/phase2_burgers_v100.csv).
Raw CSV (3D, GPU): [`benchmarks/results/phase2_burgers_3d_v100.csv`](../../benchmarks/results/phase2_burgers_3d_v100.csv).
Raw CSV (CPU baseline): [`benchmarks/results/phase2_burgers_cpu_apple_m5.csv`](../../benchmarks/results/phase2_burgers_cpu_apple_m5.csv).

This is the "at scale, on GPU" evidence requested as a direct follow-up
to task 0003 (Burgers equation + shock-capturing numerics, PR #3): the
full Burgers FVM solver (`FvmSolver` + `BurgersField` +
`MusclMinmodReconstruction` + `RusanovFlux` + SSP-RK2), running one
complete two-stage SSP-RK2 time step per measurement, on 1D grids from
one million up to one hundred million real cells, and on 3D cube grids
up to 512^3 (~134 million cells) -- mirroring
[`0003-phase1-scalar-advection-cuda-results.md`](0003-phase1-scalar-advection-cuda-results.md)'s
own methodology exactly, for the same equation-family comparison.

## Kernel and methodology

`cfe_bench_burgers_cuda` / `cfe_bench_burgers_3d_cuda`
(`benchmarks/burgers/bench_burgers_cuda.cu` / `bench_burgers_3d_cuda.cu`):
builds a periodic grid, seeds a smooth sine-based initial condition
(see those files' own header comments for why a smooth IC rather than
the shock-formation test's Riemann step is used for a throughput
benchmark), then times one full `ssp_rk2_step` call (ghost-fill +
residual assembly, twice, per the SSP-RK2 stage structure) via
`cfe::CudaParallelFor`. Median of 10 timed repetitions after one untimed
warm-up (first launch pays CUDA context/JIT costs), matching every
other CUDA benchmark in this codebase.

## Environment

| | |
|---|---|
| Hardware | NVIDIA Tesla V100-SXM2-32GB (compute capability 7.0), PSC Bridges-2 `GPU-shared` partition, node `v009` |
| CUDA toolkit / host compiler | `cuda-v100/12.9.2` + `gcc/13.3.1-p20240614` modules (same as Phase 0/1) |
| Build type | `CMAKE_BUILD_TYPE=Release`, `CFE_ENABLE_CUDA=ON`, `-DCMAKE_CUDA_ARCHITECTURES=70` |
| Date | 2026-10-02 |

Correctness was verified immediately before this run, on the same
allocation: all 87/87 unit tests passed, including the two new CUDA
correctness tests added specifically for this follow-up --
`test_burgers_cuda_matches_cpu_reference` (1D, 400 cells, 400 steps) and
`test_burgers_3d_cuda_matches_cpu_reference` (3D, 96^3 cells, 200
steps) -- GPU matching the CPU reference to `1e-9` cell-by-cell, both
using the exact shock-formation Riemann setup
`test_burgers_shock_formation.cpp`/`test_burgers_3d_sanity.cpp` already
verify on CPU. `compute-sanitizer --tool initcheck` run over the full
test suite immediately after: **0 errors**.

## Results: 1D, up to 10^8 cells

| n_cells | median ms/step | cell-updates/s |
|---|---|---|
| 1,000,000 | 0.164 | 6.10e9 |
| 10,000,000 | 1.378 | 7.26e9 |
| 50,000,000 | 6.778 | 7.38e9 |
| 100,000,000 | 12.997 | 7.69e9 |

Compare to scalar advection's own 1D V100 numbers
(`0003-...md`): at 10^8 cells, scalar advection reaches ~8.8e9
cell-updates/s vs. Burgers' 7.69e9 here -- Burgers does genuinely more
per-cell work (a `minmod` branch per face per axis, a nonlinear flux
evaluation, and the Rusanov dissipation term, vs. scalar advection's
single multiply-and-compare), so a modest (~13%) throughput reduction
at the same scale is the expected signature of that extra work, not a
regression.

## Results: 3D, up to 512^3 (~134M) cells

| n_per_axis | n_cells | median ms/step | cell-updates/s |
|---|---|---|---|
| 64 | 262,144 | 0.088 | 2.99e9 |
| 128 | 2,097,152 | 0.529 | 3.96e9 |
| 256 | 16,777,216 | 3.931 | 4.27e9 |
| 400 | 64,000,000 | 16.558 | 3.87e9 |
| 512 | 134,217,728 | 33.134 | 4.05e9 |

Compare to scalar advection's own 3D V100 numbers (`0003-...md`,
~4.4-5.7e9 cell-updates/s across the same resolution range): Burgers
lands modestly below, consistent with the same per-cell-work
explanation as the 1D case, and with all three of X/Y/Z's
`axis_flux_difference` branches active every step (same dimension-
generic residual loop Phase 1 built, now exercising
`BurgersField`/`RusanovFlux`/`MusclMinmodReconstruction` specifically
rather than `ScalarAdvectionField`/`UpwindFlux`/
`CentralDifferenceReconstruction`).

## CPU baseline (not the "at scale" claim -- see above for that)

| n_cells | serial ms/step | serial cell-updates/s | threaded ms/step | threaded cell-updates/s |
|---|---|---|---|---|
| 10,000 | 0.261 | 3.83e7 | 0.472 | 2.12e7 |
| 100,000 | 0.523 | 1.91e8 | 0.460 | 2.17e8 |
| 1,000,000 | 5.267 | 1.90e8 | 1.685 | 5.94e8 |
| 10,000,000 | 52.108 | 1.92e8 | 16.265 | 6.15e8 |

Apple M5, same methodology as
[`0004-phase1-scalar-advection-cpu-results.md`](0004-phase1-scalar-advection-cpu-results.md).
At 10M cells, Burgers' serial throughput (1.92e8/s) is noticeably below
scalar advection's post-regression serial number (0004's own ~3.03e8/s
baseline before the documented ~1.68x regression) -- again consistent
with Burgers' extra per-cell work (minmod branching defeats some
auto-vectorization the way index-mapping already does, per 0004/0005's
own findings; this was not investigated further here since GPU is this
codebase's actual "at scale" target and CPU serial throughput is
already a documented, accepted limitation for the equivalent linear
case).

## What this resolves

Closes the "is there a 3D Burgers test we can run on GPU and scale it"
follow-up raised during PR #3's review: yes, both 1D and 3D CUDA
correctness are verified (`test_burgers_cuda.cu` /
`test_burgers_3d_cuda.cu`), and both are benchmarked at the same scale
Phase 1 established for scalar advection (10^8 cells 1D, 512^3 cells
3D), on the same V100 hardware.
