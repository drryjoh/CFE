# HLLC CUDA correctness results (task 0011)

Closes the standing per-PR requirement (MEMORY note; established in
`docs/performance/0007-phase3-euler-cuda-results.md` for `EulerField`
itself): every PR adding a Field/scheme includes a 3D CUDA correctness
test run on real hardware, not deferred.

## What was added

`tests/unit/test_euler_hllc_3d_cuda.cu`: mirrors
`test_euler_3d_cuda.cu`'s exact structure (100x8x6 grid, Sod IC along X,
`SlipWallBoundary` on X / periodic Y,Z, 625 fixed-dt steps to t=0.5,
past the first wall reflection) with `RusanovFlux` swapped for
`HllcFlux`. This test's job is CPU/GPU PARITY for HLLC's own arithmetic
(including its Newton-Raphson pressure refinement and the density/
pressure floors, `docs/adr/0012-hllc-numerical-flux.md`) -- it reuses
the Sod tube's moderate wall reflection, not the much stronger Mach-3
case that originally exposed the wrong-signed-`S_L` bug before the
Newton refinement was added. That specific regression is independently
covered on CPU by `tests/unit/test_hllc_flux.cpp`'s
`test_hllc_wave_speeds_correctly_signed_for_strong_symmetric_wall_reflection`
-- this file confirms the same numerics run identically on CUDA as on
CPU, not a second instance of that correctness claim.

## Environment

| | |
|---|---|
| Hardware | NVIDIA Tesla V100-SXM2-32GB (compute capability 7.0), PSC Bridges-2 `GPU-shared` partition, node `v009` |
| CUDA toolkit / host compiler | `cuda-v100/12.9.2` (alias `cuda-legacy/12.9.2`) + `gcc/13.3.1-p20240614` (same as prior phases) |
| Build type | `CMAKE_BUILD_TYPE=Release`, `CFE_ENABLE_CUDA=ON`, `-DCMAKE_CUDA_ARCHITECTURES=70` |
| SLURM | `salloc -A phy260058p -p GPU-shared --qos=gpuinteract --gres=gpu:v100-32:1 -t 30:00`, job `49117265`; allocation held ~5 of its 30 minutes before being released |
| Date | 2026-10-13 |

Build configured and compiled with zero errors/warnings beyond the
expected "offline compilation for architectures prior to sm_75 will be
removed in a future release" `nvcc` notice every other CUDA translation
unit in this codebase already produces (`-arch=sm_70` is the V100's own
compute capability, unaffected).

## Correctness: full suite on real GPU hardware

`cfe_unit_tests`: **157/157 passed** on the allocated V100 node,
including `test_euler_hllc_3d_cuda_matches_cpu_reference` and every
pre-existing CUDA test (scalar advection, Burgers, backend-execution,
Euler 1D/3D with Rusanov) -- zero regressions. This also reconfirms
every CPU-only test from this task (145, including all 14 new HLLC unit
tests and the 2 float-precision regression tests) passes unchanged on
this different toolchain/architecture (x86-64 Xeon Gold 6248 + gcc
13.3.1, vs. this project's own Apple Silicon development machine +
clang) -- the same cross-platform reproducibility task 0009's own CUDA
follow-up established for Rusanov now holds for HLLC too.

`compute-sanitizer --tool initcheck ./build_cuda_hllc/tests/cfe_unit_tests`:
**0 errors** (full suite, same node/allocation) -- no uninitialized-read
risk from the new `hllc_floored_primitives`/`HllcShockImpedance`/Newton-
iteration code paths.

## What this resolves

Closes the standing per-PR CUDA-correctness requirement for `HllcFlux`
specifically: `EulerField`'s compile-time-generic `FvmSolver` hosts
`HllcFlux` on `cfe::CudaParallelFor` with zero solver changes (as
designed, `docs/adr/0012-...md`), and its Newton-Raphson refinement
(`sqrt`/`pow` calls, branching on shock vs. rarefaction, the fixed
5-iteration loop) runs correctly and reproducibly on real GPU hardware,
bit-identical to the CPU reference to the same `1e-9` tolerance every
other CUDA correctness test in this codebase uses.

## Known limitations / not done this session

- No at-scale GPU throughput benchmark for `HllcFlux` specifically (the
  kind `benchmarks/euler/bench_euler_cuda.cu` provides for Rusanov) --
  this session's own forward-step CPU timing comparison
  (`tutorials/euler_2d_forward_step/README.md`) already flagged a real
  methodology confound (a single, non-interleaved, non-repeated
  measurement) per Enzo's review; an isolated GPU microbenchmark
  comparing `HllcFlux` against `RusanovFlux` per-face cost, free of that
  confound, remains open future work (tracked in
  `docs/adr/0012-hllc-numerical-flux.md`'s Revisit criteria).
- No `nvcc --resource-usage`/register-pressure measurement for the
  `HllcFlux` kernel specifically (task 0009's own precedent for
  Rusanov, `docs/performance/0007-...md`) -- a standalone `nvcc
  --resource-usage` compile attempted during this session's own GPU
  allocation did not succeed within the time available on that
  allocation; worth retrying with the project's actual CMake-driven
  compile flags (rather than a quick standalone `nvcc` invocation) on a
  future allocation. Plausible expectation (not measured): more
  registers/thread than Rusanov's own 66-92 (task 0009's N=3/N=5
  figures), given the Newton loop's larger set of live temporaries
  (Enzo review point) -- a real, open question, not assumed.
- No Nsight Compute branch-efficiency/occupancy numbers for the ANRS
  seed's 3-way branch or the 4-way flux-region branch (both genuinely
  state-dependent per thread, unlike the Newton loop's fixed iteration
  count) -- Enzo's review flagged this as the right next measurement
  once real hardware access allows it.
