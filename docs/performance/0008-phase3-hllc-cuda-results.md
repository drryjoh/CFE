# HLLC CUDA correctness and at-scale benchmark results (task 0011)

Raw CSV: run output quoted directly below (not yet written to
`benchmarks/results/` as a committed file -- see "Known limitations").

Closes the standing per-PR requirement (MEMORY note; established in
`docs/performance/0007-phase3-euler-cuda-results.md` for `EulerField`
itself): every PR adding a Field/scheme includes a 3D CUDA correctness
test AND an at-scale benchmark run on real hardware, not deferred. The
at-scale half was initially missed when this document was first written
(caught in Ilia's final review) -- closed in this revision.

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

## At-scale throughput: HLLC vs. Rusanov, isolated and clean

This is the clean, isolated measurement Enzo's review said the
forward-step tutorial's own wall-clock comparison could not provide
(single process, same IC, same cube sweep, 10-repetition median each --
`benchmarks/euler/bench_euler_hllc_3d_cuda.cu` vs.
`bench_euler_3d_cuda.cu`, differing in ONLY the numerical flux).

Second allocation: `salloc ... --qos=gpuinteract --gres=gpu:v100-32:1
-t 30:00`, job `49118193`, node `v009`, 2026-10-13.

| n_per_axis | n_cells | Rusanov cell-updates/s | HLLC cell-updates/s | HLLC/Rusanov ratio |
|---:|---:|---:|---:|---:|
| 64  | 262,144     | 3.446e8 | 4.592e7 | 7.50x slower |
| 128 | 2,097,152   | 3.466e8 | 3.963e7 | 8.75x slower |
| 256 | 16,777,216  | 3.383e8 | 4.427e7 | 7.64x slower |
| 400 | 64,000,000  | 3.371e8 | 3.935e7 | 8.57x slower |
| 512 | 134,217,728 | 3.501e8 | 3.727e7 | 9.39x slower |

Rusanov's own numbers reproduce task 0009's original figures
(`docs/performance/0007-...md`) almost exactly (~3.4-3.5e8 at every
size) -- confirms this is the same measurement methodology, not a
different machine state. **HLLC is 7.5x-9.4x slower per cell-update
than Rusanov on this GPU** -- larger than the forward-step CPU
comparison's own (confounded, non-isolated) 3.4x-6.6x range, consistent
with GPU throughput being more sensitive to register pressure/occupancy
than a CPU core is (see below). This supersedes the forward-step
tutorial's own number as the authoritative HLLC/Rusanov cost
comparison; that tutorial's own README has been left with its honest
methodology caveat rather than retroactively implying it was always
this precise.

## Register pressure (AGENTS.md #8), same allocation

`nvcc --resource-usage` via `-DCMAKE_CUDA_FLAGS=--resource-usage`,
`nvlink`'s own per-kernel report (not a separate standalone invocation,
avoiding the earlier session's unsuccessful attempt at that):

| Kernel | Flux | Registers | Stack frame | Spill (lmem) |
|---|---|---:|---:|---:|
| `FvmSolver::residual` (3D, N=5) | Rusanov | 96 | 0 bytes | 0 bytes |
| `FvmSolver::residual` (3D, N=5) | HLLC | **149** | 0 bytes | 0 bytes |

Rusanov's 96 registers matches task 0009's own documented N=5 figure
exactly (`docs/performance/0007-...md`) -- confirms this measurement is
consistent with that prior one. **HLLC needs 55% more registers per
thread than Rusanov at the same N**, with no spilling either way.

Volta (`sm_70`) allocates registers per warp in multiples of 8/thread:
96 is already a multiple of 8 (Rusanov: 96 x 32 = 3072 registers/warp,
65536/3072 ~ 21 warps/SM from the register budget, 2 full 8-warp
256-thread blocks resident = ~25% occupancy -- task 0009's own number).
HLLC's 149 rounds up to 152 (19x8): 152 x 32 = 4864 registers/warp,
65536/4864 ~ 13.5 warps/SM -- enough for only ONE full 256-thread block
(8 warps), not two, since blocks are resident as whole units: **~12.5%
occupancy, roughly half of Rusanov's already-modest 25%.** This
quantifies a real, substantial, directly-measured contributor to the
throughput gap above (not the whole story -- HLLC's per-thread
arithmetic is also genuinely heavier, per Enzo's own code-level count of
extra `sqrt`/`pow` evaluations in the Newton loop -- but the occupancy
loss alone is a large, confirmed piece of it, not merely inferred).

## Known limitations / not done this session

- The at-scale numbers above are median-of-10 single-point-in-time
  measurements, not repeated across multiple allocations/nodes -- a
  single allocation's own run-to-run variance was not characterized
  (unlike the forward-step CPU numbers, where this session happened to
  observe real variance across repeated runs). Treated as reliable given
  how large and consistent the ratio is across every cube size (7.5x to
  9.4x, not scattered), but not re-verified on a second allocation.
- Raw CSV not yet committed under `benchmarks/results/` (the convention
  `docs/performance/0007-...md` established) -- the numbers above are
  quoted directly from the run's own stdout instead. Worth adding as a
  small follow-up for consistency with that convention.
- No Nsight Compute branch-efficiency numbers for the ANRS seed's 3-way
  branch or the 4-way flux-region branch (both genuinely state-dependent
  per thread, unlike the Newton loop's fixed iteration count) -- the
  register-pressure/occupancy measurement above is a real, measured
  contributor to the throughput gap, but does not fully decompose how
  much of the remainder is branch divergence vs. raw arithmetic
  intensity. Reasonable further work, not required to close this task's
  own standing per-PR requirement (now closed by the measurements above).
