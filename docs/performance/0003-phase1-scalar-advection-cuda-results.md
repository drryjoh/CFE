# Phase 1 scalar-advection CUDA benchmark results

Raw CSV: [`benchmarks/results/phase1_scalar_advection_v100.csv`](../../benchmarks/results/phase1_scalar_advection_v100.csv).

This is the "at scale" evidence requested for Phase 1
(`tasks/0002-phase1-cartesian-grid-scalar-transport.md`): the full FVM
scalar-advection solver (`FvmSolver` + `ScalarAdvectionField` +
`CentralDifferenceReconstruction` + `UpwindFlux` + SSP-RK2), running one
complete two-stage SSP-RK2 time step per measurement, on 1D grids from
one million up to one hundred million real cells.

## Kernel and methodology

`cfe_bench_scalar_advection_cuda`
(`benchmarks/scalar_advection/bench_scalar_advection_cuda.cu`): builds a
periodic 1D grid at each `nx`, seeds a sine-wave initial condition, then
times one full `ssp_rk2_step` call (ghost-fill + residual assembly, twice,
per the SSP-RK2 stage structure) via `cfe::CudaParallelFor`. Median of 10
timed repetitions after one untimed warm-up (first launch pays CUDA
context/JIT costs, matching the Phase 0 convention in
[`0002-phase0-cuda-results.md`](0002-phase0-cuda-results.md)).
`cell_updates_per_s := nx / median_step_time`.

## Environment

| | |
|---|---|
| Hardware | NVIDIA Tesla V100-SXM2-32GB (compute capability 7.0), PSC Bridges-2 `GPU-shared` partition, node `v016` |
| CUDA toolkit / host compiler | `cuda` + `gcc/13.3.1-p20240614` modules (same as Phase 0) |
| Build type | `CMAKE_BUILD_TYPE=Release`, `CFE_ENABLE_CUDA=ON`, `-DCMAKE_CUDA_ARCHITECTURES=70` |
| Date | 2026-09-29 |

Correctness was verified immediately before this run on the same
allocation: `test_scalar_advection_cuda_matches_cpu_reference`
(`tests/unit/test_scalar_advection_cuda.cu`) passed with all 48/48 unit
tests green — 256-cell grid, 50 SSP-RK2 steps, GPU matching the CPU
reference to `1e-9` cell-by-cell.

## Results

| n_cells | median ms/step | cell-updates/s |
|---|---|---|
| 1,000,000 | 0.140 | 7.12e9 |
| 10,000,000 | 1.153 | 8.68e9 |
| 50,000,000 | 5.630 | 8.88e9 |
| 100,000,000 | 11.342 | 8.82e9 |

### Observations

1. **Throughput plateaus almost immediately** — 1M cells already achieves
   7.1e9 cell-updates/s, and 10M-100M cells all land within ~2% of each
   other (8.68e9-8.88e9). The solver is memory-bandwidth-bound at every
   size tested here, not launch-overhead-bound, once the grid is large
   enough to fill the device (consistent with the occupancy analysis in
   Phase 0's CUDA results for a much simpler kernel).
2. **100 million cells complete one full SSP-RK2 step (two ghost-fills +
   two residual assemblies) in 11.3 ms.** This is the concrete "at scale"
   demonstration: a full period of this solver at a resolution far beyond
   anything the correctness/convergence tests exercise (`nx=20..256`)
   runs in real time on a single V100.
3. Each measured "step" does strictly more memory traffic than Phase 0's
   elementwise `q*q` kernel (two ghost-fills, each touching a small halo
   region, plus a 5-point-stencil residual read pattern per stage, twice)
   — so the achieved cell-updates/s is not directly comparable to Phase
   0's raw bandwidth numbers without accounting for that difference; it
   is reported here as this solver's own standalone throughput baseline.

## Bug this validates

This is also the run that confirmed the fix in commit `fa0cc88`
(`CFE_HOST_DEVICE` instead of `CFE_DEVICE` on every `Backend`-generic
kernel lambda in `ssp_rk2.hpp`, `boundary_condition.hpp`, and
`fvm_solver.hpp`): `CFE_DEVICE` expands to a device-*only* annotation
under nvcc, so a lambda meant to run under either `CpuParallelFor` (a
plain host loop) or `CudaParallelFor` (a real kernel launch) must be
`__host__ __device__`, not `__device__`-only — the device-only version
silently produced an all-zero residual when invoked from a host loop
inside a `.cu` translation unit, which is exactly what a diagnostic
single-residual-call comparison (CPU vs GPU, no time-stepping) caught:
the CPU side read as identically zero for a genuinely non-uniform
initial condition, which is not a plausible real answer. On plain
(non-CUDA) `g++` builds both macros expand to nothing, which is why this
was invisible across all 43 CPU-only unit tests throughout Phase 1's
development.

## Follow-ups not yet done

- A comparable CPU-only benchmark sweep for this same solver (smaller
  scale, matching `bench_field_update`'s CPU/GPU pairing convention) is
  not yet written.
- Multi-dimensional (2D/3D) and multi-block/AMR-relevant scaling are out
  of scope for Phase 1 (see the task file's PI direction: fixed
  refinement only, AMR-ready seams).
