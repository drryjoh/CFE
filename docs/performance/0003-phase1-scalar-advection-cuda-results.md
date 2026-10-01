# Phase 1 scalar-advection CUDA benchmark results

Raw CSV (1D): [`benchmarks/results/phase1_scalar_advection_v100.csv`](../../benchmarks/results/phase1_scalar_advection_v100.csv).
Raw CSV (3D): [`benchmarks/results/phase1_scalar_advection_3d_v100.csv`](../../benchmarks/results/phase1_scalar_advection_3d_v100.csv).

This is the "at scale" evidence requested for Phase 1
(`tasks/0002-phase1-cartesian-grid-scalar-transport.md`): the full FVM
scalar-advection solver (`FvmSolver` + `ScalarAdvectionField` +
`CentralDifferenceReconstruction` + `UpwindFlux` + SSP-RK2), running one
complete two-stage SSP-RK2 time step per measurement, on 1D grids from
one million up to one hundred million real cells, and (added below) on
3D cube grids up to 512^3 (~134 million cells).

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

## 3D results (2026-09-29 follow-up)

The 1D results above only exercise the `Axis::X` branch of
`detail::axis_flux_difference` (`src/cfe/solver/explicit/fvm_solver.hpp`).
The 1D correctness test alone does not prove the Y and Z branches, or a
direction-dependent velocity, work on real hardware. `cfe_bench_scalar_
advection_3d_cuda` (`benchmarks/scalar_advection/bench_scalar_advection_3d_cuda.cu`)
sweeps cube grids (`nx = ny = nz`) with velocity `(1.0, 0.6, 0.3)` and a
`sin(x)*sin(y)*sin(z)` initial condition, so all three axes are active
and none degenerates into a zero-flux special case. Same
warm-up/median-of-10 methodology, same V100 (node `v020`, job
`47269636`).

Correctness verified immediately before this run, same allocation:
`test_scalar_advection_3d_cuda_matches_cpu_reference`
(`tests/unit/test_scalar_advection_3d_cuda.cu`) — 32^3 grid, 30 SSP-RK2
steps, GPU matching the CPU reference to `1e-9` cell-by-cell — with all
49/49 unit tests green.

| n per axis | n_cells | median ms/step | cell-updates/s |
|---|---|---|---|
| 64 | 262,144 | 0.069 | 3.78e9 |
| 128 | 2,097,152 | 0.417 | 5.03e9 |
| 256 | 16,777,216 | 2.955 | 5.68e9 |
| 400 | 64,000,000 | 14.088 | 4.54e9 |
| 512 | 134,217,728 | 30.624 | 4.38e9 |

### Observations

1. **512^3 (~134 million cells) completes one full SSP-RK2 step in 30.6
   ms** — the 3D "at scale" demonstration, comparable in cell count to
   the 1D 10^8-cell case above but doing three times the per-cell flux
   work (X, Y, *and* Z, plus three ghost-fills per stage instead of one).
2. **Peak 3D throughput (~5.7e9 cell-updates/s at 256^3) is lower than
   1D's plateau (~8.8e9)**, consistent with observation 1: roughly 3x the
   memory traffic and arithmetic per cell for a similar total cell count
   should cost roughly 3x the time per cell-update, which is
   approximately what these numbers show (8.8e9 / 3 ~= 2.9e9, same order
   of magnitude as the measured ~4.4-5.7e9 — the actual ratio is better
   than a naive 3x because ghost-fill cost scales with surface area, not
   volume, and shrinks relative to the residual kernel as the grid grows).
3. **Throughput dips slightly at 400^3 and 512^3 relative to the 256^3
   peak** (5.68e9 -> 4.54e9 -> 4.38e9). Unlike the 1D sweep, this is not
   a flat plateau; the largest cases are big enough (400^3 padded is
   404^3 elements per field, ~2.4 GB across `q`/`stage1`/`scratch` at
   double precision) that they may be starting to press on L2/HBM
   contention in ways the smaller cubes don't -- not root-caused further
   here since correctness, not roofline optimization, was this session's
   goal.

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

## Code-review fixes re-verified on the V100 (2026-09-30)

PR #2 code review found two real correctness issues in `FvmSolver`/
`PeriodicBoundary` (see `agent_history.md`'s 2026-09-30 entry) — one of
them specifically a CUDA uninitialized-memory read (ghost-cell residual
entries were never written, and `cudaMalloc` doesn't zero-initialize).
After fixing both:

- **52/52 unit tests pass** (up from 49; 3 new tests added in the same
  response), including both CUDA correctness tests, on this V100 (job
  `47300907`, node `v012`).
- **`compute-sanitizer --tool initcheck` reports 0 errors** across the
  full test suite — direct tool confirmation the uninitialized-read fix
  actually works, using the exact verification method the review itself
  recommended, not just inferred from the fix's logic.
- **Both 1D and 3D benchmarks re-run, no meaningful regression**: 1D at
  10^8 cells, 11.30ms/step (was 11.34ms); 3D at 512^3, 31.17ms/step (was
  30.62ms) — both within a few percent, i.e. noise. The fix's CPU-side
  cost (see `docs/performance/0004-...md` Observation 4) does not show
  up on the GPU path at all.

## ssp_rk2_step made interior-cells-only, re-verified on the V100 (2026-10-01)

`ssp_rk2_step` was changed from "touch every padded cell" (round 1's
ghost-zero-fill workaround) to "only ever touch real cells, via
`FvmSolver::active_cell_index_map()`" — the actual root-cause fix,
matching standard FVM practice (fill ghosts -> compute residual on
interior cells -> integrate interior cells only) rather than inventing a
defined-but-meaningless ghost-cell residual value. See
`docs/performance/0004-...md` Observation 4 for the CPU-side story (a
real, bounded, serial-only regression traced to lost auto-vectorization).

This change surfaced a second, genuine bug, caught only by re-running
`compute-sanitizer --tool initcheck` on real hardware rather than trusting
the CPU tests alone: `stage1`/`residual_scratch` (freshly `cudaMalloc`'d,
never host-initialized) have cells that are real on one axis but ghost on
another for a 2D/3D grid -- legitimate read sources for `PeriodicBoundary::
fill_x`'s corner/edge handling, but never written by the now-interior-only
combine step. Initcheck reported **1600 errors**, entirely isolated to
`test_scalar_advection_3d_cuda_matches_cpu_reference` (the 1D test was
clean, since 1D has no second axis to go wrong on). Fixed by
zero-initializing `DeviceField` at construction (`cudaMemset` after
`cudaMalloc`, matching `cfe::Field`'s host-side `std::vector` semantics) —
re-verified clean:

- **57/57 unit tests pass** (up from 52; 5 new tests from the full
  follow-up review response), on this V100 (job `47314879`, node `v006`).
- **`compute-sanitizer --tool initcheck` reports 0 errors** again, after
  the `DeviceField` fix (it reported 1600 before the fix, confirming this
  was a real, newly-introduced bug, not a false positive).
- **Both 1D and 3D benchmarks re-run, no change at all**: 1D at 10^8
  cells, 11.48ms/step (was 11.30-11.34ms across every prior measurement);
  3D at 512^3, 30.70ms/step (was 30.62-31.17ms) — all within noise. The
  CPU-side regression and the `DeviceField` zero-init fix both have zero
  measurable cost on the GPU path.

## Follow-ups not yet done

- A comparable CPU-only benchmark sweep for this same solver (smaller
  scale, matching `bench_field_update`'s CPU/GPU pairing convention) is
  not yet written.
- The 400^3/512^3 throughput dip noted above is not root-caused (no
  Nsight Compute profiling run against the 3D kernel yet, unlike Phase
  0's memory-layout study).
- Multi-block/AMR-relevant scaling is out of scope for Phase 1 (see the
  task file's PI direction: fixed refinement only, AMR-ready seams).
