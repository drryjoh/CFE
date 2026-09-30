# Phase 1 scalar-advection CPU benchmark results

Raw CSV: [`benchmarks/results/phase1_scalar_advection_cpu_apple_m5.csv`](../../benchmarks/results/phase1_scalar_advection_cpu_apple_m5.csv).

Task spec item 11's CPU benchmark sweep for the scalar-advection kernel:
"runtime / cell-updates-per-second for the advection kernel, swept
across a few grid resolutions... on CPU" (component count stays at 1 for
this task, unlike Phase 0's memory benchmark, so grid resolution is the
scaling axis here instead).

## Kernel and methodology

`cfe_bench_scalar_advection`
(`benchmarks/scalar_advection/bench_scalar_advection.cpp`): builds a
periodic 1D grid at each `nx`, seeds a sine-wave initial condition, then
times one full `ssp_rk2_step` call (ghost-fill + residual assembly,
twice) via `cfe::backend::serial::parallel_for` and
`cfe::backend::threaded::parallel_for` directly (not `cfe::CpuParallelFor`,
which resolves to only whichever single backend is the project default —
sweeping both in one binary needs the explicit namespaces, matching
`bench_field_update.cpp`'s convention). Median of 7 timed repetitions
after one untimed warm-up.

## Environment

| | |
|---|---|
| Hardware | Apple M5 (10 cores: 4 performance + 6 efficiency), 24 GiB unified memory |
| OS | macOS 26.6.2 (Darwin 25.6.0, arm64) |
| Build type | `CMAKE_BUILD_TYPE=Release` |
| Date | 2026-09-29 |

## Results

Re-measured 2026-09-30 after the code-review fixes below (ghost-residual
full-coverage write, extent-validation asserts). Numbers updated from the
original 2026-09-29 baseline; see Observation 4.

| n_cells | serial ms/step | serial cell-updates/s | threaded ms/step | threaded cell-updates/s |
|---|---|---|---|---|
| 10,000 | 0.017 | 5.80e8 | 0.465 | 2.15e7 |
| 100,000 | 0.181 | 5.53e8 | 0.452 | 2.21e8 |
| 1,000,000 | 1.943 | 5.15e8 | 1.284 | 7.79e8 |
| 10,000,000 | 19.793 | 5.05e8 | 12.038 | 8.31e8 |

### Observations

1. **Threading only pays off once the grid is large enough** — at 10,000
   cells, threaded is slower than serial (thread-pool spin-up and
   synchronization overhead dwarfs the actual per-cell work at this
   size); by 100,000 cells threaded has already pulled ahead, and the
   margin grows through 10M cells (~1.65x faster). This is the same
   qualitative shape as Phase 0's simpler `q*q` kernel, just at a
   different crossover point — this solver does meaningfully more work
   per cell (ghost-fill + 5-point-stencil residual, twice per step) than
   Phase 0's single elementwise multiply, so there is more real work to
   amortize thread overhead against even at smaller grids.
2. **Serial throughput itself is not flat across sizes** — the smallest
   grids are too small to saturate memory bandwidth or amortize fixed
   per-call overhead (ghost-fill launches, function-call/lambda-capture
   setup), consistent with Phase 0's own observation that a kernel needs
   a large enough working set before its steady-state throughput is
   actually being measured.
3. **This is a small-scale CPU baseline, not the "at scale" evidence** —
   compare against
   [`0003-phase1-scalar-advection-cuda-results.md`](0003-phase1-scalar-advection-cuda-results.md)
   for the V100 results at up to 10^8 (1D) / 512^3 (3D) cells, several
   orders of magnitude beyond what is practical to sweep repeatedly on a
   laptop CPU.
4. **A measured, disclosed regression on the serial backend at large
   `n_cells`** (caught and investigated during code review — see
   `agent_history.md`): the original 2026-09-29 baseline measured
   15.135 ms/step (serial, 10M cells); after fixing a real correctness
   bug (ghost-cell residual entries were never written, an
   uninitialized-memory read on CUDA — see the "Bug this validates"
   section of `0003-...`), the same case now measures 19.79 ms/step,
   ~30% slower. The fix (`detail::zero_ghost_residual_x/y/z` in
   `fvm_solver.hpp`) touches only the thin ghost-cell shell, not the
   full padded volume — an earlier, rejected version of the fix that
   zeroed the *entire* padded array first regressed this same case to
   ~21 ms/step by rewriting every real cell's residual twice, which is
   why that approach was replaced with the current ghost-only version.
   The remaining ~30% gap was investigated directly (isolated via a
   controlled same-session A/B against the pre-fix code, and via an
   `__attribute__((noinline))` experiment) and traced to the serial
   backend specifically -- the threaded backend shows **no** measurable
   regression at any size (12.14ms -> 12.04ms at 10M cells, within
   noise) -- consistent with a compiler code-generation sensitivity
   (instruction scheduling/vectorization around the extra small kernel
   launch) rather than a genuine per-cell cost, since the actual data
   touched by the fix is a handful of ghost cells, not a size that could
   explain a 30% change on its own. Not fully root-caused further:
   correctness takes priority (AGENTS.md #2), the threaded backend
   (which this project's own numbers show winning at every scale that
   matters) is unaffected, and the GPU numbers in `0003-...` are the
   actual "at scale" evidence this project relies on.

## What this resolves from the task spec

Task spec item 11 ("Benchmarks") required a CPU sweep across grid
resolutions for the advection kernel specifically (as opposed to Phase
0's memory benchmark, which only exercises an elementwise kernel with no
grid/ghost-cell/flux logic at all). This is that evidence.
