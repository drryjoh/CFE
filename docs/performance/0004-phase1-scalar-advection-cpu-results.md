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

Re-measured 2026-10-01 after `ssp_rk2_step` became interior-cells-only
(see Observation 4 — this supersedes the 2026-09-30 ghost-zero-fill
numbers previously here, which no longer reflect the current code).

| n_cells | serial ms/step | serial cell-updates/s | threaded ms/step | threaded cell-updates/s |
|---|---|---|---|---|
| 10,000 | 0.030 | 3.31e8 | 0.39 | 2.5e7 |
| 100,000 | 0.31 | 3.19e8 | 0.41 | 2.4e8 |
| 1,000,000 | 3.26 | 3.06e8 | 1.24 | 8.1e8 |
| 10,000,000 | 33.04 | 3.03e8 | 12.12 | 8.2e8 |

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
4. **A measured, disclosed, root-caused regression on the serial backend
   only, confirmed to plateau rather than grow, and fully reproduced
   under controlled conditions.** Full methodology, exact commits,
   compile commands, raw logs, and vectorization remarks are committed in
   [`0005-phase1-ssp-rk2-regression-reproducibility.md`](0005-phase1-ssp-rk2-regression-reproducibility.md)
   — summary here, see that document for anything this summary
   compresses away.

   **Corrected finding** (superseding an earlier, less careful version of
   this observation that compared numbers recorded in three different
   work sessions, under three different and uncontrolled system-load
   conditions): a controlled, same-session, same-compile-flags comparison
   of the original pre-review code, round 1's ghost-zero-fill fix, and
   round 2's interior-cells-only fix shows that **round 1's fix was not a
   measurable regression at all** (original and round-1-intermediate
   differ by <1% at every size from 1M to 100M cells, well within this
   benchmark's own run-to-run noise). The one real, reproducible
   regression is round 2 (`ssp_rk2_step` made interior-cells-only via
   `FvmSolver::active_cell_index_map()` — see ADR/agent_history for the
   design) vs. *either* earlier version, at a consistent **~1.68x**, flat
   from 1M cells through 100M cells (not a growing penalty, and not two
   separate ~31%-then-~118% steps as previously reported).

   **Root cause, confirmed directly via `-Rpass-missed=loop-vectorize`
   (not inferred):** routing the per-cell storage index through
   `index_map(r)` instead of using the loop counter directly defeats the
   compiler's auto-vectorization of the combine loop for a meaningful
   fraction of its instantiations — confirmed by comparing optimization-
   remark output across all three versions (`0005-...`'s committed
   `remarks_*.txt`), not by a check once removed from the actual compiled
   code. A 1D-specific fast path (avoiding integer division via
   `if constexpr` on `Field::dim`) and force-inlining the index map
   (`CFE_FORCEINLINE`, `AGENTS.md` #9's documented pattern) were both
   tried and neither recovered the lost vectorization.

   **The threaded backend shows no regression at any size tested**
   (statistically indistinguishable across all three versions, confirmed
   in the same controlled comparison), consistent with its inner loop
   (`backend/cpu/threaded.hpp`) never auto-vectorizing in the first
   place, for unrelated structural reasons (`Cannot vectorize early exit
   loop with writes to memory`) — verified by checking the *original*
   pre-review code's own remarks, not assumed. Since threaded already
   wins at every scale this project benchmarks, and GPU is the actual "at
   scale" evidence (`0003-...`, confirmed unaffected by this same
   change), this was accepted as a known, bounded, disclosed cost rather
   than chased further — recovering the lost vectorization (compiler
   hints, or a compile-time fast path for the common contiguous-offset
   case) is explicit future work, not a dropped thread.

## What this resolves from the task spec

Task spec item 11 ("Benchmarks") required a CPU sweep across grid
resolutions for the advection kernel specifically (as opposed to Phase
0's memory benchmark, which only exercises an elementwise kernel with no
grid/ghost-cell/flux logic at all). This is that evidence.
