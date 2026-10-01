# Phase 1 ssp_rk2_step serial-regression: reproducibility record

This document exists specifically so someone else (or a future session)
can re-run the exact comparison behind the "~1.68x" serial-backend
regression claim in `0004-phase1-scalar-advection-cpu-results.md`
Observation 4, without having to reconstruct the methodology from prose.
It also **corrects** two numbers in that same Observation 4, which did
not survive a controlled re-check — see "What this corrects" below.

Raw artifacts (all committed alongside this file, in
`benchmarks/results/phase1_ssp_rk2_vectorization_repro/`):

- `run_original.log`, `run_intermediate.log`, `run_current.log` — raw
  stdout of 5 repeated full benchmark sweeps per version.
- `remarks_original.txt`, `remarks_intermediate.txt`, `remarks_current.txt`
  — full `-Rpass=loop-vectorize -Rpass-missed=loop-vectorize
  -Rpass-analysis=loop-vectorize` compiler output per version.
- `bench_scalar_advection_extended_sweep.cpp` — the actual benchmark
  source compiled for this comparison: identical to the committed
  `benchmarks/scalar_advection/bench_scalar_advection.cpp` at the
  "original" commit below, with the sweep extended to include 50M and
  100M cells (the committed benchmark only goes to 10M).

## Environment

| | |
|---|---|
| Hardware | Apple M5 (10 cores: 4 performance + 6 efficiency), 24 GiB unified memory |
| OS | macOS 26.6.2 (Darwin 25.6.0, arm64) |
| Compiler | `Apple clang version 21.0.0 (clang-2100.1.1.101)`, target `arm64-apple-darwin25.6.0` |
| Compile flags | `-O3 -DNDEBUG -std=c++20 -arch arm64 -Wall -Wextra` (matches this project's actual `CMAKE_BUILD_TYPE=Release` flags exactly, confirmed by reading the generated `flags.make`) |
| Date | 2026-10-01 |

## The three versions compared

| Label | Git commit | What it is |
|---|---|---|
| `original` | `e5df5f9` | Immediately before any PR #2 review response — `FvmSolver::residual()` writes only real cells of `out`, `ssp_rk2_step` iterates every padded cell. |
| `intermediate` | `e880a9e` | After round 1's fix: `FvmSolver::residual()` additionally zero-fills ghost-cell residual entries (`detail::zero_ghost_residual_x/y/z`); `ssp_rk2_step` itself is **byte-for-byte unchanged** from `original` (confirmed: `git diff e5df5f9 e880a9e -- src/cfe/solver/time_integration/ssp_rk2.hpp` is empty). |
| `current` | `HEAD` (`0a0a229` at time of writing) | After round 2's fix: `ssp_rk2_step` is interior-cells-only via `IndexMap`/`active_cell_index_map()`; the zero-fill workaround is removed entirely. |

Only `src/cfe/solver/explicit/fvm_solver.hpp` and
`src/cfe/solver/time_integration/ssp_rk2.hpp` were swapped between
versions; every other header (grid, field, math, backend) was the
*current* checkout's version in all three compiles. This is safe because
none of those files' public interfaces changed between `e5df5f9` and
`HEAD` — confirmed by the fact that all three versions compiled and ran
correctly with zero source changes needed beyond the two solver files
and the benchmark's own `ssp_rk2_step` call-site arguments (5 args for
`original`/`intermediate`, 7 for `current`, matching each version's real
API).

## Reproduction commands

```bash
# From a checkout of this repo, for each of the three commits above:
mkdir -p /tmp/repro/<label>/cfe/solver/explicit /tmp/repro/<label>/cfe/solver/time_integration
git show <commit>:src/cfe/solver/explicit/fvm_solver.hpp \
  > /tmp/repro/<label>/cfe/solver/explicit/fvm_solver.hpp
git show <commit>:src/cfe/solver/time_integration/ssp_rk2.hpp \
  > /tmp/repro/<label>/cfe/solver/time_integration/ssp_rk2.hpp
git show <commit>:benchmarks/scalar_advection/bench_scalar_advection.cpp \
  > /tmp/repro/<label>/bench_scalar_advection.cpp
# (then extend the nx sweep in that .cpp to add 50'000'000 and
# 100'000'000 -- see bench_scalar_advection_extended_sweep.cpp)

clang++ -O3 -DNDEBUG -std=c++20 -arch arm64 -Wall -Wextra \
  -I /tmp/repro/<label> -I src \
  /tmp/repro/<label>/bench_scalar_advection.cpp -o /tmp/repro/bench_<label>

# Vectorization remarks:
clang++ -O3 -DNDEBUG -std=c++20 -arch arm64 -Wall -Wextra \
  -I /tmp/repro/<label> -I src \
  -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize \
  -c /tmp/repro/<label>/bench_scalar_advection.cpp -o /dev/null \
  > remarks_<label>.txt 2>&1

./bench_<label>   # repeat 5x per version, same session, back to back
```

All three binaries were compiled within the same terminal session,
immediately before running all three benchmark sweeps (5 repetitions
each, interleaved as `original` x5, `intermediate` x5, `current` x5, not
alternated) — raw logs in `run_*.log`.

## Results (median of repetitions 2-5; repetition 1 is consistently an
## outlier across all three versions, a cold-start/first-touch effect,
## excluded rather than silently averaged in)

| n_cells | original (ms) | intermediate (ms) | current (ms) | current/original | current/intermediate |
|---|---|---|---|---|---|
| 10,000 | 0.0188 | 0.0187 | 0.0325 | 1.73x | 1.74x |
| 100,000 | 0.1920 | 0.1934 | 0.3343 | 1.74x | 1.73x |
| 1,000,000 | 1.9601 | 1.9534 | 3.3732 | 1.72x | 1.73x |
| 10,000,000 | 20.358 | 20.216 | 34.084 | 1.67x | 1.69x |
| 50,000,000 | 102.718 | 101.267 | 170.834 | 1.66x | 1.69x |
| 100,000,000 | 209.004 | 203.050 | 341.841 | 1.64x | 1.68x |

(10K/100K are too small for this kernel to be bandwidth-bound — see
`0004-...md` Observation 2 — so their ratios are noisier; 1M-100M is the
meaningful range, where the ratio is stable at **~1.68x, regardless of
whether the comparison is against `original` or `intermediate`.**)

Threaded backend, measured the same way (`run_*.log` has the full data):
statistically indistinguishable across all three versions at every size
— e.g. at 100M cells: 111-129ms for all three, entirely within
run-to-run noise. No regression on threaded, confirmed directly rather
than assumed.

## What this corrects

**`docs/performance/0004-...md` Observation 4 previously stated the
serial regression as "15.135ms (original) -> 19.793ms (intermediate,
+31%) -> ~33ms (current, +118% over original)".** Those three numbers
were recorded in three different work sessions, each under whatever
background system load happened to exist at the time — they were never
compared in a single controlled run until now.

**Under a controlled, same-session comparison, `original` and
`intermediate` are statistically indistinguishable** (differing by
<1% at every size from 1M to 100M cells — well within this benchmark's
own run-to-run noise, e.g. see `intermediate`'s own repetitions ranging
20.15-20.33ms at 10M cells). Round 1's zero-fill workaround
(`detail::zero_ghost_residual_x/y/z`) was **not actually a measurable
regression** — the previously-recorded 19.793ms figure was very likely
elevated by session-specific system noise, not by the code change. The
only reproducible regression is `current` vs. *either* earlier version,
at a consistent **~1.68x** (not ~1.68x vs. intermediate *and* a separate,
larger ~2.18x vs. original — those two ratios were never actually
different once measured side by side).

This does not change any correctness conclusion or the decision to
accept the regression as a documented limitation (per the request that
opened this round of work) — it only corrects the magnitude and
attributes the entire regression to round 2's interior-cells-only
`ssp_rk2_step` change specifically, not to round 1's fix at all.

## Vectorization evidence

`remarks_original.txt`, `remarks_intermediate.txt`, `remarks_current.txt`
are the complete, unedited compiler output. The clearest single data
point: `backend/cpu/serial.hpp:14`'s `for` loop (the combine-step kernel
dispatch) is instantiated once per distinct call site in each binary;
counting vectorized vs. non-vectorized instantiations of that one source
line:

| | vectorized | not vectorized |
|---|---|---|
| `original` | 6 | 6 |
| `intermediate` | 7 | 9 |
| `current` | 2 | 2 |

These raw counts mix multiple unrelated call sites (this benchmark's own
initialization loops, `Field` construction, etc., not just the
residual/combine kernels), so they are reported here as supporting,
corroborating evidence for the vectorization-loss mechanism, not as a
precisely-isolated proof on their own — the clean, directly-measured
timing comparison above is the primary evidence for the regression's
existence and size; the vectorization remarks explain *why* a smaller
set of writes (visiting fewer indices, not more) could plausibly cost
more, consistent with the independent finding (prior round) that routing
the combine loop's storage index through `index_map(r)` measurably
changes which instantiations of this loop vectorize.
