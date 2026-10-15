# Float-vs-double CPU results (task 0017)

No CUDA/GPU hardware was available for this task (same standing
limitation several recent tasks have flagged -- see
`docs/performance/0007-phase3-euler-cuda-results.md`/`0008-...md` for
the GPU-side evidence this task's own float build is meant to eventually
pair with). Everything below is CPU-only, on the development machine
this task was done on. It answers "does `-DCFE_SCALAR_TYPE=float`
actually run faster, measured, right now" -- not the GPU comparison
ROADMAP.md Phase 6 ("single/double precision studies," "CPU/GPU
performance studies") ultimately wants; that remains outstanding and is
named as this task's own natural next step, not silently dropped.

## Environment

| | |
|---|---|
| Hardware | Apple Silicon (arm64), local development machine |
| Compiler | AppleClang 21.0.0 (`Apple clang version 21.0.0 (clang-2100.1.1.101)`) |
| Build type | `CMAKE_BUILD_TYPE=Release` (`-O3 -DNDEBUG`), `CFE_ENABLE_CUDA=OFF` (no CUDA compiler found), `CFE_ENABLE_MPI=ON` (OpenMPI 5.0.11) |
| Backend | serial (`CFE_DEFAULT_BACKEND=serial`, the CMake default) -- no explicit SIMD intrinsics anywhere in this codebase's CPU backend |
| Date | 2026-10-15 |

## 1. Ground truth: does the existing suite even build/pass at float?

Configured and built fresh (`cmake -S . -B build-float -DCMAKE_BUILD_TYPE=Release -DCFE_SCALAR_TYPE=float -DCFE_BUILD_TESTS=ON -DCFE_BUILD_TUTORIALS=ON -DCFE_ENABLE_MPI=ON`, then `cmake --build build-float -j10`), separate from a `build-double/` of the same configuration at `CFE_SCALAR_TYPE=double`. Zero compile errors/warnings at either precision. `ctest` (7 registered tests, including 4 MPI tests at np=2/4/8) and the full `cfe_unit_tests` binary directly: **196/196 unit+verification tests pass at both precisions** (up from the pre-task baseline of 158, reflecting this task's own added float/double pairs -- see `agent_history.md`'s task-0017 entry for which files).

Before this task's fixes, the SAME float build already compiled and reported "158/158 passed" -- but several of those 158 tests were silently still running at `double` internally (hardcoded `using Scalar = double;`, independent of `CFE_SCALAR_TYPE`), so that earlier green result did not actually mean what it looked like it meant. See `agent_history.md` for the full list of files this affected.

## 2. Forward-step solve: float vs. double wall-clock

Standalone measurement harness (not part of the committed test/benchmark
suite -- a one-off, reusing `tests/verification/forward_step_setup.hpp`'s
own `ForwardStepBlocks`/`run_and_report` exactly as
`tutorials/euler_2d_forward_step/` does), `cells_per_unit_height=80`,
`t_final=1.0` (shorter than the tutorial's own ~4, chosen to finish in
reasonable time). Compiled directly with `g++`/AppleClang (NOT through
CMake for this one-off harness), `-O3 -DNDEBUG`, and the actual
precision-selecting preprocessor define `core/types.hpp` itself checks
(`-DCFE_SCALAR_IS_FLOAT` for the float build) -- NOT `-DCFE_SCALAR_TYPE`,
which is a CMake CACHE VARIABLE name with no meaning to a raw compiler
invocation; CMake itself is what translates `-DCFE_SCALAR_TYPE=float`
into `-DCFE_SCALAR_IS_FLOAT` for the real build (`CMakeLists.txt`'s
`target_compile_definitions(cfe_core INTERFACE CFE_SCALAR_IS_FLOAT)`,
conditioned on the cache variable). Confirmed single-threaded
(`cfe::parallel_for` resolves to the SERIAL backend without
`-DCFE_DEFAULT_BACKEND_THREADED`, which this harness does not define --
`src/cfe/backend/parallel_for.hpp`), so this measurement is not exposed
to the specific threaded-vs-threaded contention risk flagged in review
(other concurrent threaded CFE work on this machine the same evening).

Revised methodology per review (Enzo): one UNTIMED warm-up repetition
(first-touch page faulting, same convention
`bench_scalar_advection.cpp`'s own `one_step(); // warm-up` already
uses) before 5 TIMED repetitions per (flux, precision) pair, reporting
median/min/max (not a plain average that a single outlier rep could
skew):

| Flux | double median (min-max) | float median (min-max) | Speedup (median) |
|---|---|---|---|
| Rusanov | 1.5412 s (1.5282-1.5880, 855 steps) | 1.4808 s (1.4797-1.4836, 855 steps) | 1.041x (~4% wall-clock reduction) |
| HLLC | 23.6036 s (23.2964-23.7499, 861 steps) | 15.2795 s (15.1732-15.3661, 861 steps) | 1.545x (~35% wall-clock reduction) |

Spread is tight at both precisions for both fluxes (double: 3.9%/1.9% of
median for Rusanov/HLLC; float: 0.3%/1.3%) -- consistent with a
measurement that is NOT dominated by concurrent-load noise (a
contended run would be expected to show a much wider, less consistent
spread across repetitions, not a tight band around the same median run
after run). This directly re-measures and supersedes this task's
earlier, 3-repetition-with-no-warm-up pass (which had reported Rusanov
as showing "essentially no speedup," ~1.005x) -- the corrected,
warm-up-included, 5-repetition result still shows the same qualitative
picture (Rusanov: small; HLLC: substantial) with a real, small (~4%,
not ~0%) Rusanov speedup now resolved out of the earlier measurement's
noise floor.

Reading: float's speed advantage is NOT uniform. Rusanov is a short,
simple per-face calculation (a handful of arithmetic ops, one max-
wave-speed estimate) that shows only a modest ~4% benefit from float on
this CPU -- on this machine, with this un-vectorized reference
implementation, its cost is apparently not dominated by raw
floating-point throughput the way a more arithmetic-heavy kernel's is.
HLLC does substantially more arithmetic per face (an iterative/
Newton-refined pressure estimate, several square roots, multiple
branches for the wave-speed/star-state construction -- see
`docs/adr/0012-hllc-numerical-flux.md`), and shows a real, repeatable
~1.5x speedup at float. This is consistent with (but not proven to be
CAUSED by, absent an actual instruction-level profile -- see AGENTS.md
#25) float helping most where a kernel is itself compute-bound --
exactly the regime a GPU puts every kernel into, which is this
project's real target (ROADMAP.md Phase 6) and the reason this result,
while partial, is still useful evidence in that direction.

Caveat: this problem size (`cells_per_unit_height=80`, ~19k total cells
across both blocks) is small enough to be cache-resident on this
machine, not a memory-bandwidth-bound production scale -- these speedup
numbers may not generalize to a much larger run where memory bandwidth,
not compute throughput, is the bottleneck (in which case float's
smaller footprint could plausibly help MORE, not less, but that is a
hypothesis, not measured here).

`final_mass` matched to 6 significant figures between float and double
for both fluxes (4.337711 Rusanov, 4.335920-4.335921 HLLC), confirming
the float run is solving the same physical problem, not a degenerate
or diverged one.

## 3. MPI halo-exchange: message size and bandwidth at float vs. double

`benchmarks/mpi/bench_mpi_halo_exchange.cpp` (fixed this task: `Scalar`
was hardcoded `double` regardless of `cfe::scalar`, so a float build's
own run of this benchmark was previously still moving `double`-sized
messages -- see `agent_history.md`). Run at `np=2`, twice independently
(both runs agree closely -- the second run's numbers are quoted below;
raw CSVs for both are in this task's own `agent_history.md` entry).

**`size` sweep** (global_nx=256, ghost_depth=2, cross-section `n = ny =
nz` swept 8 to 128 -- a 16x range in message size at fixed rank count):

| n | double median (us) | float median (us) | double bytes | float bytes | time speedup (double/float) |
|---|---|---|---|---|---|
| 8 | 2.208 | 2.042 | 2048 | 1024 | 1.081x |
| 16 | 5.750 | 3.875 | 8192 | 4096 | 1.484x |
| 32 | 13.833 | 13.000 | 32768 | 16384 | 1.064x |
| 64 | 42.250 | 38.750 | 131072 | 65536 | 1.090x |
| 128 | 163.500 | 144.917 | 524288 | 262144 | 1.128x |

**`ghost_depth` sweep** (global_nx=256, ny=nz=64, ghost depth swept 2/4/8
-- another independent way of varying message size, at fixed
cross-section):

| ghost_depth | double median (us) | float median (us) | time speedup (double/float) |
|---|---|---|---|
| 2 | 42.125 | 38.459 | 1.095x |
| 4 | 81.334 | 73.208 | 1.111x |
| 8 | 161.958 | 144.833 | 1.118x |

`bytes_per_exchange` correctly halves at float in every row of both
sweeps (confirms the fix actually changes what is measured, not just
what is labeled).

**Checked against Enzo's review (do not conclude a mechanism from one
data point):** across BOTH independent sweeps, spanning message sizes
from ~1KB to ~512KB, the float speedup sits in a narrow, relatively FLAT
band of roughly 6-13% for every message size n >= 32 -- it does NOT grow
toward the ~2x a pure bytes-halved/bandwidth-bound regime would predict
at the largest sizes tested, and (apart from the smallest, n=8, size
sweep row) it does NOT shrink toward ~0% the way pure per-call-latency
domination would predict at smaller sizes either. This is a more
nuanced, and more honest, picture than "latency dominates small
messages" alone would suggest: the modest, roughly size-independent gap
observed from n=32 up through n=128 (and across the whole ghost_depth
sweep) is consistent with the pack/unpack loop's cost being bound more
by ELEMENT COUNT (the same number of grid values touched regardless of
`Scalar`'s byte width) than by byte count, with the smaller, genuinely
byte-proportional cost (the actual network send) contributing only a
modest fraction of the total even at the largest message tested. This
is offered as a plausible reading of the wall-clock pattern, consistent
with both sweeps -- NOT an instruction-level-profiled conclusion (no
profiler was run), and is deliberately not written into Enzo's
accumulated-knowledge file as a settled mechanism per AGENTS.md #25's
own evidentiary standard (profile before asserting mechanism).

One genuine anomaly, reproduced across both independent runs rather than
dismissed as one-off noise: the `n=16` row shows a much larger speedup
(~1.48x/1.52x across the two runs) than its neighbors `n=8`
(~1.08x/1.04x) and `n=32` (~1.06x/1.08x) in both sweeps. This is NOT
explained by the reasoning above (which would predict a smooth trend)
and is reported honestly as an unexplained, reproducible outlier at this
one specific message size -- a candidate for its own follow-up
investigation (e.g. an MPI eager-vs-rendezvous protocol threshold
crossing near this message size), not something to be smoothed over or
silently excluded from the table.

Because the reported `bandwidth_GBps` metric is `bytes / time`, and
float's bytes nearly halve while its time drops only ~6-13%, the
reported bandwidth NUMBER itself comes out LOWER at float than at
double at every message size (e.g. n=64: 3.102 GB/s double vs. 1.691
GB/s float) -- this is an artifact of what that metric divides by, not
a sign float communication is somehow slower in wall-clock terms (it is
measurably faster, per the tables above, at every size tested except
the anomalous n=16 case working in float's favor even more). A reader
comparing only the `bandwidth_GBps` column across precisions without
also checking the underlying `median_exchange_us` and `bytes` columns
could easily, and wrongly, conclude float communication regressed here;
it does not.

## Known limitations

- CPU-only (Apple Silicon, no explicit SIMD). No CUDA/GPU measurement in
  this task -- the GPU comparison is the one this project's hardware-
  acceleration goal actually cares about most, and remains open
  (tracked in ROADMAP.md Phase 6, not silently dropped).
- The forward-step speedup measurement is a single moderate resolution
  (~19k cells, cache-resident, not a memory-bandwidth-bound production
  scale) -- a directional, honestly-reported data point, not a full
  performance study (BENCHMARKS.md's own state-size/resolution-sweep
  conventions would ask for more before calling this a completed
  study). Re-measured with a warm-up repetition and 5 timed repetitions
  (median/min/max reported) specifically to rule out contamination from
  other concurrent CPU work on this machine the same evening -- the
  tight spread observed (<4% of median at every flux/precision
  combination) supports reading these numbers at face value, not as
  noise-dominated.
- The halo-exchange mechanism explanation (element-count-bound pack/
  unpack cost) is a plausible reading of the wall-clock pattern across
  two independent sweeps, not an instruction-level-profiled conclusion
  -- it should not be treated as settled without an actual profile.
