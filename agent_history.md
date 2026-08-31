# CMU-CFE Agent Development History

This file records chronological AI-assisted development work.

Do not place permanent architectural rules here. Those belong in `AGENTS.md` or an ADR.

## Entry template

```text
## YYYY-MM-DD — Short task name

Agent:
Model:

Objective:

Files changed:

Tests added:

Benchmarks run:

Performance change:

Scientific verification:

Architecture decisions:

Known limitations:

Next recommended task:
```

---

## 2026-08-30 — Repository engineering charter

Agent:
ChatGPT

Objective:
Establish initial project governance, architecture, roadmap, verification philosophy, performance requirements, ADR workflow, and the first scoped implementation task.

Files changed:
- README.md
- AGENTS.md
- ARCHITECTURE.md
- ROADMAP.md
- BENCHMARKS.md
- VERIFICATION.md
- REFERENCES.md
- CHANGELOG.md
- agent_history.md
- docs/adr/*
- tasks/0001-phase0-execution-foundation.md

Tests added:
None. Repository governance only.

Benchmarks run:
None.

Performance change:
None.

Scientific verification:
Not applicable.

Architecture decisions:
Initial architecture documented as proposals. The first permanent decisions should be ratified through ADRs as implementation evidence becomes available.

Known limitations:
No code exists yet. Execution backend, memory layout, and state storage decisions remain to be validated experimentally.

Next recommended task:
Execute `tasks/0001-phase0-execution-foundation.md`.

---

## 2026-08-30 — Phase 0 execution foundation

Agent:
Claude (Claude Code)

Model:
claude-sonnet-5

Objective:
Implement `tasks/0001-phase0-execution-foundation.md`: the smallest
performance-oriented CMake C++ architecture needed to support future
structured FVM/DG development on CPU and CUDA -- configurable scalar/index
types, host/device macros, a minimal `parallel_for` execution abstraction
(serial CPU, threaded CPU, CUDA), fixed-size math containers/operations,
contiguous compile-time-sized field storage, a benchmark harness, and unit
tests. Explicitly out of scope: any CFD physics.

Files changed:
- `CMakeLists.txt`, `.gitignore` (new project build)
- `src/cfe/core/` -- `macros.hpp` (host/device annotation abstraction),
  `types.hpp` (`cfe::scalar`, `local_index`, `global_index`),
  `component_counts.hpp` (compile-time 1/5/10/20/50/100 sweep helper)
- `src/cfe/backend/cpu/` -- `serial.hpp`, `threaded.hpp`
- `src/cfe/backend/cuda/` -- `cuda_backend.cuh`, `device_field.cuh`
  (unverified, see below)
- `src/cfe/backend/parallel_for.hpp` -- `cfe::parallel_for` default alias
- `src/cfe/math/` -- `fixed_array.hpp` (`FixedArray<Scalar,N>` +
  scalar/vector/state aliases), `operations.hpp` (componentwise ops,
  `contract`, `weight`)
- `src/cfe/field/` -- `layout.hpp` (`AoSLayout`, `SoALayout`), `field.hpp`
  (`Field`/`FieldView`)
- `src/cfe/cfe.hpp` -- convenience aggregate header
- `tests/unit/*.cpp` (+ `test_backend_execution_cuda.cu`, CUDA-only), a
  small dependency-free `test_framework.hpp`
- `benchmarks/memory/bench_field_update.cpp` (+
  `bench_field_update_cuda.cu`, CUDA-only) and their `CMakeLists.txt`
- `benchmarks/results/phase0_field_update_apple_m5.csv` -- committed
  reference baseline
- `docs/performance/0001-phase0-results.md` -- curated results + hardware
  notes
- `scripts/profile_cuda.sh` -- documented (unexecuted) register/occupancy/
  spill inspection procedure
- `tutorials/phase0_benchmark/README.md`
- `docs/adr/0001-execution-backend.md`, `docs/adr/0002-state-memory-layout.md`
  -- updated with evidence

Tests added:
22 unit tests in `tests/unit/` covering `FixedArray` arithmetic, math
operations (`contract`, `weight`, componentwise ops) in float and double,
`AoSLayout`/`SoALayout` indexing, `Field`/`FieldView` round-tripping,
serial/threaded `parallel_for` correctness (including bitwise agreement on
the all-cell square kernel), and compilation/execution for all six required
component counts. A CUDA-only test (`test_backend_execution_cuda.cu`) is
written and wired into the build behind `CFE_ENABLE_CUDA` but has not been
run. All 22 CPU tests pass (`ctest --test-dir build`, and again with
`-DCFE_SCALAR_TYPE=float -DCFE_DEFAULT_BACKEND=threaded`).

Benchmarks run:
`cfe_bench_field_update` (`q_new(i,k) = q(i,k) * q(i,k)`), all 48 required
combinations (float/double x 1/5/10/20/50/100 components x serial/threaded
x AoS/SoA), on Apple M5 (10 cores, arm64, AppleClang 21, Release build). Raw
CSV: `benchmarks/results/phase0_field_update_apple_m5.csv`; discussion in
`docs/performance/0001-phase0-results.md`. No CUDA numbers exist -- no CUDA
toolkit/GPU was available.

Performance change:
No prior baseline exists; this is the first recorded baseline. Headline
finding: AoS layout matched or beat SoA at every measured CPU component
count for this kernel, with the gap widening at large N (e.g. AoS ~4-8x
faster than SoA at N=100, serial). Threading gave sub-linear but consistent
speedups, consistent with a memory-bandwidth-bound kernel.

Scientific verification:
Not applicable -- no physics implemented. Correctness verification here
means: math/containers/storage produce hand-computed expected values
(`tests/unit/`), and serial vs. threaded execution of the same kernel agree
bitwise.

Architecture decisions:
- ADR 0001 (execution backend): updated to Accepted for the serial/threaded
  CPU backends (measured, correct, negligible abstraction overhead). CUDA
  backend implemented to the same interface but left explicitly unverified
  pending hardware access.
- ADR 0002 (state memory layout): updated with CPU evidence favoring AoS as
  a provisional recommendation, but left Proposed overall -- this ADR's own
  evidence bar requires GPU data too, and GPU coalescing considerations are
  a specific, plausible reason the CPU result might not transfer.

Known limitations:
- CUDA backend, CUDA benchmark, and CUDA unit test are unverified (no
  toolkit/hardware in this environment). This is the most important
  follow-up before any GPU-dependent work proceeds.
- Apple Silicon P-core/E-core scheduling effects on the threaded backend
  were not isolated (no thread affinity control).
- AoSoA layout was not implemented; `Field`'s `Layout` policy leaves room
  for it without an interface change.
- Threaded backend is a simple static-chunk `std::thread` split with no
  pool; fine for Phase 0's single-kernel-per-call-site usage, likely worth
  revisiting once repeated small kernels are common.

Next recommended task:
Get CUDA verified on real hardware (compile `cfe_bench_field_update_cuda`
and the CUDA unit test, run `scripts/profile_cuda.sh`, update both ADRs with
GPU evidence) before starting Phase 1 (Cartesian grid and scalar
transport), since Phase 1 explicitly requires "CPU and CUDA execution."

---

## 2026-08-31 — PR #1 review fixes

Agent:
Claude (Claude Code)

Model:
claude-sonnet-5

Objective:
Address 8 correctness/rigor items raised in review of PR #1 before merge.
No physics added, no scope expansion beyond the review list.

Files changed:
- `src/cfe/field/field.hpp` -- `Field::view() const` no longer `const_cast`s;
  added `ConstView` (`FieldView<const Scalar, ...>`) as the true read-only
  return type.
- `src/cfe/backend/cuda/cuda_check.cuh` -- new; `CFE_CUDA_CHECK` macro
  (throws `std::runtime_error` with file/line/`cudaGetErrorString` on any
  non-`cudaSuccess` result).
- `src/cfe/backend/cuda/device_field.cuh` -- `cudaMalloc`/`cudaMemcpy` calls
  now go through `CFE_CUDA_CHECK`; destructor's `cudaFree` deliberately left
  unchecked (never throw from a destructor).
- `src/cfe/backend/cuda/cuda_backend.cuh` -- `parallel_for` is now
  asynchronous (no more `cudaDeviceSynchronize()` inside every call); added
  `CFE_CUDA_CHECK(cudaGetLastError())` after every launch; added a new
  `synchronize()` function as the explicit blocking point callers must use.
- `benchmarks/memory/bench_field_update_cuda.cu`,
  `tests/unit/test_backend_execution_cuda.cu` -- updated to call
  `synchronize()` explicitly (the benchmark's timing loop would otherwise
  silently measure launch/enqueue latency, not kernel execution time, once
  `parallel_for` stopped blocking internally).
- `tests/unit/test_backend_execution_cuda.cu` -- rewritten to cover
  float/double x AoS/SoA x all six component counts (previously: double/AoS
  only).
- `tests/unit/test_field_storage.cpp` -- added a test for the
  `Field::view() const` fix (type-level static_asserts that it returns
  `ConstView` and that `operator()` returns a `const` reference, plus a
  value round-trip check).
- `CMakeLists.txt` -- `CMAKE_CXX_STANDARD`/`cxx_std_20`/`CMAKE_CUDA_STANDARD`
  bumped 17 -> 20 (see new ADR 0006); added
  `--extended-lambda` for CUDA-language sources (required for the
  `__device__`-annotated lambdas every `parallel_for` call site already
  used -- this was a real, previously-unnoticed gap since nothing has ever
  compiled this project's CUDA code with `nvcc`); explicit
  `CFE_ENABLE_CUDA=ON` with no CUDA compiler found is now
  `message(FATAL_ERROR ...)` instead of a silent downgrade to OFF (verified
  directly -- see Scientific verification below).
- `docs/adr/0001-execution-backend.md` -- corrected: threaded CPU backend is
  now documented as the Phase 0 *reference* implementation, not a settled
  production backend, because `threaded::parallel_for` creates OS threads
  and allocates its worker vector fresh on every call with no persistent
  pool. Also updated the C++ standard evidence line to reference ADR 0006.
- `docs/adr/0006-language-standard.md` -- new. Makes the C++ standard an
  explicit, evidenced decision (C++20) instead of an undocumented default,
  and is explicit about what is and is not verified (no CUDA compiler has
  ever been available to confirm `nvcc` accepts `-std=c++20` for this
  project's `.cu` files).
- `tutorials/phase0_benchmark/README.md`, `scripts/profile_cuda.sh` --
  updated C++17 -> C++20 mentions; the manual `nvcc` example in
  `profile_cuda.sh` now also includes `--extended-lambda`.

Tests added:
5 new: 1 for `Field::view() const` read-only-ness
(`test_field_storage.cpp`), 4 replacing the single old CUDA correctness
test (double/AoS only) with float/double x AoS/SoA coverage
(`test_backend_execution_cuda.cu`, still CUDA-only/unverified -- see below).
23/23 CPU tests pass (`build/tests/cfe_unit_tests`, AppleClang 21, C++20).

Benchmarks run:
Re-ran `cfe_bench_field_update` once (all 48 combinations) under C++20 as a
regression sanity check, not a new official sweep: results were consistent
with the archived Apple M5 baseline (e.g. double/100/threaded/AoS: 88.1 GB/s
here vs. 82.4 GB/s in the original run -- normal run-to-run variance, same
AoS-beats-SoA pattern, no regression). The committed reference CSV
(`benchmarks/results/phase0_field_update_apple_m5.csv`) and results doc are
left as-is; this run was verification, not a new baseline.

CUDA remains entirely unbuilt and unrun in this environment -- no CUDA
toolkit/GPU was available for this review pass either. All CUDA-related
fixes (items 2-5 below) are code-reviewed and internally consistent but
UNVERIFIED by compilation. This is unchanged from the original Phase 0
status.

Performance change:
None expected or observed; see Benchmarks run above.

Scientific verification:
Item-by-item:
1. `Field::view() const` fix verified by a passing test with compile-time
   (`static_assert`) type checks -- see Tests added.
2. `--extended-lambda` addition verified by inspection only (matches
   documented `nvcc` requirement for host/device-annotated lambdas); cannot
   be compiled here.
3/4. Async `parallel_for` + error-checking verified by inspection and by
   the CPU backends' unchanged behavior (CUDA-only code, cannot compile
   here); the benchmark/test call-site updates were specifically added
   because the async change would otherwise silently corrupt CUDA timing
   results, which is exactly the kind of bug this project's verification
   philosophy exists to prevent -- caught by reasoning about the change,
   not by running it, since running it is not possible here.
5. Expanded CUDA test coverage verified by inspection and by the analogous
   CPU test (`test_all_cell_square_update_compiles_and_runs_for_all_required_component_counts`)
   passing with the same structure.
6. **Directly verified**: `cmake -S . -B <dir> -DCFE_ENABLE_CUDA=ON` on this
   machine (no CUDA compiler present) now exits nonzero with
   `CMake Error ... CFE_ENABLE_CUDA=ON was explicitly requested but no CUDA
   compiler was found`, instead of the previous silent-downgrade-to-OFF
   behavior. Default configuration (`CFE_ENABLE_CUDA` unset) still succeeds
   and still builds/passes all CPU tests -- confirmed both ways.
7. ADR correction is a documentation change; no new test needed (the
   underlying thread-creation-per-call behavior was already correctly
   implemented and now correctly described).
8. C++20 bump verified by a full rebuild + full test pass under
   `-std=c++20` (see Tests added/Benchmarks run). CUDA-side C++20
   compatibility remains unverified, as ADR 0006 states explicitly.

Architecture decisions:
- ADR 0001 (execution backend): status corrected from unqualified
  "Accepted for serial/threaded CPU backends" to "Accepted as the Phase 0
  reference implementation" -- the threaded backend's per-call thread
  creation/allocation is now documented as a real cost, not implied away.
- ADR 0006 (language standard, new): C++20 adopted for CXX and CUDA,
  explicitly evidenced for CPU only; CUDA compatibility flagged as
  unverified pending hardware access, consistent with how ADR 0001 already
  treats the CUDA backend itself.

Known limitations:
- CUDA remains completely unverified by compilation/execution -- this was
  true before this review pass and remains true after it. It is still the
  single most important follow-up before GPU-dependent work proceeds.
- The threaded CPU backend still creates threads/allocates its worker
  vector on every call; this review documented that honestly (ADR 0001)
  rather than fixing it, since a thread-pool redesign was out of scope for
  a review-fix pass and, per AGENTS.md #2, should be driven by measurement
  under a real repeated-launch workload rather than done preemptively.
- C++20's `nvcc` compatibility for this project's actual CUDA translation
  units is asserted from general knowledge of recent CUDA toolkit releases,
  not measured here -- flagged explicitly in ADR 0006 as the first thing to
  confirm once CUDA hardware is available.

Next recommended task:
Unchanged from the prior entry: get CUDA verified on real hardware. This
review pass makes that task more self-contained than before (the
`--extended-lambda` flag, launch/copy/allocation error checking, and
explicit `synchronize()` calls are now in place ahead of time), but the
core gap -- no CUDA compilation or execution evidence exists anywhere in
this project's history -- is unchanged.
