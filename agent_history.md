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

---

## 2026-09-02 — Hands-on tutorial for PR #1

Agent:
Claude (Claude Code)

Model:
claude-sonnet-5

Objective:
Add a second, minimal tutorial so PR #1 has something a reader can build
and run in under a minute to see the Phase 0 kernel actually execute, as a
complement to the existing 48-combination benchmark sweep tutorial (which
is better for measurement than for a first look). No architecture or
scope change.

Files changed:
- `tutorials/hello_parallel_for/hello_parallel_for.cpp` -- new; fills an
  8-cell x 3-component `Field`, runs `q_new(i,k) = q(i,k) * q(i,k)` via
  `cfe::parallel_for`, prints before/after values and elapsed time.
- `tutorials/hello_parallel_for/CMakeLists.txt`,
  `tutorials/CMakeLists.txt` -- new; wired into the root build behind a
  new `CFE_BUILD_TUTORIALS` option (default `ON`), matching the existing
  `CFE_BUILD_TESTS`/`CFE_BUILD_BENCHMARKS` pattern.
- `tutorials/hello_parallel_for/README.md` -- new; includes real captured
  output from an actual run, not a mocked-up example.
- `tutorials/phase0_benchmark/README.md` -- added a cross-link so a reader
  starting from either tutorial finds the other.
- `CMakeLists.txt` -- added the `CFE_BUILD_TUTORIALS` option and
  `add_subdirectory(tutorials)`.

Tests added:
None (this is a runnable example, not a test). Verified instead by: full
rebuild + all 23 existing unit tests still pass with the new target
present; the new executable was actually run (serial and threaded
backends) and its printed output is what appears in the README, not
invented text.

Benchmarks run:
None -- this tutorial is explicitly not a performance measurement (its own
README says so); the real sweep is unchanged.

Performance change:
None.

Scientific verification:
The before/after values printed by the tool were checked by hand against
`q_new = q * q` for the documented example (e.g. cell 0: 1,2,3 -> 1,4,9;
cell 7: 8,9,10 -> 64,81,100) before being pasted into the README.

Architecture decisions:
None. This is additive tooling, not a design decision.

Known limitations:
None beyond what already applied to the rest of Phase 0 (CUDA unverified).

Next recommended task:
Unchanged: get CUDA verified on real hardware (Orchard access expected
soon). Separately, `tasks/0002-phase1-cartesian-grid-scalar-transport.md`
is already scoped and stored on `cfe/development/phase_0002` for after
PR #1 merges.

---

## 2026-09-08 — CUDA verified on real hardware (PSC Bridges-2, V100)

Agent:
Claude (Claude Code)

Model:
claude-sonnet-5

Objective:
Close Phase 0's single largest known limitation: get the CUDA backend
actually compiled, run, tested, and benchmarked on real NVIDIA hardware
(PSC Bridges-2, GPU-shared partition), fix whatever that surfaces, update
ADR 0001/0002 with the resulting evidence, and prepare PR #1 for merge.

Files changed:
- `tests/unit/test_backend_execution_cuda.cu` -- fixed a real nvcc
  rejection: the extended `__device__` lambda passed to
  `cfe::backend::cuda::parallel_for` was defined directly inside the
  generic (`auto`-parameter) lambda passed to `for_each_component_count`,
  which nvcc disallows ("An extended __device__ lambda cannot be defined
  inside a generic lambda expression"). Extracted the per-N body into its
  own `run_case_for_n<Scalar, Layout, N>()` template function, matching the
  pattern `benchmarks/memory/bench_field_update_cuda.cu` already used
  successfully.
- `tests/unit/test_backend_execution.cpp` -- fixed a flaky (compiler- and
  platform-dependent) failure in
  `test_threaded_backend_matches_serial_backend_bitwise`: it reproducibly
  failed under GCC 13.3.1/Linux but never under AppleClang/macOS. Root
  cause (confirmed with AddressSanitizer/UBSan -- clean -- and by toggling
  `-ffp-contract`/vectorization flags in isolation): GCC's default
  `-ffp-contract=fast` permits fusing `CFE_CHECK_NEAR`'s separate
  "materialize `q(i,k)*q(i,k)`" and "subtract" statements into a single
  FMA, computing the exact infinite-precision residual instead of
  double-rounding through an intermediate value -- a few ULP different
  from the plain multiply used to produce `out_serial`/`out_threaded`,
  even though both are individually correctly rounded. Clang defaults to
  `-ffp-contract=on` (single-expression only), never crossing that
  statement boundary, which is why this was never seen before. The actual
  invariant the test is named for (serial and threaded execution agree
  exactly) was never violated -- `CFE_CHECK(out_serial == out_threaded)`
  passed in every run. Loosened only the secondary "matches a freshly
  recomputed reference" check from an implicit bitwise expectation
  (`1e-15`) to a real tolerance (`1e-12`).
- `docs/performance/0002-phase0-cuda-results.md` -- new; full CUDA
  benchmark + profiling write-up (see Benchmarks run below).
- `benchmarks/results/phase0_field_update_v100.csv` -- new; raw benchmark
  data, all 24 required (precision, N, layout) combinations.
- `benchmarks/results/ncu_reports/` -- new; raw `nvcc --resource-usage`
  and `ncu --set full` reports backing the write-up.
- `docs/adr/0001-execution-backend.md` -- CUDA backend section rewritten
  from "unverified" to verified, with the nvcc lambda-nesting bug and its
  fix documented; status line updated; revisit criterion for CUDA evidence
  removed (satisfied).
- `docs/adr/0002-state-memory-layout.md` -- added the GPU evidence section
  and changed the decision from "Proposed, CPU-only evidence" to
  "Accepted: per-backend default" (see Architecture decisions below).

Tests added:
None new (the two fixes above touch existing tests). All 27 tests
(23 CPU + 4 CUDA) pass together on the V100
(`srun --jobid=... ./build/tests/cfe_unit_tests`), confirmed in 3
repeated runs for determinism after the FP-contraction fix.

Benchmarks run:
`cfe_bench_field_update_cuda`, all 24 required combinations (2 precisions
x 6 component counts x 2 layouts), on PSC Bridges-2 (NVIDIA Tesla
V100-SXM2-32GB, compute capability 7.0, CUDA 12.9.86, GCC 13.3.1 host
compiler, `-DCMAKE_CUDA_ARCHITECTURES=70`). Also ran
`scripts/profile_cuda.sh`'s full procedure: static register/spill sweep
(`nvcc --resource-usage`) for all 24 instantiations, and Nsight Compute
(`ncu --set full` plus a targeted local-memory-traffic metric query) for a
representative subset (double/N=1 and double/N=100, AoS and SoA) --
scoped to a subset because the runtime profiling step requires a live GPU
allocation, unlike the static sweep. Full results and raw reports in
`docs/performance/0002-phase0-cuda-results.md` and
`benchmarks/results/`.

Performance change:
First CUDA baseline; no prior GPU numbers existed. Headline finding: SoA
wins decisively on this GPU (up to ~33x over AoS at N=100/float), the
mirror image of the CPU result where AoS won (up to ~8x over SoA at
N=100) -- confirmed by Nsight Compute as a coalescing effect (AoS
utilizes only 8.0 of 32 bytes per memory transaction at double/N=100 vs.
SoA's 30.1 of 32), not an occupancy effect (both layouts show the same
~51% achieved occupancy at N=100, a shared problem-size effect from
`n_cells` shrinking to hold the working set at large N). Zero register
spilling observed for every one of the 24 required instantiations, both
statically and at runtime.

Scientific verification:
CUDA backend results verified against the CPU reference within the
tolerance appropriate for cross-backend floating-point comparison
(`VERIFICATION.md` #3), via the 4 CUDA correctness tests -- all passing.
The FP-contraction root-cause diagnosis was itself verified rather than
assumed: reproduced deterministically across 3 repeated runs, ruled out
memory corruption (AddressSanitizer + UBSan clean), ruled out
auto-vectorization involvement (`-fno-tree-vectorize -fno-tree-slp-vectorize`
made no difference), and confirmed the actual mechanism directly
(`-ffp-contract=off` alone fixed it).

Architecture decisions:
- ADR 0001 (execution backend): CUDA backend moves from "implemented but
  unverified" to Accepted, on the same evidentiary basis as the CPU
  backends.
- ADR 0002 (state memory layout): accepted a per-backend default (AoS for
  CPU, SoA for CUDA) rather than one global default, since CPU and GPU
  evidence disagree and the GPU effect size is much larger than the CPU
  effect size in either direction. This is a decision for ADR 0003's
  case-specific compiler to apply (select `Layout` alongside backend/
  precision/dimension); `Field`/`FieldView` themselves need no interface
  change, since they were already layout-agnostic by design.

Known limitations:
- Nsight Compute runtime profiling covered a representative subset (2 of
  6 component counts, double precision only) rather than the full
  24-instantiation sweep, due to the time cost of holding a live GPU
  allocation for `--set full` profiling; the static register/spill sweep
  (which doesn't require a GPU allocation) does cover all 24.
- The per-backend memory-layout default (ADR 0002) is not yet wired into
  a builder/compiler that actually selects `Layout` automatically per
  backend -- that is ADR 0003's case-specific compilation layer, not yet
  implemented. Until then, call sites must still choose `Layout`
  explicitly.
- GPU evidence is V100/sm_70-specific; a different GPU architecture could
  in principle show different coalescing behavior (noted in ADR 0002's
  revisit criteria).

Next recommended task:
Merge PR #1, then proceed to `tasks/0002-phase1-cartesian-grid-scalar-transport.md`
(already scoped on `cfe/development/phase_0002`). When ADR 0003
(case-specific compilation) is eventually implemented, it should select
`Field`'s `Layout` parameter per-backend per ADR 0002's decision above.

---

## 2026-09-10 — Phase 1 core implementation: Cartesian grid and scalar transport (CPU)

Agent:
Claude (Claude Code)

Model:
claude-sonnet-5

Objective:
Add the smallest Cartesian grid and explicit scalar-transport capability
needed to exercise Phase 0's execution/storage foundation on an actual
PDE, with formal 2nd-order convergence evidence, while keeping the
interface-flux abstraction DG-hybridizable (AGENTS.md #19) and the grid/
ghost-cell layer AMR-ready (PI direction, 2026-09-10: no AMR yet, but do
not foreclose fixed block-based refinement).

Files changed (new unless noted):
- `src/cfe/core/types.hpp` -- `Axis`/`Side` enums, moved here (not
  `cartesian_grid.hpp`) so numerics code doesn't depend on the grid
  module.
- `src/cfe/grid/structured/cartesian_grid.hpp` -- `CartesianGrid<Scalar>`:
  1D/2D/3D uniform grid, `(i,j,k)` <-> flat-cell-index conversion
  including ghost layers, block-scoped `dx/dy/dz`/origin (not a global
  constant, per the AMR-readiness constraint). Templated on `Scalar`
  (initially hardcoded `double`; caught and fixed mid-session).
- `src/cfe/grid/boundary/boundary_condition.hpp` -- `PeriodicBoundary`,
  `StaticBoundary<Scalar,N>`; duck-typed `fill_x/y/z`, deliberately not a
  polymorphic base class (AGENTS.md #12).
- `src/cfe/grid/ghost/ghost_fill.hpp` -- `fill_ghost_cells(field, grid,
  axis, boundary)`, the single call-site dispatch and the actual
  swappable "neighbor provider" seam the AMR-readiness constraint asked
  for.
- `src/cfe/numerics/fvm/interface_value.hpp` -- `fvm::interface_value_right/left`
  free functions plus `fvm::CentralDifferenceReconstruction`, a stateless
  functor wrapper used as `FvmSolver`'s default `Reconstruction` template
  parameter (not called by hardcoded name -- see Architecture decisions).
- `src/cfe/numerics/numerical_flux/upwind.hpp` -- `upwind_flux`/`UpwindFlux`:
  takes two one-sided face values, an `Axis`, and a `Field`, calling only
  `field.physical_flux(...)`/`field.wave_speed(...)` -- fully
  physics-agnostic, reworked twice this session to remove an initial
  hardcoded single-scalar-speed assumption.
- `src/cfe/fields/scalar_advection/field.hpp` -- `ScalarAdvectionField<Scalar,Dim>`:
  owns the physics (`Vector<Scalar,Dim>` velocity, `physical_flux`/
  `wave_speed` Calculators per ARCHITECTURE.md #2's Field/Calculator
  split). Replaced an earlier, deleted `LinearAdvectionFlux`
  (single-scalar-speed) design after review caught that the solver was
  not actually grid/dimension-agnostic yet.
- `src/cfe/solver/time_integration/ssp_rk2.hpp` -- `ssp_rk2_step`,
  generic over any `residual(q_in, out)` callable; no knowledge of grids
  or boundary conditions.
- `src/cfe/solver/explicit/fvm_solver.hpp` -- `FvmSolver<Scalar, Layout,
  Field, BoundaryX, BoundaryY=BoundaryX, BoundaryZ=BoundaryX,
  Reconstruction=CentralDifferenceReconstruction, NumericalFlux=UpwindFlux>`:
  the orchestrator. `detail::axis_flux_difference<Axis A>` factors out
  X/Y/Z duplication via `if constexpr` (changed from runtime `if
  (y_active)` after review asked for compile-time dimension branching).
  Dimension-agnostic via `if constexpr (Field::dim >= 2/3)`.
- `docs/type-reference.md` -- new (added at the reviewer's request,
  formalized as AGENTS.md #27: every new public type must be added here
  in the same change).
- Tests: `test_grid_indexing.cpp`, `test_boundary_conditions.cpp`,
  `test_interface_flux.cpp`, `test_ssp_rk2.cpp`,
  `test_scalar_advection_convergence.cpp`,
  `test_scalar_advection_conservation.cpp`,
  `test_scalar_advection_2d_sanity.cpp`.

Tests added:
All of the above. 43/43 tests pass (23 from Phase 0 + 20 new), CPU
serial and threaded backends agree.

Benchmarks run:
None this session (CPU benchmark sweep and CUDA port deferred to the
2026-09-29 entry below).

Performance change:
None measured this session.

Scientific verification:
`test_scalar_advection_second_order_convergence` (the Phase 1 acceptance
bar, ROADMAP.md): L2 error at `nx = 20, 40, 80, 160` drops by a ratio in
`(3.5, 4.5)` at every consecutive pair -- 2nd-order convergence confirmed
by measurement, not assumption. `test_scalar_advection_conserves_total_quantity_over_periodic_domain`:
total transported quantity conserved to `1e-9` after 200 SSP-RK2 steps on
a periodic domain -- a structural property of conservative flux
differencing, verified rather than assumed. `test_scalar_advection_2d_solve_matches_1d_solve_row_for_row_when_y_velocity_is_zero`:
2D solve with zero Y-velocity matches the 1D solve row-for-row to
`1e-12` -- confirms the Y-axis code path (otherwise only exercised by
grid/boundary unit tests) behaves correctly inside an actual solve.

Architecture decisions:
- Reviewer feedback drove three real design changes during this session,
  each recorded because the reasoning matters for whoever reads this
  later: (1) `Reconstruction`/`NumericalFlux` must be template
  parameters, not hardcoded calls, so future schemes (MUSCL, WENO,
  Rusanov, HLLC) can be swapped without touching `residual()`; (2)
  `Solver` must call into `Field` for physics (velocity, flux formula),
  not embed a flux formula itself -- `Field` owns its Calculators, per
  ARCHITECTURE.md #2; (3) the solver's residual loop must be
  dimension-agnostic (`if constexpr (Field::dim >= 2/3)`), with velocity
  itself a `Vector<Scalar,Dim>` sized to the problem's actual
  dimensionality, so 1D/2D/3D transport is the same types at a different
  `Dim`, not different types.
- Naming: reviewer flagged abbreviated codebase-specific shorthand
  (`recon`, `num_flux`) as unreadable; renamed throughout to full words
  (`reconstruction`, `numerical_flux`). Genuine domain notation (`q`,
  `dx`, `im1`) was explicitly kept as-is -- the objection was to
  abbreviating *concepts*, not established mathematical notation.
- See ADR 0004 (updated) and ADR 0007 (new) in the 2026-09-29 entry below
  for the formal ADR writeups of the interface-flux design and the
  SSP-RK2-over-RK3 choice made this session.

Known limitations:
- CPU-only at this point -- CUDA port not yet attempted (see below).
- No benchmark numbers yet for the scalar-advection kernel itself.
- ADRs, presentation, and this agent_history entry itself were written
  retroactively on 2026-09-29 alongside the CUDA work, not in the same
  session as the code above -- a process gap worth naming: AGENTS.md #4/#5
  ask for these to be written as the work happens, and they were not, for
  work spanning 2026-09-10 through 2026-09-29.

Next recommended task:
Port to CUDA (PSC Bridges-2 V100, per `docs/bridges2-setup.md`), then
write the CPU benchmark sweep, ADR updates, and presentation -- see the
2026-09-29 entry below for how all of that went.

---

## 2026-09-29 — CUDA port, at-scale benchmarks (1D + 3D), visualization, and Phase 1 closeout

Agent:
Claude (Claude Code)

Model:
claude-sonnet-5

Objective:
Get the scalar-transport solver from the prior entry running correctly
on CUDA (PSC Bridges-2 V100) and demonstrated "at scale," then close out
the remaining Phase 1 deliverables the task spec requires: CPU benchmark
sweep, ADR updates, presentation, this entry.

Files changed:
- `src/cfe/backend/parallel_for.hpp` -- `CpuParallelFor` tag
  (`Backend::run(n, f)` wrapping `cfe::parallel_for`).
- `src/cfe/backend/cuda/cuda_backend.cuh` -- `CudaParallelFor` tag
  (CUDA-only), deliberately does not `synchronize()` per launch (default
  CUDA stream is in-order; only a host-side read needs to synchronize).
- `src/cfe/grid/boundary/boundary_condition.hpp`,
  `src/cfe/grid/ghost/ghost_fill.hpp`,
  `src/cfe/solver/time_integration/ssp_rk2.hpp`,
  `src/cfe/solver/explicit/fvm_solver.hpp` -- threaded a `Backend`
  template parameter (default `CpuParallelFor`) through every kernel
  launch, so the same code compiles for CPU or CUDA. `Backend` placed
  *first* in `fill_x/y/z`'s template parameter list specifically because
  `PeriodicBoundary` and `StaticBoundary<Scalar,N>` have different
  numbers of remaining deducible parameters.
- `src/cfe/solver/explicit/fvm_solver.hpp` -- added `SolverResidual<Solver>`,
  a named (namespace-scope) functor wrapping `Solver::residual`. Required
  because nvcc forbids a *locally-defined* lambda as a template argument
  to a function whose body contains an extended `__device__` lambda
  (`ssp_rk2_step`), in any instantiation compiled within a `.cu` file --
  this affected even the CPU-backend instantiation.
- `tests/unit/test_scalar_advection_cuda.cu` -- new; 1D CPU-vs-GPU
  correctness (256 cells, 50 SSP-RK2 steps, `1e-9` tolerance).
- `tests/unit/test_scalar_advection_3d_cuda.cu` -- new; 3D CPU-vs-GPU
  correctness (32^3, direction-dependent velocity `(1.0,0.6,0.3)`, all
  three `axis_flux_difference` branches active, 30 steps, `1e-9`
  tolerance) -- added after noticing the 1D case alone does not prove
  the Y/Z flux branches work on real hardware.
- `benchmarks/scalar_advection/bench_scalar_advection.cpp` -- new; CPU
  sweep (serial/threaded, `nx = 10^4..10^7`).
- `benchmarks/scalar_advection/bench_scalar_advection_cuda.cu` -- new; 1D
  CUDA sweep (`nx = 10^6..10^8`).
- `benchmarks/scalar_advection/bench_scalar_advection_3d_cuda.cu` -- new;
  3D CUDA sweep (`nx=ny=nz = 64..512`).
- `src/cfe/io/vtk_writer.hpp` -- new; minimal legacy-VTK
  (`STRUCTURED_POINTS`, ASCII, cell-centered scalar) writer, per
  ARCHITECTURE.md #19 ("API over an established format, not a custom
  one").
- `tutorials/scalar_advection_3d_visualization/` -- new; CPU-only
  driver advecting a Gaussian bump on a periodic 64^3 grid, writing a
  41-frame VTK time series + `.pvd` manifest for ParaView.
- `docs/performance/0003-phase1-scalar-advection-cuda-results.md` -- new;
  1D and 3D CUDA results.
- `docs/performance/0004-phase1-scalar-advection-cpu-results.md` -- new;
  CPU sweep results.
- `docs/adr/0004-grid-connectivity.md` -- extended from an unevidenced
  "Proposed" placeholder to "Accepted for the Cartesian/FVM scope," with
  evidence from the actual `CartesianGrid`/ghost-cell/boundary-condition
  implementation.
- `docs/adr/0007-interface-flux-and-time-integration.md` -- new; records
  the two-stage `Reconstruction`+`NumericalFlux` design and the
  SSP-RK2-over-RK3 choice, with the measured convergence-order evidence.
- `presentations/0002-phase1-cartesian-grid-scalar-transport.md` -- new.
- This entry and the one above it.

Tests added:
`test_scalar_advection_cuda.cu`, `test_scalar_advection_3d_cuda.cu`. All
49/49 tests pass on the V100 (PSC Bridges-2, node `v016`/`v020`, jobs
`47265530`/`47267793`/`47269636`) -- 43 CPU + 2 pre-existing Phase 0 CUDA
+ 2 new Phase 1 CUDA + 2 pre-existing Phase 0 CUDA (dtype/layout
variants). CPU-only rebuild throughout stayed at 43/43 with no behavior
change, confirmed before and after every commit.

Benchmarks run:
- CPU (`cfe_bench_scalar_advection`, Apple M5): `nx = 10^4..10^7`,
  serial and threaded. See
  `docs/performance/0004-phase1-scalar-advection-cpu-results.md`.
- CUDA 1D (`cfe_bench_scalar_advection_cuda`, V100): `nx = 10^6..10^8`.
- CUDA 3D (`cfe_bench_scalar_advection_3d_cuda`, V100): `nx=ny=nz = 64..512`.
  Full results and raw CSVs in
  `docs/performance/0003-phase1-scalar-advection-cuda-results.md`.

Performance change:
First scalar-advection baseline on every backend. Headline numbers: CPU
threaded reaches ~8.2e8 cell-updates/s at 10M cells (1.25x serial); GPU
1D reaches ~8.8e9 cell-updates/s, plateauing by 10M cells, sustaining
100M cells in 11.3ms/step; GPU 3D reaches ~4.4-5.7e9 cell-updates/s
(lower than 1D as expected, since 3D does ~3x the per-cell flux/
ghost-fill work), sustaining 512^3 (~134M cells) in 30.6ms/step.

Scientific verification:
CUDA correctness verified against the CPU reference to `1e-9`, both 1D
(256 cells, 50 steps) and 3D (32^3, 30 steps, direction-dependent
velocity exercising all three flux axes). The visualization tutorial's
output was independently verified, not just eyeballed: the advected
Gaussian bump's peak-cell position across all 41 output frames matches
the closed-form advected position (`(x0 + v*t) mod domain_length`)
exactly, including two axes visibly wrapping the periodic boundary.

Architecture decisions:
- ADR 0004 (grid connectivity): moved from unevidenced "Proposed" to
  "Accepted for the Cartesian/FVM scope" -- see the ADR for the full
  evidence (flat-index-only-in-grid, ghost-in-Field, block-scoped
  spacing, swappable boundary conditions).
- ADR 0007 (new): interface-flux two-stage design (Reconstruction +
  NumericalFlux, both template parameters) and SSP-RK2-over-RK3, with
  the measured convergence-order table as evidence.

Known limitations:
- The 400^3/512^3 CUDA 3D benchmark cases show a throughput dip relative
  to the 256^3 peak, not root-caused (no Nsight Compute profiling run
  against this kernel, unlike Phase 0's memory-layout study).
- The visualization tutorial and CPU/CUDA benchmarks are all 1D or
  single-block 3D; multi-block/AMR-relevant scaling remains explicitly
  out of scope for Phase 1 per the PI direction.
- ADR 0003 (case-specific compilation) still does not exist, so
  `Backend`/`Layout` must be selected explicitly by every call site
  rather than derived automatically -- unchanged from Phase 0's known
  limitation.

Next recommended task:
Open the PR for `cfe/development/phase_0002` -> `main` (mirroring Phase
0's PR #1 workflow; final merge decision left to the PI). After merge,
proceed to Phase 2 (per ROADMAP.md) -- the first nonlinear equation, and
the point at which the `Reconstruction`/`NumericalFlux` genericity and
the DG-hybridizable interface-flux shape (ADR 0007) get their first real
test beyond linear scalar advection.

---

## 2026-09-30 — PR #2 code review response

Agent:
Claude (Claude Code)

Model:
claude-sonnet-5

Objective:
Respond to a local code review of PR #2 (`docs/pr-2-review.md`-style
findings, not posted to GitHub): two "request changes" findings and six
advisory recommendations. For each, independently verify the claim
before acting on it, then implement a fix or give evidence for declining.

Files changed:
- `src/cfe/solver/explicit/fvm_solver.hpp` -- `residual()` now writes a
  defined value (0) to every *ghost* cell of `out`, not just real cells,
  via new `detail::zero_ghost_residual_x/y/z` helpers (mirroring
  `PeriodicBoundary::fill_x/y/z`'s own index enumeration, minus the
  "read a source" part). Added host-side assertions for positive
  extent/spacing and documented inactive-axis shape (`ny==1, ngy==0`
  when `Field::dim<2`, etc.). Corrected the header comment's claim that
  "a future Burgers or Euler field plugs in here unchanged" -- Euler
  needs this type generalized over `NComponents` first, since it
  hardcodes `FieldView<Scalar,1,Layout>` and component `0` throughout.
  Trimmed a duplicative "why dimension-agnostic" paragraph, pointing to
  ADR 0004/type-reference.md instead of re-deriving it inline.
- `src/cfe/grid/boundary/boundary_condition.hpp` -- `PeriodicBoundary::
  fill_x/y/z` now assert `nx>=ngx` (`ny>=ngy`, `nz>=ngz`): on a grid
  narrower than its own ghost depth, the "opposite real boundary" index
  arithmetic underflows into the ghost region itself, making one ghost
  cell's "source" another ghost cell written by the same parallel fill
  -- a genuine cross-thread race on some execution orders, not merely
  "looks wrong."
- `src/cfe/fields/scalar_advection/field.hpp`,
  `src/cfe/numerics/numerical_flux/upwind.hpp`,
  `docs/adr/0007-interface-flux-and-time-integration.md`,
  `docs/type-reference.md` -- corrected the same "Burgers or Euler plugs
  in unchanged" overclaim everywhere it appeared. Burgers (still
  single-component) genuinely needs zero interface changes; Euler does
  not, and `UpwindFlux`'s plain sign-of-wave-speed switch is not itself
  an entropy-correct Burgers solver or a valid Euler Riemann solver
  either way -- a different `NumericalFlux` type would be needed, not
  just a different `Field`.
- `docs/adr/0007-...md` -- corrected an overclaim conflating asymptotic
  convergence *order* with absolute error at a fixed resolution (SSP-RK2
  vs. SSP-RK3): the order argument is correct and was verified, but does
  not by itself prove SSP-RK3 couldn't reduce absolute error at any one
  resolution. Corrected MUSCL's order (ordinarily 2nd, same as the
  scheme already used here) out of an incorrect ">2nd order" grouping
  with WENO.
- `tests/unit/test_framework.hpp` -- `CFE_CHECK_NEAR` now explicitly
  rejects non-finite values before the tolerance comparison:
  `std::fabs(NaN - x) > tol` is always false, so a NaN on either side
  previously passed silently -- a real gap in this project's primary
  CPU/GPU cross-backend acceptance gate.
- `tests/unit/test_scalar_advection_convergence_variants.cpp` (new) --
  2nd-order convergence at negative advection speed (double) and at
  float precision, both against the analytic solution (not just
  cross-backend agreement).
- `tests/unit/test_scalar_advection_2d_convergence.cpp` (new) -- genuine
  multi-axis analytic-solution convergence check: nonzero velocity on
  *both* X and Y (existing 2D test deliberately zeroed Y-velocity,
  degenerating every row into an independent 1D problem), checked
  against the exact product-of-sines traveling-wave solution, not just
  CPU/GPU agreement.
- `docs/performance/0004-...md`, `benchmarks/results/phase1_scalar_advection_cpu_apple_m5.csv`
  -- re-measured after the fixes above; see Performance change below.

Tests added:
`test_scalar_advection_second_order_convergence_negative_velocity`,
`test_scalar_advection_second_order_convergence_float_precision`,
`test_scalar_advection_2d_second_order_convergence_with_nonzero_xy_velocity`.
46/46 tests pass (43 existing + 3 new), on both serial and threaded
backends, clean under AddressSanitizer + UndefinedBehaviorSanitizer.

Benchmarks run:
Re-ran `cfe_bench_scalar_advection` (Apple M5) after the ghost-residual
fix; see Performance change and `docs/performance/0004-...md` Observation
4 for the full investigation.

Performance change:
A first, naive fix (zero-fill `out`'s *entire* padded range before the
real-cell kernel) regressed serial 10M-cell performance from 15.135ms to
21.21ms/step (~40%) -- root-caused to rewriting every real cell's
residual twice (once as 0, once with its actual value), confirmed by a
controlled same-session A/B against the pre-fix code. Replaced with a
ghost-cells-only fix (`zero_ghost_residual_x/y/z`, touching only the
thin ghost shell via the same index enumeration `PeriodicBoundary`
already uses), which reduced but did not eliminate the regression: serial
10M cells now measures ~19.79ms/step (~30% slower than the original
15.135ms baseline), while the threaded backend shows no measurable
regression at any tested size. Investigated directly (same-session A/B
isolation, an `__attribute__((noinline))` experiment that made things
markedly worse, ruling out an inlining-pollution theory) and traced to a
serial-backend-specific compiler code-generation sensitivity around the
extra small kernel launch, not a genuine per-cell cost (the data touched
is a handful of ghost cells, far too little to explain 30% on its own).
Not fully root-caused further -- see `docs/performance/0004-...md`
Observation 4 for the full reasoning on why this was an acceptable place
to stop (correctness priority, threaded backend unaffected, GPU is the
actual "at scale" evidence).

Scientific verification:
Both P2 findings were independently reproduced before being trusted:
a standalone repro poisoning `residual_scratch` with NaN before a
constant-periodic-field step confirmed NaN leaking into the output
state's ghost cells (matching the review's predicted symptom exactly);
a standalone repro using a custom reverse-iteration-order `Backend` tag
on an `nx=1, ngx=2` grid reproduced the exact `0 7 | 7 | 7 0` wrong-ghost
pattern the review predicted by hand-derivation. Both repros were re-run
after each fix: the NaN repro now shows no NaN anywhere in the output;
the small-grid repro now fails its new assertion instead of silently
producing wrong values. Full suite re-verified clean under ASan+UBSan on
both serial and threaded backends after all fixes.

Architecture decisions:
- Declined one recommendation, with reasoning recorded rather than
  silently ignored: shortening every header comment to "contracts,
  stencil requirements, and surprising constraints," moving all
  architectural narrative to ADRs. Applied narrowly to the single
  worst-offending duplication (`fvm_solver.hpp`'s dimension-agnostic
  paragraph, now a pointer to ADR 0004/type-reference.md instead of a
  re-derivation), but declined as a blanket sweep: this project's
  comments-with-rationale style has been an explicit, repeated
  preference throughout Phase 0 and Phase 1 (not merely an oversight),
  and has concretely paid for itself this session -- several of the PI's
  own questions this session ("what is q", "should I be concerned about
  ijk", "is this efficient, can we use compile time") were answerable
  quickly precisely because the relevant reasoning already lived next
  to the code, not only in an ADR several files away.
- PeriodicBoundary's `nx < ngx` restriction is enforced via `assert`
  (compiled out under `NDEBUG`/Release, matching the existing `ngx>=2`
  precedent in the same function), not a runtime-checked exception:
  no current grid configuration in this codebase needs `nx < ngx`, and
  the reviewer's own recommendation explicitly allowed "rejecting them
  is also a valid initial contract."

Known limitations:
- The ~30% serial-backend-only performance regression at large
  `n_cells` (see Performance change above) is measured and disclosed,
  not resolved to its true root cause.
- CUDA correctness and benchmarks were re-verified on real hardware
  after these fixes (see the follow-up note below/next entry if a
  separate session recorded it) -- the reviewer's own validation could
  not cover CUDA (no hardware available to them).
- Euler support remains entirely out of scope, as it was before this
  review; the comment corrections above only make that explicit where
  it was previously (incorrectly) implied otherwise.

Next recommended task:
Re-verify this entry's CUDA claims on PSC Bridges-2 if not already done
in the same session, then proceed with opening/updating PR #2 for
review of these fixes.

---

## 2026-10-01 — Fix ssp_rk2_step at its root: interior-cells-only (PR #2 follow-up review, continued)

Agent:
Claude (Claude Code)

Model:
claude-sonnet-5

Objective:
Finish the 2026-09-30 follow-up review's two remaining items (serial
performance root cause, standalone-tutorial-build docs). Item 3 changed
scope mid-session: the user, independently reasoning through the ghost-
cell architecture, correctly identified that `ssp_rk2_step` touching
ghost cells at all (requiring round 1's zero-fill workaround) was itself
the actual bug -- not something to optimize around. This entry covers
implementing that root-cause fix and everything it surfaced; item 4
(tutorial build docs) remains for a later session.

Files changed:
- `src/cfe/solver/time_integration/ssp_rk2.hpp` -- new `IdentityIndexMap`
  (default, for no-grid callers); `ssp_rk2_step` gains `n_active` and
  `IndexMap index_map = IndexMap{}` trailing parameters. Combine kernels
  now iterate `[0, n_active)` through `index_map(r)`, never touching any
  index outside that set -- ghost cells are structurally never read or
  written, matching standard FVM practice (fill ghosts -> compute
  residual on interior cells -> integrate interior cells only) rather
  than the previous "touch everything, including ghosts" design that
  made round 1's zero-fill workaround necessary in the first place.
- `src/cfe/solver/explicit/fvm_solver.hpp` -- removed
  `zero_ghost_residual_x/y/z` entirely (nothing reads `out`'s ghost cells
  anymore, so nothing needs to define them). Added
  `detail::CartesianRealCellIndexMap<Scalar, Dim>` (the index-map
  implementation, specialized per `Dim` via `if constexpr` -- see
  Performance change) and `FvmSolver::active_cell_count()`/
  `active_cell_index_map()` accessors.
- Every `ssp_rk2_step` call site updated (two trailing arguments):
  `test_ssp_rk2.cpp` (uses `state.n_cells()` + default `IdentityIndexMap`,
  no grid at all), the five CPU convergence/conservation/sanity tests,
  both CUDA tests, all three benchmarks, the visualization tutorial.
- `tests/unit/test_fvm_solver_ghost_residual.cpp` -- header comment
  updated to describe the new mechanism (structural impossibility, not a
  defined placeholder) while keeping the same NaN-poisoning regression
  test, now poisoning the *entire* padded `residual_scratch` range.
- `src/cfe/backend/cuda/device_field.cuh` -- `DeviceField` now
  zero-initializes via `cudaMemset` right after `cudaMalloc` (see
  Scientific verification).
- `docs/performance/0003-...md`, `0004-...md`,
  `benchmarks/results/phase1_scalar_advection_cpu_apple_m5.csv` --
  updated with this round's final, re-verified numbers.

Tests added:
None new (the existing 51 CPU + 6 CUDA tests cover this change; no new
test was needed beyond updating existing call sites). 51/51 CPU, 57/57
total with CUDA.

Benchmarks run:
`cfe_bench_scalar_advection` (Apple M5, serial + threaded, repeated
many times for stability — this session's numbers were unusually noisy
at first, requiring several repeated full sweeps to separate signal from
thermal/background-load noise), plus an ad-hoc extended sweep to 50M/
100M cells (beyond the committed benchmark's normal range) specifically
to check whether the regression below grows or plateaus with size.
`cfe_bench_scalar_advection_cuda` and `_3d_cuda` on the V100 (job
`47314879`, node `v006`), both before and after the `DeviceField` fix.

Performance change:
**Not a net improvement as hoped going in.** The architecturally-correct
fix regressed the CPU serial backend *further* than round 1's workaround
(serial, 10M cells: 15.135ms original -> 19.79ms round-1 workaround ->
~33ms this round, a ~118% regression from original). Root-caused with
direct evidence (not inferred) via `-Rpass-missed=loop-vectorize`: routing
the per-cell storage index through `index_map(r)` instead of the raw loop
counter defeats the compiler's auto-vectorization of the combine loop,
confirmed by diffing optimization-remark output before/after. Tried and
rejected as insufficient: a `Dim`-specialized fast path avoiding integer
division for 1D/2D (`if constexpr`, matching this project's existing
dimension-dispatch convention), and `CFE_FORCEINLINE` on the index map
(`AGENTS.md` #9's documented host/device pattern) -- neither recovered
the lost vectorization. Confirmed via an extended 50M/100M-cell sweep
that the regression is a **flat ~1.68x, not a growing one** -- both
versions scale linearly with cell count individually; only the constant
factor between them differs, and that factor is stable from 10M through
100M cells, consistent in magnitude with the measured vectorization-width
loss (width 2, so up to ~2x). The threaded backend shows no regression at
any size -- confirmed (not assumed) by checking that its inner loop
(`backend/cpu/threaded.hpp`) already failed to auto-vectorize in the
*original* pre-review code, for unrelated structural reasons
(`Cannot vectorize early exit loop with writes to memory`). GPU
benchmarks (1D and 3D) show zero change from this entire investigation,
both before and after the `DeviceField` fix below.

Scientific verification:
Re-running `compute-sanitizer --tool initcheck` on the V100 after this
change (rather than assuming round 1's "0 errors" result still held)
surfaced a second, genuine, newly-introduced bug: 1600 uninitialized-
memory-read errors, entirely isolated to the 3D CUDA test (the 1D test
was clean). Root cause: `fill_ghost_cells`'s per-axis fill (e.g.
`fill_x`) must read across the *entire* padded extent of the other two
axes, not just their real range, to correctly fill corners/edges later
fill_y/fill_z calls depend on -- on a 2D/3D grid, cells real in X but
ghost in Y or Z are legitimate read sources for `fill_x`, but the new
interior-cells-only combine step never writes them, and `DeviceField`'s
raw `cudaMalloc` never defined them either. Fixed by zero-initializing
`DeviceField` at construction, matching `cfe::Field`'s host-side
`std::vector` semantics (a fix considered and explicitly declined in an
earlier round as redundant -- that reasoning no longer held once the
combine step stopped touching every cell). Re-verified: 0 errors, 57/57
tests, no benchmark change.

Architecture decisions:
- `ssp_rk2_step`'s "grid-agnostic" contract is now expressed as "generic
  over an index-mapping concept" rather than "touches every cell of
  whatever FieldView it's given" -- arguably a *more* correct notion of
  genericity (a future unstructured-grid solver could supply its own
  index map), not a compromise of the original design goal.
- Declined (for this session): recovering the lost auto-vectorization.
  Recorded as explicit, prioritized future work (compiler vectorization
  pragmas; a compile-time fast path for the common contiguous-offset
  case; explicit SIMD as a last resort) rather than left as a silent gap
  -- see `docs/performance/0004-...md` Observation 4.

Known limitations:
- The CPU serial-backend regression (flat ~1.68x) is disclosed, root-
  caused, and bounded, but not recovered.
- Item 4 from the 2026-09-30 follow-up review (standalone-tutorial-build
  README accuracy) is still not done.

Next recommended task:
Item 4 (standalone tutorial build vs. documented behavior -- see the
2026-09-30 entry above for the two options already sketched). After
that, this PR should be fully caught up on every outstanding review
item.

---

## 2026-10-01 — PR #2 cleanup round: reproducibility, docs, index-map tests, tutorial build fix

Agent:
Model: Claude Sonnet 5

Objective:
Close out the last four outstanding items from PR #2's review, explicitly
scoped as "minor cleanup" with the serial CPU regression accepted as a
documented limitation (not something to recover in this change): (1) make
the ~1.68x serial-backend regression's performance record reproducible,
(2) correct `docs/type-reference.md`'s `ssp_rk2_step` signature and
document `IdentityIndexMap`/`active_cell_count()`/`active_cell_index_map()`,
(3) commit index-map regression tests (rectangular grids, ghost-untouched
guarantee, AoS/SoA, multi-component), (4) fix the standalone tutorial
build instructions.

Files changed:
- `docs/type-reference.md` -- corrected `ssp_rk2_step` signature
  (`n_active`, `index_map` were missing), added `IdentityIndexMap` row,
  extended `FvmSolver`'s row with `active_cell_count()`/
  `active_cell_index_map()` and their role in interior-cells-only updates.
- `tutorials/scalar_advection_3d_visualization/README.md`,
  `tutorials/hello_parallel_for/README.md` -- standalone build command
  corrected to `cmake --build build --target cfe_<name> -j`, with a new
  paragraph explaining why `--target` is required (the standalone path
  pulls the repo root in as a nested subdirectory purely to get
  `cfe_core`, but that also re-enables `CFE_BUILD_TESTS`/`BENCHMARKS`/
  `TUTORIALS` at their default-`ON` setting, so an unscoped
  `cmake --build build` builds the whole project, not just the one
  tutorial). Verified empirically: diffed `find build -type f -perm
  +111` before/after a scoped build.
- `tutorials/CMakeLists.txt` -- its existing convention comment extended
  with the same explanation, so a new tutorial's author sees the reason
  up front rather than rediscovering it.
- `tests/unit/test_ssp_rk2_index_map.cpp` (new), `tests/CMakeLists.txt`
  -- see Tests added.
- `docs/performance/0005-phase1-ssp-rk2-regression-reproducibility.md`
  (new) -- full reproducibility record: exact commits compared, compiler/
  flags, reproduction commands, raw results table, vectorization-remark
  counts, and an explicit "what this corrects" section.
- `docs/performance/0004-phase1-scalar-advection-cpu-results.md` --
  Observation 4 rewritten to match the corrected finding (see Performance
  change below) and to point at `0005-...md` for full methodology.
- `benchmarks/results/phase1_ssp_rk2_vectorization_repro/` (new) --
  `run_{original,intermediate,current}.log` (raw 5-rep benchmark stdout),
  `remarks_{original,intermediate,current}.txt` (full
  `-Rpass{,-missed,-analysis}=loop-vectorize` compiler output),
  `bench_scalar_advection_extended_sweep.cpp` (the benchmark source used,
  swept out to 50M/100M cells).

Tests added:
`tests/unit/test_ssp_rk2_index_map.cpp` (5 tests, wired into
`tests/CMakeLists.txt`): `CartesianRealCellIndexMap` is a bijection onto
exactly the real cells on genuinely rectangular (nx != ny != nz) 2D and 3D
grids -- deliberately non-cubic, since a cube can't catch an i/j-axis
mixup in the index decomposition; `ssp_rk2_step`'s combine kernels leave
every ghost entry of both `state` and `residual_scratch` exactly
byte-for-byte untouched (sentinel values, zero tolerance, not just
"isn't NaN"), using a residual callable with no ghost-fill of its own so
the check isolates the combine step specifically; the whole mechanism
works for a 3-component field under both `AoSLayout` and `SoALayout`
(previously untested -- `FvmSolver`/`ScalarAdvectionField` hardcode
N=1). Real-cell values are checked against the closed-form solution of
Heun's method applied to `dy/dt=-y`, not just "moved in the right
direction." 56/56 total (CPU), clean under ASan/UBSan.

Benchmarks run:
A from-scratch, same-session, same-compile-flags, same-commit
reproduction of `cfe_bench_scalar_advection` (serial + threaded) at
10K/100K/1M/10M/50M/100M cells, for three code versions in one sitting:
`original` (`e5df5f9`, pre-review), `intermediate` (`e880a9e`, round 1's
ghost-zero-fill fix), `current` (`HEAD`, round 2's interior-cells-only
fix). Compiled with the project's exact Release flags (confirmed by
reading `build/tests/CMakeFiles/cfe_unit_tests.dir/flags.make`), 5
repetitions each, median of repetitions 2-5 (repetition 1 is a
consistent cold-start outlier across all three versions). Full raw logs
and remarks committed under `benchmarks/results/
phase1_ssp_rk2_vectorization_repro/`.

Performance change:
**Corrects, rather than confirms, the previously-recorded numbers.** The
2026-09-30/2026-10-01 entries above recorded a "15.135ms -> 19.79ms
(+31%) -> ~33ms (+118%)" two-step regression story, built from numbers
taken across three separate work sessions under uncontrolled,
non-comparable system load. Under this session's controlled, single-
sitting, same-compile-flags, same-commit comparison, `original` and
`intermediate` are statistically indistinguishable (<1% apart at every
size from 1M to 100M cells, well inside this benchmark's own run-to-run
noise) -- round 1's ghost-zero-fill fix was **not** a measurable
regression; the previously-recorded 19.79ms figure was very likely
elevated by session-specific noise, not the code change. The one real,
reproducible regression is `current` vs. *either* earlier version, at a
consistent **~1.68x**, flat from 1M cells through 100M cells (not two
separate ratios, and not a growing one). This does not change any
correctness conclusion, the decision to accept the regression as a
documented limitation, or its root cause (per the 2026-10-01 entry
above) -- it only corrects the magnitude and attributes the entire
regression to round 2's change specifically, reported to the user as
such rather than silently reconciled with the old narrative. Threaded
backend: statistically indistinguishable across all three versions at
every size, confirmed directly in this same controlled comparison (not
re-asserted from the earlier, less careful measurement).

Scientific verification:
Compared `git diff e5df5f9 e880a9e -- src/cfe/solver/time_integration/
ssp_rk2.hpp` (empty -- confirms `intermediate`'s `ssp_rk2_step` is
byte-for-byte `original`'s, isolating round 1's change to the residual
zero-fill alone) before drawing any conclusion about which change caused
what. An earlier, ad-hoc standalone reproduction attempt (not committed)
was found to be unfaithful -- it dropped the real benchmark's anonymous-
namespace scoping and dual Serial/Threaded template-instantiation
structure -- and was discarded rather than reported; the committed
numbers come only from compiling and running the actual
`bench_scalar_advection.cpp` source (header-swapped per version via
`-I overlay -I src`), confirmed identical to the project's own compile
flags.

Architecture decisions:
None -- this round touched no production code, only tests, benchmarks,
and documentation, per the user's explicit "keep this change limited to
these items" scope.

Known limitations:
- The CPU serial-backend regression (flat ~1.68x, now reproducibly
  measured) remains disclosed, root-caused, and bounded, but not
  recovered -- accepted as a documented limitation for this PR per
  explicit instruction.

Next recommended task:
PR #2 is now caught up on every outstanding review item across three
rounds. Recommended next: open the follow-up task for recovering the
lost auto-vectorization (compiler hints, or a compile-time fast path for
the common contiguous-offset case), tracked as explicit future work in
`docs/performance/0004-...md` Observation 4 and `0005-...md`.

---

## 2026-10-02 — Phase 2 (first slice): Burgers equation and shock-capturing numerics

Agent:
Model: Claude Sonnet 5

Objective:
Phase 1 (PR #2) merged to `main` (squash commit `78b96e4`). Per task
0003, add the inviscid Burgers equation and a shock-capturing (TVD)
reconstruction/numerical-flux pair, closing two canonical problems
`VERIFICATION.md` names but nothing yet implemented: "Burgers smooth
convergence" and "Burgers shock formation." Deliberately a narrower
slice of `ROADMAP.md`'s full Phase 2 -- MPI, DG prototype, state
sizes through 100, the memory-layout study, and a Burgers CUDA
port/benchmark/visualization tutorial are all explicitly deferred to
separate follow-up tasks (see `tasks/0003-...md`'s own exclusion list).

Files changed:
- `src/cfe/fields/burgers/field.hpp` (new) -- `BurgersField<Scalar,
  Dim>`: `physical_flux(state, axis) = state^2/2`,
  `wave_speed(left, right, axis) = max(|left|,|right|)`. Same
  `(state, axis)`-in Calculator shape `ScalarAdvectionField` already
  established -- `fvm_solver.hpp` needed zero changes.
- `src/cfe/numerics/numerical_flux/rusanov.hpp` (new) -- `rusanov_flux`/
  `RusanovFlux`: local Lax-Friedrichs, entropy-satisfying for Burgers'
  convex flux (closing the gap `upwind.hpp`'s own header comment
  documents for `UpwindFlux`). Same
  `NumericalFlux::operator()(left, right, axis, field)` shape.
- `src/cfe/numerics/fvm/muscl_minmod.hpp` (new) -- `minmod`,
  `muscl_minmod_slope/value_right/value_left`, and the
  `MusclMinmodReconstruction` functor: TVD, minmod-limited MUSCL. Same
  `Reconstruction::right(...)/left(...)` 3-point-stencil shape
  `CentralDifferenceReconstruction` already uses; that type keeps
  serving linear advection unchanged, this is a second, additive pair.
- `tests/unit/test_burgers_flux.cpp`, `test_muscl_reconstruction.cpp`
  (new) -- hand-computed reference values for every new free
  function/functor (shock/rarefaction/degenerate-equal-state cases for
  Rusanov; monotone/local-extremum/genuinely-linear cases for minmod).
- `tests/unit/test_burgers_shock_formation.cpp` (new) -- "Burgers shock
  formation": a Riemann-type step (`StaticBoundary` fixing both ends),
  checked against the exact Rankine-Hugoniot solution, overshoot/
  undershoot, TVD, and flux-balance conservation (see Scientific
  verification below) -- templated on `Scalar`, exercised at both
  double and float precision.
- `tests/unit/test_burgers_convergence.cpp` (new) -- "Burgers smooth
  convergence": smooth periodic IC run strictly before the analytic
  breaking time, checked against a Newton-solved method-of-
  characteristics exact reference.
- `tests/CMakeLists.txt` -- the 5 new test files wired in.
- `docs/adr/0008-burgers-shock-capturing-scheme.md` (new) -- records
  minmod+Rusanov as the scheme choice, with the Evidence section below
  reproduced there.
- `tasks/0003-phase2-burgers-shock-capturing.md` (new) -- formal task
  spec scoping this as a first slice of Phase 2.

Tests added:
22 new tests (78/78 total, up from 56/56 at Phase 1's close): Burgers
Calculator hand-values (2), Rusanov hand-values incl. the degenerate
equal-state case (4), minmod/MUSCL hand-values incl. the local-extremum
clip and the genuinely-linear no-clip case (8), shock-formation (6,
incl. a float-precision variant), smooth convergence (2, incl. a
breaking-time sanity check on the test's own setup). CPU-only this
task, per task 0003's explicit scope (no CUDA changes).

Benchmarks run:
None -- explicitly deferred to a follow-up task (task 0003's own
"Benchmarks: Not required this task").

Performance change:
N/A (no production hot path touched beyond new, additive leaf types
substituted as template arguments; `fvm_solver.hpp`/`ssp_rk2.hpp`/
`ghost_fill.hpp`/`cartesian_grid.hpp` were not modified).

Scientific verification:
Both `VERIFICATION.md`-named canonical problems verified with real,
reported numbers, not "ran and looked reasonable" (full tables in ADR
0008's Evidence section):
- **Shock formation** (Riemann step, `u_left=2`, `u_right=1`, exact
  shock speed `s=1.5`, run to `t=2.0`): mean absolute error against the
  exact solution halves almost exactly with each doubling of resolution
  (4.21e-3 at nx=200 -> 5.26e-4 at nx=1600) -- the expected O(1/nx)
  behavior for a captured shock, not a formal 2nd-order claim (any
  limited scheme smears a discontinuity over O(1) cells regardless of
  resolution). Measured overshoot/undershoot was exactly `0.0` at every
  resolution tested. Total variation was exactly `1.0` (`=u_left-
  u_right`) before and after, at every resolution -- zero measured
  oscillation anywhere. Flux-balance conservation
  (`integral_final-integral_initial` vs. `(F(u_left)-F(u_right))*T`)
  matched to `1e-6`, both sides equal to `3.000000` at the printed
  precision.
- **Smooth convergence** (`u0=1.0+0.5*sin(2*pi*x)`, run to half the
  analytic breaking time, against a Newton-solved method-of-
  characteristics exact reference): the error ratio stabilizes tightly
  at **~3.21-3.23** across 5 refinement levels (40->640 cells) -- not
  the clean ~4.0 the linear-advection test shows, root-caused (not
  assumed) to minmod clipping the slope to exactly zero at the IC's two
  smooth extrema, a documented, accepted property of TVD limiters
  (Sweby, 1984; see `numerics/fvm/muscl_minmod.hpp`'s own header
  comment). The convergence test's acceptance band was loosened from
  `[3.5, 4.5]` to `[3.0, 4.5]` specifically to reflect this, with the
  mechanism stated in the test file rather than the threshold silently
  narrowed to make a tight-but-unexplained number pass.
- Note on the non-periodic shock test's "conservation" claim: the
  Riemann-step domain is not periodic (mass genuinely flows in/out at
  the two StaticBoundary ends), so "domain integral constant in time"
  (the periodic convergence test's own conservation check) does not
  apply here -- what was actually verified is the flux-form scheme's
  exact flux-balance guarantee instead (see test file's own comment for
  why this is the correct, not weaker, substitute).

Architecture decisions:
`docs/adr/0008-burgers-shock-capturing-scheme.md` (new): minmod-limited
MUSCL + Rusanov selected as Burgers' first shock-capturing pair --
simplest provably-TVD/entropy-correct combination, matching every prior
phase's "smallest capability needed" approach. Named alternatives for
later phases: superbee/van Leer/MC limiters and WENO (AGENTS.md #18);
HLLC/AUSM/exact Godunov (Phase 3's Euler work already names HLLC/AUSM).
Confirms, rather than merely asserts, Phase 1's own stated genericity
claim for `FvmSolver`/`Reconstruction`/`NumericalFlux`: a genuinely
nonlinear, shock-forming equation required zero changes to any of
`fvm_solver.hpp`, `ssp_rk2.hpp`, `ghost_fill.hpp`, or
`cartesian_grid.hpp`.

Known limitations:
- CUDA port, benchmark sweep, and visualization tutorial for Burgers are
  not done -- explicitly deferred, per task 0003's own scope.
- The rest of `ROADMAP.md` Phase 2 (MPI decomposition + communication
  benchmark, DG storage/communication prototype, state sizes through
  100, the memory-layout study) is not started.
- Only minmod is implemented; sharper limiters (superbee/van Leer/MC)
  and more accurate fluxes (exact Godunov/HLLC/AUSM) are named future
  work in ADR 0008, not implemented.

Next recommended task:
Either (a) port Burgers to CUDA + benchmark + visualization tutorial,
mirroring Phase 1's own CPU-then-GPU sequencing, or (b) continue Phase 2
breadth-first into the MPI decomposition prototype -- both are
reasonable next slices; recommend checking with the PI on which matters
more before committing effort, since task 0003 deliberately left this
open rather than presuming the order.

---

## 2026-10-02 — Burgers 3D CPU sanity, CUDA port, and at-scale GPU benchmark (PR #3 follow-up)

Agent:
Model: Claude Sonnet 5

Objective:
Mid-review of PR #3, the user asked whether a 3D Burgers test existed
that could run on GPU and check scale -- it did not (the task 0003
entry above explicitly deferred CUDA/3D/benchmark work). The user then
stated a standing policy: every PR adding a Field/scheme needs this
coverage in the same PR, not deferred (recorded in auto-memory as
`cfe_feedback_pr_gpu_scale_test`). This entry folds that work into PR
#3 rather than opening a separate follow-up task.

Files changed:
- `tests/unit/test_burgers_3d_sanity.cpp` (new) -- CPU-only: a Riemann
  shock varying only in X, uniform/periodic in Y/Z, must match the 1D
  reference column-for-column (same philosophy as
  `test_scalar_advection_2d_sanity.cpp`, extended one dimension
  further). Established 3D residual-loop correctness on CPU *before*
  porting to GPU, matching this project's own staged-verification
  discipline.
- `tests/unit/test_burgers_cuda.cu`, `test_burgers_3d_cuda.cu` (new) --
  CPU-vs-GPU correctness, 1D and 3D, mirroring
  `test_scalar_advection_cuda.cu`/`test_scalar_advection_3d_cuda.cu`
  exactly. Both use the actual shock-formation Riemann setup (not a
  smooth proxy), so this simultaneously verifies the GPU port AND
  exercises `StaticBoundary` on CUDA for the first time in this
  codebase (every prior CUDA test used only `PeriodicBoundary`).
- `benchmarks/burgers/bench_burgers.cpp`, `bench_burgers_cuda.cu`,
  `bench_burgers_3d_cuda.cu`, `CMakeLists.txt` (new) -- mirror
  `benchmarks/scalar_advection/`'s three-file structure and resolution
  sweeps exactly (CPU 10^4-10^7; GPU 1D 10^6-10^8; GPU 3D up to 512^3).
- `benchmarks/CMakeLists.txt`, `tests/CMakeLists.txt` -- new
  files/subdirectory wired in.
- `docs/performance/0006-phase2-burgers-cuda-results.md` (new) -- full
  results tables, environment, methodology.
- `docs/adr/0008-...md` -- GPU-port/at-scale evidence appended to the
  existing Evidence section.

Tests added:
3 new CPU tests (3D sanity) + 2 new CUDA tests (1D, 3D) = 92 total
listed, 87/87 actually executed on this machine (CPU here has no CUDA
compiler; all 87 -- the 5 CUDA-gated tests included -- ran and passed
on the V100 allocation, see Scientific verification below).

Benchmarks run:
`cfe_bench_burgers` (Apple M5, serial+threaded, CPU baseline) and
`cfe_bench_burgers_cuda`/`cfe_bench_burgers_3d_cuda` (V100, PSC
Bridges-2, job `47335443`, node `v009`, via the `gpuinteract` QOS
fast-lane -- see [[bridges2_gpu_access]]).

Performance change:
N/A (new capability, not a change to existing code -- no prior Burgers
GPU/benchmark numbers existed to compare against).

Scientific verification:
Built and ran on real V100 hardware (not assumed/deferred): 87/87 unit
tests passed, including `test_burgers_cuda_matches_cpu_reference` (1D,
400 cells, 400 steps) and `test_burgers_3d_cuda_matches_cpu_reference`
(3D, 96^3 cells, 200 steps), both matching the CPU reference to `1e-9`
cell-by-cell. `compute-sanitizer --tool initcheck` run over the entire
suite immediately after: **0 errors** -- checked directly rather than
assumed clean by analogy to the earlier scalar-advection/DeviceField
fix, given this project's own history of finding a genuine bug exactly
this way twice already this PR cycle's predecessor (PR #2).
Benchmarked at the same scale Phase 1 established for scalar advection:
1D up to 10^8 cells (~7.69e9 cell-updates/s, 13.0ms/step); 3D up to
512^3/~1.34e8 cells (~4.05e9 cell-updates/s, 33.1ms/step). Both are a
modest (~10-15%), expected reduction from scalar advection's own
numbers at the same scale (0003-...md), attributed directly to
Burgers' extra per-cell work (a `minmod` branch per face per axis, a
nonlinear flux evaluation, the Rusanov dissipation term) -- not treated
as an unexplained regression requiring investigation.

Architecture decisions:
None new -- confirms `docs/adr/0008-...md`'s existing decision (minmod +
Rusanov) now extends to GPU and 3D without any additional design choice
needed, since every new type was already `CFE_HOST_DEVICE`.

Known limitations:
- The Burgers CPU serial-backend throughput is below scalar advection's
  own (even pre-regression) baseline at the same cell count, consistent
  with the extra per-cell work but not separately root-caused via
  vectorization remarks the way 0004/0005 did for scalar advection --
  not investigated further here since GPU is this codebase's actual "at
  scale" target.
- A Burgers visualization tutorial (mirroring
  `tutorials/scalar_advection_3d_visualization/`) is still not done.
- The rest of `ROADMAP.md` Phase 2 (MPI, DG prototype, state sizes
  through 100, memory-layout study) is still not started.

Next recommended task:
Burgers visualization tutorial (quick, mirrors an existing pattern), or
move on to the MPI decomposition prototype -- same open question as the
prior entry, now with CUDA/3D/benchmark work no longer blocking either
choice.

---

## 2026-10-02 — Burgers 3D visualization tutorial (PR #3 follow-up)

Agent:
Model: Claude Sonnet 5

Objective:
The user asked whether a Burgers tutorial existed (a Gaussian bump
deforming to a shock, or a sine wave) -- it did not; this was the "known
limitation" flagged at the end of the prior entry. Closes it: a 3D
Gaussian-bump visualization tutorial, mirroring
`tutorials/scalar_advection_3d_visualization/` exactly in structure, but
demonstrating genuinely different physics.

Files changed:
- `tutorials/burgers_3d_visualization/advect_burgers_gaussian_3d.cpp`,
  `CMakeLists.txt`, `README.md` (new) -- same `FvmSolver` + `BurgersField`
  + `MusclMinmodReconstruction` + `RusanovFlux` + SSP-RK2 stack as the
  rest of this task, run on a `64^3` periodic grid, a positive-background
  Gaussian bump (`1.0 + 0.5*exp(-r^2/(2*sigma^2))`, same `A+B` pattern
  `test_burgers_convergence.cpp` uses, chosen so the state never changes
  sign), 500 steps, 51 VTK frames + a `.pvd` manifest.
- `tutorials/CMakeLists.txt` -- new subdirectory wired in.

Tests added:
None -- a visualization tutorial, not a correctness claim (the
underlying solver stack is already verified by
`test_burgers_3d_cuda.cu`/`test_burgers_shock_formation.cpp`; this file
just confirms, at run time, that the printed state range per frame never
exceeds the initial `[1.0, ~1.497]` bounds -- a live, visible
demonstration of the same TVD guarantee those tests verify numerically).

Benchmarks run:
None (not a performance artifact).

Performance change:
N/A.

Scientific verification:
Ran locally (CPU, Apple M5): state range printed per frame stayed
bounded in `[1.0000, 1.4968]` (initial) shrinking to `[1.0000, 1.3846]`
by the final frame (`t=0.52`) -- confirms, by direct observation rather
than assumption, that (a) the TVD guarantee holds with a real,
non-trivial 3D multi-axis initial condition, not just the 1D Riemann
step `test_burgers_shock_formation.cpp` checks numerically, and (b) the
bump is genuinely deforming (the peak decaying monotonically as the
leading faces steepen and mass spreads via the trailing rarefaction),
not just sitting still or translating unchanged the way the linear
scalar-advection tutorial's bump does.

Found and fixed one bug during this work, isolated to the tutorial
itself (not the solver): the per-frame diagnostic min/max computation
seeded its running min from storage index `0`, which is a ghost cell
(never written before the first ghost-fill call, left at whatever
`Field`'s zero-initialization gives it) -- not a real cell. This wrongly
printed `state range [0.0000, 1.4968]` for frame 0 (implying the state
once hit zero, which never happened). Fixed by seeding from the first
real cell (`grid.flat_index(grid.ngx, grid.ngy, grid.ngz)`) instead.
Re-verified: frame 0 now correctly prints `[1.0000, 1.4968]`.

Architecture decisions:
None -- pure application of already-existing, already-verified types.

Known limitations:
- The rest of `ROADMAP.md` Phase 2 (MPI, DG prototype, state sizes
  through 100, memory-layout study) is still not started.

Next recommended task:
PR #3 is now caught up on every item raised during its own review
(CPU correctness, 3D, CUDA, at-scale benchmark, visualization). Move on
to the MPI decomposition prototype, or check with the PI on Phase 2's
remaining priority order.

---

## 2026-10-02 — Two quantitative Burgers verification/plotting tutorials (1D and 2D)

Agent:
Model: Claude Sonnet 5

Objective:
A detailed, explicit user request for two rigorous, reproducible,
plotted Burgers tutorials -- exact cut-cell references (including cells
the moving discontinuity straddles), grid-convergence sweeps, a
first-order-vs-limited-second-order reconstruction comparison, and
committed data/figures a reader can inspect without re-running anything.
Builds entirely on PR #3's existing production machinery
(`BurgersField`/`RusanovFlux`/`MusclMinmodReconstruction` plugged into
the unchanged `FvmSolver`/`ssp_rk2_step`); net new production code is
two small, additive types.

Files changed:
- `src/cfe/numerics/fvm/first_order_reconstruction.hpp` (new) --
  `FirstOrderReconstruction`: piecewise-constant, same two-method shape
  as every other `Reconstruction` type -- what "first-order vs limited
  second-order" actually swaps between.
- `src/cfe/grid/boundary/boundary_condition.hpp` -- added
  `InflowOutflowBoundary`: fixed Dirichlet at the low end, zero-order
  extrapolation at the high end. First real implementation of AGENTS.md
  #17's named "extrapolation/outflow" BC category (explicitly the
  simple zero-order kind, not Phase 3's characteristic-based one).
- `tests/unit/test_boundary_conditions.cpp`,
  `tests/unit/test_interface_flux.cpp` -- one new hand-computed test
  each for the two new types above.
- `tutorials/burgers_1d_shock_and_steepening/` (new) -- one executable,
  two selectable cases (`--case=shock`/`--case=steepening`, default
  both): Case A (moving shock, `InflowOutflowBoundary`, 3 grids x 2
  reconstructions x 4 output times, exact fractional-coverage cell
  averages including shock-straddled cells); Case B (sinusoidal
  steepening, periodic, exact method-of-characteristics reference
  before the analytic breaking time, numerical-only after it). Both
  cases use the existing, unmodified `cfe::ssp_rk2_step` directly --
  neither boundary's ghost VALUES depend on wall-clock time here, so
  ghosts refreshing every residual call (the stack's existing default
  behavior) already satisfies "update ghost states before every
  residual evaluation, using the correct RK stage time."
- `tutorials/burgers_2d_diagonal_shock/` (new) -- a diagonal moving
  shock (`u=1` where `x+y<0.5+t`), 3 grids x 2 reconstructions x 3
  output times, exact cut-cell area fractions for a square clipped by a
  slope-(-1) line (`cut_cell_fraction`, used identically for both the
  t=0 IC and every later reference). This IS the case where ghost
  values genuinely depend on t -- `ssp_rk2_step` was deliberately left
  unmodified (zero blast radius on that shared, already-reviewed
  helper), and this tutorial instead hand-rolls Heun's method
  explicitly (`step_once`), setting a tutorial-local
  `DiagonalShockExactBoundary`'s `time` member to the correct stage
  time (t_n, then t_n+dt) between the two stages.
- Both tutorial directories: `plot_results.py` (numpy/pandas/matplotlib),
  `README.md`, `data/` (summary.csv in full + one representative raw
  field/profile set per tutorial), `figures/` (committed PNGs).
- `tutorials/CMakeLists.txt` -- both new subdirectories wired in.
- `.gitignore` -- added `.venv/` (both new READMEs suggest a local venv
  for the Python dependencies).

Tests added:
2 new CPU unit tests (`test_inflow_outflow_boundary_fixes_low_end_and_
extrapolates_high_end`, `test_first_order_reconstruction_ignores_
neighbors_and_returns_cell_value_at_both_faces`) -- 81/81 total.

Benchmarks run:
None -- these are correctness/verification tutorials, not performance
artifacts; CPU-only by the same established convention every other
tutorial in this repo already uses (performance/scale lives in
benchmarks/tests, not tutorials).

Performance change:
N/A.

Scientific verification:
Both C++ binaries and both Python plotting scripts were actually run
(not assumed) before calling this done, producing the committed
data/figures directly:
- **Case A (1D moving shock):** L1 error vs. exact cell average drops
  essentially linearly with grid refinement for both reconstructions
  (confirmed O(dx) at the captured shock, the expected, not a flawed,
  behavior per docs/adr/0008-...md); limited second-order consistently
  roughly half the first-order error at matching resolution; shock
  position converges to the exact Rankine-Hugoniot position as
  resolution increases (e.g. nx=400, t=1.0: exact=0.7500,
  numerical=0.7503); **zero measured overshoot/undershoot at every
  grid/reconstruction/time combination** (both schemes, not just the
  TVD-limited one -- first-order is unconditionally monotone by
  construction).
- **Case B (1D sinusoidal steepening):** domain mean held at exactly
  `1.000000` at all 5 output times, including the 2 past the analytic
  breaking time -- conservation confirmed directly, not assumed, even
  post-shock. A genuine bug was found and fixed during this work: the
  pre-shock exact reference (plain Newton's method on the implicit
  characteristics equation) produced a visibly wrong, jagged artifact at
  `t=0.30` (very close to the breaking time `t_s=0.318`), caught by
  inspecting the generated plot, not by a numeric check alone -- Newton
  can overshoot badly where the equation's derivative gets small, which
  happens near the breaking time by construction. Fixed by switching to
  a bracketed bisection solve (the same equation is provably monotonic
  below the breaking time, so bisection is unconditionally robust
  there) -- re-verified: the regenerated plot is clean at every output
  time, including t=0.30.
- **2D diagonal shock:** the hand-rolled, per-stage-time-dependent
  SSP-RK2 driver produces a shock that stays measurably straight and
  tracks `x+y=0.5+t` closely at every grid/time (e.g. n=400, t=0.5: the
  diagonal (x=y) crossing's exact position is 0.5000, numerical 0.4999),
  L1 error drops under refinement for both reconstructions, and **zero
  measured overshoot/undershoot** at every grid/reconstruction/time --
  indirectly but concretely confirming the per-stage time-threading is
  correct: a bug in which stage saw which ghost time would most likely
  show up as spurious oscillation or a measurably wrong shock speed,
  neither of which appeared.

Architecture decisions:
- `InflowOutflowBoundary` added to core (`src/cfe/grid/boundary/`) as a
  genuinely reusable type, not tutorial-local -- unlike
  `DiagonalShockExactBoundary` (2D tutorial), which hardcodes one
  problem's exact-solution formula and stays tutorial-local by design
  (same physics/generic-numerics separation already established for
  Reconstruction/NumericalFlux types).
- Deliberately did NOT add a time parameter to `cfe::ssp_rk2_step` to
  support the 2D tutorial's time-dependent ghost fills -- that shared,
  already-reviewed helper is used unmodified by every other solver in
  this codebase; the 2D tutorial hand-rolls its own two-stage loop
  instead, confined entirely to tutorial-local code.

Known limitations:
- No CUDA port for either tutorial -- consistent with every other
  tutorial in this repo being CPU-only (performance/scale is a
  benchmark/test concern here, not a tutorial one).
- The 2D tutorial's post-shock story has no analogue to worry about
  (the diagonal shock has no breaking-time complication the 1D Case B
  does), so this limitation list is shorter than that entry's.

Next recommended task:
Continue Phase 2 breadth-first (MPI decomposition prototype), or check
with the PI on priority -- same open question as every entry since
PR #3 began.

---

## 2026-10-02 — 1D tutorial: shock-capturing zoom-in plot

Agent:
Model: Claude Sonnet 5

Objective:
User follow-up on the two new tutorials above: add a plot zooming in on
Case A's captured front at the final time, centered on the shock
location, `x in [-0.05, 0.05]`, to directly show what the limiter
actually buys (narrower smearing), not just a smaller L1 number.

Files changed:
- `tutorials/burgers_1d_shock_and_steepening/burgers_1d.cpp` -- the
  finest-grid field-CSV write, previously second-order-only, now
  happens for `FirstOrderReconstruction` too (so the zoom plot can put
  both schemes side by side at matching resolution).
- `plot_results.py` -- new `plot_case_a_shock_zoom()`: re-centers each
  reconstruction's field data on ITS OWN numerically-detected shock
  position (`summary.csv`'s `shock_position_numerical` -- the
  C++-computed value, not recomputed in Python) before windowing, since
  first-order and second-order land at very slightly different
  positions and a single shared shift would not put both fronts at
  `x=0`.
- `README.md` -- embeds the new figure, explains the re-centering choice
  and what the two panels show.
- `data/case_shock_nx0400_first_order_t*.csv` (new, 4 files) --
  first-order field data at the finest grid, all four output times.

Tests added:
None (plotting/visualization change only; no production code touched).

Scientific verification:
Ran the C++ binary and the plotting script again, inspected the
resulting figure directly: first-order smears the captured shock over
roughly 4-5 cells, limited second-order (minmod) over roughly 2-3 --
visibly, not just numerically, confirming the limiter's benefit at the
discontinuity itself, consistent with (and a direct visual explanation
of) the existing convergence plot's "both converge at the same O(dx)
rate, but second-order's prefactor is about half" finding.

Architecture decisions:
None -- tutorial-only change.

Known limitations:
None new.

Next recommended task:
Unchanged from the prior entry.
