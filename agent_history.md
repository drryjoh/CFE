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
