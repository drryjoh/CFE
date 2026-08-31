# ADR 0001: Minimal execution backend abstraction

**Status:** Accepted for serial/threaded CPU backends; CUDA backend implemented but unverified  
**Date:** 2026-08-30 (updated 2026-08-30, Phase 0 evidence)

## Context

CMU-CFE must execute efficiently on multicore CPUs and NVIDIA GPUs while keeping physics and numerical code independent of backend-specific launch syntax.

Apple Silicon is also an important development target.

The code should remain sufficiently transparent that kernel behavior and compiler output can be inspected directly.

## Options considered

### A. Minimal CMU-CFE execution abstraction

Provide a small interface such as:

```cpp
cfe::parallel_for(range, lambda);
```

with backend-specific implementations.

Advantages:

- direct control;
- small abstraction surface;
- easy inspection;
- avoids committing early to a large portability framework.

Disadvantages:

- CMU-CFE must maintain backend code;
- additional accelerators require new backends.

### B. Adopt a portability framework immediately

Examples could include Kokkos, RAJA, SYCL, or similar systems.

Advantages:

- existing portability ecosystem;
- potentially easier additional-backend support.

Disadvantages:

- dependency and abstraction cost;
- may constrain architecture before CMU-CFE requirements are measured;
- can make low-level performance behavior less transparent to students.

## Evidence (Phase 0)

Implemented under `src/cfe/backend/`:

- `cpu/serial.hpp` -- a plain `for` loop.
- `cpu/threaded.hpp` -- `std::thread`-based static chunking, no pool.
- `cuda/cuda_backend.cuh` + `cuda/device_field.cuh` -- templated
  `__global__` kernel launch and device-memory field storage, confined to
  `.cuh`/`.cu` files per AGENTS.md #9.

Both CPU backends compiled cleanly (`-Wall -Wextra`, AppleClang 21, C++17)
and produced identical results to each other and to a hand-computed
reference (`tests/unit/test_backend_execution.cpp`:
`test_threaded_backend_matches_serial_backend_bitwise`,
`test_all_cell_square_update_compiles_and_runs_for_all_required_component_counts`).
All 6 required component counts (1/5/10/20/50/100) and both precisions
(float/double) compiled through the `cfe::parallel_for` call sites used in
tests and benchmarks -- see `docs/performance/0001-phase0-results.md`.

Overhead: the abstraction itself is a function template around either a raw
loop or `std::thread`; there is no virtual dispatch and no per-call
allocation once the field storage exists, so there is no abstraction
overhead distinct from "an equivalent hand-written loop/thread pool would
have." The threaded backend gave a consistent, if sub-linear (bandwidth-
bound), speedup over serial across every measured configuration (see
results doc).

**CUDA backend: unverified.** No CUDA toolkit or NVIDIA GPU was available in
the Phase 0 development environment (`nvcc` not found; CMake's
`check_language(CUDA)` returned not-found and `CFE_ENABLE_CUDA` was forced
`OFF`). `cuda_backend.cuh`, `device_field.cuh`,
`benchmarks/memory/bench_field_update_cuda.cu`, and
`tests/unit/test_backend_execution_cuda.cu` were written to the same
interface and reviewed, but have not been compiled by `nvcc`, let alone run
or benchmarked. Register/occupancy/spill inspection
(`scripts/profile_cuda.sh`) is likewise documented but unexecuted.

## Decision

Accepted for the CPU backends: the minimal `cfe::parallel_for` abstraction
(serial + threaded) compiles with no observed overhead and produces correct,
reproducible results across all required precisions and component counts.

The CUDA backend remains a reviewed but unverified proposal. Do not treat
CUDA support as validated until it has actually been compiled and run on
CUDA hardware; the next task that has access to an NVIDIA GPU should do so
before any physics work depends on it.

## Consequences

Physics and numerics code should not contain raw launch syntax.

Backend APIs must remain intentionally small.

Because CUDA is unverified, any near-term task targeting GPU execution
should budget time to first get `cfe_bench_field_update_cuda` and
`cfe_unit_tests` (CUDA-enabled) compiling and passing before building
anything physics-related on top of the CUDA backend.

## Revisit criteria

Revisit if:

- backend maintenance becomes disproportionate;
- another framework provides measured performance/portability benefits;
- additional accelerators become a near-term requirement;
- CUDA compilation/execution evidence becomes available and contradicts the
  design assumptions above (revisit this ADR's status to fully "Accepted"
  once that evidence exists, or "Rejected"/revised if it does not hold up).
