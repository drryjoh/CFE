# ADR 0001: Minimal execution backend abstraction

**Status:** Accepted as the Phase 0 reference implementation for serial/threaded/CUDA backends -- not yet a settled production CPU backend, see the threaded-backend caveat below  
**Date:** 2026-08-30 (updated 2026-08-31, PR #1 review correction; updated 2026-09-08, CUDA verified on PSC Bridges-2)

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

Both CPU backends compiled cleanly (`-Wall -Wextra`, AppleClang 21, C++20 as
of the PR #1 review -- see ADR 0006 -- originally verified under C++17) and
produced identical results to each other and to a hand-computed
reference (`tests/unit/test_backend_execution.cpp`:
`test_threaded_backend_matches_serial_backend_bitwise`,
`test_all_cell_square_update_compiles_and_runs_for_all_required_component_counts`).
All 6 required component counts (1/5/10/20/50/100) and both precisions
(float/double) compiled through the `cfe::parallel_for` call sites used in
tests and benchmarks -- see `docs/performance/0001-phase0-results.md`.

Overhead: the abstraction itself is a function template around either a raw
loop or `std::thread`, so there is no virtual-dispatch cost. There is no
per-call allocation for the *field storage* -- that is allocated once,
outside any parallel loop, as intended. **The threaded backend itself is a
different story: `threaded::parallel_for` (`src/cfe/backend/cpu/
threaded.hpp`) creates a fresh `std::vector<std::thread>` and spawns new
`std::thread` objects on every single call -- there is no persistent thread
pool.** This was not measured in isolation (the benchmark harness calls
`parallel_for` once per timed repetition, not in a tight per-timestep loop,
so this cost is folded into the reported numbers rather than isolated from
them). It is an acceptable cost for Phase 0's single-launch-per-benchmark
usage pattern, but it is a real, unamortized cost that would matter once
`parallel_for` starts being called many times per timestep (the expected
Phase 1+ usage pattern for FVM/DG stages). The threaded backend gave a
consistent, if sub-linear (bandwidth-bound), speedup over serial across
every measured configuration despite this overhead (see results doc) --
which is worth noting precisely because it means the *true* achievable
threaded speedup, with thread creation removed from the critical path, is
at least as good as what was measured, likely better.

**CUDA backend: verified on real hardware (2026-09-08).** Built and run on
PSC Bridges-2 (NVIDIA Tesla V100-SXM2-32GB, compute capability 7.0, CUDA
12.9.86, GCC 13.3.1 host compiler, `-DCMAKE_CUDA_ARCHITECTURES=70`). All 4
CUDA correctness tests (`tests/unit/test_backend_execution_cuda.cu`: both
precisions x both layouts) pass, matching the CPU reference within the
tolerance appropriate for cross-backend floating-point comparison
(VERIFICATION.md #3). `cfe_bench_field_update_cuda` ran the full required
sweep (6 component counts x 2 precisions x 2 layouts); `scripts/
profile_cuda.sh` was executed (static register/spill sweep for all 24
instantiations, plus Nsight Compute occupancy/coalescing/local-memory
profiling for a representative subset). See
`docs/performance/0002-phase0-cuda-results.md` for full results; headline
findings: zero register spilling at any required component count, and SoA
substantially outperforms AoS on this GPU (opposite of the CPU result --
see ADR 0002).

Fixed one real nvcc-specific bug found during this verification:
`test_backend_execution_cuda.cu` originally defined its extended
`__device__` lambda directly inside the generic (`auto`-parameter) lambda
passed to `for_each_component_count`, which nvcc rejects ("An extended
__device__ lambda cannot be defined inside a generic lambda expression").
Fixed by extracting the per-N body into its own template function, matching
the pattern already used in `bench_field_update_cuda.cu`.

## Decision

Accepted for the CPU backends **as the Phase 0 reference implementation**:
the minimal `cfe::parallel_for` abstraction (serial + threaded) compiles
with no abstraction-dispatch overhead and produces correct, reproducible
results across all required precisions and component counts.

This is deliberately not the same claim as "settled production CPU
backend." The threaded backend's current implementation creates OS threads
and allocates its worker vector fresh on every call, with no pool -- see
the Evidence section above. That is a legitimate Phase 0 choice (AGENTS.md
#2: build the simplest thing that works, replace based on measurement, not
anticipation) and it does not invalidate the correctness results, but it
does mean this implementation should not be assumed adequate once
`parallel_for` is called at the frequency later phases will require.

The CUDA backend is now accepted on the same basis: it compiles, runs, and
produces correct results on real NVIDIA hardware (V100), with no register
spilling observed at any required state size. This does not by itself
imply a memory-layout default -- see ADR 0002, where CPU and GPU evidence
disagree.

## Consequences

Physics and numerics code should not contain raw launch syntax.

Backend APIs must remain intentionally small.

CUDA is now verified; near-term tasks targeting GPU execution can build on
the CUDA backend directly rather than budgeting time to first get it
compiling.

The threaded CPU backend should not be treated as production-ready for
repeated per-timestep kernel launches. A task that introduces a real
per-timestep execution loop (Phase 1's scalar transport is the first
candidate) should measure whether thread-creation overhead is actually
material at that call frequency before deciding whether a persistent
thread pool is warranted -- per AGENTS.md #2, that redesign should be
driven by measurement of the real access pattern, not assumed now.

## Revisit criteria

Revisit if:

- backend maintenance becomes disproportionate;
- another framework provides measured performance/portability benefits;
- additional accelerators become a near-term requirement;
- the threaded backend is measured under a realistic repeated-launch
  workload (e.g. once real timestep code exists) and thread-creation
  overhead proves material -- at that point replace the per-call
  `std::thread` spawn with a persistent thread pool and update this ADR's
  status to reflect a production-ready CPU backend rather than a Phase 0
  reference implementation.
