# Task 0001: Phase 0 Execution Foundation

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `docs/adr/0001-execution-backend.md`
4. `docs/adr/0002-state-memory-layout.md`
5. `BENCHMARKS.md`
6. `VERIFICATION.md`
7. `agent_history.md`

before making changes.

## Objective

Establish the smallest performance-oriented architecture necessary to support future structured finite-volume and DG development on CPUs and NVIDIA GPUs.

**Do not implement CFD in this task.**

Do not implement:

- scalar advection;
- Burgers;
- compressible flow;
- chemistry;
- AMR;
- unstructured grids;
- MPI decomposition.

## Required functionality

Implement:

1. Modern C++ project structure using CMake.
2. Configurable floating-point `scalar` type supporting:
   - `float`;
   - `double`.
3. Independent configurable index type.
4. Host/device annotation abstraction.
5. Minimal execution abstraction:

```cpp
cfe::parallel_for(...)
```

with:

- serial CPU backend;
- threaded CPU backend if straightforward;
- CUDA backend.

6. Lightweight fixed-size mathematical containers required for:
   - scalar;
   - vector;
   - fixed-size state.
7. Mathematical functions including:
   - componentwise operations;
   - `contract`;
   - `weight`.
8. A contiguous field-storage abstraction supporting compile-time component counts.
9. Compilation and execution for component counts:

```text
1
5
10
20
50
100
```

10. A simple all-cell operation such as:

```cpp
q_new(i,k) = q(i,k) * q(i,k);
```

or equivalent.

The purpose is to exercise memory and execution architecture, not solve a PDE.

11. Benchmark infrastructure measuring:
   - total runtime;
   - cell updates/s;
   - scalar updates/s;
   - effective memory bandwidth where meaningful.
12. CUDA profiling instructions for:
   - register count;
   - occupancy;
   - local-memory usage;
   - register spilling.
13. Unit tests for:
   - mathematical operations;
   - fixed-size containers;
   - field storage;
   - backend execution correctness.
14. One tutorial demonstrating the benchmark on:
   - CPU;
   - CUDA GPU when available.

## Architecture constraints

Do not allocate inside parallel loops.

Do not use virtual functions inside kernels.

Do not introduce external execution portability frameworks.

Do not optimize only for five-component states.

Do not assume AoS, SoA, or AoSoA is optimal without measurement.

Structure storage so alternative layouts can be benchmarked later.

Do not introduce CUDA syntax outside CUDA backend/abstraction code unless unavoidable.

Keep device-callable mathematical functions small and inlinable.

## Tests

At minimum:

- math operations produce expected results;
- CPU backend produces expected results;
- CUDA backend produces expected results when available;
- all required state sizes compile;
- float and double compile;
- storage indexing is correct.

## Benchmarks

Run available benchmarks for:

```text
components = 1, 5, 10, 20, 50, 100
precision  = float, double
```

on available CPU and CUDA hardware.

If a required hardware target is unavailable, document that fact. Do not invent performance numbers.

## Architecture decisions

Update ADR 0001 with evidence.

Update ADR 0002 with evidence if enough data exists to select an initial default memory layout. Otherwise leave it Proposed and record what remains unknown.

## Completion report

At the end report:

1. architecture created;
2. files added/changed;
3. tests performed;
4. benchmark results;
5. CUDA register/spill observations;
6. unresolved design questions;
7. ADR changes;
8. recommended next task.

Append the same work to `agent_history.md`.
