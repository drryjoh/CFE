# Tutorial: running the Phase 0 execution/memory benchmark

This tutorial builds CMU-CFE's Phase 0 foundation and runs the benchmark
that exercises it: a single all-cell kernel,

```cpp
q_new(i, k) = q(i, k) * q(i, k);
```

across every required precision, component count, backend, and storage
layout. There is no physics here on purpose (see
`tasks/0001-phase0-execution-foundation.md`) -- the point is to observe the
execution and memory architecture itself.

## 1. Prerequisites

- A C++17 compiler (this was developed against Apple clang 21 / arm64; any
  recent GCC or Clang should work).
- CMake >= 3.18. If your system does not have one:
  ```bash
  pip3 install --user cmake
  ```
- Optional, for the CUDA path: the NVIDIA CUDA toolkit (`nvcc`) and an
  NVIDIA GPU.

## 2. Build (CPU)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CMake auto-detects whether a CUDA compiler is available
(`check_language(CUDA)` in the top-level `CMakeLists.txt`) and only builds
the CUDA backend/targets when it finds one, so this works unmodified on a
CPU-only machine.

Useful configure-time options (all have sensible defaults):

| Option | Values | Meaning |
|---|---|---|
| `CFE_SCALAR_TYPE` | `float` \| `double` | Project-wide default `cfe::scalar` (math/field/backend templates are exercised in *both* precisions regardless -- see `tests/unit/test_math_operations.cpp`) |
| `CFE_DEFAULT_BACKEND` | `serial` \| `threaded` | What `cfe::parallel_for` (the unqualified alias) resolves to |
| `CFE_ENABLE_CUDA` | `ON`/`OFF` | Force the CUDA backend; auto-detected off if no `nvcc` is found |
| `CFE_BUILD_TESTS` | `ON`/`OFF` | Build `cfe_unit_tests` |
| `CFE_BUILD_BENCHMARKS` | `ON`/`OFF` | Build the benchmark executables |

## 3. Run the unit tests

```bash
ctest --test-dir build --output-on-failure
```

or run the test binary directly for per-test PASS/FAIL output:

```bash
./build/tests/cfe_unit_tests
```

## 4. Run the benchmark (CPU)

```bash
./build/benchmarks/memory/cfe_bench_field_update
```

This prints a CSV header followed by one row per (precision, component
count, backend, layout) combination -- 2 x 6 x 2 x 2 = 48 rows -- to stdout.
Columns:

```
precision,n_components,backend,layout,n_cells,repetitions,median_ms,
cell_updates_per_s,scalar_updates_per_s,bandwidth_gb_s
```

Redirect to a file to keep a record:

```bash
./build/benchmarks/memory/cfe_bench_field_update > my_run.csv
```

A committed reference run lives at
`benchmarks/results/phase0_field_update_apple_m5.csv`, discussed in
`docs/performance/0001-phase0-results.md`.

## 5. Run the benchmark (CUDA, when available)

```bash
cmake -S . -B build -DCFE_ENABLE_CUDA=ON
cmake --build build --target cfe_bench_field_update_cuda -j
./build/benchmarks/memory/cfe_bench_field_update_cuda
```

**This path was written but not exercised during Phase 0 development**: no
CUDA toolkit or NVIDIA GPU was available in that environment. If you run it,
please add the resulting CSV and a short writeup to `docs/performance/` and
update `docs/adr/0001-execution-backend.md` / `0002-state-memory-layout.md`,
which currently only have CPU evidence.

For register/occupancy/spill inspection of the CUDA kernel, see
`scripts/profile_cuda.sh`.

## 6. What to look for

- Compare `AoS` vs `SoA` rows at the same `precision`/`n_components`: on the
  CPU this benchmark was developed on, `AoS` matched or beat `SoA`, with the
  gap widening as `n_components` grows (see
  `docs/performance/0001-phase0-results.md` for the numbers and why). Do not
  assume this holds on your hardware, or on a GPU -- measure it.
- Compare `serial` vs `threaded` at fixed `n_components`/`layout`: this
  kernel is memory-bandwidth bound, so expect sub-linear speedup relative to
  core count once the memory bus saturates.
- `bandwidth_gb_s` is computed as one read of `q` plus one write of
  `q_new` per scalar element, divided by the median wall-clock time. It is
  a lower bound on achieved bandwidth (it does not count e.g. any
  cache-line traffic wasted by a poorly-strided access pattern), which is
  itself part of why `AoS` vs `SoA` differ.
