# Phase 0 benchmark results

Raw CSV: [`benchmarks/results/phase0_field_update_apple_m5.csv`](../../benchmarks/results/phase0_field_update_apple_m5.csv).

## Kernel

`q_new(i,k) = q(i,k) * q(i,k)` over every cell `i` and component `k`
(`cfe_bench_field_update`, `benchmarks/memory/bench_field_update.cpp`). This is
the "simple all-cell operation" required by the task spec; it is a
memory-bandwidth exercise, not a physics kernel.

For each (precision, component count) pair the working set is sized so that
each of `q`/`q_new` occupies roughly 64 MiB (`n_cells = 64MiB / (N * sizeof(Scalar))`,
floored at 1024 cells), comfortably larger than cache. Each reported time is
the median of 7 timed repetitions after one untimed warm-up pass; allocation
and initialization happen once, outside every timed region.

## Environment

| | |
|---|---|
| Hardware | Apple M5 (10 cores: 4 performance + 6 efficiency), 24 GiB unified memory |
| OS | macOS (Darwin 25.4.0, arm64) |
| Compiler | Apple clang version 21.0.0 (clang-2100.1.1.101) |
| Build type | `CMAKE_BUILD_TYPE=Release` |
| CFE options | `CFE_SCALAR_TYPE=double` (project default; both precisions benchmarked via templates regardless), `CFE_ENABLE_CUDA=OFF` (auto-detected off, no `nvcc`/NVIDIA GPU present) |
| Date | 2026-08-30 |

**NVIDIA GPU / CUDA:** no CUDA toolkit or NVIDIA GPU was available in this
environment. `CFE_ENABLE_CUDA` was auto-detected `OFF`
(`check_language(CUDA)` found no compiler), so the CUDA backend
(`src/cfe/backend/cuda/`) and its benchmark
(`benchmarks/memory/bench_field_update_cuda.cu`) were written and reviewed but
**not compiled or executed**. No CUDA performance numbers exist for Phase 0;
none are invented below. This is the single biggest open item carried into
the next task (see ADR 0001 and ADR 0002).

## Results — CPU (Apple Silicon), serial and threaded backends

All 48 required combinations (2 precisions x 6 component counts x 2 backends
x 2 layouts) compiled and ran. Selected rows (full data in the CSV):

| precision | N | backend | layout | n_cells | median ms | cell updates/s | scalar updates/s | bandwidth (GB/s) |
|---|---|---|---|---|---|---|---|---|
| double | 1 | serial | AoS | 8,388,608 | 1.269 | 6.61e9 | 6.61e9 | 105.7 |
| double | 1 | threaded | AoS | 8,388,608 | 1.156 | 7.26e9 | 7.26e9 | 116.2 |
| double | 20 | serial | AoS | 419,430 | 1.725 | 2.43e8 | 4.86e9 | 77.8 |
| double | 20 | serial | SoA | 419,430 | 4.346 | 9.65e7 | 1.93e9 | 30.9 |
| double | 20 | threaded | AoS | 419,430 | 1.529 | 2.74e8 | 5.49e9 | 87.8 |
| double | 20 | threaded | SoA | 419,430 | 1.939 | 2.16e8 | 4.33e9 | 69.2 |
| double | 100 | serial | AoS | 83,886 | 1.754 | 4.78e7 | 4.78e9 | 76.5 |
| double | 100 | serial | SoA | 83,886 | 7.534 | 1.11e7 | 1.11e9 | 17.8 |
| double | 100 | threaded | AoS | 83,886 | 1.628 | 5.15e7 | 5.15e9 | 82.4 |
| double | 100 | threaded | SoA | 83,886 | 2.564 | 3.27e7 | 3.27e9 | 52.3 |
| float | 100 | serial | AoS | 167,772 | 1.779 | 9.43e7 | 9.43e9 | 75.4 |
| float | 100 | serial | SoA | 167,772 | 14.503 | 1.16e7 | 1.16e9 | 9.3 |
| float | 100 | threaded | AoS | 167,772 | 1.523 | 1.10e8 | 1.10e10 | 88.1 |

### Observations

1. **AoS was faster than SoA at every component count on this CPU, and the
   gap widens with N.** At N=1 the two layouts are within noise of each
   other (both backends), because there is only one component per cell and
   the layouts are identical. From N=20 upward, AoS holds roughly flat
   bandwidth (~76-88 GB/s across N and both backends) while SoA degrades
   substantially -- at N=100/double/serial, SoA is ~4.3x slower than AoS
   (17.8 vs 76.5 GB/s); at N=100/float/serial, SoA is ~8.1x slower (9.3 vs
   75.4 GB/s). This is consistent with this kernel's access pattern: each
   loop iteration over cell `i` touches all N components of that cell. AoS
   keeps those N values contiguous (one cache-line neighborhood); SoA
   scatters them across N independent strides of length `n_cells`, so per
   iteration the kernel is opening N far-apart memory streams instead of
   one.
2. **Threading helped, but not linearly with core count.** The threaded
   backend (`std::thread`, static contiguous chunking, no pool) improved
   AoS runs by roughly 1.1x-1.6x over serial, and helped SoA runs more
   (up to ~2.9x at N=100/double). This is the expected signature of a
   memory-bandwidth-bound kernel on a machine with far more cores (10) than
   the observed speedup: once enough threads are saturating the memory
   controller, additional threads stop helping. It also suggests SoA's
   single-threaded penalty is partly a prefetching/stream-count problem that
   more concurrent streams can partially hide -- another reason not to
   generalize from one thread count.
3. **No layout was uniformly best at N=1, only at N>=20 does AoS's advantage
   become large enough to be an actionable default.** This matches
   AGENTS.md's instruction not to assume a layout is optimal without
   measurement: N=1 genuinely does not distinguish the layouts.
4. **float vs double:** relative AoS/SoA behavior is consistent across
   precisions; float benchmarks use twice as many cells for the same target
   working-set size, as expected.

### What remains unknown

- **GPU behavior.** ADR 0002 explicitly requires CPU *and* NVIDIA GPU
  evidence before a layout can be a global default; only CPU evidence exists.
  GPU coalescing behavior is a different memory-access model (SoA is
  frequently *better* for coalesced GPU access precisely because consecutive
  threads then read consecutive addresses per component), so **this result
  should not be assumed to transfer to CUDA.**
- **Apple Silicon P-core/E-core scheduling effects** on the threaded backend
  were not isolated (`std::thread::hardware_concurrency()` reports all 10
  cores; no affinity control was attempted). This may partially explain the
  sub-linear threaded speedups above and is not disentangled from pure
  memory-bandwidth saturation.
- **AoSoA** was not implemented or measured in Phase 0 (spec scope: AoS/SoA
  only needed to be "structured so alternative layouts can be benchmarked
  later" -- `cfe::Field`'s `Layout` template parameter satisfies that without
  requiring a third implementation yet).
- Register count, occupancy, local-memory usage, and spilling (task spec
  item 12) are CUDA-specific measurements and could not be collected; see
  `scripts/profile_cuda.sh` for the documented procedure to run once
  hardware is available.
