# Phase 0 CUDA benchmark results

Raw CSV: [`benchmarks/results/phase0_field_update_v100.csv`](../../benchmarks/results/phase0_field_update_v100.csv).

This is the CUDA counterpart to
[`0001-phase0-results.md`](0001-phase0-results.md), which was written before
any CUDA hardware was available and explicitly left GPU behavior as the
biggest open item. It is filled in here with real hardware evidence.

## Kernel and methodology

Same kernel and protocol as the CPU results: `q_new(i,k) = q(i,k) * q(i,k)`
over every cell `i` and component `k`
(`cfe_bench_field_update_cuda`, `benchmarks/memory/bench_field_update_cuda.cu`),
working set sized to ~64 MiB per field, median of 7 timed repetitions after
one untimed warm-up (warm-up matters more on CUDA than CPU: the first launch
pays CUDA context/JIT costs). Bandwidth is computed the same way (2x the
field size, read + write, divided by median time).

## Environment

| | |
|---|---|
| Hardware | NVIDIA Tesla V100-SXM2-32GB (compute capability 7.0), PSC Bridges-2 `GPU-shared` partition, node `v004` |
| Host CPU | Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz (not performance-relevant here; the kernel runs entirely on-device) |
| CUDA toolkit | 12.9.86 (module `cuda-v100/12.9.2`, alias for `cuda-legacy/12.9.2`) |
| Host compiler | GCC 13.3.1 (module `gcc/13.3.1-p20240614`) |
| Build type | `CMAKE_BUILD_TYPE=Release`, `CFE_ENABLE_CUDA=ON`, `-DCMAKE_CUDA_ARCHITECTURES=70` |
| Date | 2026-09-05 |

This is the first time the CUDA backend has run on real NVIDIA hardware.
Everything under `src/cfe/backend/cuda/` and its tests/benchmarks were
written and reviewed but unverified prior to this.

## Results — CUDA backend, all required (precision, N) combinations

All 24 required combinations (2 precisions x 6 component counts x 2 layouts)
compiled and ran (full data in the CSV):

| precision | N | layout | n_cells | median ms | bandwidth (GB/s) |
|---|---|---|---|---|---|
| float | 1 | AoS | 16,777,216 | 0.179 | 749.9 |
| float | 1 | SoA | 16,777,216 | 0.179 | 750.0 |
| float | 20 | AoS | 838,860 | 3.256 | 41.2 |
| float | 20 | SoA | 838,860 | 0.209 | 643.6 |
| float | 100 | AoS | 167,772 | 8.179 | 16.4 |
| float | 100 | SoA | 167,772 | 0.246 | 546.6 |
| double | 1 | AoS | 8,388,608 | 0.175 | 764.8 |
| double | 1 | SoA | 8,388,608 | 0.175 | 764.8 |
| double | 20 | AoS | 419,430 | 2.048 | 65.5 |
| double | 20 | SoA | 419,430 | 0.193 | 694.7 |
| double | 100 | AoS | 83,886 | 4.458 | 30.1 |
| double | 100 | SoA | 83,886 | 0.198 | 678.0 |

### Observations

1. **SoA wins decisively on the GPU, the opposite of the CPU result, and the
   gap is far larger than anything measured on CPU.** At N=1 the layouts are
   identical (single component, ~750-765 GB/s either way) and agree with each
   other almost exactly, as expected. From N=5 upward SoA pulls ahead and the
   gap widens with N: at N=100/float, SoA is **~33x faster than AoS** (546.6
   vs 16.4 GB/s); at N=100/double, SoA is **~22.5x faster** (678.0 vs 30.1
   GB/s). This is the mirror image of the CPU result (AoS ~4-8x faster than
   SoA at N=100), and it is exactly the mechanism the CPU writeup flagged as
   unverified: adjacent GPU threads process adjacent cells `i`, so under SoA
   they read/write *consecutive addresses* for a given component `k` --
   textbook coalesced access. Under AoS, adjacent threads' accesses for a
   fixed `k` are `n_components` scalars apart, so the memory controller sees
   a strided, uncoalesced pattern that gets dramatically worse as N grows.
2. **AoS bandwidth degrades almost monotonically with N**; SoA bandwidth
   stays roughly flat (~550-765 GB/s) across the entire N=1..100 sweep. SoA's
   flatness indicates the kernel remains memory-bandwidth-bound and
   well-coalesced regardless of state size; AoS's collapse indicates the
   uncoalesced-access penalty, not raw data volume, is what dominates AoS's
   cost at large N.
3. **Peak context**: the V100-SXM2-32GB's published peak HBM2 bandwidth is
   ~900 GB/s. SoA's ~680-765 GB/s at this kernel's arithmetic intensity
   (1 FLOP per 2 loads + 1 store) represents roughly 75-85% of published peak
   -- a reasonable achieved fraction for a simple bandwidth-bound kernel, not
   claimed here as "near peak" without this stated reference (BENCHMARKS.md
   #9). AoS at N=100 (~16-30 GB/s) is roughly 2-3% of peak.
4. **float vs double behaves consistently with the CPU results**: relative
   AoS/SoA behavior is the same shape across both precisions; float uses
   twice as many cells for the same ~64 MiB target working set, as expected.

## Register / occupancy / spill evidence (task spec item 12)

Procedure: `scripts/profile_cuda.sh`, part 1 (`nvcc --resource-usage`), run
against every required `(Scalar, N, Layout)` instantiation of
`bench_field_update_cuda.cu`'s `parallel_for_kernel`, compiled for `sm_70`.
This step is a static compile-time analysis and does not require holding a
GPU allocation.

| precision | N | AoS registers | SoA registers |
|---|---|---|---|
| double | 1 | 10 | 10 |
| double | 5 | 18 | 26 |
| double | 10 | 18 | 26 |
| double | 20 | 18 | 26 |
| double | 50 | 18 | 28 |
| double | 100 | 20 | 28 |
| float | 1 | 10 | 10 |
| float | 5 | 16 | 24 |
| float | 10 | 16 | 24 |
| float | 20 | 16 | 24 |
| float | 50 | 16 | 26 |
| float | 100 | 16 | 24 |

**Every one of the 24 instantiations reported `0 bytes stack frame, 0 bytes
spill stores, 0 bytes spill loads`.** No register spilling occurs at any
required component count, for either precision or layout. Register counts
stay low (10-28) and do not scale linearly with N -- consistent with the
kernel's per-component loop not being fully unrolled into N simultaneously
live values, so the "large reacting-flow state" register-pressure concern
motivating this measurement (AGENTS.md #7, BENCHMARKS.md #8) does not
materialize for this simple elementwise kernel. This is a favorable initial
result, not a guarantee that stays true once physics kernels (flux
calculators, chemistry source terms) do meaningfully more per-component
work than a single multiply.

SoA using consistently more registers than AoS at the same N (e.g. 28 vs 18
at double/N=50) is worth noting alongside observation 1 above: SoA is
*both* the faster layout *and* the higher-register one here, so register
count alone would have been a misleading proxy for this kernel's actual
performance -- the achieved-bandwidth measurement is what actually matters.

<!-- Runtime occupancy/local-memory-traffic evidence (profile_cuda.sh parts
2-3, via Nsight Compute) to be appended once collected. -->

## What this resolves from ADR 0001 / ADR 0002

- **ADR 0001 (execution backend):** the CUDA backend now has verified
  correctness (4/4 CUDA unit tests passing on this V100, see
  `agent_history.md`) and verified performance behavior. It can move from
  "unverified" to accepted evidence for the CUDA half of the minimal
  execution abstraction.
- **ADR 0002 (state memory layout):** CPU and GPU evidence now both exist,
  and they **disagree** -- AoS is better on this CPU, SoA is better on this
  GPU, and the GPU effect size is much larger. This is exactly the situation
  ADR 0002 anticipated ("do not hard-code physics APIs to a single layout")
  and argues for keeping `Field`'s `Layout` template parameter rather than
  picking one global default: a CPU-only build should likely default to AoS,
  a CUDA build should likely default to SoA, decided at the same
  case-specific-compilation layer (ADR 0003) that already selects backend
  and precision.
