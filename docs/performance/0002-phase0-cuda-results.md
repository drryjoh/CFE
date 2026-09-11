# Phase 0 CUDA benchmark results

Raw CSV: [`benchmarks/results/phase0_field_update_v100.csv`](../../benchmarks/results/phase0_field_update_v100.csv).
Raw `nvcc --resource-usage` and `ncu` reports:
[`benchmarks/results/ncu_reports/`](../../benchmarks/results/ncu_reports/).

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

## Runtime occupancy / local-memory traffic / achieved bandwidth (Nsight Compute)

Procedure: `scripts/profile_cuda.sh`, parts 2-4, via `ncu --set full` (occupancy,
memory workload analysis) and a targeted `ncu --metrics
l1tex__t_bytes_pipe_lsu_mem_local_op_{ld,st}.sum` query (local-memory/spill
traffic). Unlike the static register sweep above, this requires a live GPU
allocation, so it was run on a representative subset rather than all 24
instantiations x 8 launches (192 total) each: `double`/AoS and `double`/SoA at
`N=1` (identical layouts, sanity baseline) and `N=100` (the largest gap
observed in the raw benchmark above). `--kernel-name-base mangled -k
"regex:..."` selected one specific `(Scalar, N, Layout)` instantiation's first
launch per profiling run.

| | N=1, AoS | N=100, AoS | N=100, SoA |
|---|---|---|---|
| Achieved bandwidth | 793.9 GB/s (89.4% of peak) | 213.6 GB/s (23.7% of peak) | 728.8 GB/s (82.3% of peak) |
| Achieved occupancy | 91.2% | 50.7% | 50.8% |
| Theoretical occupancy | 100% | 100% | 100% |
| Waves per SM | 51.2 | 0.51 | 0.51 |
| Bytes used per 32-byte sector (global loads) | -- | 8.0 (25%) | 30.1 (94%) |
| Local-memory (spill) traffic | -- | 0 bytes ld, 0 bytes st | -- |

### Observations

1. **The N=1 -> N=100 occupancy drop is a problem-size effect, not a layout
   effect.** Both N=100 layouts land at the same ~51% achieved occupancy and
   0.51 waves/SM ("this kernel grid is too small to fill the available
   resources on this device" per Nsight Compute's own launch-statistics
   warning) -- because `n_cells` is intentionally shrunk as N grows to hold
   each field at ~64 MiB (see Methodology), there are simply fewer thread
   blocks to launch at N=100 than at N=1. Layout does not change this.
2. **The AoS vs SoA gap at N=100 is a pure coalescing effect on top of that
   shared occupancy ceiling.** Nsight Compute quantifies this directly: AoS
   utilizes only 8.0 of the 32 bytes transmitted per memory sector (a stride
   between threads wastes 75% of every transaction), while SoA utilizes 30.1
   of 32 bytes (94%, close to ideal). This is precisely the coalesced-access
   mechanism hypothesized in observation 1 above, now measured rather than
   inferred, and Nsight Compute's own optimization estimate agrees: it
   reports fixing AoS's uncoalesced pattern at this instantiation would be
   worth an estimated 73.9% speedup.
3. **Zero local-memory (spill) traffic at runtime**, corroborating the static
   `nvcc --resource-usage` result above with a direct runtime measurement at
   the largest required state size (double/N=100/AoS): `0 bytes` for both
   `l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum` and `..._st.sum`.
4. **N=1's 793.9 GB/s (89.4% of the V100-SXM2-32GB's ~888 GB/s published HBM2
   peak)** is the cleanest achieved-bandwidth reference this kernel produces
   on this GPU: full occupancy (51.2 waves/SM), identical layouts, no
   coalescing penalty possible with a single component.

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
