# ADR 0002: State memory layout

**Status:** Proposed (CPU evidence collected; GPU evidence outstanding)  
**Date:** 2026-08-30 (updated 2026-08-30, Phase 0 evidence)

## Context

CMU-CFE must perform well for small states and for reacting-flow states containing tens to approximately one hundred components.

A human-friendly nested state representation must not force inefficient physical storage.

## Options considered

- Array of Structures (AoS)
- Structure of Arrays (SoA)
- Array of Structures of Arrays (AoSoA)
- backend-specific layouts behind a common view

## Evidence required

Before acceptance, benchmark representative operations for state sizes:

```text
1, 5, 10, 20, 50, 100
```

on:

- CPU;
- NVIDIA GPU.

Measure:

- throughput;
- memory bandwidth;
- register count;
- local memory;
- spilling;
- vectorization/coalescing behavior.

## Evidence (Phase 0, CPU only)

Implemented `cfe::Field<Scalar, NComponents, Layout>` (`src/cfe/field/field.hpp`)
with a `Layout` policy template parameter (`src/cfe/field/layout.hpp`):
`AoSLayout` (`index = cell * N + component`) and `SoALayout`
(`index = component * n_cells + cell`). Both are exercised by the same
`q_new(i,k) = q(i,k) * q(i,k)` benchmark
(`benchmarks/memory/bench_field_update.cpp`) across component counts
1/5/10/20/50/100, precisions float/double, and backends serial/threaded, on
Apple M5 (see `docs/performance/0001-phase0-results.md` for the full
table and environment; raw data in
`benchmarks/results/phase0_field_update_apple_m5.csv`).

Summary of the CPU result: **AoS matched or beat SoA at every measured
component count**, and the gap widened with N. At N=1 the layouts are
equivalent (only one component exists). From N>=20, AoS held roughly flat
bandwidth (~76-88 GB/s across backends) while SoA degraded -- at
N=100/double/serial, SoA measured ~4.3x slower than AoS (17.8 vs 76.5 GB/s);
at N=100/float/serial, ~8.1x slower (9.3 vs 75.4 GB/s). The threaded backend
narrowed but did not eliminate the gap. This is consistent with the kernel's
per-cell access pattern: AoS keeps a cell's N components contiguous, so one
loop iteration touches one cache-line neighborhood; SoA scatters them across
N independent strides of length `n_cells`, opening N far-apart memory
streams per iteration.

**No GPU evidence exists.** No CUDA toolkit/NVIDIA GPU was available in the
Phase 0 environment. This matters here specifically because GPU coalescing
favors consecutive threads reading consecutive addresses -- which is a
property SoA typically provides and AoS typically does not -- so the CPU
result above must not be assumed to transfer to CUDA. This ADR's own
"Evidence required" section explicitly calls for both CPU and GPU data
before a layout is accepted as a global default; only half of that exists.

## Decision

Still not fully accepted, by this ADR's own evidence bar: CPU data alone is
not sufficient given the required CPU + GPU comparison above, and GPU
coalescing considerations are a plausible reason the CPU-favored layout
(AoS) could be the wrong default on CUDA.

Provisional recommendation for CPU-only configurations: default new
`cfe::Field` usage to `AoSLayout` unless/until GPU evidence says otherwise,
since it is at least as good as SoA at every CPU-measured point and
meaningfully better at large N. This is a recommendation, not an accepted
default -- `Field`'s `Layout` template parameter keeps the choice
overridable per call site precisely because of this.

The initial implementation continues to provide contiguous storage and
semantic state views (`FieldView`) without hard-coding physics APIs to a
single layout.

## Consequences

The state interface and storage implementation remain separable
(`Field` owns storage; `FieldView` is the pointer-based accessor physics/
numerics code would actually use, independent of which `Layout` or which
memory space -- host or device -- backs it).

Any code choosing `AoSLayout` as a default ahead of GPU evidence should be
prepared to revisit that choice; do not hard-code assumptions that depend on
AoS being globally best.

## Revisit criteria

Accept a specific default once GPU evidence exists. Specifically:

- run `benchmarks/memory/bench_field_update_cuda.cu` (currently unbuilt, see
  ADR 0001) across the same sweep, on real CUDA hardware;
- if CUDA shows SoA (or another layout) is meaningfully better there,
  either pick a default per-backend rather than one global default, or
  weigh the CPU/GPU tradeoff explicitly and document the reasoning here;
- revisit the working AoS recommendation above if any future measured
  kernel (not just the elementwise-square microbenchmark) shows different
  layout sensitivity -- this kernel has no cross-component reduction, and a
  kernel that does (e.g. `contract`-heavy physics) may behave differently.
