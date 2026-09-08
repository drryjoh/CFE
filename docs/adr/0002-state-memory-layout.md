# ADR 0002: State memory layout

**Status:** Accepted -- per-backend default (AoS for CPU, SoA for CUDA), not one global default; see Decision  
**Date:** 2026-08-30 (updated 2026-08-30, Phase 0 CPU evidence; updated 2026-09-08, Phase 0 GPU evidence)

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

## Evidence (Phase 0, CPU: Apple M5)

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

## Evidence (Phase 0, GPU: NVIDIA V100, 2026-09-08)

`bench_field_update_cuda.cu` was run on PSC Bridges-2 (Tesla V100-SXM2-32GB)
across the same required sweep (see
`docs/performance/0002-phase0-cuda-results.md` and
`benchmarks/results/phase0_field_update_v100.csv`). Result: **SoA wins
decisively, the opposite of CPU, and by a much larger margin.** At N=1 the
layouts are equivalent (750-765 GB/s either way). From N=5 upward SoA pulls
ahead and the gap widens with N: at N=100/float, SoA is ~33x faster than AoS
(546.6 vs 16.4 GB/s); at N=100/double, ~22.5x faster (678.0 vs 30.1 GB/s).
Nsight Compute profiling confirms the mechanism directly rather than by
inference: at double/N=100, AoS utilizes only 8.0 of 32 bytes per memory
transaction (uncoalesced, a stride between threads) vs SoA's 30.1 of 32
(94%, near-ideal coalescing) -- consecutive GPU threads process consecutive
cells, so SoA gives them consecutive addresses per component while AoS
scatters each thread's access `n_components` scalars apart. Register
pressure was not a confound: zero spilling was observed for every
`(Scalar, N, Layout)` instantiation on this GPU (see ADR 0001).

This is exactly the situation this ADR's evidence bar was designed to
surface: CPU and GPU evidence now both exist, and they disagree, with the
GPU effect size substantially larger than the CPU effect size measured in
either direction.

## Decision

**Accepted: the memory-layout default is per-backend, not global.**

- CPU-only configurations default `cfe::Field` usage to `AoSLayout` (CPU
  evidence: matches or beats SoA at every measured point, up to ~8x at
  N=100).
- CUDA configurations default to `SoALayout` (GPU evidence: up to ~33x
  faster than AoS at N=100, with Nsight Compute directly attributing this
  to coalesced vs. strided memory access).

`Field`'s `Layout` template parameter already supports this without any
interface change: the decision is *which layout the case-specific compiler
(ADR 0003) selects by default per backend*, not a change to `Field`/
`FieldView` themselves, which remain layout-agnostic. Do not pick one
single global default (neither AoS nor SoA) -- either measured platform
would be meaningfully wrong for the other.

## Consequences

The state interface and storage implementation remain separable
(`Field` owns storage; `FieldView` is the pointer-based accessor physics/
numerics code would actually use, independent of which `Layout` or which
memory space -- host or device -- backs it). This separability is what
makes a per-backend default free to implement: no physics/numerics call
site needs to change based on which backend it's compiled for.

ADR 0003 (case-specific compilation) should select `Layout` as part of its
backend-driven compile-time configuration, alongside precision and
dimension, rather than requiring call sites to specify it explicitly.

## Revisit criteria

- Revisit the per-backend default if any future measured kernel (not just
  the elementwise-square microbenchmark) shows different layout
  sensitivity -- this kernel has no cross-component reduction, and a kernel
  that does (e.g. `contract`-heavy physics, flux calculators with more
  arithmetic per component) may behave differently on either platform.
- Revisit if AoSoA is implemented and measured -- it was out of Phase 0's
  scope (see `docs/performance/0001-phase0-results.md`) but could plausibly
  narrow or close either platform's gap.
- Revisit the CUDA default if profiling on a different GPU architecture
  (the Nsight Compute evidence above is V100/sm_70-specific) shows
  different coalescing behavior.
