# ADR 0002: State memory layout

**Status:** Proposed  
**Date:** 2026-08-30

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

## Decision

No physical memory layout is accepted yet.

The initial implementation should provide contiguous storage and semantic state views without hard-coding physics APIs to a single layout.

## Consequences

The state interface and storage implementation must be separable.

## Revisit criteria

Accept a default after Phase 0 performance evidence is available.
