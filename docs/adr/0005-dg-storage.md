# ADR 0005: DG storage and execution

**Status:** Proposed  
**Date:** 2026-08-30

## Context

CMU-CFE will support DG `p=1...4`, initially on Cartesian elements and later on unstructured grids.

Storage must support efficient element-local operations, surface communication, and future multimethod DG/FVM coupling.

## Questions to resolve experimentally

- degree-of-freedom ordering;
- element-major versus component-major layouts;
- GLL collocation storage;
- volume/surface kernel fusion;
- state-size effects;
- register pressure;
- treatment of geometry data.

## Decision

No permanent DG storage scheme is accepted yet.

Create a benchmarkable prototype before locking the layout.

## Revisit criteria

Accept after DG `p=1` and `p=2` kernels are benchmarked on CPU and CUDA.
