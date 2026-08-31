# ADR 0001: Minimal execution backend abstraction

**Status:** Proposed  
**Date:** 2026-08-30

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

## Decision

Proposed initial decision: use a minimal CMU-CFE execution abstraction with serial/threaded CPU and CUDA backends.

Do not treat this as Accepted until Phase 0 demonstrates acceptable compile behavior and benchmark overhead.

## Consequences

Physics and numerics code should not contain raw launch syntax.

Backend APIs must remain intentionally small.

## Revisit criteria

Revisit if:

- backend maintenance becomes disproportionate;
- another framework provides measured performance/portability benefits;
- additional accelerators become a near-term requirement.
