# CMU-CFE Verification Requirements

Scientific verification is required for numerical functionality.

## 1. Verification hierarchy

Prefer the following progression:

1. algebra/unit test;
2. manufactured solution;
3. analytic solution;
4. canonical numerical benchmark;
5. physical validation.

A visually plausible solution is insufficient evidence.

## 2. Required verification categories

### Unit tests

Test:

- math primitives;
- state accessors;
- geometry;
- indexing;
- boundary transformations;
- thermodynamic operations;
- flux functions;
- source terms;
- derivatives.

### Conservation tests

Where appropriate verify:

- mass;
- momentum;
- energy;
- species;
- interface conservation.

### Convergence tests

For every discretization with a claimed order, perform mesh/time refinement and measure observed order.

Record:

- grid sequence;
- error norm;
- expected order;
- observed order;
- asymptotic behavior.

### Regression tests

Store compact reference quantities that detect unintended changes.

Do not use regression tests as substitutes for convergence or analytic verification.

### Canonical problems

Planned examples include:

- scalar translation;
- Burgers smooth convergence;
- Burgers shock formation;
- Sod shock tube;
- Shu-Osher;
- forward-facing step/ramp;
- flat plate;
- Taylor-Green vortex;
- 0D chemistry;
- laminar flame;
- detonation.

## 3. CPU/GPU comparison

When the same algorithm runs on CPU and GPU, compare scientifically meaningful quantities within tolerances appropriate for floating-point non-associativity.

Do not require bitwise agreement unless explicitly designed for it.

## 4. Precision verification

Important capabilities should be tested in:

- single precision;
- double precision.

Document where single precision is insufficient for a verification target.

## 5. Derivative verification

Analytic derivatives and Jacobian components should be checked against finite-difference or complex-step references when possible.

Derivative tests should use tolerances appropriate for truncation and roundoff.

## 6. Chemistry verification

Chemistry requires independent tests for:

- thermodynamic identities;
- reaction source terms;
- elemental conservation;
- species conservation constraints;
- equilibrium/known reactor behavior where applicable;
- Jacobian consistency.

Transport + chemistry verification must not be the first test of chemistry correctness.

## 7. DG/FVM interface verification

Multimethod coupling should test:

- conservation across interfaces;
- constant-state preservation;
- smooth-solution accuracy;
- wave transmission/reflection;
- method-transition stability.

## 8. Moving-frame verification

Grid recycling should test:

- constant-state preservation;
- periodic/simple transported profile;
- indexing consistency after many recycling events;
- conservation where physically applicable;
- equivalence to a larger static-domain reference where possible.

## 9. Test naming

Tests should state what scientific property they verify.

Prefer:

```text
test_scalar_advection_second_order_convergence
test_hllc_preserves_stationary_contact
test_grid_recycling_preserves_constant_state
```

over generic names such as:

```text
test_solver1
test_flux
```

## 10. Acceptance principle

AI-generated numerical code is accepted only after evidence establishes its behavior.
