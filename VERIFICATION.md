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

#### Canonical problems with no closed-form exact solution

Some canonical problems (Shu-Osher is the first case in this codebase;
Taylor-Green vortex and flat plate will likely need the same treatment
later) have no analytic oracle for their disturbed/nonlinear region --
that is often the entire point of the test. Accepted practice for these
(established by `tests/verification/test_euler_shu_osher.cpp`, task
0012):

- compare against a converged fine-grid reference solution run with the
  SAME code (and, where more than one numerical method is being
  compared, the same method per reference -- a Rusanov run is verified
  against its own Rusanov fine-grid reference, not against an HLLC one),
  not an analytic oracle;
- use an exact solution in any subregion that genuinely still has one
  (e.g. the region ahead of a shock, where the initial condition itself
  remains an exact, time-independent solution until the shock arrives)
  as a real, blocking, oracle-based check -- do not treat "no exact
  solution for the problem as a whole" as "no exact check is possible
  anywhere in it";
- use a resolution-converging error metric (error shrinks as resolution
  increases toward the fine reference) rather than a single
  fixed-resolution pass/fail;
- a scriptable proxy (e.g. a feature/extremum count) stands in for
  "matches a published figure's qualitative structure" where useful --
  do not attempt to script a direct comparison against a digitized
  literature plot.

This derived-metric-oracle pattern (compare against a known analytic or
converged-reference scalar, never a raw per-cell field dump) is also
what `docs/adr/0013-tutorial-data-commit-policy.md` relies on to justify
NOT committing raw snapshot data for tutorials/verification cases:
committed `figures/*.png` and `data/**/summary.csv` already carry
everything a reviewer or a future reader needs to check a result, the
same way these tests do.

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
