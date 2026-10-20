# ADR 0014: Dual-precision (float AND double) verification is a standing PR requirement

**Status:** Accepted
**Date:** 2026-10-15

## Context

`cfe::scalar` (`core/types.hpp`) has existed since Phase 0 specifically so
a single codebase can build and run at either `float` or `double`
(`-DCFE_SCALAR_TYPE=float|double`), independent of the fact that
physics/field/backend headers themselves stay templated on `Scalar` (not
hardcoded to `cfe::scalar`) so both precisions can be exercised explicitly
in the same build (AGENTS.md #11). The motivation is squarely
performance: `float` is the precision GPUs (and SIMD CPU execution more
generally) are fastest at, and AGENTS.md #1/#7 make measured performance
a load-bearing project goal, not an afterthought.

Task 0017 (the first task to actually audit this end-to-end since Phase
0/1/2) found that this promise had quietly gone untested at the
test/tutorial/benchmark layer, even though every production physics/
numerics header itself was already fully `Scalar`-generic (confirmed:
zero hardcoded `double` in `fields/euler/field.hpp`,
`numerics/numerical_flux/{hllc,rusanov}.hpp`, `grid/boundary/*.hpp`).
Specifically:

- `tests/unit/test_hllc_flux.cpp`, `test_euler_3d_sanity.cpp`,
  `test_forward_step_sanity.cpp` hardcoded `using Scalar = double;` at
  file scope -- building the whole project with
  `-DCFE_SCALAR_TYPE=float` did not change what these tests exercised
  AT ALL, since they define their own fixed `Scalar`, independent of
  `cfe::scalar`. A `float` build therefore looked "fully green" while
  silently not testing the `float` path for HLLC, 3D Euler, or the
  forward-step assembly.
- `tests/verification/test_euler_sod_shock_tube{,_hllc}.cpp`,
  `test_euler_shu_osher.cpp`, `test_euler_propagating_shock.cpp`,
  `test_euler_isolated_contact_hllc.cpp`,
  `test_forward_step_interface_conservation.cpp` had the same gap, at
  tight (1e-8 to 1e-13) tolerances specifically calibrated for double's
  own roundoff floor -- exactly the tolerances most likely to need
  revisiting at `float`, and least likely to be revisited by accident.
- `benchmarks/mpi/bench_mpi_halo_exchange.cpp` and the two MPI strong-
  scaling tutorials hardcoded the SIMULATION/field precision to `double`
  regardless of `cfe::scalar` -- a `float` build's own benchmark numbers
  were measuring a `double` halo exchange/solve mislabeled as `float`.

Only `test_fixed_array.cpp` (Phase 0) already did this correctly: a
`template <class Scalar>` check function, instantiated once at `float`
and once at `double`, both wired into the test registry.

See `agent_history.md`'s task-0017 entry for the full remediation (which
files were fixed, the measured float-vs-double tolerance deltas, and the
measured float-vs-double wall-clock speedup) and
`docs/type-reference.md` for anything new introduced.

`VERIFICATION.md` #4 ("Precision verification") already stated this as
a general principle ("Important capabilities should be tested in single
precision; double precision. Document where single precision is
insufficient for a verification target.") -- this ADR is what actually
operationalizes it into an enforceable, per-PR requirement with a
concrete file-level convention, rather than leaving it as an
unenforced aspiration the way the `Scalar`/`cfe::scalar` split itself
had been. Per that same requirement's second sentence: task 0017 found
NO verification target in its scope where `float` was insufficient at
any reasonable tolerance -- every check in
`tests/verification/test_euler_sod_shock_tube{,_hllc}.cpp`,
`test_euler_shu_osher.cpp`, `test_euler_propagating_shock.cpp`,
`test_euler_isolated_contact_hllc.cpp`, and
`test_forward_step_interface_conservation.cpp` passes at `float` with a
measured, documented margin (see `agent_history.md`'s task-0017 entry
for the specific numbers) -- itself a documented finding, not a gap.

## Decision

**Every PR that introduces or changes numerical production code (a
Field, a Calculator, a numerical flux, a reconstruction scheme, a
boundary condition, a time integrator, or anything else templated on
`Scalar`) must pass its relevant tests under BOTH
`-DCFE_SCALAR_TYPE=float` and `-DCFE_SCALAR_TYPE=double`, in the same
PR -- not deferred to a follow-up.** This is the same standing-policy
shape this project already uses for GPU coverage ("every PR adding a
Field/scheme needs a 3D CUDA correctness test + at-scale benchmark in
the same PR," `agent_history.md`'s 2026-10-02 entry, ADR 0011 Decision/
limitations section, ADR 0012's evidence section) -- recorded here as
its own ADR, and in AGENTS.md #11, because (per that same precedent)
leaving it as a private recollection in `agent_history.md` alone let it
go unenforced for three Phases before task 0017 caught it.

Convention for a new or touched test file (see
`tests/unit/test_fixed_array.cpp`'s original pattern, and task 0017's
`test_hllc_flux.cpp`/`test_euler_sod_shock_tube*.cpp`/etc. for the
multi-check-per-file version of it):

1. Each check is a `template <class Scalar> void check_X(...)` function,
   not a file-scope `using Scalar = double;` alias.
2. Register it under **two** `CFE_TEST` entries, `..._double` and
   `..._float`, each instantiating the template.
3. Any oracle/reference solution that is a property of the PROBLEM, not
   of the simulation (an exact Riemann solver, a Rankine-Hugoniot jump,
   a published constant) stays plain `double` regardless of `Scalar`,
   same convention this codebase already used before this ADR (e.g.
   `sod_shock_tube_exact.hpp`) -- it is a reference value, not state
   being verified.
4. A tolerance that must widen at `float` is derived from a MEASURED
   float-vs-double error (print it, run it, read the real number), not
   guessed -- the same evidentiary standard AGENTS.md #25 already
   requires project-wide. Many tolerances, once actually measured,
   needed NO widening at all (task 0017's own finding: truncation-
   error-dominated checks like exact-solution density samples or L1-
   convergence trends were nearly IDENTICAL at float and double; only
   roundoff-dominated checks -- exact conservation, bit-for-bit cross-
   checks -- needed a float-epsilon-scaled widening). Do not loosen a
   tolerance "to be safe" without first measuring whether it is
   actually needed.
5. Benchmarks/tutorials: if the hardcoded type is the
   SIMULATION/field precision, it must track `cfe::scalar` (`using
   Scalar = cfe::scalar;`, `#include "cfe/core/types.hpp"`), the
   convention `tutorials/burgers_3d_visualization/` already
   established. If it is a reported METRIC (wall-clock time, a
   `std::chrono::duration<double>`), it stays plain `double`
   regardless of `cfe::scalar` -- same existing convention.

This does not mean every test in the repository must grow a float
counterpart in one pass -- task 0017 is scoped to the Phase 3/4 Euler/
HLLC/forward-step files the gap was found in (see its own task file and
`agent_history.md` entry for the explicit list); the earlier Phase 1/2
Burgers/ScalarAdvection unit tests (e.g. `test_euler_flux.cpp`,
`test_burgers_*.cpp`, `test_scalar_advection_*.cpp`) were NOT touched by
this task and remain a known, named residual gap, not silently closed.
The STANDING RULE this ADR records is forward-looking: it governs PRs
from this point on, not a mandate to retrofit every pre-existing file in
one sweep.

## Alternatives considered

- **Leave `cfe::scalar`'s dual-precision promise as an aspirational
  design note, verified ad hoc.** Rejected: this is exactly the status
  quo task 0017 found had silently decayed over three Phases -- an
  unenforced aspiration is indistinguishable from an untested one.
- **Require only ONE precision's tests to pass, with the other treated
  as "best effort."** Rejected: `core/types.hpp`'s own documented
  rationale is that physics/field/backend components stay templated on
  `Scalar` specifically so both precisions can be explicitly exercised
  -- treating one as second-class defeats that rationale and directly
  contradicts the user's hardware-acceleration goal (float path
  untested is float path not trustworthy to actually ship on GPU
  hardware).
- **A single shared macro/helper that generates both `CFE_TEST` entries
  automatically**, to cut the double-registration boilerplate. Considered
  (Snorlax's review scope for task 0017) and deliberately NOT adopted:
  the six verification files task 0017 touched do not share enough
  structure (different tolerance-derivation reasoning, different oracle
  shapes, different per-check logic) for a single macro to stay readable
  -- see task 0017's own Snorlax review notes in `agent_history.md` for
  the specific files considered and why a forced abstraction was judged
  worse than the explicit two-`CFE_TEST`-per-check repetition
  `test_fixed_array.cpp` already established.

## Consequences

- A new numerical PR's review checklist (CLAUDE.md's workflow step 4,
  Violet/Snorlax/Sally/Enzo) now explicitly includes: "does this PR's
  new/changed test coverage pass at both `-DCFE_SCALAR_TYPE=float` and
  `=double`?" -- the same footing as the existing CUDA-coverage
  checklist item, not a softer suggestion.
- Slightly more test-registry entries per numerical check (two
  `CFE_TEST`s instead of one) -- accepted; task 0017's own measurement
  (196 vs. the prior 158 unit+verification tests, all passing, total
  suite wall-clock unchanged at ~10-18s) shows this cost is negligible
  next to the correctness value.
- This is a TEST-SUITE policy, not a change to any production header's
  public interface -- no entries needed in `docs/type-reference.md` for
  this ADR itself (see task 0017's `agent_history.md` entry for whether
  any incidental new type was introduced alongside it).

## Revisit criteria

- If a future numerical component genuinely cannot be made to pass at
  `float` at any reasonable tolerance (not merely "wasn't tried"), that
  is itself a documented finding (per this ADR's own evidentiary
  standard) -- escalate to Violet/Amelia rather than silently excluding
  it from this rule.
- If the per-check `CFE_TEST` pair pattern starts to feel genuinely
  repetitive across many MORE files than task 0017's six (i.e. the
  "alternatives considered" shared-macro rejection above stops holding
  as the dual-precision file count grows), revisit the shared-helper
  question with the larger set of examples in hand.
