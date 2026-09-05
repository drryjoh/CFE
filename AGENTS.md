# CMU-CFE Agent Engineering Rules

This file contains durable rules for all AI agents and human contributors working on CMU-CFE.

Do not use this file as a development diary. Chronological work belongs in `agent_history.md`. Architectural rationale belongs in `ARCHITECTURE.md` and `docs/adr/`.

## 1. Project mission

CMU-CFE is a performance-oriented CFD research environment designed to make experimentation with numerical methods and physical models easy without making expensive computation slow.

The code should support education, research discovery, and eventually production-scale simulations of compressible chemically reacting flows.

## 2. Priority order

When requirements conflict, use this order:

1. correctness;
2. measured performance;
3. architectural clarity;
4. testability;
5. extensibility;
6. portability;
7. convenience.

Never trade correctness for speed.

Never call code optimized without measurement.

## 3. Required reading before changes

Before modifying the repository:

1. read `AGENTS.md`;
2. read `ARCHITECTURE.md`;
3. read relevant ADRs in `docs/adr/`;
4. read `agent_history.md`;
5. inspect relevant tests;
6. inspect relevant benchmarks;
7. identify whether the change affects public interfaces, memory layout, numerical behavior, precision, communication, or performance-critical kernels.

## 4. Required actions after changes

After a change:

1. compile every affected configuration;
2. run relevant unit tests;
3. run relevant regression tests;
4. run convergence or verification tests when numerical behavior changed;
5. run benchmarks when performance-sensitive code changed;
6. add tests for every new feature;
7. add or update a tutorial for user-facing functionality;
8. update documentation;
9. append an entry to `agent_history.md`;
10. create or update an ADR for major architectural decisions;
11. add a presentation summary under `presentations/` for the PR (see
    section 26).

## 5. Definition of done

A feature is complete only when:

- it compiles;
- relevant tests pass;
- expected numerical behavior is demonstrated;
- performance-sensitive portions are benchmarked;
- no unexplained regression exists;
- documentation is updated;
- a tutorial exists when applicable;
- `agent_history.md` is updated.

"Code generated successfully" is not a definition of done.

## 6. Scope discipline

When given a task:

- implement only the requested scope;
- do not opportunistically add unrelated capabilities;
- do not redesign unrelated interfaces;
- prefer the smallest architecture that satisfies the requirement;
- stop before making a major unrequested architectural decision and write an ADR proposal instead.

## 7. Performance rules

Performance is a scientific requirement.

Measure, where applicable:

- wall-clock time;
- throughput;
- latency;
- memory bandwidth;
- kernel-launch overhead;
- MPI communication;
- GPU occupancy;
- register count;
- local-memory traffic;
- register spilling;
- strong scaling;
- weak scaling.

Performance-sensitive changes require benchmark evidence.

Do not assume CPU optimization improves GPU performance.

Do not assume GPU optimization improves Apple Silicon performance.

## 8. Large-state requirement

CMU-CFE must not be designed only around five-equation compressible-flow states.

Representative state-size benchmarks must include:

- 1 component;
- 5 components;
- 10 components;
- 20 components;
- 50 components;
- 100 components.

Large reacting-flow states must be considered from the beginning.

Register pressure and spilling must be monitored on GPU kernels.

## 9. Hardware and backend rules

Initial hardware targets:

- x86-64 multicore CPUs;
- Apple Silicon CPU execution;
- NVIDIA CUDA GPUs;
- distributed-memory systems using MPI.

Algorithms must be separated from execution policy.

CUDA-specific syntax must remain inside backend abstractions wherever practical.

Preferred style:

```cpp
CFE_HOST_DEVICE
CFE_FORCEINLINE
auto square(const scalar& a)
{
    return a * a;
}
```

and:

```cpp
cfe::parallel_for(range, [=] CFE_DEVICE (auto i)
{
    // kernel body
});
```

Do not introduce Kokkos, RAJA, SYCL, AMReX, or another portability framework without an ADR and measured justification.

## 10. Memory rules

Performance-critical storage should favor:

- contiguous memory;
- coalesced GPU access;
- vectorizable CPU access;
- efficient MPI packing;
- predictable indexing.

Avoid:

- pointer-heavy object graphs;
- per-cell heap allocation;
- hidden allocations in kernels;
- hidden host-device transfers;
- virtual dispatch in hot kernels unless benchmarked and justified.

Do not assume AoS, SoA, or AoSoA is optimal without measurement.

Unified/managed memory may be supported but must not be required for high performance.

## 11. Scalar and index types

Floating-point precision and indexing are separate concepts.

Initial floating-point scalar types:

```cpp
float
double
```

Define independent index types for local and global indexing as required.

Do not use `long` or `long long` as floating-point precision options.

## 12. State representation

Physics states should have compile-time-known structure whenever practical.

Physics code should use semantic accessors:

```cpp
auto rho = density(q);
auto mom = momentum(q);
auto E   = total_energy(q);
```

rather than scattering raw positional indices throughout the code.

The logical physics interface must not dictate a poor physical memory layout.

Tuple-like semantic views are acceptable. Pointer-heavy or deeply nested storage is not assumed acceptable.

## 13. Math rules

Low-level mathematical functionality belongs under `src/math/`.

Math operations needed in kernels must be host/device callable.

Examples include:

- componentwise operations;
- dot products;
- contractions;
- dyadic products;
- tensor products;
- norms;
- transpose;
- determinant;
- inverse for supported dimensions.

Required semantic operations include:

```cpp
contract(A, B)
```

for contraction and:

```cpp
weight(A, B)
```

for componentwise weighting.

Every mathematical primitive requires unit tests.

Analytic derivatives should live near the corresponding operation.

## 14. Separation of concerns

Use the following conceptual hierarchy:

```text
Solver
 ├── Grid
 ├── Fields
 │    └── Calculator
 ├── Numerics
 │    ├── Reconstruction / Basis
 │    └── Numerical Flux
 ├── Time Integrator
 ├── Diagnostics
 └── Execution Backend
```

Definitions:

- **Field**: what equations are solved.
- **Calculator**: what the physics means.
- **Numerics**: how the equations are discretized.
- **Solver**: how the calculation is orchestrated.
- **Backend**: where computation executes.

Do not duplicate physics across numerical schemes.

Do not duplicate reconstruction inside fields.

Do not place MPI communication inside physics calculators.

## 15. Grid rules

Initial implementation targets Cartesian box grids.

The long-term architecture must support structured, nonuniform, AMR, and unstructured hybrid grids.

Do not over-generalize the first Cartesian implementation.

Do not design future grid interfaces around the assumption that all connectivity can always be reconstructed from `(i,j,k)`.

Unstructured connectivity should eventually use compact contiguous adjacency structures suitable for GPU execution.

## 16. MPI and parallelism

MPI is the primary distributed-memory decomposition mechanism.

Within each MPI partition, use the selected CPU or accelerator backend.

Halo exchange must be separated from physical calculations.

Communication and computation should eventually permit overlap where useful.

Do not require graph coloring globally. Gather formulations, atomics, ownership approaches, or coloring may be selected per algorithm based on benchmark evidence.

## 17. Boundary conditions

Finite-volume methods may initially use ghost cells.

Finite-element methods should use appropriate boundary states and/or numerical fluxes.

Required early boundary capabilities include:

- periodic;
- Dirichlet;
- Neumann;
- extrapolation/outflow;
- reflective/slip;
- no-slip where appropriate.

Moving-frame/grid-recycling behavior is a separate capability and must not be conflated with ordinary static boundary conditions.

## 18. Numerical methods

Target methods include:

### Finite volume
- MUSCL;
- PPM;
- high-order WENO;
- second-order viscous discretization initially;
- higher-order viscous discretizations later.

### Finite element
- DG first;
- CG later where useful;
- DG polynomial orders `p = 1...4`;
- collocated Gauss-Lobatto-Legendre points where appropriate.

Numerical schemes must be documented with exact formulations and references.

Avoid ambiguous names such as "8th-order WENO" without recording the exact reconstruction and formal order.

## 19. Hybrid DG/FVM

Multimethod DG/FVM simulation is a central research objective.

Hybridization logic belongs outside the individual DG and FVM implementations.

The architecture must eventually permit different regions to use different methods with explicit interfaces between them.

Begin with simple 1D static-region demonstrations before dynamic reacting-flow hybridization.

## 20. Fields and calculators

A field defines a system of equations and its state.

A calculator provides small, fast, composable physics operations such as:

- primitive/conserved conversion;
- equation of state;
- physical fluxes;
- viscous fluxes;
- source terms;
- thermodynamic properties;
- transport properties;
- stability limits;
- artificial-viscosity information;
- analytic derivatives;
- Jacobian components.

Performance-critical calculator functions must not allocate dynamic memory.

## 21. Chemistry

Chemistry will use ChemGen-generated kernels.

ChemGen should eventually provide:

- species thermodynamics;
- reaction rates;
- source terms;
- derivatives;
- Jacobian information.

Target chemistry integration modes:

1. explicit coupled;
2. explicit Strang split;
3. implicit coupled;
4. implicit Strang split.

Generated chemistry should compile directly into case-specific executables where practical.

## 22. Differentiability

Functions that will participate in implicit methods must have accessible analytic derivatives or perturbation operators.

Automatic differentiation is not a core requirement.

Derivative functionality should be compile-time removable when unused.

Do not impose derivative storage or computation cost on explicit configurations unnecessarily.

## 23. Machine learning

ML belongs under `src/ml/` and must not become a dependency of the core CFD library.

Support both:

- inline/fused inference for sufficiently small models;
- separate batched inference kernels for larger models.

Choose based on measurement.

Do not fuse ML into CFD kernels when it causes unacceptable register pressure, occupancy loss, duplicated work, or synchronization cost.

## 24. Case-specific compilation

Performance is favored over unnecessary runtime generality.

Compile-time specialization should be used when it materially improves performance.

Likely compile-time configuration includes:

- backend;
- precision;
- dimension;
- field type;
- species mechanism/count;
- major numerical method;
- DG polynomial order where required;
- derivative support.

Likely runtime configuration includes:

- CFL;
- final time;
- output frequency;
- diagnostic positions;
- initial-condition parameters;
- boundary values where appropriate.

**Configuration determines types. Types do not parse configuration.**

C++ physics kernels must not read YAML or other case configuration directly.

## 25. AI-specific rule

AI-generated code is accepted by evidence, not by confidence.

For numerical development, prefer this acceptance sequence:

```text
compile
-> unit test
-> conservation
-> formal convergence
-> canonical verification
-> performance profile
-> regression baseline
```

The compiler, tests, convergence data, canonical solutions, and profilers are the referees.

## 26. Presentation summaries

Every PR must include a markdown file under `presentations/`, named to match
the PR/task it covers (e.g. `presentations/0001-phase0-execution-foundation.md`
for `tasks/0001-phase0-execution-foundation.md`). This is a distinct
document from `agent_history.md` and the ADRs: those are engineering
records; this one is a weekly/biweekly progress-review script for a
non-CS audience.

Audience: bright students without a CS/CE background. They know GPUs are
"fast" but do not necessarily know what a kernel, a thread, a cache, or a
memory layout is. Do not assume prior exposure to any of these terms --
define each one in plain language the first time it is used.

Format: write it so it can be copied directly into presentation slides.
That means:

- short, slide-sized sections with a clear title each (one idea per
  section, not one giant wall of text);
- bullet points over paragraphs;
- plain-language analogies where they help (a mesh cell, a thread, a
  memory layout are all things worth grounding in an everyday comparison);
- numbers/results presented as the takeaway ("AoS was up to 8x faster"),
  not as a raw data dump -- the CSV/results doc is already the place for
  the full data.

Content: explain, at a conceptual level and in this rough order --

1. what problem this PR's code is actually solving, in one sentence a
   non-CS student would understand;
2. what the core execution pattern does (e.g. "loop over every cell and do
   the same small operation" -- this is what `parallel_for` is) and why
   that pattern matters for performance work;
3. why something was tested locally/on CPU first rather than jumping
   straight to GPU;
4. what any benchmark/test comparison actually showed and why it matters
   (e.g. AoS vs. SoA), stated as a takeaway, not just a result;
5. what remains untested/unverified and why (e.g. "needs real GPU
   hardware"), framed as the natural next step rather than a shortcoming.

A presentation file for a PR whose story continues in a later PR (e.g.
Phase 0's CUDA backend, pending GPU access) should be appended to when that
follow-up evidence arrives, not abandoned in favor of a new file, unless
the follow-up is large enough to deserve its own PR and its own
presentation file -- use judgment; the goal is a presentation file a
student could read start-to-finish and understand the state of that
specific piece of work.
