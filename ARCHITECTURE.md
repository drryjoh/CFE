# CMU-CFE Architecture

This document describes the current intended architecture of CMU-CFE. It may evolve through Architecture Decision Records.

## 1. Architectural objective

CMU-CFE should allow physics, numerical methods, grids, execution backends, diagnostics, and future ML components to compose without forcing unnecessary changes in one another.

The architecture should make scientific experimentation inexpensive while keeping storage, execution, and performance-critical kernels constrained and inspectable.

## 2. Core decomposition

```text
Solver
 ├── Grid
 ├── Field(s)
 │    └── Calculator(s)
 ├── Numerics
 │    ├── Reconstruction / Basis
 │    ├── Volume Terms
 │    └── Numerical Flux
 ├── Time Integrator
 ├── Boundary Handling
 ├── Diagnostics
 ├── I/O
 └── Execution Backend
```

### Solver

The solver orchestrates:

- initialization;
- decomposition;
- halo exchange;
- boundary updates;
- residual evaluation;
- source integration;
- time stepping;
- diagnostics;
- output;
- checkpoints;
- stopping criteria.

The solver must not contain physical formulas that belong in calculators or numerical reconstruction formulas that belong in numerics.

### Field

A field defines the equations being solved and the semantic state representation.

Expected field families include:

- scalar advection;
- Burgers;
- heat;
- wave;
- compressible Navier-Stokes;
- incompressible Navier-Stokes;
- compressible reacting Navier-Stokes;
- low-Mach reacting flow;
- RANS;
- generated/custom fields.

### Calculator

A calculator provides physics-specific device-callable functions:

- primitive/conserved conversion;
- equation of state;
- inviscid flux;
- viscous flux;
- source terms;
- thermodynamics;
- transport;
- wave speeds;
- stability constraints;
- artificial viscosity;
- derivatives;
- Jacobian components.

Calculators should be small, inlinable, composable, and free of dynamic allocation in hot paths.

### Numerics

Numerics describe spatial discretization independently of physics where mathematically appropriate.

Target structure:

```text
numerics/
    fvm/
        reconstruction/
            muscl/
            ppm/
            weno/
        viscous/
    fem/
        dg/
        cg/
    numerical_flux/
        rusanov/
        hllc/
        ausm/
        average/
    hybrid/
```

Desired conceptual APIs include:

```cpp
fvm(field).residual(...)
dg(field).surface_term(...)
dg(field).volume_term(...)
```

Exact syntax is not fixed.

### Execution backend

Backend-specific execution and memory behavior belongs under:

```text
backend/
    cpu/
    cuda/
    mpi/
```

The primary execution abstraction should remain small.

Conceptually:

```cpp
cfe::parallel_for(range, lambda);
```

Backend abstractions should hide host/device attributes, launch details, and implementation-specific execution policy from physics and numerics.

## 3. Proposed repository layout

```text
CMU-CFE/

    AGENTS.md
    ARCHITECTURE.md
    README.md
    ROADMAP.md
    BENCHMARKS.md
    VERIFICATION.md
    REFERENCES.md
    CHANGELOG.md
    agent_history.md

    docs/
        adr/
        theory/
        verification/
        performance/

    tasks/

    cmake/
    scripts/

    src/
        backend/
            cpu/
            cuda/
            mpi/

        math/
            scalar/
            vector/
            tensor/
            derivative/

        grid/
            structured/
                cartesian/
                nonuniform/
                amr/
            unstructured/
                connectivity/
                importer/
                    gmsh/
                    cgns/
            partition/
            boundary/
            moving_frame/

        fields/
            scalar_advection/
            burgers/
            heat/
            wave/
            compressible_navier_stokes/
            reacting_navier_stokes/
            multiphysics/

        chemistry/
            chemgen/

        numerics/
            fvm/
            fem/
            numerical_flux/
            hybrid/

        solver/
            explicit/
            implicit/
            source_split/
            moving_frame/

        diagnostics/
        ml/
        io/
        api/

    python/
        cfe.py
        builder/
        api/

    tests/
        unit/
        regression/
        convergence/
        verification/
        performance/

    benchmarks/
        math/
        memory/
        flux/
        reconstruction/
        residual/
        mpi/
        chemistry/

    tutorials/
    cases/
```

## 4. Data architecture

### Scalar types

Floating-point precision should be configurable independently from indexing.

Initial:

```cpp
using scalar = double;
```

or compile-time equivalent.

Required initial floating-point modes:

- `float`;
- `double`.

Indexing should use separate types such as:

```cpp
using local_index  = ...;
using global_index = ...;
```

### State semantics versus storage

Physics-facing code should use semantic accessors:

```cpp
auto rho = density(state);
auto mom = momentum(state);
auto E   = total_energy(state);
```

The storage representation must remain performance-oriented and may use SoA, AoS, AoSoA, or another contiguous organization.

Logical tuple-like structure must not force nested physical storage.

### Species

Species count should be compile-time known whenever the mechanism is selected at build time.

Conceptual type:

```cpp
fixed_array<scalar, n_species>
```

Large-state behavior must be benchmarked at 1, 5, 10, 20, 50, and 100 components.

## 5. Math layer

`src/math/` owns low-level mathematical operations and derivatives.

Examples:

```cpp
contract(A, B)
weight(A, B)
dot(A, B)
outer(A, B)
norm(A)
```

All operations required in GPU kernels must be host/device callable.

## 6. Grid architecture

### Initial scope

Start with:

- 1D Cartesian box grids;
- 2D Cartesian box grids;
- 3D Cartesian box grids.

Provide enough boundary/neighbor information to test FVM and DG communication.

### Structured roadmap

Later add:

- nonuniform structured grids;
- lightweight block AMR.

Initial AMR target:

- second-order Cartesian FVM;
- block-based where practical;
- 3–4 refinement levels;
- simple base grids such as `10 x 10 x 10`.

AMR is not intended to dominate the code architecture.

### Unstructured roadmap

Future support:

- Gmsh;
- CGNS;
- simplices;
- quadrilaterals;
- hexahedra;
- prisms;
- pyramids;
- hybrid connectivity.

Connectivity should use compact contiguous adjacency arrays suitable for GPU execution.

## 7. Domain decomposition

Primary hierarchy:

```text
global domain
    -> MPI partitions
        -> local cells/faces
            -> CPU threads or GPU threads
```

Halo exchange belongs outside physics calculators.

MPI packing should avoid repeated allocation.

Communication/computation overlap should be supported when measurements justify it.

## 8. FVM execution

Candidate execution forms should be benchmarked:

- cell gather;
- face loops with atomics;
- ownership-based updates;
- coloring where beneficial.

No single race-avoidance method is mandatory.

A conceptual cell formulation:

```cpp
cfe::parallel_for(cells, [=] CFE_DEVICE (auto cell)
{
    residual(cell) = ...;
});
```

## 9. DG execution

DG is an early first-class method.

Initial targets:

- tensor-product Cartesian elements;
- polynomial orders `p = 1...4`;
- collocated GLL points where appropriate;
- surface and volume kernels kept separable enough for benchmarking;
- stabilization through documented limiter/artificial-viscosity methods.

The DG representation must not prevent future unstructured element support.

## 10. DG/FVM multimethod

Hybridization should live in a separate layer:

```text
numerics/hybrid/
```

It should define:

- region/method selection;
- state transfer;
- interface coupling;
- static selection first;
- dynamic selection later.

The first demonstration should be small, likely 1D.

## 11. Boundary handling

Early finite-volume boundaries may use ghost cells.

Early finite-element boundaries use numerical flux/boundary state formulations.

Required early types:

- periodic;
- Dirichlet;
- Neumann;
- outflow/extrapolation;
- reflective/slip;
- no-slip when relevant.

Boundary metadata belongs to the grid/domain description while physics-specific interpretation belongs to fields/numerics.

Concretely, this means not every `BoundaryCondition` implementation can stay
physics-agnostic the way Phase 1's `PeriodicBoundary`/`StaticBoundary` are
(they only ever copy or write raw component values, never interpret them).
A characteristic-based (partial-specification) inflow/outflow condition --
e.g. specifying temperature at a subsonic inlet while extrapolating pressure
from the interior, the split between the two decided by which way each
characteristic points -- needs an equation of state and wave-speed
information to do that split, and needs to read the adjacent interior
cell's state, not just a stored constant or the opposite boundary. Such a
type takes a `Field` parameter and calls into it (the same shape
`NumericalFlux` already uses `Field`'s Calculators, see numerics/numerical_flux/),
while still conforming to the same swappable `fill_x/fill_y/fill_z`
ghost-cell-filling interface every other `BoundaryCondition` does -- the
grid-traversal/backend-dispatch infrastructure does not need to change,
only this one BC type becomes Field-aware. See ROADMAP.md Phase 3.

## 12. Moving frame / grid recycling

CMU-CFE requires a special capability for propagating waves.

Initial 1D behavior:

1. track a feature;
2. estimate propagation speed periodically;
3. shift/recycle the computational window;
4. remove downstream cells;
5. insert upstream cells;
6. initialize inserted state;
7. preserve indexing and communication consistency.

The implementation must distinguish:

- physical mesh motion;
- coordinate-frame transformation;
- discrete grid recycling.

## 13. Time integration

Initial:

- SSP explicit Runge-Kutta.

Source integration must support:

- directly coupled explicit source terms;
- Strang splitting.

Future:

- fully implicit;
- implicit chemistry;
- implicit Strang splitting;
- elliptic solves for incompressible systems.

Analytic derivatives and Jacobian information should be part of physics/numerical APIs before implicit systems are added.

## 14. Chemistry

ChemGen is the intended chemistry/thermodynamics generator.

Generated code should be suitable for direct compilation into case-specific executables.

Chemistry modes:

1. explicit coupled;
2. explicit Strang split;
3. implicit coupled;
4. implicit Strang split.

## 15. Multiphysics

Multiple fields should communicate through a multiphysics coupling layer.

Example:

```text
solid heat conduction
        <- interface ->
reacting compressible gas
```

The coupling layer owns:

- interface variables;
- transfer;
- conservation constraints;
- coupling frequency;
- explicit/implicit coupling policy.

## 16. Machine learning

ML should be optional and isolated.

Proposed structure:

```text
ml/
    model/
    generated/
    inference/
    coupling/
```

Support:

- inline inference for tiny generated models;
- batched standalone kernels for larger models.

Execution strategy is benchmark-driven.

## 17. Diagnostics

Target diagnostics:

- point sampler;
- line sampler;
- plane sampler;
- box sampler;
- boundary sampler;
- iso-surface extraction.

For DG, geometric search/interpolation maps should be created once where possible and reused.

Diagnostics should avoid full-domain copies when possible.

## 18. Case builder

Expected case structure:

```text
Case/
    domain/
    configuration/
```

Expected command:

```bash
python cfe.py --run Case
```

The builder should:

1. read configuration;
2. separate compile-time from runtime parameters;
3. generate a deterministic build identity;
4. reuse the executable if compile-relevant configuration has not changed;
5. compile only required capabilities where practical;
6. run the case.

Example compile identity:

```text
CUDA-double-3D-burgers50-FVM-MUSCL
```

A hash may be appended for reproducibility.

### Compile-time candidates

- backend;
- precision;
- dimension;
- field;
- species mechanism/count;
- numerical method;
- DG order where appropriate;
- derivative support.

### Runtime candidates

- CFL;
- final time;
- output interval;
- diagnostic positions;
- initial-condition values;
- boundary values where appropriate.

Configuration parsing belongs in the builder layer, not inside physics kernels.

## 19. Native file format

Do not design a custom binary file format prematurely.

Initially place a CFE API over an established portable format.

Long-term data requirements include:

- structured/unstructured topology;
- fields;
- species metadata;
- DG polynomial metadata;
- precision;
- partition information;
- checkpoints.

## 20. Architectural principle

The architecture should make the following progression possible without rewriting the core:

```text
scalar transport
-> Burgers
-> Euler
-> Navier-Stokes
-> reacting Navier-Stokes
```

while independently progressing:

```text
Cartesian FVM
-> high-order FVM
-> DG
-> DG/FVM multimethod
-> unstructured hybrid grids
```

and:

```text
CPU
-> CUDA
-> MPI + accelerator
-> additional backends
```

## 21. Memory layout: ownership, views, and block-structured grids

This section records how `Field`/`FieldView`/`CartesianGrid`/`FvmSolver`/
`Backend` relate today, where the block-structured extension (Phase 4)
and `BlockSet` sit in that picture, and the open question of whether/when
that extension should ever unify with MPI domain decomposition. It is a
reference, synthesizing decisions already recorded in ADR 0002
(state memory layout), ADR 0009 (MPI domain decomposition), and ADR 0011
(forward-step block interface) -- see those ADRs for the evidence behind
each decision; this section does not re-decide anything they already
settled, and introduces no new decision of its own.

Where this section itself should live was agreed with Amelia **before**
it was written (per CLAUDE.md's "consult before a consequential design
decision" rule), the same way ADR 0011 records that its own design
direction was agreed with Amelia before that task's implementation: she
was asked to choose among (a) a new `docs/architecture/` file, (b) this
`ARCHITECTURE.md` addition, or (c) a new ADR, and recommended (b) --
`ARCHITECTURE.md` §1 already describes itself as "current intended
architecture" that "may evolve"; this section's part on MPI unification
is deliberately an open question rather than a decision, which does not
fit an ADR's decided/accepted status; and `docs/architecture/` is not an
existing convention in this repo (`docs/adr/`, `docs/theory/`,
`docs/verification/`, `docs/performance/` are). See
`agent_history.md`'s entry for this task for the full exchange.

### Ownership versus view

Every buffer-holding type in this codebase splits into exactly one of two
roles:

- **Owning storage** allocates once, in its constructor, and never inside
  a parallel loop (AGENTS.md #10): `cfe::Field<Scalar, NComponents,
  Layout>` (one `std::vector<Scalar>`, `field/field.hpp`),
  `cfe::backend::cuda::DeviceField<...>` (one `cudaMalloc`'d buffer,
  `backend/cuda/device_field.cuh`, ADR 0002), and -- as of this task --
  `cfe::BlockSet<Scalar, NComponents, Layout>` (one `std::vector<Scalar>`
  sized for several blocks at once, `field/block_set.hpp`, see below).
- **Non-owning view**: `cfe::FieldView<Scalar, NComponents, Layout>` is
  exactly `(Scalar* data, cfe::Index n_cells)` -- trivially copyable,
  with zero assumption that it owns, or sits at the start of, its own
  allocation. This is what every piece of per-cell numerics
  (`FvmSolver::residual()`, every `BoundaryCondition::fill_x/y/z`,
  `ssp_rk2_combine_stage1/2`) actually consumes. None of that code can
  tell a `Field`-provided view from a `DeviceField`-provided view from a
  `BlockSet`-provided view -- they are indistinguishable at the call site,
  which is the single fact this whole section rests on.

`CartesianGrid<Scalar>` is neither of these: it holds no `Scalar` data at
all, only shape (cell counts, ghost depth, spacing, origin) and the
`flat_index(i,j,k)` arithmetic that turns a padded 3-index into the flat
`cell` index a `FieldView` expects. `Backend` (`CpuParallelFor`/
`CudaParallelFor`) is orthogonal again: it decides where the loop over
cells executes, never what the pointers mean.

### Single-grid versus block-structured: the same consumption pattern

Phases 1-3 (scalar advection, Burgers, single-block Euler) all have
exactly one `CartesianGrid` and one `Field` per buffer (state, stage1,
scratch). Phase 4's forward-facing-step case (ADR 0011) is CFE's first
domain that is not one rectangular box: two `CartesianGrid` blocks,
joined at their shared face by `BlockInterfaceBoundary`, each block still
using the exact same `FvmSolver`/reconstruction/numerical-flux stack a
single-block case uses. ADR 0011 Decision 1 already covers why this is
two blocks rather than one masked grid -- not repeated here.

Going from one block to several did not touch `FvmSolver`,
`BlockInterfaceBoundary`, or `ssp_rk2_combine_stage1/2`, because none of
them were ever written against "the one buffer this process owns" --
they were already written against "a `FieldView`, however it got here."
Multiple blocks are consumed by running the existing single-block
machinery once per block (`ForwardStepBlocks` owns one `Solver1`/
`Solver2` pair, `tests/verification/forward_step_setup.hpp`) and bridging
them at the shared face with a `BoundaryCondition` that reads the
neighbor block's `FieldView` directly (same process, no copy) -- the
numerics layer never had to learn that "domain" could mean more than one
box.

### Where `BlockSet` sits

Before this task, each block's three buffers (state, stage1, scratch)
were three independent `cfe::Field` allocations, so a two-block case
meant six unrelated heap allocations for what is conceptually three
buffers, each decomposed across more than one block. `cfe::BlockSet<
Scalar, NComponents, Layout>` (`field/block_set.hpp`) instead owns ONE
`std::vector<Scalar>` sized for the sum of every block's cell count, and
hands out a `FieldView` per block (`view(block)`) at the right pointer
offset into that single allocation -- block `b+1`'s region begins exactly
where block `b`'s ends.

This needed zero changes to `FvmSolver`, `BlockInterfaceBoundary`, or
`ssp_rk2_combine_stage1/2` for precisely the reason the previous
subsection gives: they already only ever consume a `FieldView`, with no
assumption about where its pointer lives or what else shares the
allocation behind it. `BlockSet` is additive infrastructure under
`field/`, at the same level as `Field`/`DeviceField` -- a second owning
type beside them, not a change to the view type or to any consumer.

Mirroring ADR 0011's own stated limits: `BlockSet` is sized from
whatever list of per-block cell counts its caller passes (today, always
two, one per forward-step block) and knows nothing about block geometry,
adjacency, or how many axes blocks share -- that bookkeeping still lives
entirely in `BlockInterfaceBoundary`/`ForwardStepBlocks::step()`, per
ADR 0011's own documented scope (exactly two blocks, one shared axis).
Extending to three-or-more blocks or multi-axis joins is still the open
item ADR 0011 already names, unchanged by this task.

### How this connects to MPI domain decomposition -- an open question, not a decision

`CartesianPartition`/`MpiHaloBoundary` (ADR 0009) and
`BlockInterfaceBoundary` (ADR 0011) are the same pattern at a different
level: a structured sub-region of a larger domain, bridged to its
neighbor(s) by a ghost-cell exchange at the shared face, behind the same
`BoundaryCondition` duck-typed interface. That resemblance is exactly why
a reader with an HPC background recognizes the shape immediately. It is
not, however, the same pattern at the same level:

- An MPI rank's local sub-domain is a genuinely separate allocation in a
  genuinely separate address space (possibly a separate machine); its
  "ghost exchange" is `MPI_Sendrecv` over the network, and nothing about
  that can become "one contiguous allocation" the way `BlockSet` makes
  two same-process blocks contiguous.
- `BlockSet`'s contiguity is a same-process, single-address-space
  property. It says nothing about, and does not require, each block
  running in its own rank.

Whether these two mechanisms should ever be unified into one
sub-domain abstraction -- e.g. a block-structured case whose blocks can
each optionally be MPI-decomposed, or an MPI rank whose local domain can
itself be multiple blocks -- is a standing open architectural question,
**not decided by this task or this document**. `BlockSet` does not
generalize past today's same-process, two-block case to answer it either
way. Two separately-scoped, not-yet-scheduled items already name pieces
of this space without committing to any unification: `agent_history.md`
(tasks 0006/0007) notes GPU-aware MPI for the existing single-grid
`CartesianPartition`/`MpiHaloBoundary` path remains out of scope; ADR
0011's own "Known limitations" names a possible future MPI-aware
multi-block boundary (making `BlockInterfaceBoundary` itself cross-rank)
as a separate, also-not-yet-built capability. Revisiting the unification
question would be motivated by a case that actually needs both
block-structured geometry AND multi-rank scale at once, or by either of
those two items landing in a shape that makes unification clearly
cheaper than maintaining two mechanisms -- neither has happened yet, so
this section only names the question, not an order or a plan for
answering it.

### Where AMR and unstructured grids fit

Not scoped or designed here. `ROADMAP.md`'s "Early 2027" lightweight
Cartesian AMR and eventual unstructured-grid support (see also this
file's #6/#20) are future directions that a block-structured-aware
memory layout may eventually need to accommodate (e.g. per-block
resolution, parent/child block relationships) -- but extending
`BlockSet` or `CartesianPartition` toward either is not needed by, and is
not part of, this task.
