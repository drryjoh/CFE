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
auto rho = density(q);
auto mom = momentum(q);
auto E   = total_energy(q);
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
