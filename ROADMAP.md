# CMU-CFE Development Roadmap

This roadmap describes intended milestones, not immutable promises. Scientific correctness and measured performance take precedence over calendar dates.

Current planning baseline: **August 30, 2026**.

## Phase 0 — Repository and execution foundation

### Target: September 6, 2026

Deliver:

- repository structure;
- CMake build;
- `AGENTS.md`;
- ADR workflow;
- test framework;
- benchmark framework;
- scalar/index abstractions;
- host/device macros;
- CPU execution abstraction;
- CUDA execution abstraction;
- fixed-size math/state primitives;
- contiguous field storage prototype.

Acceptance:

- CPU build and tests pass;
- CUDA build and tests pass when CUDA is available;
- state sizes 1, 5, 10, 20, 50, and 100 benchmarked;
- register/spill inspection procedure documented.

## Phase 1 — Cartesian grid and scalar transport

### Target: September 20, 2026

Deliver:

- 1D/2D/3D Cartesian grids;
- static boundaries;
- periodic boundaries;
- ghost-cell infrastructure for FVM;
- scalar advection;
- SSP-RK integration;
- CPU and CUDA execution;
- formal convergence tests.

Acceptance:

- known-order convergence;
- conservation checks;
- CPU/GPU result agreement within defined tolerance;
- no unexplained performance regressions.

## Phase 2 — Large-state Burgers and communication

### Target: September 30, 2026

Deliver:

- nonlinear Burgers equation;
- 3D Cartesian execution;
- independent transported components;
- state sizes through 100;
- memory-layout study;
- first MPI decomposition prototype;
- communication benchmark;
- basic DG communication/storage prototype.

Acceptance:

- correctness and convergence;
- GPU profiling for register pressure/spilling;
- throughput scaling versus state size;
- documented memory-layout decision or open ADR.

## Phase 3 — Shocked compressible Euler

### Target: October 15, 2026

Deliver:

- compressible Euler field/calculator;
- MUSCL;
- numerical flux infrastructure;
- Rusanov;
- HLLC and/or AUSM-family implementation;
- Sod shock tube;
- Shu-Osher problem;
- DG `p=1` and `p=2` prototype where feasible.

Acceptance:

- canonical shock benchmarks;
- conservation checks;
- positivity/stability behavior documented;
- CPU/GPU agreement.

## Phase 4 — 3D shocked flow and moving frame

### Target: October 31, 2026

Deliver:

- 3D compressible Euler;
- forward-facing ramp/step target case;
- moving-frame/grid-recycling prototype;
- 1D propagating shock demonstration;
- feature tracking and periodic propagation-speed estimate.

Acceptance:

- recycling preserves solution consistency;
- no cumulative indexing corruption;
- performance cost quantified.

## Phase 5 — Compressible Navier-Stokes

### Target: November 15, 2026

Deliver:

- second-order viscous fluxes;
- PPM;
- flat-plate infrastructure;
- Taylor-Green vortex infrastructure;
- DG `p=1...4` progression where feasible.

Acceptance:

- viscous convergence study;
- canonical flat-plate comparison;
- TGV conservation/dissipation diagnostics.

## Phase 6 — Verified 3D compressible solver

### Target: November 30, 2026

Deliver:

- verified 3D Cartesian compressible Navier-Stokes;
- single/double precision studies;
- CPU/GPU performance studies;
- 10–15 tutorials total;
- documented benchmark baselines.

Acceptance:

- canonical cases repeatably reproduce expected behavior;
- performance baseline is recorded;
- no known architectural blocker to chemistry.

## Phase 7 — Chemistry integration

### Target: December 15, 2026

Deliver:

- ChemGen integration;
- thermodynamics;
- reaction-source kernels;
- derivatives/Jacobian interfaces;
- 0D reactor/unit verification;
- 1D flame;
- 1D detonation;
- explicit chemistry;
- Strang-split chemistry.

Acceptance:

- chemistry verified independently from transport;
- species conservation tests;
- thermodynamic consistency tests;
- performance versus species count measured;
- register/spill behavior measured.

## Phase 8 — Reacting flow and first multimethod demo

### Target: December 31, 2026

Deliver:

- 2D reacting-flow demonstration;
- 2D detonation target;
- moving detonation frame;
- first static DG/FVM multimethod demonstration, even if 1D;
- reacting flat-plate foundation.

Acceptance:

- stable verified demonstration;
- interface conservation characterized;
- limitations documented.

## Early 2027 targets

- 3D detonation;
- 2D/3D reacting flat plate;
- solid/gas reacting interface;
- splitter plate;
- dynamic DG/FVM hybridization;
- unstructured grid infrastructure;
- Gmsh/CGNS import;
- unstructured DG;
- lightweight Cartesian AMR;
- implicit source integration;
- broader MPI scaling;
- ML inference experiments.

## Scope guardrails

Do not allow these to delay the September execution foundation:

- custom file formats;
- generalized AMR;
- generalized unstructured infrastructure;
- implicit solvers;
- complex ML frameworks;
- GUI development;
- broad runtime plugin systems.

The September objective is to prove that the core execution and storage model performs for both small and large states.
