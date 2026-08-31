# CMU-CFE References

This file tracks scientific and software references used to define algorithms in CMU-CFE.

Every implemented numerical method should identify the exact formulation used. Do not rely only on method names when multiple variants exist.

## Finite volume

### PeleC

CMU-CFE finite-volume development will draw from methods used in PeleC, particularly PPM-style finite-volume reconstruction and second-order viscous discretization.

Before implementation, record the exact PeleC algorithm/source location and any deviations used by CMU-CFE.

### MUSCL

Add the exact reconstruction, limiter, and citation when selected.

### PPM

Add the exact PPM formulation, limiting procedure, and citation when selected.

### WENO

Do not label an implementation only as "8th-order WENO."

Record:

- reconstruction family;
- stencil;
- candidate polynomial order;
- nonlinear weights;
- smoothness indicators;
- formal order;
- boundary treatment;
- reference.

## Numerical fluxes

Planned families include:

- Rusanov/local Lax-Friedrichs;
- HLLC;
- AUSM-family;
- central/average fluxes.

Each implementation should cite its exact formulation.

## Discontinuous Galerkin

DG development will draw from prior work by:

- Eric Ching;
- Andrew Kercher;
- Ryan Johnson;
- collaborators.

A known relevant paper is:

Andrew D. Kercher, Andrew Corrigan, and David A. Kessler, *The Moving Discontinuous Galerkin Finite Element Method with Interface Condition Enforcement for Compressible Viscous Flows*.

Add complete bibliographic metadata before the method is implemented from this work.

For every DG stabilization method, document:

- basis/nodes;
- quadrature;
- split form if used;
- numerical flux;
- limiter;
- artificial viscosity;
- positivity treatment;
- citation.

## Chemistry

### ChemGen

ChemGen is intended to generate:

- thermodynamics;
- chemical source terms;
- derivatives;
- Jacobian information.

Document ChemGen version/interface and generated-model provenance when integrated.

## Grid formats

Potential external formats:

- Gmsh;
- CGNS.

Record exact supported versions when importers are implemented.

## Performance tools

Expected NVIDIA tools may include:

- compiler resource reports;
- Nsight Systems;
- Nsight Compute.

Record reproducible profiling commands in `docs/performance/` once CUDA infrastructure exists.

## Reference policy

When implementing a scientific method:

1. add the primary reference here or in the relevant module documentation;
2. identify the exact variant;
3. document departures from the reference;
4. add a verification test consistent with the method's claimed behavior.
