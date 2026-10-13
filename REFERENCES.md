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

### Rusanov/local Lax-Friedrichs (implemented, Phase 2/3)

`src/cfe/numerics/numerical_flux/rusanov.hpp`. Rusanov, "The calculation
of the interaction of non-steady shock waves with obstacles," 1961; see
also Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
3rd ed., Sec. 10.5, or LeVeque, *Finite Volume Methods for Hyperbolic
Problems*, Sec. 10.3-10.4.

### HLLC (implemented, task 0011)

`src/cfe/numerics/numerical_flux/hllc.hpp`. Toro, Spruce, Speares,
"Restoration of the contact surface in the HLL Riemann solver," Shock
Waves 4, 1994; reproduced as Toro, *Riemann Solvers and Numerical
Methods for Fluid Dynamics*, 3rd ed., Ch. 10 (eq. 10.37-10.43). Wave-speed
estimate: Toro's ANRS (Sec. 9.5.2, adaptive PVRS/TRRS/TSRS) used as a
seed only, refined by 5 fixed Newton-Raphson iterations (Toro
eq. 4.5-4.8) toward the exact two-shock/two-rarefaction `p*` -- see
`docs/adr/0012-hllc-numerical-flux.md` for why the non-iterative
estimate alone was insufficient (a silent, severe robustness bug at a
strong Mach-3 wall reflection) and for a second, separate robustness fix
(density/pressure floors guarding against a near-vacuum
componentwise-MUSCL-reconstruction artifact at a strong 2D multi-wave
interaction).

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
