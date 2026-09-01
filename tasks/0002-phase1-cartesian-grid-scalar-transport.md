# Task 0002: Phase 1 — Cartesian Grid and Scalar Transport

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `ROADMAP.md` (Phase 1)
4. `docs/adr/0001-execution-backend.md`
5. `docs/adr/0002-state-memory-layout.md`
6. `docs/adr/0004-grid-connectivity.md`
7. `docs/adr/0006-language-standard.md`
8. `VERIFICATION.md`
9. `agent_history.md`

before making changes.

## Objective

Add the smallest Cartesian grid and explicit scalar-transport capability
needed to exercise the Phase 0 execution/storage foundation on an actual
(if simple) PDE, with formal convergence evidence -- while keeping the
neighbor-communication and interface-flux abstractions generic enough that
Phase 2's large-state work and eventual DG development (AGENTS.md #19: DG
and FVM must remain hybridizable, with hybridization logic living outside
both) do not require redesigning this layer.

**Do not implement in this task:**

- Burgers or any nonlinear equation (Phase 2);
- compressible Euler or Navier-Stokes (Phase 3/Phase 5);
- diffusive/viscous fluxes of any kind (Phase 5 -- this task is
  convective/advective only);
- DG storage or execution (Phase 2 prototype / Phase 3);
- MPI or domain decomposition (Phase 2);
- unstructured grids or AMR;
- chemistry;
- more boundary condition types than static and periodic (Dirichlet,
  Neumann, outflow, reflective, no-slip are listed in AGENTS.md #17 as
  eventually required, but are not needed for scalar advection and are out
  of scope here).

## Required functionality

1. 1D/2D/3D Cartesian grid representation. Structured, uniform spacing
   initially. Per AGENTS.md #15: do not over-generalize this into
   unstructured connectivity, but do not hard-code interfaces around the
   assumption that all connectivity is always reconstructible from
   `(i,j,k)` either.
2. Ghost-cell infrastructure for FVM neighbor access. **Must be generic
   over `NComponents`** and must sit on top of Phase 0's
   `Field<Scalar, NComponents, Layout>` -- do not build a parallel
   scalar-only storage path. This task only exercises N=1, but the grid/
   ghost-cell/BC layer itself must not assume N=1.
3. Boundary conditions: static (fixed value) and periodic only.
4. A method-agnostic "interface value" contract: something that can
   produce the state at a cell/element face and combine two sides' values
   into a numerical flux. Per AGENTS.md #19, hybridization logic belongs
   outside individual DG/FVM implementations -- this task only needs an
   FVM-side implementation (1-ring neighbor, central-difference-style
   reconstruction), but the shape of this interface must not bake in
   "FVM cell-average" as the only thing that could ever sit on either side
   of it. Do not implement a DG side now; just don't foreclose it.
5. Linear scalar advection equation, single transported component.
6. SSP-RK time integration. Pick SSP-RK2 or SSP-RK3 and document which and
   why (e.g. stability region vs. cost).
7. CPU execution, reusing Phase 0's `cfe::parallel_for` (serial and
   threaded backends).
8. CUDA execution. Orchard GPU access is expected to be available for this
   task -- if it is not yet by the time this task runs, document that
   explicitly and proceed CPU-only, matching Phase 0's precedent (do not
   invent numbers; do not block the rest of the task on it).
9. Formal convergence test: verify the expected 2nd-order accuracy via a
   grid-refinement study (e.g. halve `dx` repeatedly, confirm error drops
   ~4x each time). "It ran and looked reasonable" does not satisfy this.
10. A conservation check appropriate to linear advection (e.g. total
    transported quantity is conserved over a periodic domain to within
    floating-point tolerance).

## Architecture constraints

- Ghost-cell/BC/grid code must be generic over component count (AGENTS.md
  #8/#10/#12, the large-state requirement) even though this task only
  exercises a scalar -- Phase 2 needs this layer to already support N up
  to 100 without modification.
- The interface-value/flux abstraction must be shaped so that a future DG
  element (multiple states/nodes per element, not a single cell average)
  could plausibly implement the same contract without this task's code
  needing to be redesigned. This is a constraint on interface shape, not
  an instruction to build DG now.
- Halo/ghost-cell exchange must stay separated from the physics calculator
  itself (AGENTS.md #14, #16) -- this also anticipates Phase 2's MPI halo
  exchange reusing the same seam.
- Moving-frame/grid-recycling behavior is a separate future capability and
  must not be conflated with these static/periodic boundary conditions
  (AGENTS.md #17).
- Do not allocate inside parallel loops.
- Do not introduce external portability/mesh frameworks.

## Tests

At minimum:

- grid indexing/connectivity is correct in 1D/2D/3D, including that ghost
  cells resolve to the correct neighbor;
- static and periodic boundary conditions produce the correct ghost-cell
  values at domain edges;
- the interface-value/numerical-flux calculation matches a hand-computed
  reference for a simple case;
- SSP-RK time integration is correct (e.g. verified against a known
  analytic ODE solution, or as part of the PDE convergence study below);
- **the scalar advection solver demonstrates the expected 2nd-order
  convergence rate under grid refinement** -- this is the Phase 1
  acceptance bar per `ROADMAP.md`, not optional;
- conservation holds over a periodic domain;
- CPU serial and threaded backends agree;
- CUDA backend correctness when hardware is available; otherwise
  documented as unverified, matching ADR 0001's existing precedent.

## Benchmarks

- Runtime / cell-updates-per-second for the advection kernel, swept across
  a few grid resolutions (this task's new relevant scaling axis, since
  component count stays at 1) on CPU, and on CUDA if Orchard access has
  landed.
- If CUDA hardware is unavailable when this task starts, document that
  fact plainly rather than blocking the rest of the task -- update this
  section once GPU access lands, following the same pattern as
  `docs/performance/0001-phase0-results.md`.

## Architecture decisions

- Update ADR 0004 (grid connectivity) with evidence from the actual
  Cartesian implementation -- it is currently an unevidenced "Proposed"
  placeholder.
- Open a new ADR (or extend ADR 0004) for the ghost-cell/boundary-condition
  storage design if a nontrivial choice is made (e.g. how ghost cells are
  represented relative to `Field`).
- Record explicitly, in whichever ADR covers the interface-flux
  abstraction, how and why it was kept method-agnostic per AGENTS.md #19,
  so whoever picks up DG later has the reasoning, not just the code.

## Presentation

Per AGENTS.md #26, add `presentations/0002-phase1-cartesian-grid-scalar-transport.md`
for the same non-CS student audience as `presentations/0001-...`. Cover, at
minimum: what a "ghost cell" is and why it exists (in plain terms -- a
fake extra cell at the edge that lets every real cell run the exact same
kernel, boundary cells included, without special-casing code); what
"convergence order" means and why halving the grid spacing should quarter
the error for a 2nd-order method; and what the grid-refinement study
actually showed.

## Completion report

At the end report:

1. architecture created;
2. files added/changed;
3. tests performed;
4. convergence study results (observed order vs. expected order);
5. benchmark results (CPU, and CUDA if available);
6. CUDA register/spill observations if hardware was available;
7. unresolved design questions;
8. ADR changes;
9. recommended next task.

Append the same work to `agent_history.md`.
