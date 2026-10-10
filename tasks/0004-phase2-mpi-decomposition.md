# Task 0004: Phase 2 — First MPI Domain-Decomposition Prototype

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `ROADMAP.md` (Phase 2)
4. `docs/adr/0004-grid-connectivity.md`
5. `docs/adr/0009-mpi-domain-decomposition.md`
6. `VERIFICATION.md`
7. `agent_history.md`

before making changes.

## Objective

Implement the first MPI domain-decomposition prototype and
communication benchmark -- 1D slab decomposition along X, blocking halo
exchange -- closing the "first MPI decomposition prototype" and
"communication benchmark" items `ROADMAP.md` Phase 2 names. Both task
0001 and task 0003 explicitly deferred MPI as "its own follow-up task";
this is that follow-up.

The existing architecture was built with this moment in mind: ADR 0004
and `grid/ghost/ghost_fill.hpp`'s own header comment already name the
`fill_x/fill_y/fill_z` boundary-condition seam as where an MPI
halo-exchange provider plugs in, with zero change needed to `FvmSolver`,
`fill_ghost_cells`, `CartesianGrid`, or `ssp_rk2_step`. This task follows
that seam literally.

This task is deliberately a narrower slice of `ROADMAP.md`'s full Phase 2
("Large-state Burgers and communication"), not all of it.

**Do not implement in this task:**

- 2D/3D block decomposition (diagonal/corner-neighbor exchange) --
  Y/Z-axis slab decomposition or full block decomposition is a separate,
  structurally similar follow-up task;
- non-blocking (`Isend`/`Irecv`) communication/computation overlap --
  "eventually," per AGENTS.md #16, not required for a first prototype;
- GPU-aware MPI / combining `MpiHaloBoundary` with `CudaParallelFor` --
  this type is CPU-only for v1;
- multi-rank `BurgersField` (or any state-dependent-CFL field). Its
  `wave_speed()` is `max|u|` computed over the whole initial condition --
  if computed per-rank over only a local slice, different ranks could
  pick different `dt` for the same timestep, a real correctness break.
  Fixing this needs an `MPI_Allreduce(MAX)` across ranks, not yet
  implemented (see ADR 0009's Consequences section). This task's
  correctness oracle uses `ScalarAdvectionField` instead, whose
  `wave_speed()` is a fixed constant needing no synchronization;
- a non-periodic domain's two true physical-boundary ranks getting a
  real (non-zero, non-stale) ghost value from `MpiHaloBoundary` alone --
  it does not compose with a second boundary condition for that one true
  edge; only the fully periodic case is exercised/tested here;
- a DG storage/communication prototype, state sizes through 100, or a
  memory-layout study (the rest of `ROADMAP.md` Phase 2 -- each its own
  follow-up task);
- any multi-node Bridges-2 scaling beyond the 1/2/4/8 single-node sweep;
- unstructured grids, AMR, chemistry, compressible Euler.

## Required functionality

- `src/cfe/grid/partition/slab_partition.hpp`: `SlabPartition` +
  `make_slab_partition(global_nx, rank, size, periodic)` -- splits a
  global cell count across ranks (remainder-to-the-first-ranks rule),
  computing each rank's local cell count, global offset, and left/right
  neighbor ranks (periodic wraparound, or `kNoNeighbor` for a true
  physical boundary). Deliberately has no MPI dependency.
- `src/cfe/backend/mpi/mpi_environment.hpp` -- RAII `MPI_Init`/
  `MPI_Finalize` guard + `rank()`/`size()` helpers.
- `src/cfe/backend/mpi/mpi_datatype.hpp` -- `Scalar` -> `MPI_Datatype`
  mapping (`float`/`double`).
- `src/cfe/grid/boundary/mpi_halo_boundary.hpp`: `MpiHaloBoundary
  <Scalar,N>` -- same duck-typed `fill_x`/`fill_y`/`fill_z` shape every
  boundary type implements (`fill_y`/`fill_z` throw if ever actually
  called -- this type is X-axis-only by design, but must still exist and
  type-check for `fill_ghost_cells`'s runtime `switch` to compile). Packs
  the X ghost slab into a flat buffer (general strided loop, no
  contiguous-1D fast path), exchanges via two blocking `MPI_Sendrecv`
  calls (one per direction), unpacks into the matching ghost range.
  Buffers sized once per grid shape, never shrunk.
- `CFE_ENABLE_MPI` CMake option, auto-detected via `find_package(MPI
  COMPONENTS CXX)`, mirroring the existing `CFE_ENABLE_CUDA` pattern
  exactly (opt-in, `FATAL_ERROR` if explicitly requested but not found,
  else silent OFF).

## Architecture constraints

- Zero changes to `FvmSolver`, `fill_ghost_cells`, `CartesianGrid`, or
  `ssp_rk2_step` -- this is the whole point of ADR 0004's seam.
- Halo exchange kept out of physics Calculators (AGENTS.md #16).
- No allocation inside the per-timestep hot path -- `MpiHaloBoundary`'s
  send/recv buffers are sized once, on first use per grid shape.
- `MpiHaloBoundary` lives in `grid/boundary/` (not `backend/mpi/`),
  consistent with every other boundary type's precedent of calling into
  a backend header without owning that backend's logic itself -- see
  ADR 0009 for the full reasoning.

## Tests

- `tests/unit/test_slab_partition.cpp` -- pure index arithmetic, runs
  in the ordinary single-process `cfe_unit_tests` binary, no MPI
  toolchain needed.
- `tests/mpi/test_mpi_halo_exchange.cpp` -- a separate executable
  (needs `mpirun`/`mpiexec`), registered as `ctest` entries at `np=2`
  and `np=4` via CMake's `MPIEXEC_*` variables. Decomposes
  `ScalarAdvectionField` over a periodic domain and checks the result is
  bit-identical (tolerance 0) to an independently-computed single-rank
  reference run of the same global problem -- not just "close" to the
  analytic exact solution. Only built/registered when `CFE_ENABLE_MPI`
  is ON.

## Benchmarks

- `benchmarks/mpi/bench_mpi_halo_exchange.cpp` -- isolates just the
  halo-exchange step (not a full timestep); three sweeps (vs. rank
  count 1/2/4/8, vs. problem size using a 2D/3D domain decomposed only
  along X, vs. ghost depth 2/4/8), CSV output.

## Architecture decisions

`docs/adr/0009-mpi-domain-decomposition.md` -- decomposition granularity
(1D slab/X), exchange mechanism (blocking `Sendrecv`), and
`MpiHaloBoundary`'s location (`grid/boundary/`).

## Completion report

Files added/changed; test results (CPU-only build unaffected; local
verification via Homebrew OpenMPI on the dev machine at np=1,2,3,4,5,8,
including a deliberate-bug sanity check that the oracle actually has
teeth; explicitly flag whether PSC Bridges-2 verification has happened
yet, same wording pattern used for CUDA tests before first Bridges-2
access); benchmark results/CSV (local sanity run vs. authoritative
Bridges-2 run); the Burgers CFL/Allreduce risk, documented not silently
dropped; recommended next task (2D/3D block decomposition, non-blocking
overlap, or the DG communication prototype).
