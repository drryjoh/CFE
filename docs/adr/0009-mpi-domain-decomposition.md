# ADR 0009: MPI domain decomposition (first prototype)

**Status:** Accepted
**Date:** 2026-10-09 (updated 2026-10-10: PSC Bridges-2 verification)

## Context

AGENTS.md #16 names MPI as "the primary distributed-memory decomposition
mechanism" and requires halo exchange to be separated from physical
calculations. `docs/adr/README.md` has reserved "MPI ownership/
communication strategy" as a topic since Phase 0, with nothing written
until now. ADR 0004 (grid connectivity) already identified
`grid/ghost/ghost_fill.hpp`'s `fill_x`/`fill_y`/`fill_z` boundary-type
seam as the intended MPI plug-in point but deferred the actual decision.
`ROADMAP.md`'s Phase 2 names "first MPI decomposition prototype" and
"communication benchmark" as required deliverables; both task 0001 and
task 0003 explicitly excluded MPI from their own scope, deferring it to
"its own follow-up task" -- this is that follow-up (task 0004).

Three genuinely separable decisions need recording: how much of the
domain to decompose in this first slice, what communication mechanism
to use, and where the new boundary type should live in the source tree.

## Options considered

### Decomposition granularity

**Option A: 1D slab decomposition along X only.**

Advantages:

- `CartesianGrid::flat_index(i,j,k) = i + j*padded_nx() + k*padded_nx()*padded_ny()`
  makes X the unit-stride axis -- the structurally hardest case for halo
  packing (strided gather/scatter once `ny`/`nz` > 1, not a contiguous
  end-of-array run). Proving the exchange correct here first, then
  extending to other axes, is the harder-first order of operations.
- Matches "first MPI decomposition prototype" exactly -- a minimal next
  slice, not full production decomposition.
- `FvmSolver`'s `BoundaryX`/`BoundaryY`/`BoundaryZ` are three
  independent template parameters (`fvm_solver.hpp`), so the new type
  plugs in as `BoundaryX` alone with zero change to `FvmSolver`,
  `fill_ghost_cells`, or `CartesianGrid` -- `BoundaryY`/`BoundaryZ` stay
  `PeriodicBoundary` for a 2D/3D problem decomposed only along X.

Disadvantages:

- Does not exercise corner/edge-neighbor (diagonal) communication a
  full 2D/3D block decomposition would eventually need.

### Option B: full 2D/3D block decomposition

Advantages:

- More representative of eventual production-scale decomposition.

Disadvantages:

- Several times the design/test surface (diagonal-neighbor bookkeeping,
  corner-cell pack/unpack) for a "first prototype" -- explicitly
  deferred by the roadmap's own phrasing.

## Evidence

- `src/cfe/grid/partition/slab_partition.hpp` + `tests/unit/
  test_slab_partition.cpp`: 6 tests covering even/uneven splits,
  periodic/non-periodic neighbor assignment, and the single-rank edge
  case. Pure index arithmetic, no MPI dependency -- passes today
  (96/96 total unit tests, zero regressions in the existing suite).
- `src/cfe/grid/boundary/mpi_halo_boundary.hpp` + `tests/mpi/
  test_mpi_halo_exchange.cpp`: verified locally (OpenMPI via Homebrew,
  this being a dev machine, not the target cluster) at 1, 2, 3, 4, 5,
  and 8 ranks -- bit-identical (tolerance 0) to an independently
  computed single-rank reference run of the same global problem at
  every rank count, including non-power-of-two counts (3, 5) that
  exercise the remainder-cell split under real communication, not just
  the pure-math unit test. The oracle was confirmed to have teeth: a
  deliberate one-line indexing bug (off-by-one in the packed high-face
  real-cell index) was introduced, rebuilt, and immediately caught
  (160/160 cells mismatched at np=4), then reverted and re-verified
  passing.
- `benchmarks/mpi/bench_mpi_halo_exchange.cpp`: ran locally at 1, 2, 4,
  8 ranks (`benchmarks/results/phase2_mpi_halo_exchange_apple_m5.csv`)
  -- confirms the benchmark harness itself produces well-formed,
  monotonic-with-message-size timing data. This local run is a
  sanity check only, not the authoritative scaling result (an
  oversubscribed laptop is not representative hardware).
- **2026-10-10: verified on real PSC Bridges-2 hardware** (RM-shared
  partition, job `49005352`, node `r193`, 8 CPUs, GCC 13.3.1 +
  OpenMPI 5.0.8 -- see "Bridges-2 execution notes" below for the
  module/launch details). CPU-only (`CFE_ENABLE_MPI=OFF`) build
  completely unaffected. With `CFE_ENABLE_MPI=ON`: `test_slab_partition`
  passes as part of the full `cfe_unit_tests` suite; `test_mpi_
  halo_exchange` passes bit-identically via the registered `ctest`
  entries at `np=2`/`np=4`, and via a manual sweep at `np=1,2,4,8,16`
  (the last with `--oversubscribe`, since this allocation only had 8
  real cores). `bench_mpi_halo_exchange` ran its full rank-count/size/
  ghost-depth sweep at 1/2/4/8 real (non-oversubscribed) ranks --
  `benchmarks/results/phase2_mpi_halo_exchange_bridges2_rm_shared_cpu.csv`.
  This is a CPU-only RM-shared node, not a V100 GPU node specifically
  (that pool was fully allocated cluster-wide at the time -- not a
  policy limit; `gpuinteract` QOS allows up to 8 GPUs/user) -- entirely
  appropriate since this task's code has zero CUDA dependency.

### Bridges-2 execution notes (new, not previously documented)

Two environment-specific gotchas surfaced only on the cluster, neither
a code defect:

1. **Nested `srun` job steps restrict visible cores.** Launching
   `ctest`/`mpirun` via `srun --jobid=<id> [--cpus-per-task=N] bash -c
   "..."` creates a *job step* with its own sub-cgroup that, on this
   cluster, exposed only 1 CPU to `mpirun`'s slot detection regardless
   of `--cpus-per-task` -- `mpirun` then refused to launch 2+ ranks
   ("not enough slots"), and explicit `--host <node>:N` hit a
   mapping/binding failure against that same restricted cpuset. Fix:
   connect directly to the allocated compute node (`ssh <node>`, no
   nested `srun` step) before invoking `ctest`/`mpirun` -- this exposes
   the job's full core allocation correctly. (A bare `srun -n N
   ./binary` without a job step wrapper was also tried first and does
   NOT work as an mpirun substitute here: this OpenMPI build's PMIx
   plugin fails to load under this cluster's Slurm PMIx version, and
   PMI2 falls back to launching `N` independent single-rank
   "singletons" instead of one coordinated `N`-rank job -- `mpirun` is
   the correct launcher on this cluster, not raw `srun`.)
2. **RM-shared is not an interconnect-optimized fabric.** The measured
   halo-exchange bandwidth (~0.7-1.4 GB/s) is lower than a dedicated
   HPC interconnect would give -- expected and fine for a CPU-only
   shared node whose primary job is general compute, not communication
   benchmarking; not a code performance regression to chase.

### Exchange mechanism

**Option A: blocking `MPI_Sendrecv`, two calls per `fill_x`.**

Advantages:

- Simplest correct implementation. Each call's send and recv happen
  concurrently from the calling rank's point of view, so a ring of
  these calls cannot deadlock at any rank count -- no odd/even-rank
  ordering trick needed.
- Matches AGENTS.md #16's own framing: comm/compute overlap "should
  eventually permit" -- not "must," for a first prototype.

Disadvantages:

- No hiding of communication latency behind computation.

**Option B: non-blocking `Isend`/`Irecv` with overlap.**

Advantages:

- The eventual real goal once communication is shown to be a
  bottleneck.

Disadvantages:

- Real added complexity (request lifetime management) not justified
  without first measuring that latency is actually a problem
  (AGENTS.md #2: build for measured, not speculative, needs).

### `MpiHaloBoundary`'s location

**Option A: `grid/boundary/` (alongside `PeriodicBoundary`/`StaticBoundary`).**

Advantages:

- Consistent with every existing boundary type's own precedent of
  calling into a backend header (`Backend::run`) from `grid/boundary/`
  without owning that backend's logic itself. Discoverable at the one
  call site (`fill_ghost_cells`) that already dispatches on boundary
  type.
- The raw MPI plumbing (communicator bootstrap, datatype mapping) still
  lives in `backend/mpi/` -- only the pack/unpack + `Sendrecv` call
  itself lives in `grid/boundary/`, mirroring how CUDA's reusable
  primitives (`backend/cuda/device_field.cuh`) are called into by
  non-`backend/` code elsewhere.

Disadvantages:

- Couples one file in `grid/boundary/` to `<mpi.h>` (mitigated: gated
  behind `#ifndef CFE_ENABLE_MPI` / `#error`, same confinement
  AGENTS.md #9 already requires of CUDA headers).

**Option B: `backend/mpi/`.**

Advantages:

- Keeps all MPI-aware code under one directory tree.

Disadvantages:

- Breaks the established "a boundary type lives in `grid/boundary/`"
  pattern for no architectural reason; `ARCHITECTURE.md`'s `backend/mpi/`
  is framed as general backend-level plumbing (parallel to
  `backend/cpu`/`backend/cuda`), not as "where every MPI-touching type
  must live."

## Decision

1D slab decomposition along X (Option A), blocking `MPI_Sendrecv`
(Option A), `MpiHaloBoundary` in `grid/boundary/` (Option A) -- backed
by `backend/mpi/mpi_environment.hpp` (process bootstrap) and
`backend/mpi/mpi_datatype.hpp` (Scalar -> MPI_Datatype mapping) as the
reusable plumbing it calls into.

Status is **Accepted**: verified first locally (this dev machine, via
Homebrew OpenMPI) and then for real on PSC Bridges-2 (2026-10-10,
RM-shared CPU node -- see Evidence above), both passing in full.

## Consequences

**Easier**: a 2D/3D problem decomposed only along X works today with
zero changes to `FvmSolver`/`ssp_rk2_step`/`CartesianGrid` -- exactly
the "swappable neighbor provider" seam ADR 0004 anticipated paid off as
designed. `SlabPartition`'s MPI-independence means its correctness is
fully covered by ordinary single-process unit tests, with the
MPI-dependent surface area kept to exactly one new boundary type plus
two small backend headers.

**Harder**: a non-periodic domain's two true physical-boundary ranks
get a genuine no-op `Sendrecv` on their outward side from
`MpiHaloBoundary` alone -- this type does not compose with a second
boundary condition to give that one true edge a real value. Only the
fully periodic case is exercised/tested in this task; a future non-periodic
MPI use case needs that composition designed and built, not assumed to
fall out for free.

**Future constraint**: `BurgersField`'s state-dependent CFL sizing
(`dt` from `max|u|` over the whole initial condition) is NOT
MPI-decomposition-safe as written -- if computed per-rank over only a
local slice of a non-uniform IC, different ranks could pick different
`dt` for the same timestep, a real correctness break. This needs an
`MPI_Allreduce(MAX)` across ranks, not yet implemented. This task's
correctness oracle deliberately uses `ScalarAdvectionField` instead
(its `wave_speed()` is a fixed constant, needing no synchronization) --
see `tasks/0004-phase2-mpi-decomposition.md`'s "Do not implement" list.

## Revisit criteria

- Revisit decomposition granularity when a 2D/3D block-decomposition
  task begins (diagonal/corner-neighbor exchange).
- Revisit blocking-vs-overlap when a benchmark run on real target
  hardware (Bridges-2, multi-node) actually shows communication-bound
  (not latency-floor-bound at small scale) behavior at the rank counts
  that matter for production runs.
- Revisit `MpiHaloBoundary`'s location if a second MPI-aware boundary
  type is ever added and the `grid/boundary/` coupling to `<mpi.h>`
  becomes a real build-time or dependency-isolation problem (not
  observed so far).
