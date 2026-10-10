# Task 0006: Phase 2 — MPI-Safe Burgers via Collective CFL (`MPI_Allreduce(MAX)`)

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `ROADMAP.md` (Phase 2)
4. `docs/adr/0009-mpi-domain-decomposition.md` (including this task's
   own amendment)
5. `VERIFICATION.md`
6. `agent_history.md`

before making changes.

## Objective

Close the one correctness gap task 0004/0005 both named and
deliberately deferred: `BurgersField`'s state-dependent CFL sizing
(`dt` from `max|u|` over the whole initial condition) was not safe to
decompose across MPI ranks, because each rank computed that maximum
over only its own local slice. Add the missing `MPI_Allreduce(MAX)`
collective and prove, with a real correctness test (not just a
compile-time check), that Burgers now decomposes correctly.

**Do not implement in this task:**

- non-blocking communication/computation overlap -- still "eventually,"
  unchanged from tasks 0004/0005;
- 2D/3D Burgers decomposition -- this task's correctness test is 1D
  (periodic sinusoidal steepening); extending to 2D/3D Burgers is
  possible with the same fix but not re-verified here;
- non-periodic domains / a composed true-physical-edge boundary --
  unchanged limitation from tasks 0004/0005;
- GPU-aware MPI;
- a strong-scaling tutorial for Burgers specifically -- the existing
  `tutorials/mpi_scalar_advection_3d_strong_scaling/` already
  demonstrates the scaling story; this task is about correctness, not
  a second scaling demonstration.

## Required functionality

- `src/cfe/backend/mpi/mpi_reduce.hpp`: `cfe::backend::mpi::
  allreduce_max<Scalar>(Scalar local_value, MPI_Comm comm =
  MPI_COMM_WORLD)` -- a thin wrapper over `MPI_Allreduce(..., MPI_MAX,
  ...)`, reusing `mpi_datatype_for<Scalar>()`.
- `tests/mpi/test_mpi_burgers_steepening.cpp`: decomposes the periodic
  sinusoidal-steepening-into-shock case (same IC/parameters as
  `tutorials/burgers_1d_shock_and_steepening/`'s Case B) run PAST the
  analytic breaking time, so a genuine shock exists. Deliberately a
  non-uniform-amplitude IC (`sin(2*pi*x)` peaks at `x=0.25`, troughs at
  `x=0.75`) chosen so that at 4 ranks, two of them see a local `max|u|`
  ~33% smaller than the true global one -- large enough to matter, not
  a rounding-noise case.

## Architecture constraints

- Zero changes to `FvmSolver`, `fill_ghost_cells`, `MpiHaloBoundary`,
  `CartesianPartition`, or `ssp_rk2_step` -- the fix is entirely at the
  call site that computes `dt`, the same place every existing
  single-rank Burgers test/tutorial/benchmark already computes
  `max_abs_u0`.

## Tests

`tests/mpi/test_mpi_burgers_steepening.cpp`, registered at `np=2` and
`np=4`. Same bit-identical-vs-independently-computed-reference oracle
as the existing MPI correctness tests. Verified to have teeth via a
deliberate sabotage: skipping the `allreduce_max` call (using the
rank-local maximum directly) was tried and reverted -- it did not just
produce a slightly-wrong-but-passing result, it caused ranks to
compute **different step counts** for the same nominal final time,
which desynchronized their blocking `Sendrecv` calls and **hung** the
job (one rank's loop exits while its neighbor is still blocked waiting
for a `Sendrecv` partner that will never call again) -- a more severe
failure mode than silently wrong output, underscoring why this fix is
not optional for any state-dependent-CFL field.

## Benchmarks

None new.

## Architecture decisions

Amendment to `docs/adr/0009-mpi-domain-decomposition.md` (not a new ADR
number -- closes a gap that ADR already named).

## Completion report

At the end report:

1. files added/changed;
2. test results (CPU-only unaffected; local + Bridges-2, explicitly
   distinguishing which);
3. the sabotage-test finding (hang, not just wrong output) and why that
   matters;
4. remaining known limitations (2D/3D Burgers decomposition not
   re-verified; non-periodic composition still unsupported; GPU-aware
   MPI still out of scope);
5. recommended next task.
