# Task 0016: Checkpoint/restart for the forward-step resolution sweep

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md` (especially #19 "Native file format" and #21
   "Memory layout: ownership, views, and block-structured grids")
3. `agent_history.md`
4. `docs/adr/0011-forward-step-block-interface.md`
5. `tests/verification/forward_step_setup.hpp`
6. `src/cfe/io/vtk_writer.hpp`
7. `src/cfe/field/block_set.hpp`

before making changes.

## Motivation

Verifying a change against `tutorials/euler_2d_forward_step`'s full
resolution sweep (7 resolutions x 2 fluxes) requires re-running every
case from `t=0` every single time, including the most expensive case
(`res288_hllc`, roughly 12,000 steps), even when the change being
verified (e.g. a pure rename) provably cannot affect physics. This makes
routine re-verification slow and prone to interruption.

The fix: add checkpoint/restart capability, so the expensive tail end of
a long run can be re-verified by loading a checkpoint taken some steps
before the target and only simulating the remaining steps, instead of
the whole run from scratch.

**Scoped down deliberately, per explicit user instruction:** this task
builds checkpoint/restart as its own generic, CPU-only (serial/threaded
backend) capability. It does **not** wire MPI into Euler or
`BlockInterfaceBoundary`/forward-step -- that is separate, already-named
future work ("Task B: MPI-aware multi-block interface boundary," still
unstarted). MPI today only connects to `BurgersField`/
`ScalarAdvectionField` via `CartesianPartition`+`MpiHaloBoundary<Scalar,
N>`; this task does not change that.

## What this task builds

1. **A restart-file format and read/write functions**
   (`src/cfe/io/restart_file.hpp`): a versioned binary format
   serializing, per block: grid metadata (extents, ghost depth, spacing,
   origin -- enough to reconstruct/verify a `CartesianGrid`) and the
   field's raw real-cell data (every component; ghost cells excluded),
   plus a file-level step count and simulation time. See that header's
   own doc comment for the full design rationale, including the
   ghost-cell-exclusion assumption and why it is safe (`FvmSolver::
   residual()` refills every ghost cell from real-cell data before ever
   reading it).

2. **Restart support for the forward-step case**
   (`tests/verification/forward_step_setup.hpp`): `ForwardStepBlocks`
   gained `run_n_steps(n, cfl)` (a step-count-bounded adaptive-CFL
   advance, as opposed to `run_to`'s time-bounded one), plus
   `write_restart(path, step, time)`/`read_restart(path)` wrapping
   `cfe::io::write_restart_checkpoint`/`read_restart_checkpoint` for both
   blocks at once.

3. **A small, separate CLI driver**
   (`tutorials/euler_2d_forward_step/forward_step_2d_restart_demo.cpp`),
   not a rewrite of the existing sweep's `main()`: `--mode=checkpoint`
   runs `res288_hllc` from scratch, writes a checkpoint 1000 steps before
   a (step-count-bounded) target, and continues to completion, reporting
   real wall-clock time; `--mode=resume` loads that checkpoint into a
   fresh object and runs just the remaining 1000 steps, reporting real
   wall-clock time for that alone. The actual measured numbers are in
   this tutorial's `README.md`.

4. **Correctness property (non-negotiable):** a continuous run of N
   steps produces the IDENTICAL final state (bit-for-bit) to: run
   (N-1000) steps, write restart, read restart, run 1000 more steps.
   Checked directly and end-to-end against the real two-block case
   (`tests/unit/test_forward_step_restart.cpp`, both Dim=2 and Dim=3),
   including a direct (not inferred) check that ghost cells are left
   untouched by `read_restart`. File-format-level round-trip/rejection
   tests are in `tests/unit/test_restart_file.cpp`.

5. **"Nightly on PSC" workflow -- documentation/script only.**
   `tutorials/euler_2d_forward_step/regenerate_restart_checkpoints.sh`
   performs a full from-scratch sweep and produces checkpoint files for
   the expensive cases; `README.md`'s "Checkpoint/restart" section
   documents the intended workflow (full sweep + checkpoint generation
   overnight on PSC Bridges-2, unattended; local development/verification
   uses the resulting restart files for fast tail-only re-checks). This
   task does **not** SSH into Bridges-2 or set up a live cron job --
   that remains separate, explicit follow-up work the user would need to
   request and supervise directly.

## Review workflow

Touches solver/memory-layout/I-O and introduces a new persistent file
format: full review cycle (Violet, Snorlax, Enzo, Sally, then Ilia's
independent final review and Amelia's alignment check) -- see
`agent_history.md`'s entry for this task for the full trail and outcome.

## Known limitations (see `agent_history.md`'s entry and `README.md` for
the full, current account)

- CPU-only (serial/threaded backend); no CUDA restart path exists or was
  requested.
- MPI-aware multi-block restart is explicitly out of scope (see
  Motivation above) -- "Task B" remains a separate, unstarted task.
- The restart file format is not portable across machines/builds
  (no byte-swapping, exact `sizeof(Scalar)`/component-count/grid-shape
  match required) -- a same-machine/same-build checkpoint, not an
  interchange format. Revisit if/when a reacting-flow restart genuinely
  needs to move between machines with different precision.
- `regenerate_restart_checkpoints.sh` documents the PSC Bridges-2 nightly
  workflow but does not execute it remotely -- no live SSH/cron
  automation was set up in this task.
