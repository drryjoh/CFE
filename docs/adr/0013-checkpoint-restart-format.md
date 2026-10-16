# ADR 0013: Checkpoint/restart binary format

**Status:** Accepted
**Date:** 2026-10-15

## Context

Task 0016 (`tasks/0016-checkpoint-restart.md`) adds the ability to
resume a deterministic explicit-scheme CFE run from a point partway
through, to avoid re-simulating from `t=0` when re-verifying an
expensive case (motivating example: `tutorials/euler_2d_forward_step`'s
`res288_hllc` resolution-sweep case, ~12,000 steps / tens of minutes).

This requires persisting enough state to resume bit-identically:
grid shape, field data, and a step/time bookkeeping pair. It is CFE's
first binary (non-ASCII, non-visualization) native file format.
ARCHITECTURE.md #19 warns against designing a custom binary format
prematurely and asks for a CFE API over an established portable format
instead -- but also explicitly lists "checkpoints" among the long-term
native-format data requirements that format is meant to eventually
cover. This ADR records why a checkpoint is the case where a bespoke
format is the right call now, not a violation of that principle.

## Decision 1: a bespoke binary format, not an extension of the existing VTK writer

### Options considered

**Option A (selected): a new, purpose-built binary format**
(`src/cfe/io/restart_file.hpp`), read and written only by CFE itself.

**Option B (rejected): extend `io/vtk_writer.hpp`'s legacy-VTK writer to
also carry ghost-adjacent/step/time metadata**, so checkpointing reuses
the one existing native-format writer instead of adding a second one.

Disadvantages of Option B (why rejected):

- `io/vtk_writer.hpp`'s own header comment already scopes it away from
  this use ("not for checkpointing a production-scale run... a binary/
  XML writer would replace this if that need arises") -- it is ASCII,
  one scalar component at a time, real cells only, explicitly for
  human/ParaView-inspectable visualization snapshots. Extending it to
  carry full multi-component conserved state plus step/time metadata
  while keeping it ASCII would make it slow to write/read at the exact
  moment this feature needs it to be fast (a checkpoint write/read
  happens in the hot path of "how much time does re-verification cost");
  making it binary to fix that would turn it into a different format in
  behavior, just sharing a name.
- A checkpoint and a visualization snapshot have opposite requirements:
  a snapshot is read by a human/ParaView, on any machine, long after it
  was written, and only needs to be dense enough to look at in 3D; a
  checkpoint is read back only by CFE itself, on the same machine/build
  that wrote it, as fast as possible, bit-exact in every component. Zero
  users want both at once from a single file.
- A restart file also needs a question a visualization snapshot never
  asks: "does this file actually match the grid I just built?" (format
  version, `sizeof(Scalar)`, component count, exact grid shape) -- a
  reader that silently proceeds on a mismatch would misread into the
  wrong memory layout or resume the wrong resolution's checkpoint
  without any signal. This has no VTK-format equivalent to reuse.

Option A costs a second small I/O header, but is honest about the
format actually needed, keeps `vtk_writer.hpp`'s existing meaning
(visualization snapshot) intact for its own users, and gives
`kRestartFormatVersion` (ARCHITECTURE.md #19's own explicit ask for
checkpoints) a real place to live.

## Decision 2: what is serialized, and the ghost-cell-exclusion assumption

Per block: grid shape (extents, ghost depth, spacing, origin) and every
REAL cell's raw conserved state (every component) -- ghost cells are
excluded. File-level: a step count and simulation time.

This relies on one assumption: every ghost cell in this codebase is
fully reconstructible from real-cell data alone, because
`FvmSolver::residual()` (`solver/explicit/fvm_solver.hpp`) calls
`fill_ghost_cells()` at the START of every residual evaluation, before
reading any state -- a ghost cell's value is therefore never state that
must survive a restart; it is always recomputed, from real cells plus
the boundary condition (and, for `BlockInterfaceBoundary`, the
neighbor block's own real cells -- also restored), before it is next
read, whether the process ran continuously or just resumed.

This is not merely asserted: `tests/unit/test_forward_step_restart.cpp`
checks it directly two ways -- (a) a snapshot-before/after-read_restart
comparison across all four of this case's ghost-fill cases (fixed
inflow, slip wall, cross-block interface, riser-face fallback) confirms
`read_restart` genuinely never touches ghost storage, and (b) the
bit-identical continuous-vs-restart-resume comparison across the full
remaining run confirms that whatever those ghost cells are refilled
to after restart is bit-identical to what the continuous run had at the
same point -- if ghost reconstruction were not exact, this test would
fail at the very first post-restart step, not silently pass.

Excluding ghosts also roughly halves (for 2 ghost layers on 2 active
axes at this task's tested resolutions) the data written/read per
checkpoint versus including them, directly reducing the I/O cost this
feature exists to keep small relative to the steps it saves.

## Decision 3: step count/time are free-standing metadata, not solver state

Every field's solve computes its next `dt` from the CURRENT real-cell
state alone (adaptive CFL via `max_wave_speed()`), never from an
elapsed-time value. Step count and simulation time are therefore not
state the solve depends on -- they round-trip purely so a resumed
driver can report "resumed at step 11000, t=3.67" instead of restarting
that bookkeeping at zero. This is why `ForwardStepBlocks::run_n_steps`
(the restart-friendly entry point this task adds) is step-count-bounded
rather than time-bounded like the existing `run_to`: `run_to`'s
`t_final` clamping (`if (t + dt > t_final) dt = t_final - t`) DOES
depend on the accumulated `t` value, so resuming a `run_to`-driven run
bit-identically would require restoring that exact accumulated time and
re-deriving the same clamping decision -- solvable, but unneeded
complexity this task does not require, since the user's own description
of the desired workflow was already step-count-based ("checkpoint ~1000
steps before the target, then only simulate those last ~1000 steps").

## Decision 4: not portable across machines/builds

No byte-swapping; `sizeof(Scalar)`, component count, and exact grid
shape are checked and rejected on mismatch rather than coerced. This is
a same-machine, same-build-configuration checkpoint -- explicitly NOT
the long-term "checkpoints" interchange format ARCHITECTURE.md #19
anticipates as a future need; it is a first, minimal implementation of
that need, versioned (`kRestartFormatVersion`) so it can evolve without
breaking old files, but not yet solving cross-machine portability or
reacting-flow species metadata. A future need to move a restart file
between machines with different precision, or to add species metadata,
is the trigger to revisit this -- not a reason to generalize ahead of
it now (AGENTS.md #6).

## Evidence

- **Format-level round-trip/rejection**
  (`tests/unit/test_restart_file.cpp`): single-block and multi-block
  round-trips confirm exact real-cell data recovery and genuine
  ghost-cell exclusion (destination ghosts are poisoned with a sentinel
  different from the source's own ghost poison, and come back
  unchanged); format-version/scalar-size/component-count/grid-shape
  mismatches are confirmed to throw rather than silently misread.
- **Bit-identical restart-resume, real case**
  (`tests/unit/test_forward_step_restart.cpp`): the actual
  `ForwardStepBlocks` two-block/`BlockInterfaceBoundary`/`BlockSet`
  assembly, Dim=2 and Dim=3, confirms the core property end-to-end (see
  Decision 2's evidence above).
- **Real measured speedup, `res288_hllc`**
  (`tutorials/euler_2d_forward_step/forward_step_2d_restart_demo.cpp`):
  see that tutorial's `README.md` "Checkpoint/restart" section for the
  actual wall-clock numbers this produced (full-run cost vs.
  checkpoint-load-plus-tail cost).
- Full CPU unit/verification suite passing with these additions, zero
  regressions to any pre-existing test.

## Consequences

**Easier:** re-verifying an expensive, already-checkpointed case's tail
after a change provably unrelated to physics no longer requires
re-running the whole case from scratch. Any future case wanting the
same capability has a tested, documented format and a small, reusable
API (`write_restart_checkpoint`/`read_restart_checkpoint`,
block-count-generic) to build on, plus a template for what a
restart-friendly step-count-bounded run loop looks like
(`run_n_steps`).

**Harder:** `run_to`-driven (time-bounded) production runs are not
directly restart-resumable as-is; a future case wanting that would need
either a `t_start`-aware `run_to` variant or to accept
`run_n_steps`-style step-count bounding as the restart-compatible
convention (Decision 3).

**Known limitations (named, not silently dropped):**

- CPU-only; no CUDA restart path.
- Not portable across machines/builds (Decision 4).
- MPI-aware multi-block restart (the forward-step case's own two-block
  assembly, if it ever gains MPI decomposition) is out of scope -- see
  `tasks/0016-checkpoint-restart.md`'s Motivation section and the
  still-unstarted "Task B: MPI-aware multi-block interface boundary."
- `regenerate_restart_checkpoints.sh` documents the intended PSC
  Bridges-2 nightly workflow but does not execute it remotely.

## Revisit criteria

- A reacting-flow field needing species metadata in the checkpoint, or
  a restart file needing to move between machines with different
  precision, would justify bumping `kRestartFormatVersion` and
  extending the format -- the version field exists specifically so that
  change does not have to be a breaking surprise for old files.
- A future production driver wanting to resume a `run_to`-bounded
  (rather than `run_n_steps`-bounded) run bit-identically would need
  the `t_start`-aware `run_to` variant named in Consequences above.
- MPI-aware multi-block restart becomes relevant once "Task B"
  (MPI-aware `BlockInterfaceBoundary`) is undertaken -- this format's
  per-block design (a list of grids/views, not hardcoded to exactly
  two) was chosen specifically so extending it to a per-rank restart
  file later would not require a redesign, but that extension is not
  built here.
