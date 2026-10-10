# Task 0008: Phase 2 — 3D MPI Isosurface Videos + Burgers Run-Duration Tuning

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `ROADMAP.md` (Phase 2)
4. `VERIFICATION.md`
5. `agent_history.md`

before making changes.

## Objective

User asked for an MP4 isosurface visualization (viewed from outside
the simulated box, camera orbiting, showing the isosurface move) of
both 3D MPI strong-scaling tutorials' VTK output, since manually
post-processing them in ParaView each time was tedious. Also asked to
run the Burgers tutorial longer so its steepening is more dramatic to
watch, since its existing final time (matched to `tutorials/
burgers_3d_visualization/`'s own, `t~=0.521`) turned out to only just
clear the IC's analytic shock-formation time (`t_s~=0.396`) -- barely
past breaking, not a clearly-developed front.

**Do not implement in this task:**

- a shared/common rendering library across the two tutorials -- each
  gets its own self-contained `render_isosurface_video.py`, consistent
  with every other pair of tutorials in this repo being independently
  self-contained rather than sharing code;
- changing `tutorials/scalar_advection_3d_strong_scaling/`'s own
  physics parameters (final time, IC) -- only its isosurface video was
  added; the user's "run it longer" request was specific to Burgers.

## Required functionality

- `tutorials/mpi_burgers_3d_strong_scaling/render_isosurface_video.py`
  and the equivalent for the scalar-advection sibling tutorial: stitch
  every rank's own VTK tile for a frame into one full-domain
  `pv.ImageData` (merging raw cell data BEFORE any cell-to-point
  averaging, to avoid a seam at rank boundaries -- averaging
  independently per piece first would compute each piece's own
  boundary-face point values only from cells on that one side), extract
  one isosurface per frame at a FIXED threshold (not re-centered on
  each frame's own peak) via PyVista's `contour()` (VTK's marching
  cubes under the hood), render from a slowly-orbiting camera outside
  the domain, encode to `figures/isosurface.mp4` via PyVista's
  `open_movie`/`write_frame` (backed by `imageio`/`imageio-ffmpeg`, no
  separate system ffmpeg install required).
- `tutorials/mpi_burgers_3d_strong_scaling/mpi_burgers_3d.cpp`:
  `kFinalTime` increased from `0.521` to `1.2` (~3x the analytic
  breaking time for this IC), `kOutputFrames` from 20 to 40 for a
  smoother video. Full strong-scaling re-verification required since
  this changes the tutorial's own timing numbers.

## Architecture constraints

No production code changed -- this task touches only tutorial-local
driver code (one `.cpp` constant change) and new Python post-processing
scripts, neither of which cross into `src/cfe/`.

## Tests

None new. Re-verification of existing invariants at the new duration:
TVD boundedness (`state` stays in the IC's own `[1.0, 1.49921]` bounds)
checked across every frame/rank again at `t=1.2`, not assumed to still
hold just because it held at the old, shorter duration.

## Benchmarks

None new -- the strong-scaling tutorial itself is re-run at the new
duration (local + Bridges-2), not a separate benchmark.

## Architecture decisions

None -- no new design decisions; this is tooling + a tuning parameter.

## Completion report

At the end report:

1. files added/changed;
2. why `t=1.2` was chosen (analytic breaking-time calculation) and
   confirmation TVD boundedness still holds at every frame/rank;
3. the re-measured strong-scaling numbers (local + Bridges-2) at the
   new duration, since the final time change invalidated the
   previously-committed ones;
4. a plain description of what the two isosurface videos show and why
   the periodic "ghost fragment" artifacts in the Burgers one are real
   physics, not a bug;
5. recommended next task.
