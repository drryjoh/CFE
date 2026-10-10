# Tutorial: 3D MPI Burgers strong scaling

Task 0006 added the `MPI_Allreduce(MAX)` fix (`backend/mpi/
mpi_reduce.hpp`) that makes `BurgersField` safe to decompose, and
proved it correct with `tests/mpi/test_mpi_burgers_steepening.cpp`.
This tutorial is the visual/quantitative follow-up: the same
`FvmSolver` + `BurgersField<Scalar,3>` + `MusclMinmodReconstruction` +
`RusanovFlux` + SSP-RK2 stack `tutorials/burgers_3d_visualization/`
already uses, decomposed across an increasing MPI rank count, both
**running correctly** (a shock forms and crosses rank boundaries
cleanly) and **scaling** (more ranks finish the same fixed problem
faster) -- the direct sibling of `tutorials/
mpi_scalar_advection_3d_strong_scaling/`, read that one first for the
strong-vs-weak-scaling background this README doesn't repeat.

## What it does

A Gaussian bump (`state = 1.0 + 0.5*exp(-r^2/(2*0.12^2))`, identical IC
and constants to `tutorials/burgers_3d_visualization/`) is simulated on
a periodic 128^3 grid to the same physical final time that single-rank
tutorial reaches in 500 steps at its own (lower) resolution, decomposed
across however many MPI ranks the binary is launched with via
`CartesianPartition` + `MpiHaloBoundary` (task 0005). Unlike the linear
scalar-advection sibling tutorial, Burgers' characteristic speed is the
local state itself, so the bump does not just translate: its leading
faces steepen into a shock, its trailing faces spread into a smooth
rarefaction fan -- independently along all three axes at once.

**This is also where task 0006's fix is directly exercised, not just
invoked for show**: the bump's peak sits inside only one (or a few, at
higher rank counts) rank's own local block, so most ranks see a
meaningfully smaller local `max|u|` than the true global one -- sizing
`dt` from the wrong, local-only value would desynchronize ranks' step
counts and hang the run (confirmed by deliberately breaking this in
task 0006's own correctness test). The one-line fix,
`cfe::backend::mpi::allreduce_max`, is what this file's own `dt`
computation calls before anything else.

Each rank writes **its own local block only** as a separate VTK file
(`frame_NNNN_rankNN.vtk`) using its true physical origin -- these tile
together exactly, same as the sibling tutorial (verified numerically
again for this tutorial, not just assumed from the shared machinery).
Opening the whole series in ParaView and watching it play lets you
**see** the shock crossing a rank boundary with no visible seam.

Wall-clock time for the whole run (compute + periodic VTK writes,
synchronized with `MPI_Barrier`) is printed as one CSV row per
invocation: `ranks,px,py,pz,global_n,n_steps,wall_clock_s`.

## Build and run

Requires `-DCFE_ENABLE_MPI=ON`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCFE_ENABLE_MPI=ON
cmake --build build --target cfe_mpi_burgers_3d -j
```

## Running in parallel

Same as the sibling tutorial: an ordinary MPI program, `-n` picks the
rank count, `(px,py,pz)` is chosen automatically. One run by itself:

```bash
mpirun -n 4 build/tutorials/mpi_burgers_3d_strong_scaling/cfe_mpi_burgers_3d
```

Add `--oversubscribe` (OpenMPI) if your machine has fewer cores than
the rank count requested -- correctness is unaffected, only timing.

**To measure strong scaling**, run that same command at several `-n`
values and collect the result:

```bash
cd tutorials/mpi_burgers_3d_strong_scaling
mkdir -p data
for np in 1 2 4 8; do
  mpirun -n $np ../../build/tutorials/mpi_burgers_3d_strong_scaling/cfe_mpi_burgers_3d \
    > /tmp/run_np${np}.out
  if [ "$np" -eq 1 ]; then
    grep -A1 '^ranks' /tmp/run_np${np}.out > data/summary.csv
  else
    grep -A1 '^ranks' /tmp/run_np${np}.out | tail -1 >> data/summary.csv
  fi
done
```

## Regenerating the figures

```bash
python3 -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install numpy pandas matplotlib
python3 plot_results.py
```

Writes `figures/expected_scaling_reference.png` (conceptual, no data
needed -- see the sibling tutorial's README for what it means), plus,
once `data/summary.csv` exists, `figures/strong_scaling_time.png` and
`figures/strong_scaling_speedup.png`.

## What the numbers show

| Machine | 1 rank | 2 ranks | 4 ranks | 8 ranks |
|---|---|---|---|---|
| This dev laptop (Apple Silicon, 10 cores, `mpirun --oversubscribe`) | 62.62 s | 32.36 s (1.94x) | 18.83 s (3.33x) | 24.80 s (2.53x) |
| **PSC Bridges-2 (RM-shared, dedicated cores)** | **252.14 s** | **125.81 s (2.00x)** | **64.24 s (3.92x)** | **34.39 s (7.33x)** |

The committed `data/summary.csv` and `figures/*.png` are the
**Bridges-2 numbers** -- the authoritative result, and it is
**near-ideal**: 100% efficiency at 2 ranks, 98% at 4, still 92% at 8.
This is the same textbook strong-scaling result the sibling
scalar-advection tutorial found on the same cluster, now confirmed for
the equation that actually needed task 0006's fix to be safe to
decompose at all.

The dev laptop's numbers tell the same instructive, different story as
the sibling tutorial: good scaling up to 4 ranks, then falling off at
8 as communication overhead becomes a larger fraction of each rank's
shrinking local workload (a shared, non-dedicated machine, not a
dedicated cluster node) -- expected, not a bug, and directly confirmed
by the cluster numbers above showing that same effect is far smaller
on real dedicated hardware. Burgers does more per-cell work than
scalar advection (minmod slope limiting, a nonlinear flux, Rusanov
dissipation), so the absolute times are larger at every rank count on
both machines, but the qualitative scaling shape matches.

Bridges-2's per-core single-threaded speed for this workload was
notably slower than the laptop's (the 1-rank time is ~4x the laptop's,
not faster as might be naively expected from "it's a supercomputer") --
a reminder that a cluster's value is in dedicated, numerous, reliably-
scaling cores, not necessarily a faster single core; the comparison
that matters is each machine's own speedup curve (the parenthesized
multipliers above), not raw wall-clock time across different hardware.

**TVD/boundedness check** (same guarantee every single-rank Burgers
test in this repo already verifies numerically): the state never
exceeds its initial maximum (1.49921) or drops below its initial
minimum (1.0) at any rank, any frame -- confirmed directly from the
committed VTK output's own `CELL_DATA`, not just assumed from the
scheme's design.

## What is committed vs. regenerated

`data/summary.csv` and all three PNGs under `figures/` are committed in
full. VTK frames are **not** committed (regenerate by running the
binary once) -- for interactive viewing in ParaView, not for this
README to embed.

## Where to go next

- For the collective CFL fix this tutorial exercises:
  `src/cfe/backend/mpi/mpi_reduce.hpp`,
  `docs/adr/0009-mpi-domain-decomposition.md`'s Burgers-fix amendment.
- For the correctness proof (numerical, not just visual) that a
  decomposed Burgers shock matches a single-process reference exactly:
  `tests/mpi/test_mpi_burgers_steepening.cpp`.
- For the decomposition machinery itself:
  `src/cfe/grid/partition/cartesian_partition.hpp`,
  `src/cfe/grid/boundary/mpi_halo_boundary.hpp`.
- For the sibling tutorial's own strong-vs-weak-scaling background and
  expected-scaling reference plot explanation:
  `tutorials/mpi_scalar_advection_3d_strong_scaling/README.md`.
