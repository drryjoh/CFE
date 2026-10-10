# Tutorial: 3D MPI strong scaling

Task 0004 built the first MPI domain-decomposition prototype (1D slab
along X) and a communication-only benchmark. Task 0005 generalized that
to full 3D block decomposition (`grid/partition/cartesian_partition.hpp`,
`grid/boundary/mpi_halo_boundary.hpp`). This tutorial is the
full-solver demonstration both were building toward: the same
`FvmSolver` + `ScalarAdvectionField<Scalar,3>` + SSP-RK2 stack every
other scalar-advection tutorial/test in this repo uses, run at a fixed
global resolution across an increasing MPI rank count.

## What is strong scaling?

**Strong scaling** fixes the total amount of work (here: one 128^3
grid, one initial condition, one final time -- identical across every
run) and asks: does splitting that *same* fixed pile of work across
more MPI ranks finish it faster? This is different from **weak
scaling**, which grows the problem size *along with* the rank count so
each rank's own share of the work stays the same size -- weak scaling
asks "can we solve a bigger problem in the same time with more
hardware?", not "can we solve this problem faster?". This tutorial
demonstrates strong scaling specifically: `(px,py,pz)` -- how many ranks
sit along each axis -- changes with the rank count, but `global_n=128`
never does.

## What it does

A Gaussian bump (`state = 1.0 + 0.5*exp(-r^2/(2*0.08^2))`, same
convention `tutorials/scalar_advection_3d_visualization/` uses)
translates diagonally (`velocity=(1,1,1)`) across a periodic 128^3
grid for a fixed final time, decomposed across however many MPI ranks
the binary is launched with via `CartesianPartition` +
`MpiHaloBoundary` (task 0005). `(px,py,pz)` is chosen automatically
from the rank count (most "cube-like" factorization -- e.g. 1 rank ->
1x1x1, 2 -> 2x1x1 or 1x1x2, 4 -> 2x2x1, 8 -> 2x2x2).

Each rank writes **its own local block only** as a separate VTK file
(`frame_NNNN_rankNN.vtk`) using its true physical origin -- these tile
together exactly (verified numerically: at 8 ranks, each piece is a
64^3-cell block at one of the 8 corners of the unit cube, with no gaps
or overlaps) when all loaded together in ParaView.

Wall-clock time for the whole run (compute + periodic VTK writes,
synchronized with `MPI_Barrier` so every rank starts the timed region
together) is printed as one CSV row per invocation:
`ranks,px,py,pz,global_n,n_steps,wall_clock_s`.

## Build and run

Requires `-DCFE_ENABLE_MPI=ON` (see repo root `README.md`/`CMakeLists.txt`
for the MPI auto-detect/gate pattern).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCFE_ENABLE_MPI=ON
cmake --build build --target cfe_mpi_scalar_advection_3d -j
```

Run once per rank count and collect the CSV (this binary prints its own
header every time, so only the data row after the first run is kept):

```bash
cd tutorials/mpi_scalar_advection_3d_strong_scaling
mkdir -p data
for np in 1 2 4 8; do
  mpirun -n $np ../../build/tutorials/mpi_scalar_advection_3d_strong_scaling/cfe_mpi_scalar_advection_3d \
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

Writes `figures/strong_scaling_time.png` (wall-clock vs. rank count,
log-log, against an ideal `T(1)/ranks` reference line) and
`figures/strong_scaling_speedup.png` (speedup and parallel efficiency).

## What the numbers show

| Machine | 1 rank | 2 ranks | 4 ranks | 8 ranks |
|---|---|---|---|---|
| This dev laptop (Apple Silicon, 10 cores, `mpirun --oversubscribe`) | 8.70 s | 4.69 s (1.85x) | 3.33 s (2.61x) | 3.84 s (2.26x) |
| PSC Bridges-2 (RM-shared, real cores, see below) | *pending* | | | |

**Scaling improves up to 4 ranks, then falls off (and even regresses
slightly at 8) on this dev laptop.** This is expected, not a bug: as
rank count grows, each rank's own local block shrinks (at 8 ranks, each
owns only a 64^3 slice of the original 128^3 domain), so the fraction
of total work spent on halo exchange (communication) relative to
interior computation grows -- exactly the overhead
`benchmarks/mpi/bench_mpi_halo_exchange.cpp` (task 0004) already
measured in isolation. A laptop's shared memory bus and non-uniform
core types (performance vs. efficiency cores, on Apple Silicon) also
make rank-to-rank timing less uniform than on dedicated cluster
hardware. **The authoritative scaling curve needs real, demonstrably
single-purpose cluster cores** -- see the row above, to be filled in
from a PSC Bridges-2 run (same "local sanity check first, cluster
numbers for the record" pattern every GPU benchmark in this repo
already follows).

## What is committed vs. regenerated

`data/summary.csv` (tiny, just the four measured rows) and both PNGs
under `figures/` are committed in full. The VTK frames themselves are
**not** committed (even the smallest, 1-rank case would be ~20 ASCII
files per run) -- regenerate them by running the binary once; they are
for interactive viewing in ParaView, not for this README to embed.

## Things to try

- **Change `kGlobalN`** (`mpi_scalar_advection_3d.cpp`) -- a larger
  global grid raises the interior-work-to-halo-exchange ratio, which
  should push the point of diminishing returns to a higher rank count.
- **Change `kFinalTime`/`kCfl`** -- more steps means more halo
  exchanges for the same interior work per step, so communication
  overhead (and the point where scaling falls off) becomes relatively
  more significant with a longer run at the same resolution.
- Compare this tutorial's numbers directly against task 0004's
  `bench_mpi_halo_exchange` sweep at a similar cross-section size --
  the fraction of this tutorial's wall-clock time explainable by
  communication alone is a direct, measured answer, not a guess.

## Where to go next

- For the decomposition machinery this tutorial is built on:
  `src/cfe/grid/partition/cartesian_partition.hpp`,
  `src/cfe/grid/boundary/mpi_halo_boundary.hpp`.
- For the correctness proof that decomposition changes nothing about
  the answer: `tests/mpi/test_mpi_halo_exchange_3d.cpp`.
- For the communication-only cost in isolation:
  `benchmarks/mpi/bench_mpi_halo_exchange.cpp`.
- For the scheme-choice rationale (1D slab -> full 3D block
  decomposition, why no diagonal/corner exchange is needed):
  `docs/adr/0009-mpi-domain-decomposition.md`.
