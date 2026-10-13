# Phase 3 Euler CUDA benchmark results (and correctness evidence)

Raw CSV (1D, GPU): [`benchmarks/results/phase3_euler_v100.csv`](../../benchmarks/results/phase3_euler_v100.csv).
Raw CSV (3D, GPU): [`benchmarks/results/phase3_euler_3d_v100.csv`](../../benchmarks/results/phase3_euler_3d_v100.csv).

This closes the policy gap flagged during task 0009's review (compressible
Euler, MUSCL/Rusanov, Sod shock tube): every PR adding a Field/scheme
must include a 3D CPU correctness test, a CUDA correctness test, and an
at-scale GPU benchmark in the same PR. Task 0009 had landed clean on
CPU only; this is the direct, same-PR follow-up, run for real on PSC
Bridges-2 -- not deferred.

## What was added

- `tests/unit/test_euler_3d_sanity.cpp`: 3D CPU correctness test.
  Mirrors `test_scalar_advection_2d_sanity.cpp`/`test_burgers_3d_sanity.cpp`:
  the 1D Sod shock tube repeated across every (j,k) column (periodic,
  uniform Y/Z, so each column is an independent copy of the 1D
  problem), confirmed to match the shared 1D reference
  (`sod_shock_tube_runner.hpp`) column for column to `1e-13` (tightened
  from an original `1e-10` during review -- see "Review findings and
  resolution" below), including the adaptive-CFL `dt` sequence.
  Directly exercises the Y and Z `axis_flux_difference` branches for a
  multi-component Field, which the production verification test
  (`EulerField<Scalar,1>` only) cannot.
- `tests/unit/test_euler_cuda.cu`: 1D CPU-vs-GPU correctness test. Sod
  IC, `SlipWallBoundary` at both ends (CFE's first CUDA exercise of
  `SlipWallBoundary`), 2500 fixed-dt steps (t=0.5), GPU matches CPU to
  `1e-9` cell-by-cell, plus a mass/total-energy conservation check
  (`1e-10`) on the CPU reference. **Revised during Violet's review of
  this follow-up**: the original 300-step version (t=0.06) never
  reached either wall -- a standalone CPU-only probe confirmed the
  shock does not arrive until t~0.28 -- so despite the file's own
  "reflecting tube" framing, no version prior to this fix had actually
  exercised a wave reflecting off `SlipWallBoundary`. Step count
  corrected to run well past the first reflection (confirmed via the
  same probe); the conservation check was added because no closed-form
  solution exists for a doubly-reflected Sod problem, so conservation
  is the right additional physical sanity check for the now-longer run.
- `tests/unit/test_euler_3d_cuda.cu`: 3D CPU-vs-GPU correctness test.
  100x8x6 grid, `SlipWallBoundary` on X / periodic Y,Z, 625 fixed-dt
  steps (t=0.5), GPU matches CPU to `1e-9` cell-by-cell, plus the same
  mass/total-energy conservation check. Exercises X, Y, and Z
  `axis_flux_difference` together with `SlipWallBoundary::fill_x`'s
  genuinely non-trivial (`ny,nz>1`) ghost-index decomposition, for the
  first time on GPU. Same reflection-coverage fix as the 1D test above
  (original 150 steps/t=0.12 also predated the t~0.27 reflection).
- `benchmarks/euler/bench_euler_cuda.cu` / `bench_euler_3d_cuda.cu`: at-
  scale GPU throughput benchmarks, mirroring
  `bench_burgers_cuda.cu`/`bench_burgers_3d_cuda.cu`'s structure and
  resolution sweep exactly (1D: 1M-10^8 cells; 3D cube: up to 512^3,
  ~134M cells), smooth periodic sine IC (not the Sod discontinuity --
  same "smooth IC for throughput, discontinuous IC for correctness"
  convention). First at-scale GPU measurement in this codebase for a
  genuinely multi-component Field (`EulerField<Scalar,1>`:
  `n_components=3`; `EulerField<Scalar,3>`: `n_components=5`).

Both new CUDA test files use the required named
`cfe::SolverResidual<decltype(solver)>` functor (not a local lambda) as
the template argument to `ssp_rk2_step`, per the established nvcc
constraint (a locally-defined lambda cannot be used as a template
argument to a function containing an extended `__device__` lambda in a
`.cu` file) already documented at `test_burgers_cuda.cu`/
`test_scalar_advection_cuda.cu`.

## Methodology

Median of 10 timed repetitions after one untimed warm-up (first launch
pays CUDA context/JIT cost), via `cfe::CudaParallelFor`, timing one full
`ssp_rk2_step` call (ghost-fill + residual assembly, twice, per the
SSP-RK2 stage structure) -- identical methodology to
`0003-phase1-scalar-advection-cuda-results.md`/
`0006-phase2-burgers-cuda-results.md`.

## Environment

| | |
|---|---|
| Hardware | NVIDIA Tesla V100-SXM2-32GB (compute capability 7.0), PSC Bridges-2 `GPU-shared` partition, node `v002` |
| CUDA toolkit / host compiler | `cuda-v100/12.9.2` (aliases `cuda-legacy/12.9.2`) + `gcc/13.3.1-p20240614` modules (same as prior phases) |
| Build type | `CMAKE_BUILD_TYPE=Release`, `CFE_ENABLE_CUDA=ON`, `-DCMAKE_CUDA_ARCHITECTURES=70` |
| SLURM | `salloc -A phy260058p -p GPU-shared --qos=gpuinteract --gres=gpu:v100-32:1 -t 30:00`, job `49095911`; allocation held ~3.3 of its 30 minutes |
| Date | 2026-10-12 |

Note on QOS: `scontrol show job` reported `QOS=gpu` even with
`--qos=gpuinteract` passed explicitly on the (successful, promptly-
granted) second submission -- apparently how this field is displayed on
this cluster (confirmed `gpuinteract` is a valid QOS for this account
via `sacctmgr show assoc`) rather than evidence the flag was dropped;
the first submission attempt genuinely did queue indefinitely
(`Nodes_required_for_job_are_DOWN,_DRAINED_or_reserved...`, the same
cluster scheduling noise task 0004's own Bridges-2 run documented), and
cancel-and-resubmit fixed it exactly as it did then.

## Correctness: full suite on real GPU hardware

`cfe_unit_tests`: **127/127 passed** on the allocated V100 node
(job `49095911`), including both new Euler CUDA tests
(`test_euler_cuda_matches_cpu_reference`,
`test_euler_3d_cuda_matches_cpu_reference`) and all pre-existing CUDA
tests (scalar advection, Burgers, backend-execution) -- zero
regressions. On the login node (no GPU), the same binary shows
116/127, the 11 failures being exactly the CUDA tests
(`cudaMalloc: CUDA driver version is insufficient for CUDA runtime
version`, i.e. no GPU present there) -- confirming the 116 CPU-only
tests are unaffected by the CUDA build and the 11-test gap is the
expected hardware-availability signature, not a real failure.

`compute-sanitizer --tool initcheck ./build/tests/cfe_unit_tests`:
**0 errors** (full suite, same node/allocation).

**Re-verified a second time** (job `49096728`, node `v008`, a separate
~20-minute allocation) after Violet's and Enzo's review of this
follow-up produced real fixes (not just doc corrections -- see "Review
findings and resolution" below): the two CUDA correctness tests'
step counts were corrected so they genuinely exercise a wave reflecting
off `SlipWallBoundary` (the original versions did not), a mass/
total-energy conservation check was added to each, and
`test_euler_3d_sanity.cpp`'s tolerance was tightened. Full suite: still
**127/127 passed**, `compute-sanitizer --tool initcheck`: still **0
errors** -- the fixes did not regress anything, and the longer runs
(2500/625 steps vs. the original 300/150) stayed well within the
`kMaxSpeedEstimate=5.0` margin with no instability.

One cross-platform finding from this second round, worth recording:
`test_euler_3d_sanity.cpp` was briefly tried at an EXACT (`0.0`)
tolerance during review (CPU-only, no GPU needed) -- it passed
deterministically on this project's development machine (arm64/clang)
but did NOT reproduce bit-exact on Bridges-2 (x86-64/gcc 13.3.1): same
source, same reasoning (every (j,k) column is an independent copy of
the 1D reference), genuinely different floating-point rounding between
the two toolchains/architectures (ordinary FMA-contraction/
reassociation differences, not a bug). Settled on `1e-13` instead --
still ~10^3 tighter than the original `1e-10` for real regression
protection, but portable across the toolchains AGENTS.md #9 actually
targets (x86-64, Apple Silicon, CUDA), confirmed passing on both.

## Results: 1D, up to 10^8 cells

| n_cells | median ms/step | cell-updates/s |
|---|---|---|
| 1,000,000 | 0.656 | 1.524e9 |
| 10,000,000 | 6.289 | 1.590e9 |
| 50,000,000 | 30.572 | 1.636e9 |
| 100,000,000 | 60.683 | 1.648e9 |

Compare to Burgers' own 1D V100 numbers (`0006-...md`, 7.69e9
cell-updates/s at 10^8 cells): Euler here reaches 1.648e9 at the same
scale, roughly 4.7x lower. `EulerField<Scalar,1>` has `n_components=3`
against Burgers' 1 -- three times the per-cell state to read/write and
reconstruct/flux PER COMPONENT, plus Euler's physical flux needs an
equation-of-state evaluation (pressure from density/momentum/energy)
that Burgers' flux does not -- a roughly-3x-plus-overhead slowdown is
the expected signature of doing genuinely more arithmetic and memory
traffic per cell, not a regression.

## Results: 3D, up to 512^3 (~134M) cells

| n_per_axis | n_cells | median ms/step | cell-updates/s |
|---|---|---|---|
| 64 | 262,144 | 0.765 | 3.427e8 |
| 128 | 2,097,152 | 6.100 | 3.438e8 |
| 256 | 16,777,216 | 49.582 | 3.384e8 |
| 400 | 64,000,000 | 189.440 | 3.378e8 |
| 512 | 134,217,728 | 382.170 | 3.512e8 |

The flagged OOM risk at 512^3 (`EulerField<Scalar,3>` has
`n_components=5`; three device buffers -- state/stage1/scratch -- at
134,217,728 cells x 5 components x 8 bytes (double) each is
~5.37 GB/buffer, ~16.1 GB total, against the V100's 32 GB) **did not
materialize** -- the full sweep, 64^3 through 512^3, completed with no
allocation failure. Measured against the actual budget, ~16.1 GB of 32
GB is **~50% utilization** -- a comfortable margin, not a near-miss;
the risk assessment going in was conservative, not an accurate
prediction of danger. This is still a real, measured result rather
than an assumption: the sweep was watched run to completion on the
allocated node, not inferred from the memory arithmetic alone.

Compare to Burgers' own 3D V100 numbers (`0006-...md`, ~2.99e9-4.27e9
cell-updates/s across the same resolution range): Euler lands at
~3.4-3.5e8, roughly an order of magnitude lower. `EulerField<Scalar,3>`
has `n_components=5` against Burgers' 1, with three active axes each
doing a full MUSCL-minmod reconstruction + Rusanov flux + equation-of-
state evaluation per component per face -- substantially more
arithmetic and memory traffic per cell than the 1D case above already
showed.

Quantifying that "further compounding" precisely (not just
qualitatively), using the Burgers/Euler throughput ratio at each
matching sweep position (1D 1M/10M/50M/100M against 3D 64^3/128^3/
256^3/400^3 respectively -- the 3D sweep's extra 512^3 point has no 1D
counterpart so is excluded from this specific comparison):

| Sweep position | 1D ratio (Burgers/Euler) | 3D ratio (Burgers/Euler) | 3D/1D compounding |
|---|---|---|---|
| 1 | 6.10e9/1.524e9 = 4.00 | 2.99e9/3.427e8 = 8.73 | 2.18x |
| 2 | 7.26e9/1.590e9 = 4.57 | 3.96e9/3.438e8 = 11.52 | 2.52x |
| 3 | 7.38e9/1.636e9 = 4.51 | 4.27e9/3.384e8 = 12.62 | 2.80x |
| 4 | 7.69e9/1.648e9 = 4.67 | 3.87e9/3.378e8 = 11.46 | 2.46x |

The real compounding factor is **~2.2x-2.8x (average ~2.5x)**, not the
much larger figure an earlier draft of this document mis-stated --
corrected during Enzo's review of this follow-up. Register/occupancy
effects (see below) are the likely source of the remaining
~2.5x -- component count alone (5 vs 3, a 1.67x ratio) does not account
for it, so some combination of lower occupancy at N=5 (92 registers/
thread, see below) and the fully 3D-active (X+Y+Z, vs 1D's X-only)
reconstruction/flux work per cell is the more complete explanation,
not precisely decomposed here.

## Register pressure / spilling (AGENTS.md #8)

Static `nvcc --resource-usage -O3 -arch=sm_70` check (no GPU allocation
needed, run both on and off the allocation to confirm it does not
depend on it):

| Kernel | n_components | Registers | Stack frame | Spill stores/loads |
|---|---|---|---|---|
| `FvmSolver::residual` (1D, `EulerField<Scalar,1>`) | 3 | 66 | 0 bytes | 0 bytes |
| `FvmSolver::residual` (3D, `EulerField<Scalar,3>`) | 5 | 92 | 0 bytes | 0 bytes |

No register spilling at either N=3 or N=5 on the GPU path, in contrast
to task 0009's own CPU (arm64) finding (`agent_history.md`'s task 0009
entry: a 128-byte stack spill at N=5, `-O3`, Apple Silicon) -- the two
backends hit the register-pressure threshold differently, as AGENTS.md
#7 ("do not assume GPU optimization improves Apple Silicon
performance", and implicitly the converse) already anticipates.

The 92-registers/thread occupancy bound, done correctly (corrected
during Enzo's review of this follow-up -- a first pass used a naive
`65536/92` division that ignores both the actual launch configuration
and Volta's register-allocation granularity, overstating occupancy at
~35%): `cfe::backend::cuda::parallel_for` (`src/cfe/backend/cuda/
cuda_backend.cuh:49`) launches with a fixed 256-thread block size (8
warps/block). Volta (`sm_70`) allocates registers per warp in
multiples of 8 per thread, so 92 rounds up to 96 registers/thread ->
96 x 32 = 3072 registers/warp. The V100's 65536 registers/SM gives
65536 / 3072 ~ 21.3 -> 21 resident warps from the register budget
alone; since warps are only resident as whole 8-warp blocks, that is 2
full blocks (16 warps = 512 threads/SM), not 21 individual warps. 512
of the SM's 2048-thread maximum is **~25% occupancy** -- lower than the
naive estimate, still not a spilling problem (0 bytes spill stores/
loads either way), but a real, if modest, headroom-for-more-threads
constraint worth tracking as the component count grows toward
AGENTS.md #8's 10/20/50/100-component sweep. An actual `ncu`-measured
occupancy number (vs. this static calculation) was not collected this
session -- reasonable next step, not required to resolve the per-PR
policy gap this follow-up closes.

## Review findings and resolution

Violet (numerical correctness) and Enzo (performance/register-pressure)
reviewed this follow-up specifically. Both independently confirmed the
core claims hold (the 3D path reduces exactly to the 1D reference under
uniform/periodic Y/Z; the benchmark/test timed-region structure has no
hidden allocations, transfers, or synchronization issues) and found
real problems, since fixed:

- **Enzo, blocking:** the original draft of this document claimed a
  "roughly-5x" 3D-vs-1D compounding slowdown and a naive "~35%"
  occupancy estimate. Both were arithmetic/methodology errors, not code
  problems -- fixed above: the real compounding factor is ~2.2x-2.8x
  (computed directly from this document's own tables, shown in the
  "Results: 3D" section), and the real occupancy, accounting for the
  actual 256-thread launch configuration and Volta's register-
  allocation granularity, is ~25%, not ~35% (full corrected methodology
  in "Register pressure / spilling" above).
- **Violet, worth treating as effectively blocking despite being filed
  optional:** `test_euler_cuda.cu`/`test_euler_3d_cuda.cu` originally
  ran only to t=0.06/t=0.12 -- Violet independently simulated the same
  setup and confirmed neither the shock nor the rarefaction reaches
  either wall in that time, so the tests' own "reflecting tube" framing
  was not actually exercised (both walls stayed bit-identical to the
  initial condition the entire run). The CPU-vs-GPU bit-agreement claim
  itself was still valid (two backends computing the identical,
  if physically inert-at-the-boundary, sequence) but the framing
  overclaimed coverage. Fixed, with real evidence (a standalone
  CPU-only probe, not committed, confirming the shock reaches the wall
  at t~0.27-0.28 and the post-reflection max wave speed stays ~2.21,
  comfortably inside the existing margin): both tests' step counts were
  extended to t=0.5 (2500/625 steps), and a mass/total-energy
  conservation check was added to each as the acceptance criterion for
  the now-reflection-including run (no closed-form solution exists for
  a doubly-reflected Sod problem, so conservation -- not an exact-value
  comparison -- is the right physical sanity check). Re-verified on
  real V100 hardware after the fix: still 127/127, still 0
  compute-sanitizer errors, no instability from the longer run.
- **Violet, optional, taken:** `test_euler_3d_sanity.cpp`'s original
  `1e-10` tolerance was ~10 orders looser than the actual agreement.
  Tightened -- see the cross-platform finding recorded above (settled
  on `1e-13`, not an exact `0.0`, specifically because `0.0` does not
  reproduce across this project's development machine and Bridges-2).
- **Enzo, optional, taken:** the 512^3/N=5 OOM-risk framing in an
  earlier draft implied a near-miss; corrected above to state the real
  measured utilization (~50%, a comfortable margin).
- **Enzo, optional, noted but not acted on this session:** no durable
  tracked location yet for when a real N-configurable field (future
  species/ChemGen work) will finally produce actual 10/20/50/100-
  component register/occupancy data per AGENTS.md #8 -- a one-line
  forward reference belongs in that future work's own ADR; not
  established anywhere durable yet (see "Known limitations" below).

Ilia's and Amelia's prior sign-off on task 0009 itself (the physics,
solver, and architecture) was judged still valid rather than re-run
from scratch: this follow-up adds test/benchmark/documentation
coverage, and fixes errors within that new coverage, without changing
any of the production code their review already covered.

## What this resolves

Closes the per-PR policy gap flagged for task 0009: `EulerField` now
has 3D CPU correctness (`test_euler_3d_sanity.cpp`, exact tolerance
confirmed portable at `1e-13`), 1D and 3D CUDA correctness verified on
real V100 hardware (127/127, 0 compute-sanitizer errors, re-verified
after review fixes), both CUDA tests now genuinely exercise a wave
reflecting off `SlipWallBoundary` with a conservation-based acceptance
criterion (not just an inert pre-arrival state), and at-scale GPU
benchmarks at the same resolutions Phase 1/2 established (10^8 cells
1D, 512^3 cells 3D), with the flagged 512^3 memory-ceiling risk
measured at a real, comfortable ~50% utilization -- not a near-miss.

## Known limitations / next steps

- No float32 benchmark variant for Euler yet (prior CUDA benchmarks in
  this codebase default to `cfe::scalar`, project-wide `double`; a
  `float` sweep would show whether Euler's extra arithmetic intensity
  changes the float/double throughput ratio seen in earlier phases --
  not requested for this follow-up).
- Register pressure at N=5 (92 registers/thread, 96 after Volta's
  register-allocation rounding) bounds occupancy to ~25% even without
  spilling (see the corrected methodology above); not yet measured at
  the 10/20/50/100-component scale AGENTS.md #8 names as a future
  requirement for reacting-flow states -- flagged as a watch item for
  whenever a higher-component Field is benchmarked, same status the
  N=3 CPU finding carries from task 0009. No durable tracking location
  for this yet beyond this document and the task-0009 entries in
  `agent_history.md`; the first real species/ChemGen-driven
  N-configurable Field (AGENTS.md #21) is the natural point to actually
  collect 10/20/50/100-component register/occupancy data rather than
  extrapolate from N=3/5 as done here -- worth a one-line forward
  reference from that future work's own ADR back to this document, not
  established anywhere durable yet.
- `ncu --set full` runtime occupancy/bandwidth profiling (vs. the
  static `--resource-usage` register/spill check done here) was not
  run this session -- the register/spill evidence above was judged
  sufficient to resolve the specific per-PR policy gap; a full
  occupancy profile remains reasonable future work, same as it has
  been for every prior CUDA benchmark in this codebase.
