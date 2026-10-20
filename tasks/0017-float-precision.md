# Task 0017: Float precision — closing the dual-precision testing gap

Read and follow:

1. `AGENTS.md`
2. `ARCHITECTURE.md`
3. `ROADMAP.md` (Phase 6's "single/double precision studies")
4. `agent_history.md`
5. `docs/adr/0012-hllc-numerical-flux.md`

before making changes.

## Objective

`cfe::scalar` (`core/types.hpp`, set via `-DCFE_SCALAR_TYPE=float|double`)
exists specifically so CFE can run its full pipeline in single precision
(`float`) to exploit hardware that is meaningfully faster at float32
(GPUs especially, but also SIMD CPU throughput generally), without
silently "mixing types" anywhere that would force conversions or leave
float an untested second-class path. All production physics/numerics
headers are already fully generic over `Scalar` — the gap is at the
test/tutorial/benchmark layer, where some files intentionally pin
`double` (a deliberate, correct choice for an exact-oracle reference)
and others pin it incidentally (a real gap, since that silently defeats
`-DCFE_SCALAR_TYPE=float` for exactly the files most likely to need
retuning at that precision).

## Scope

1. Establish ground truth: configure/build/run the full suite at
   `-DCFE_SCALAR_TYPE=float` (tests, tutorials, MPI) before touching any
   code, and report exactly what breaks (if anything).
2. Close the unit-test gap: `tests/unit/test_hllc_flux.cpp`,
   `test_euler_3d_sanity.cpp`, `test_forward_step_sanity.cpp` hardcoded
   `using Scalar = double;` with no `float` counterpart, unlike
   `test_fixed_array.cpp`'s established dual-instantiation pattern. Add
   explicit `<float>` instantiations alongside the existing `<double>`
   ones.
3. Close the tight-tolerance verification-test gap:
   `tests/verification/test_euler_sod_shock_tube{,_hllc}.cpp`,
   `test_euler_shu_osher.cpp`, `test_euler_propagating_shock.cpp`,
   `test_euler_isolated_contact_hllc.cpp`,
   `test_forward_step_interface_conservation.cpp` only instantiate
   `double`, at tolerances (1e-8 to 1e-13) calibrated for double's own
   roundoff floor. Add a float-precision variant of each with a
   tolerance derived from a MEASURED float-vs-double error, not guessed.
4. Fix benchmarks/tutorials that hardcode `double` where the hardcoded
   type is the simulation/field precision (should track `cfe::scalar`):
   `benchmarks/mpi/bench_mpi_halo_exchange.cpp`,
   `tutorials/mpi_burgers_3d_strong_scaling/mpi_burgers_3d.cpp`,
   `tutorials/mpi_scalar_advection_3d_strong_scaling/mpi_scalar_advection_3d.cpp`.
   Reported-metric values (wall-clock timing) stay `double` regardless.
5. Measure the actual float speedup on available hardware (CPU only;
   no CUDA hardware available for this task) and report a real
   float-vs-double wall-clock number, not an assumption.
6. Institutionalize: add "every PR introducing or changing numerical
   production code must pass its relevant tests under both
   `-DCFE_SCALAR_TYPE=float` and `=double`" as a standing requirement in
   `AGENTS.md` #11, with an ADR for the full rationale, matching this
   project's existing "every PR adding a Field/scheme needs a 3D CUDA
   correctness test + at-scale benchmark" convention.

## Explicit non-goals

- Do NOT mechanically replace every hardcoded `double` with
  `cfe::scalar` — an exact-oracle reference (e.g.
  `sod_shock_tube_exact.hpp`, `rankine_hugoniot_jump.hpp`) is correctly
  `double` regardless of `Scalar`, and collapsing that would be a
  regression in dual-precision coverage, not a fix.
- Do NOT retrofit every pre-existing test file in the repository in one
  pass (e.g. the earlier Phase 1/2 Burgers/ScalarAdvection unit tests)
  — this task is scoped to the Phase 3/4 Euler/HLLC/forward-step files
  the gap was actually found in; see `docs/adr/0014-...md` for why the
  standing rule this task adds is forward-looking, not a retrofit
  mandate.

## Acceptance

- Full suite builds and passes at both `-DCFE_SCALAR_TYPE=float` and
  `=double` (ground truth reported before and after).
- Every new float-precision test's tolerance is justified in-file by a
  measured number, not a guess.
- A real, measured float-vs-double wall-clock comparison is reported,
  including cases where float shows no measurable speedup — an honest
  partial result is acceptable; an unmeasured claim is not.
- `AGENTS.md`, a new ADR, `agent_history.md`, and a presentation summary
  are all updated in the same change.
