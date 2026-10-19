# Task 0015: `cfe::Index` -- a general-purpose runtime index type

## Status

Done. Implemented, reviewed (Violet, Snorlax, Sally, Enzo, then Ilia
and Amelia), two rounds of real blocking findings fixed: (1) the
NTTP-mirror-exception applied inconsistently at 23 sites, and (2) an
evidence-integrity problem Ilia caught in the `euler_2d_forward_step`
tutorial verification (stale git-tracked baseline data mistaken for
fresh run output across three worktrees) -- corrected with a genuine,
complete 14/14 resolution/flux rerun in untracked scratch space. See
`agent_history.md`'s 2026-10-14 Task 0015 entry for the full account.

## Motivation

`std::size_t` is used throughout CFE (~98 files, ~1200 occurrences at
the time of this task) for every purpose that happens to share that
type: generic runtime array/loop indexing, buffer sizes, flat-index
results, AND compile-time non-type template parameters (arity/dimension
constants). The raw builtin gives a reader no signal about which of
these a given `std::size_t` is, and the project already has a
precedent for naming a concept instead of using a raw builtin directly
(`cfe::scalar`, `core/types.hpp`).

The user asked for a new type, `cfe::Index`, to be the canonical name
for the *runtime* case specifically: buffer offsets, element counts,
flat-index results, `Backend::run` iteration indices -- i.e. "generic
array/loop indexing" as used today via raw `std::size_t`.

**`cfe::Index` is explicitly NOT a replacement for `local_index`
(`int32_t`, partition-local) or `global_index` (`int64_t`, distributed)
-- those stay exactly as they are** (AGENTS.md #11: independent index
types for local/global indexing). Those two types are also currently
unused anywhere outside `core/types.hpp` itself (grepped before
starting) -- this task does not introduce their first use either; it
only adds the third, general-purpose name alongside them.

## Acceptance criteria

1. `src/cfe/core/types.hpp` gains `using Index = std::size_t;` next to
   `scalar`/`local_index`/`global_index`, with a comment distinguishing
   its purpose from those two.
2. Every `std::size_t` occurrence that denotes a genuine **runtime**
   index/count/offset is renamed to `cfe::Index` (outside `namespace
   cfe`) or bare `Index` (inside it, matching how every `src/cfe/**`
   file is already wrapped in `namespace cfe { ... }` or a nested
   namespace under it).
3. Occurrences that are **compile-time** arity/dimension constants (see
   "What stays `std::size_t`" below) are left untouched, applied
   consistently codebase-wide, not file-by-file.
4. Zero behavior change: same width, same signedness, no new
   overloads, no ABI change. Verified by full rebuild + full test suite
   + byte-identical tutorial output diffs (not just "it's a typedef so
   it must be fine").
5. `docs/type-reference.md` gets a `cfe::Index` entry.
6. Full review trail given the reach of the change: Violet, Snorlax,
   Sally, Enzo, then Ilia and Amelia.

## Scope rules (decided before the mechanical pass, to keep it
mechanical rather than file-by-file judgment)

### What becomes `Index`

- Runtime buffer sizes/counts: `n_cells`, `bytes`, `n_active`,
  `block_cell_counts`, `total_scalars`, `chunk`, `begin`/`end`, etc.
- Loop variables, including ones that iterate up to a compile-time
  bound (e.g. `for (std::size_t c = 0; c < NComponents; ++c)` -- `c`
  becomes `Index` even though the loop bound stays `std::size_t`; see
  below). Rationale: `c` is still a different *value* every call/
  iteration, which is exactly what `Index` denotes; the bound it is
  compared against is a separate, compile-time-fixed quantity.
- Flat-index results used to subscript storage, even when computed by
  a `constexpr` function from a compile-time quantity --
  `Field::momentum_index(Axis)` / `Field::energy_index()`
  (`fields/euler/field.hpp`) and every local copy of their result
  (`normal_component`, `component`, `m_d`'s index operand in
  `numerics/numerical_flux/hllc.hpp`,
  `grid/boundary/slip_wall_boundary.hpp`,
  `grid/boundary/block_interface_boundary.hpp`). These are genuinely
  runtime-capable: `hllc.hpp:235` and `:286` call
  `Field::momentum_index(static_cast<Axis>(d))` with `d` a runtime loop
  variable, not a compile-time literal -- so, unlike `NComponents`/
  `Dim` themselves, the *value* really can vary from call to call. This
  is different from "a `constexpr` function that happens to be callable
  at compile time" (`Dim`/`N` themselves never do); it is included for
  consistency across the `momentum_index`/`energy_index` pair even
  though `energy_index()` alone (`Dim + 1`, no parameter) would also be
  defensible left alone -- splitting the pair was judged worse for
  readability than converting both.
- `std::vector<std::size_t>` / `std::initializer_list<std::size_t>`
  holding runtime-determined counts (`field/block_set.hpp`'s
  `block_cell_counts_`/`scalar_offsets_`, tutorial grid-resolution
  lists like `burgers_2d.cpp`'s `kGrids`).
- `IdentityIndexMap`/custom `IndexMap` callables
  (`solver/time_integration/ssp_rk2.hpp`): the "r-th active cell ->
  storage index" translation is a runtime mapping.
- Backend iteration index parameters (`Backend::run`/`parallel_for`
  callback parameters), once the shadowing template-parameter name
  issue below is resolved.

### What stays `std::size_t` (NOT renamed)

Decided with Snorlax before the mechanical pass (see `agent_history.md`
Task 0015 entry for her full reasoning) and applied uniformly:

- Non-type template parameters that are compile-time arity/dimension
  constants baked into the type itself: `N`/`NComponents`/`Dim` in
  `FixedArray<Scalar, N>`, `Vector<Scalar, Dim>`, `State<Scalar,
  NComponents>` (`math/fixed_array.hpp`), `Field`/`FieldView`/
  `BlockSet`/`DeviceField<Scalar, NComponents, Layout>`, `FvmSolver<Scalar,
  Dim>`, the `Dim` parameter of `{Burgers,Euler,ScalarAdvection}Field`,
  and the `N` parameter threaded through `numerics/numerical_flux/
  {rusanov,hllc}.hpp` and `grid/boundary/{boundary_condition,
  slip_wall_boundary,mpi_halo_boundary}.hpp`/`grid/ghost/ghost_fill.hpp`.
  Rule of thumb applied: any `std::size_t` appearing inside a
  `template <...>` parameter list in this codebase is one of these
  arity constants (confirmed by inspection -- no counterexample found).
- `constexpr`/`static constexpr` members or local copies that directly
  expose one of those same compile-time constants with no runtime
  variation: `Field::n_components()`/`::n_components`, `::dim`,
  `FixedArray::size()`, a local `constexpr std::size_t n_components =
  FieldViewT::n_components();` (`ssp_rk2.hpp`), `constexpr std::size_t N
  = Field::n_components;` (`fvm_solver.hpp`, `tests/verification/
  forward_step_setup.hpp`), `constexpr std::size_t Dim = Field::dim;`
  (`hllc.hpp`). These are "how many", mirroring the NTTP itself, not "which
  position" -- unlike `momentum_index`/`energy_index` above.
- `std::integral_constant<std::size_t, N>` (`core/component_counts.hpp`):
  direct interop with a standard-library type whose first template
  argument must match `N`'s own (unchanged) type.
- `sizeof(...)` expressions that are naturally `std::size_t`-typed and
  immediately consumed, not stored (none found that aren't already
  folded into a `bytes`/similar runtime value, which itself becomes
  `Index` -- e.g. `device_field.cuh`'s `n_cells_ * NComponents *
  sizeof(Scalar)` assigned to `const std::size_t bytes` -> `bytes`
  becomes `Index`, `sizeof(Scalar)` itself is untouched since `sizeof`
  always yields `std::size_t` by language definition).

### Template-parameter name shadowing (Snorlax Q2, resolved before the
mechanical pass)

`backend/parallel_for.hpp`, `backend/cpu/serial.hpp`,
`backend/cpu/threaded.hpp`, `backend/cuda/cuda_backend.cuh`, and three
benchmark structs that mimic the same `Backend::run` shape
(`benchmarks/{burgers,scalar_advection,results/
phase1_ssp_rk2_vectorization_repro}/...`) all declare `template <class
Index, class Functor>` -- using the identifier `Index` for a generic,
deducible type parameter (today instantiated with whatever the caller
passes, typically `std::size_t`). Once `cfe::Index` is a concrete named
type, reusing the exact same spelling for an unrelated generic
parameter is a readability hazard (unlike the `Scalar`/`scalar`
precedent, which disambiguates by case). Renamed to `IndexT` in all
seven places, including the benchmark mimics, for consistency between
production and benchmark code. Pure identifier rename, zero behavior
change.

### Deliberately out of scope (named, not silently papered over)

- `tests/mpi/test_mpi_{burgers_steepening,halo_exchange}.cpp` use
  `std::size_t global_i`/`global_nx` as local variable names for a
  distributed cell index -- a naming *coincidence* with the project's
  existing (currently unused anywhere) `global_index` type. Converting
  these call sites to actually adopt `cfe::global_index` would be a
  real, separate design decision (first real use of that type, with its
  own correctness implications for a 400-/160-cell problem that fits
  comfortably in 32 bits anyway) -- out of scope for a rename pass per
  the task's own framing ("a rename pass, not a redesign"); flagged for
  Enzo/Violet to comment on, not fixed here.

## Verification plan

- Full CPU rebuild, every CMake configuration this repo supports.
- Full CPU unit test suite rerun; expect byte-identical results (pure
  typedef).
- Full MPI test suite rerun.
- Every tutorial rerun, output diffed byte-identical against the
  committed baseline.
- CUDA: no CUDA toolchain available on this machine (same standing
  limitation as the PR #7 cleanup task) -- `.cu`/`.cuh` files are
  inspected by eye for the same mechanical rules, not compiled or run.
  Stated honestly as an open item, not silently skipped.

## Review trail

To be filled in `agent_history.md` as each review actually completes:
Violet (zero behavior change), Snorlax (pre-consulted above on the two
judgment calls; post-diff pass for naming consistency and scope
creep), Sally (tutorial/test readability), Enzo (zero overhead by
construction; narrower-type opportunities papered over by this rename,
if any), Ilia (independent final review), Amelia (alignment with the
`cfe::scalar` precedent).
