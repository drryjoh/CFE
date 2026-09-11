# CMU-CFE Type Reference

A quick-reference map of the core types under `src/cfe/`: what each one
is, where it lives, and what its key members are. This is a living
document — update it whenever a type's public shape changes or a new
one is added; do not let it drift from the code. It is a reference, not
a tutorial: for the reasoning behind a design, see the relevant ADR under
`docs/adr/`.

*Covers through Phase 1's scalar-advection solver (grid, boundary
conditions, interface-value/numerical-flux, SSP-RK2, the
`ScalarAdvectionField`/`FvmSolver` split).*

## `cfe::core` — precision, indexing, host/device macros

| Type / macro | File | What it is |
|---|---|---|
| `cfe::scalar` | `core/types.hpp` | Project-wide default floating-point precision (`float` or `double`), selected at configure time via `CFE_SCALAR_TYPE`. Most code is templated on `Scalar` directly rather than using this alias, so both precisions can be exercised in one build. |
| `cfe::local_index` | `core/types.hpp` | `std::int32_t`. Within-partition/rank indices. |
| `cfe::global_index` | `core/types.hpp` | `std::int64_t`. Indices that may need to address a distributed problem larger than 2^31 elements. |
| `cfe::Axis` | `core/types.hpp` | `{X=0, Y=1, Z=2}`. Lives in `core/` (not `grid/`) so numerics code (`numerics/numerical_flux/upwind.hpp`) can use it without depending on the grid module. Values are stable so `static_cast<std::size_t>(axis)` indexes a per-axis `Vector`/`FixedArray` directly. |
| `cfe::Side` | `core/types.hpp` | `{Low, High}`. |
| `CFE_HOST_DEVICE`, `CFE_DEVICE`, `CFE_HOST`, `CFE_GLOBAL` | `core/macros.hpp` | Expand to `__host__ __device__` / `__device__` / `__host__` / `__global__` when compiled by `nvcc` (`__CUDACC__` defined), and to nothing otherwise. **The only place any code may know whether it's being compiled by nvcc** — physics/numerics code must use these macros, never raw CUDA attributes. |
| `CFE_FORCEINLINE` | `core/macros.hpp` | `__forceinline__` under nvcc, `inline __attribute__((always_inline))` under GCC/Clang, plain `inline` otherwise. |
| `cfe::ComponentCounts` | `core/component_counts.hpp` | `std::index_sequence<1,5,10,20,50,100>` — the required component-count sweep (AGENTS.md #8). |
| `cfe::for_each_component_count(f)` | `core/component_counts.hpp` | Calls `f(std::integral_constant<std::size_t, N>{})` once per required `N`. `f` must be a generic (`auto`-parameter) lambda; if it launches CUDA work, the body must call out to a separate template function rather than define the extended `__device__` lambda directly inside the generic lambda — nvcc rejects that (see `agent_history.md`, 2026-09-08 entry). |

## `cfe::math` — fixed-size containers and operations

| Type / function | File | What it is |
|---|---|---|
| `cfe::FixedArray<Scalar, N>` | `math/fixed_array.hpp` | Allocation-free, trivially-copyable `Scalar[N]`. The one building block for scalars, vectors, and physics states. `operator[]`, `data()`, `+=`/`-=`/`*=`/`/=`. |
| `cfe::ScalarValue<Scalar>` | `math/fixed_array.hpp` | Alias for `FixedArray<Scalar, 1>`. |
| `cfe::Vector<Scalar, Dim>` | `math/fixed_array.hpp` | Alias for `FixedArray<Scalar, Dim>`. |
| `cfe::State<Scalar, NComponents>` | `math/fixed_array.hpp` | Alias for `FixedArray<Scalar, NComponents>` — a full per-cell physics state. |
| `operator+`, `operator-`, `operator*`, `operator/`, `componentwise_multiply` | `math/operations.hpp` | Componentwise arithmetic on `FixedArray`. |
| `cfe::contract(a, b)` | `math/operations.hpp` | Full contraction to a scalar: `sum_k a[k]*b[k]` (generalized dot product). |
| `cfe::weight(a, w)` | `math/operations.hpp` | Componentwise scaling of `a` by `w`; returns a `FixedArray`, not a scalar (unlike `contract`). |

## `cfe::field` — contiguous storage

| Type | File | What it is |
|---|---|---|
| `cfe::AoSLayout`, `cfe::SoALayout` | `field/layout.hpp` | Index policies: `index(cell, component, n_cells, n_components)`. AoS = `cell*n_components+component` (grouped by cell); SoA = `component*n_cells+cell` (grouped by quantity). No common base class — duck-typed, matching `BoundaryCondition`'s style. Per ADR 0002: AoS is the CPU default, SoA the CUDA default (measured, not assumed). |
| `cfe::Field<Scalar, NComponents, Layout=AoSLayout>` | `field/field.hpp` | Owning host-side storage: one `std::vector<Scalar>` sized `n_cells*NComponents`, allocated once in the constructor. `operator()(cell, component)`, `view()`/`view() const`. |
| `cfe::FieldView<Scalar, NComponents, Layout>` | `field/field.hpp` | Non-owning, trivially-copyable pointer + `n_cells`. Safe to capture by value into a `CFE_DEVICE` lambda. Equally valid over host or device memory — this is what makes grid/boundary/numerics code backend-agnostic. |
| `cfe::backend::cuda::DeviceField<Scalar, NComponents, Layout>` | `backend/cuda/device_field.cuh` | Device-memory counterpart to `Field`. `copy_from_host`/`copy_to_host`, `view()` returns the *same* `FieldView` type CPU code uses. `.cu`-only. |

## `cfe::backend` — execution abstraction

| Type / function | File | What it is |
|---|---|---|
| `cfe::parallel_for(n, f)` | `backend/parallel_for.hpp` | Alias for whichever CPU backend is the project default (`CFE_DEFAULT_BACKEND`); the backend-agnostic call site physics/numerics code should use. |
| `cfe::backend::serial::parallel_for(n, f)` | `backend/cpu/serial.hpp` | Plain sequential `for` loop, `f(i)` for `i` in `[0,n)`. |
| `cfe::backend::threaded::parallel_for(n, f, n_threads=0)` | `backend/cpu/threaded.hpp` | Static contiguous chunking across `std::thread`s. No persistent pool — spawns fresh threads every call (documented cost, see ADR 0001). |
| `cfe::backend::cuda::parallel_for(n, f, block_size=256)` | `backend/cuda/cuda_backend.cuh` | One GPU thread per index. **Asynchronous** — returns once the launch is queued, does not wait for completion. `.cu`-only; call explicitly, never through the `cfe::parallel_for` alias. |
| `cfe::backend::cuda::synchronize()` | `backend/cuda/cuda_backend.cuh` | Blocks until all queued device work completes. Required before reading a `parallel_for` result on the host, or before ending a timed region. |
| `CFE_CUDA_CHECK(expr)` | `backend/cuda/cuda_check.cuh` | Throws `std::runtime_error` with file/line/`cudaGetErrorString` on any non-`cudaSuccess` CUDA API result. Every `cudaMalloc`/`cudaMemcpy`/launch-error-check in this codebase goes through this. |

## `cfe::grid` — Cartesian grid, boundary conditions, ghost cells (Phase 1)

| Type / function | File | What it is |
|---|---|---|
| `cfe::CartesianGrid<Scalar>` | `grid/structured/cartesian_grid.hpp` | Describes one block: real cell counts, ghost-layer depth (`std::size_t`), and spacing (`dx`/`dy`/`dz`, type `Scalar`) per dimension, plus interior origin. Templated on `Scalar` (not hardcoded `double`) so a `float`-precision solver carries no hidden conversion for its grid spacing. `flat_index(i,j,k)` is the *only* place `(i,j,k)` becomes the flat `cell` index `Field`/`FieldView` expect — ghost cells live in the same contiguous array as real cells, at padded indices outside `[n_ghost, n_ghost+n_cells)`. Spacing/extents are scoped to this one object (not global) so a second block at a different resolution — fixed-block AMR — doesn't require redesigning this type. |
| `cfe::PeriodicBoundary` | `grid/boundary/boundary_condition.hpp` | Wraps one axis's ghost layer from the opposite real boundary. `fill_x`/`fill_y`/`fill_z`, each dispatched via `cfe::parallel_for`. Stateless. |
| `cfe::StaticBoundary<Scalar, N>` | `grid/boundary/boundary_condition.hpp` | Writes a fixed `State<Scalar,N>` into every ghost cell on the low/high side of one axis (the two sides may differ). Same `fill_x`/`fill_y`/`fill_z` shape as `PeriodicBoundary` — duck-typed, no common base (AGENTS.md #12: no virtual functions inside kernels). |
| `cfe::fill_ghost_cells(field, grid, axis, boundary)` | `grid/ghost/ghost_fill.hpp` | The single call site solver code uses to fill ghost cells. This is the actual swappable seam: solver code never touches `(i,j,k)±1` indexing directly, so a future MPI halo-exchange or coarse-fine-AMR-interpolation provider is a new `boundary` type at this same call shape, not a redesign. |

## `cfe::numerics` — interface reconstruction and flux combinator (Phase 1)

Physics-agnostic by design: neither of these knows what PDE it's being used
for. The physics lives in `cfe::fields` instead (below).

| Type / function | File | What it is |
|---|---|---|
| `cfe::fvm::interface_value_right(q_left, q_self, q_right)` / `cfe::fvm::interface_value_left(...)` | `numerics/fvm/interface_value.hpp` | This cell's own value at its `+axis`/`-axis` face: central-difference linear extrapolation (`q_self ± (q_right - q_left)/4`), using only its immediate ("1-ring") neighbors. Together, the FVM implementation of "produce my value at a face from my own representation" — a future DG element could implement the same shape from its own internal DOFs instead. |
| `cfe::fvm::CentralDifferenceReconstruction` | `numerics/fvm/interface_value.hpp` | Stateless functor wrapping the two functions above into the `.right(...)`/`.left(...)` shape `FvmSolver` is generic over (its default `Reconstruction`). A future MUSCL/PPM/WENO reconstruction is a new type with this same shape. |
| `cfe::upwind_flux(left_value, right_value, axis, field)` / `cfe::UpwindFlux` | `numerics/numerical_flux/upwind.hpp` | Combines two independently-computed face values into one flux by upwind selection on the sign of `field.wave_speed(left_value, right_value, axis)`, then evaluates `field.physical_flux(...)` on the upwind side. Never knows what those two Calculator methods actually compute — that's `Field`'s job. `UpwindFlux` is the functor form (`FvmSolver`'s default `NumericalFlux`); a future Rusanov/HLLC/AUSM flux is a new type with the same `operator()(left, right, axis, field)` shape. |

## `cfe::fields` — the physics (Phase 1)

Where the physics-specific Calculator functions actually live (ARCHITECTURE.md
#2's "Field" concept), kept separate from the physics-agnostic numerics above.

| Type | File | What it is |
|---|---|---|
| `cfe::ScalarAdvectionField<Scalar, Dim>` | `fields/scalar_advection/field.hpp` | The linear scalar-advection equation `dQ/dt + velocity·grad(Q) = 0`. Holds a `Vector<Scalar, Dim>` velocity (one component per active axis — a 1D problem uses `Dim=1`, 2D uses `Dim=2`, etc.) and provides the two Calculator methods `NumericalFlux` needs: `physical_flux(q, axis) -> velocity[axis]*q` and `wave_speed(q_left, q_right, axis) -> velocity[axis]`. Exposes `static constexpr std::size_t dim = Dim` so `FvmSolver` can select 1D/2D/3D behavior with `if constexpr` at compile time rather than a per-cell runtime check. A future Burgers/Euler field supplies its own Calculator methods under this same `(state, axis)`-in shape. |

## `cfe::solver` — orchestration, no physics (Phase 1)

| Type / function | File | What it is |
|---|---|---|
| `cfe::ssp_rk2_step<Scalar>(q, stage1, residual_scratch, dt, residual)` | `solver/time_integration/ssp_rk2.hpp` | Advances `q` in place by one SSP-RK2 (Heun's method) step. Generic over a `residual(q_in, out)` callable computing `out := dQ/dt`; has no knowledge of grids or boundary conditions. `stage1`/`residual_scratch` are caller-allocated scratch storage of the same shape as `q`, reused every call. Chosen over SSP-RK3 because the paired spatial scheme is 2nd-order. |
| `cfe::FvmSolver<Scalar, Layout, Field, BoundaryX, BoundaryY=BoundaryX, BoundaryZ=BoundaryX, Reconstruction=fvm::CentralDifferenceReconstruction, NumericalFlux=UpwindFlux>` | `solver/explicit/fvm_solver.hpp` | Assembles a flux-form residual for a single-component conservation law. Dimension-agnostic: the residual loop and ghost-fill both run over however many axes are active, decided at compile time from `Field::dim` (`if constexpr`, not a per-cell runtime branch). Contains no physics itself — only orchestrates `Field` (physics), `Reconstruction`+`NumericalFlux` (numerics), and `BoundaryX`/`Y`/`Z` (ghost-cell filling), calling only their public interfaces. Requires `grid.ngx/ngy/ngz >= 2` on every active axis (each cell's residual needs both its faces, and each face's two one-sided reconstructions each reach one cell further out). |

## Test framework

| Macro | File | What it is |
|---|---|---|
| `CFE_TEST(name)` | `tests/unit/test_framework.hpp` | Declares and registers a test function. |
| `CFE_CHECK(condition)` | `tests/unit/test_framework.hpp` | Throws `AssertionFailure` (caught by the runner, printed as `[FAIL]`) if `condition` is false. |
| `CFE_CHECK_NEAR(a, b, tol)` | `tests/unit/test_framework.hpp` | Same, for `|a-b| > tol`. See `agent_history.md` (2026-09-08) for a real gotcha: don't assert bitwise equality between two *independently recomputed* values of the same expression — GCC's `-ffp-contract=fast` can make that fail even when both sides are individually correctly rounded. Use a real tolerance for that comparison; reserve exact `==` for values that must be bitwise identical by construction (e.g. two backends' outputs from the same input).
