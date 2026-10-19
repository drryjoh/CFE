// Generic explicit FVM solver (task spec item 5, ARCHITECTURE.md #2:
// "Solver" orchestrates Grid + Field(s) + Numerics + Time Integrator; it
// contains no physics itself). Assembles a flux-form residual for a
// conservation law dQ/dt + div(F(Q)) = 0, generic over
// `Field::n_components` (Phase 3: generalized from a hardcoded
// single-component assumption -- see fields/scalar_advection/field.hpp's
// and tasks/0003-...md's own comments flagging this as a prerequisite
// for a multi-component field like `EulerField`, fields/euler/field.hpp)
// from:
//   - `Field` -- the physics. N=1 fields (`ScalarAdvectionField`,
//     `BurgersField`) use a `Scalar`-in/`Scalar`-out Calculator shape;
//     N>1 fields (`EulerField`) use a `State<Scalar,N>`-in/-out shape,
//     since a system's flux for any one component genuinely depends on
//     the whole state. `detail::axis_flux_difference` below dispatches
//     between the two shapes via `if constexpr (N == 1)` -- zero
//     behavior change for existing N=1 fields (same calls, same
//     formula, just routed through a length-1 `State` on the way in/out).
//   - `Reconstruction` -- per-side, per-component face-value
//     reconstruction (numerics/fvm/interface_value.hpp,
//     numerics/fvm/muscl_minmod.hpp). Always scalar-in/-out, applied
//     independently to each conserved component ("componentwise"
//     reconstruction, not characteristic reconstruction) -- this is
//     unchanged from N=1 usage, just looped over components for N>1.
//   - `NumericalFlux` -- combines two face STATEs into one flux STATE,
//     using only Field's Calculator methods
//     (numerics/numerical_flux/upwind.hpp, rusanov.hpp);
//   - `BoundaryX`/`BoundaryY`/`BoundaryZ` -- ghost-cell filling per axis
//     (grid/boundary/boundary_condition.hpp).
// None of these know about each other's internals -- this type is the
// only place they're wired together, and only at the granularity of
// calling each one's public interface.
//
// Dimension-agnostic: the residual loop runs over however many of X/Y/Z
// the grid actually has active (`Field::dim`, `if constexpr`, no runtime
// branch) -- see ADR 0004 and docs/type-reference.md for the design
// rationale, not repeated here. Component count (`Field::n_components`)
// is an orthogonal compile-time axis of genericity -- see ADR 0010.
//
// Stencil requirement: computing a cell's full residual along one axis needs
// both its faces on that axis, and each face's two one-sided
// reconstructions each reach one cell further out -- so the full
// footprint along an active axis spans i-2..i+2. This requires **two**
// ghost layers on every active axis, even though each individual
// reconstruction call only ever reads its own immediate ("1-ring")
// neighbor.
#pragma once

#include <cassert>
#include <cstddef>

#include "cfe/backend/parallel_for.hpp"
#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/grid/ghost/ghost_fill.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/math/fixed_array.hpp"
#include "cfe/numerics/fvm/interface_value.hpp"
#include "cfe/numerics/numerical_flux/upwind.hpp"

namespace cfe {

namespace detail {

// The one-axis flux-form residual contribution, identical in structure
// for X/Y/Z -- only which neighbor is "+-1"/"+-2" along `Axis A` differs,
// resolved at compile time (`if constexpr`) so this fully inlines with no
// runtime branching per axis, exactly as if each axis's version had been
// written out by hand.
//
// Returns a `State<Scalar, Field::n_components>` (every component's
// residual contribution along this one axis at once) -- for N=1 fields
// this is a length-1 State wrapping exactly the same scalar value the
// pre-generalization version of this function returned directly.
template <Axis A, class Scalar, class FieldViewT, class Reconstruction, class NumericalFlux,
          class Field>
CFE_HOST_DEVICE
State<Scalar, Field::n_components> axis_flux_difference(FieldViewT state, const CartesianGrid<Scalar>& grid,
                                                          Index i, Index j, Index k,
                                                          const Reconstruction& reconstruction,
                                                          const NumericalFlux& numerical_flux,
                                                          const Field& field)
{
  constexpr std::size_t N = Field::n_components;

  Index cell_m2, cell_m1, cell_c, cell_p1, cell_p2;
  Scalar spacing;
  if constexpr (A == Axis::X) {
    cell_m2 = grid.flat_index(i - 2, j, k);
    cell_m1 = grid.flat_index(i - 1, j, k);
    cell_c = grid.flat_index(i, j, k);
    cell_p1 = grid.flat_index(i + 1, j, k);
    cell_p2 = grid.flat_index(i + 2, j, k);
    spacing = grid.dx;
  } else if constexpr (A == Axis::Y) {
    cell_m2 = grid.flat_index(i, j - 2, k);
    cell_m1 = grid.flat_index(i, j - 1, k);
    cell_c = grid.flat_index(i, j, k);
    cell_p1 = grid.flat_index(i, j + 1, k);
    cell_p2 = grid.flat_index(i, j + 2, k);
    spacing = grid.dy;
  } else {
    cell_m2 = grid.flat_index(i, j, k - 2);
    cell_m1 = grid.flat_index(i, j, k - 1);
    cell_c = grid.flat_index(i, j, k);
    cell_p1 = grid.flat_index(i, j, k + 1);
    cell_p2 = grid.flat_index(i, j, k + 2);
    spacing = grid.dz;
  }

  // Read this axis's 5-point stencil, every component, into local
  // State vectors -- the componentwise reconstruction loop below reads
  // each component's own 5 raw values out of these.
  State<Scalar, N> state_m2, state_m1, state_c, state_p1, state_p2;
  for (Index c = 0; c < N; ++c) {
    state_m2[c] = state(cell_m2, c);
    state_m1[c] = state(cell_m1, c);
    state_c[c] = state(cell_c, c);
    state_p1[c] = state(cell_p1, c);
    state_p2[c] = state(cell_p2, c);
  }

  // Componentwise reconstruction: each conserved component gets its own
  // independent 3-point `Reconstruction::right/left` call, exactly the
  // same call every N=1 field already made -- just looped over
  // components. This is NOT characteristic reconstruction (no
  // eigenvector projection); see ADR 0010 for why that choice is
  // acceptable for this slice.
  //
  // Right face: left value is this cell's own reconstruction, right
  // value is the next cell's own reconstruction.
  // Left face: left value is the previous cell's own reconstruction,
  // right value is this cell's own reconstruction.
  State<Scalar, N> state_left_of_right_face, state_right_of_right_face, state_left_of_left_face,
      state_right_of_left_face;
  for (Index c = 0; c < N; ++c) {
    state_left_of_right_face[c] = reconstruction.right(state_m1[c], state_c[c], state_p1[c]);
    state_right_of_right_face[c] = reconstruction.left(state_c[c], state_p1[c], state_p2[c]);
    state_left_of_left_face[c] = reconstruction.right(state_m2[c], state_m1[c], state_c[c]);
    state_right_of_left_face[c] = reconstruction.left(state_m1[c], state_c[c], state_p1[c]);
  }

  State<Scalar, N> result;
  if constexpr (N == 1) {
    // Legacy scalar-Calculator shape (`ScalarAdvectionField`,
    // `BurgersField`): call `numerical_flux`/`field` with bare `Scalar`
    // arguments, exactly as before generalization -- bit-identical
    // formula and call sequence to the pre-Phase-3 version of this
    // function.
    const Scalar flux_right =
        numerical_flux(state_left_of_right_face[0], state_right_of_right_face[0], A, field);
    const Scalar flux_left =
        numerical_flux(state_left_of_left_face[0], state_right_of_left_face[0], A, field);
    result[0] = -(flux_right - flux_left) / spacing;
  } else {
    // Vector (system) Calculator shape (`EulerField`): `numerical_flux`
    // takes/returns full `State<Scalar, N>` vectors, calling `field`'s
    // vector-shaped `physical_flux`/`wave_speed` exactly once per face
    // (not once per component) -- see numerics/numerical_flux/rusanov.hpp.
    const State<Scalar, N> flux_right =
        numerical_flux(state_left_of_right_face, state_right_of_right_face, A, field);
    const State<Scalar, N> flux_left =
        numerical_flux(state_left_of_left_face, state_right_of_left_face, A, field);
    for (Index c = 0; c < N; ++c) {
      result[c] = -(flux_right[c] - flux_left[c]) / spacing;
    }
  }
  return result;
}

// Maps a dense "active cell number" r in [0, nx*ny*nz) to the padded
// flat storage index real cells actually live at. This is the mechanism
// that lets `ssp_rk2_step` (solver/time_integration/ssp_rk2.hpp) update
// only real cells, never ghost cells: ghost layers interleave with real
// cells on every active axis for a 2D/3D grid, so "the first N storage
// indices" is not the same set as "the N real cells" except by
// coincidence in 1D -- this performs the same local_i/j/k decomposition
// `FvmSolver::residual()`'s own kernel below already does, factored out
// so `ssp_rk2_step` can reuse it without knowing what a `CartesianGrid`
// is (it only ever sees this as an opaque callable).
//
// Templated on `Dim` (matching `Field::dim`, resolved via `if constexpr`,
// not a runtime branch -- the same pattern `FvmSolver::y_active`/
// `z_active` already use) specifically to avoid paying for integer
// division/modulo by axes that do not exist: a naive always-3D
// decomposition (`r % nx`, `(r/nx) % ny`, `r/(nx*ny)`) divides and mods
// by runtime values on *every* call regardless of dimensionality, which
// measured as a severe regression on the 1D benchmark case specifically
// (where the correct answer is just `ngx + r`, a single addition) --
// caught in code review; see agent_history.md.
template <class Scalar, std::size_t Dim>
struct CartesianRealCellIndexMap
{
  CartesianGrid<Scalar> grid;

  CFE_HOST_DEVICE
  CFE_FORCEINLINE
  Index operator()(Index r) const
  {
    if constexpr (Dim == 1) {
      return grid.flat_index(grid.ngx + r, grid.ngy, grid.ngz);
    } else if constexpr (Dim == 2) {
      const Index local_i = r % grid.nx;
      const Index local_j = r / grid.nx;
      return grid.flat_index(grid.ngx + local_i, grid.ngy + local_j, grid.ngz);
    } else {
      const Index local_i = r % grid.nx;
      const Index local_j = (r / grid.nx) % grid.ny;
      const Index local_k = r / (grid.nx * grid.ny);
      return grid.flat_index(grid.ngx + local_i, grid.ngy + local_j, grid.ngz + local_k);
    }
  }
};

}  // namespace detail

template <class Scalar, class Layout, class Field, class BoundaryX, class BoundaryY = BoundaryX,
          class BoundaryZ = BoundaryX, class Reconstruction = fvm::CentralDifferenceReconstruction,
          class NumericalFlux = UpwindFlux, class Backend = CpuParallelFor>
struct FvmSolver
{
  CartesianGrid<Scalar> grid;
  Field field;
  BoundaryX boundary_x;
  BoundaryY boundary_y{};
  BoundaryZ boundary_z{};
  Reconstruction reconstruction{};
  NumericalFlux numerical_flux{};

  // Computes out := dQ/dt for every real cell. Fills ghost cells on every
  // active axis in-place first, so `state` must be mutable storage, not a
  // read-only view.
  // Which axes are active is a property of `Field::dim` (a compile-time
  // constant, see fields/scalar_advection/field.hpp), not a runtime grid
  // check -- so the branches below are `if constexpr`, resolved once at
  // compile time rather than once per cell/thread. `grid.ngy`/`grid.ngz`
  // must still agree with `Field::dim` at runtime (asserted below); the
  // compile-time flag is what a 1D problem's dimensionality actually is,
  // and the grid is expected to match it.
  static constexpr bool y_active = Field::dim >= 2;
  static constexpr bool z_active = Field::dim >= 3;
  static constexpr std::size_t n_components = Field::n_components;

  void residual(FieldView<Scalar, n_components, Layout> state, FieldView<Scalar, n_components, Layout> out) const
  {
    assert(grid.nx > 0 && grid.dx > Scalar(0) && "FvmSolver requires a positive extent and spacing on X");
    assert(grid.ngx >= 2 && "FvmSolver needs at least 2 ghost layers on every active axis");
    if constexpr (y_active) {
      assert(grid.ny > 0 && grid.dy > Scalar(0) &&
             "FvmSolver requires a positive extent and spacing on Y when Field::dim >= 2");
      assert(grid.ngy >= 2 && "FvmSolver needs at least 2 ghost layers on every active axis");
    } else {
      assert(grid.ny == 1 && grid.ngy == 0 &&
             "An inactive Y axis (Field::dim < 2) must have ny=1, ngy=0 -- CartesianGrid's own "
             "documented shape for an unused dimension");
    }
    if constexpr (z_active) {
      assert(grid.nz > 0 && grid.dz > Scalar(0) &&
             "FvmSolver requires a positive extent and spacing on Z when Field::dim >= 3");
      assert(grid.ngz >= 2 && "FvmSolver needs at least 2 ghost layers on every active axis");
    } else {
      assert(grid.nz == 1 && grid.ngz == 0 &&
             "An inactive Z axis (Field::dim < 3) must have nz=1, ngz=0 -- CartesianGrid's own "
             "documented shape for an unused dimension");
    }

    fill_ghost_cells<Scalar, n_components, Layout, BoundaryX, Backend>(state, grid, Axis::X, boundary_x);
    if constexpr (y_active) {
      fill_ghost_cells<Scalar, n_components, Layout, BoundaryY, Backend>(state, grid, Axis::Y, boundary_y);
    }
    if constexpr (z_active) {
      fill_ghost_cells<Scalar, n_components, Layout, BoundaryZ, Backend>(state, grid, Axis::Z, boundary_z);
    }

    const CartesianGrid<Scalar> grid = this->grid;
    // Copied into locals (rather than capturing `this`) so the lambda
    // below holds self-contained, trivially-copyable state -- capturing
    // `this` would capture a host pointer, which breaks once this same
    // residual() body is reused from a CUDA translation unit later.
    const Field field = this->field;
    const Reconstruction reconstruction = this->reconstruction;
    const NumericalFlux numerical_flux = this->numerical_flux;

    // `out` is only ever written for real cells here -- never ghost
    // cells, and intentionally so: `ssp_rk2_step`'s combine kernels only
    // ever visit real cells too (via `active_cell_index_map()` below),
    // so a ghost-cell entry of `out` is never read by anything. An
    // earlier version of this function wrote a defined placeholder into
    // every ghost cell of `out` so a then-grid-agnostic `ssp_rk2_step`
    // (which used to visit every padded cell indiscriminately) would
    // never read undefined memory -- that was a real, measured
    // performance cost for a value nothing needed (caught in code
    // review; see agent_history.md). Fixed at the actual source instead:
    // `ssp_rk2_step` no longer visits ghost cells at all.
    Backend::run(grid.nx * grid.ny * grid.nz, [=] CFE_HOST_DEVICE(Index linear) mutable {
      const Index local_i = linear % grid.nx;
      const Index local_j = (linear / grid.nx) % grid.ny;
      const Index local_k = linear / (grid.nx * grid.ny);

      const Index i = grid.ngx + local_i;
      const Index j = grid.ngy + local_j;
      const Index k = grid.ngz + local_k;

      State<Scalar, n_components> total_flux_difference = detail::axis_flux_difference<Axis::X>(
          state, grid, i, j, k, reconstruction, numerical_flux, field);

      if constexpr (y_active) {
        total_flux_difference += detail::axis_flux_difference<Axis::Y>(
            state, grid, i, j, k, reconstruction, numerical_flux, field);
      }
      if constexpr (z_active) {
        total_flux_difference += detail::axis_flux_difference<Axis::Z>(
            state, grid, i, j, k, reconstruction, numerical_flux, field);
      }

      const Index cell = grid.flat_index(i, j, k);
      for (Index c = 0; c < n_components; ++c) {
        out(cell, c) = total_flux_difference[c];
      }
    });
  }

  // How many real cells `ssp_rk2_step` should integrate, and the
  // storage-index mapping it should use to reach them -- pass both as
  // `ssp_rk2_step`'s trailing arguments so its combine kernels visit
  // only real cells, never ghost cells. See detail::CartesianRealCellIndexMap's
  // own doc comment above for why a plain cell count alone isn't enough
  // on a 2D/3D grid.
  Index active_cell_count() const { return grid.nx * grid.ny * grid.nz; }
  detail::CartesianRealCellIndexMap<Scalar, Field::dim> active_cell_index_map() const
  {
    return detail::CartesianRealCellIndexMap<Scalar, Field::dim>{grid};
  }
};

// A named (namespace-scope) functor wrapping `Solver::residual` as a
// `residual(state_in, out)` callable for `ssp_rk2_step`. Required specifically
// for CUDA: nvcc forbids passing a *locally-defined* lambda (a closure
// type local to a function, e.g. `auto residual = [&](...){...}` written
// inside `main()` or a test body) as a template argument to a function
// whose body contains an extended `__device__` lambda -- which
// `ssp_rk2_step` always does, in any `.cu` translation unit, regardless
// of which `Backend` a given call site actually selects. A named functor
// at namespace scope is not a local type, so it sidesteps the
// restriction. CPU-only call sites may still use a local lambda if they
// are compiled as plain `.cpp` (not `.cu`); use this whenever the same
// call site must also work compiled by nvcc.
template <class Solver>
struct SolverResidual
{
  const Solver* solver;

  template <class FieldViewT>
  void operator()(FieldViewT in, FieldViewT out) const
  {
    solver->residual(in, out);
  }
};

}  // namespace cfe
