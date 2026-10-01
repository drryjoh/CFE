// Generic explicit FVM solver (task spec item 5, ARCHITECTURE.md #2:
// "Solver" orchestrates Grid + Field(s) + Numerics + Time Integrator; it
// contains no physics itself). Assembles a flux-form residual for a
// single-component conservation law dQ/dt + div(F(Q)) = 0 from:
//   - `Field` -- the physics (Phase 1's linear scalar advection; a
//     future Burgers field, also single-component, plugs in here
//     unchanged -- Euler would NOT: this type hardcodes
//     `FieldView<Scalar, 1, Layout>` and reads/writes component `0`
//     only throughout, so a genuinely multi-component state (Euler's
//     density/momentum/energy) needs this solver generalized over
//     `NComponents` first, not just a new `Field`);
//   - `Reconstruction` -- per-side face-value reconstruction
//     (numerics/fvm/interface_value.hpp);
//   - `NumericalFlux` -- combines two face values into one flux, using
//     only Field's Calculator methods (numerics/numerical_flux/upwind.hpp);
//   - `BoundaryX`/`BoundaryY`/`BoundaryZ` -- ghost-cell filling per axis
//     (grid/boundary/boundary_condition.hpp).
// None of these know about each other's internals -- this type is the
// only place they're wired together, and only at the granularity of
// calling each one's public interface.
//
// Dimension-agnostic: the residual loop runs over however many of X/Y/Z
// the grid actually has active (`Field::dim`, `if constexpr`, no runtime
// branch) -- see ADR 0004 and docs/type-reference.md for the design
// rationale, not repeated here.
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
#include "cfe/field/field.hpp"
#include "cfe/grid/ghost/ghost_fill.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/numerics/fvm/interface_value.hpp"
#include "cfe/numerics/numerical_flux/upwind.hpp"

namespace cfe {

namespace detail {

// The one-axis flux-form residual contribution, identical in structure
// for X/Y/Z -- only which neighbor is "+-1"/"+-2" along `Axis A` differs,
// resolved at compile time (`if constexpr`) so this fully inlines with no
// runtime branching per axis, exactly as if each axis's version had been
// written out by hand.
template <Axis A, class Scalar, class FieldViewT, class Reconstruction, class NumericalFlux,
          class Field>
CFE_HOST_DEVICE
Scalar axis_flux_difference(FieldViewT state, const CartesianGrid<Scalar>& grid, std::size_t i,
                            std::size_t j, std::size_t k, const Reconstruction& reconstruction,
                            const NumericalFlux& numerical_flux, const Field& field)
{
  Scalar state_m2, state_m1, state_c, state_p1, state_p2, spacing;
  if constexpr (A == Axis::X) {
    state_m2 = state(grid.flat_index(i - 2, j, k), 0);
    state_m1 = state(grid.flat_index(i - 1, j, k), 0);
    state_c = state(grid.flat_index(i, j, k), 0);
    state_p1 = state(grid.flat_index(i + 1, j, k), 0);
    state_p2 = state(grid.flat_index(i + 2, j, k), 0);
    spacing = grid.dx;
  } else if constexpr (A == Axis::Y) {
    state_m2 = state(grid.flat_index(i, j - 2, k), 0);
    state_m1 = state(grid.flat_index(i, j - 1, k), 0);
    state_c = state(grid.flat_index(i, j, k), 0);
    state_p1 = state(grid.flat_index(i, j + 1, k), 0);
    state_p2 = state(grid.flat_index(i, j + 2, k), 0);
    spacing = grid.dy;
  } else {
    state_m2 = state(grid.flat_index(i, j, k - 2), 0);
    state_m1 = state(grid.flat_index(i, j, k - 1), 0);
    state_c = state(grid.flat_index(i, j, k), 0);
    state_p1 = state(grid.flat_index(i, j, k + 1), 0);
    state_p2 = state(grid.flat_index(i, j, k + 2), 0);
    spacing = grid.dz;
  }

  // Right face: left value is this cell's own reconstruction, right value
  // is the next cell's own reconstruction.
  const Scalar state_left_of_right_face = reconstruction.right(state_m1, state_c, state_p1);
  const Scalar state_right_of_right_face = reconstruction.left(state_c, state_p1, state_p2);
  const Scalar flux_right = numerical_flux(state_left_of_right_face, state_right_of_right_face, A, field);

  // Left face: left value is the previous cell's own reconstruction,
  // right value is this cell's own reconstruction.
  const Scalar state_left_of_left_face = reconstruction.right(state_m2, state_m1, state_c);
  const Scalar state_right_of_left_face = reconstruction.left(state_m1, state_c, state_p1);
  const Scalar flux_left = numerical_flux(state_left_of_left_face, state_right_of_left_face, A, field);

  return -(flux_right - flux_left) / spacing;
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
  std::size_t operator()(std::size_t r) const
  {
    if constexpr (Dim == 1) {
      return grid.flat_index(grid.ngx + r, grid.ngy, grid.ngz);
    } else if constexpr (Dim == 2) {
      const std::size_t local_i = r % grid.nx;
      const std::size_t local_j = r / grid.nx;
      return grid.flat_index(grid.ngx + local_i, grid.ngy + local_j, grid.ngz);
    } else {
      const std::size_t local_i = r % grid.nx;
      const std::size_t local_j = (r / grid.nx) % grid.ny;
      const std::size_t local_k = r / (grid.nx * grid.ny);
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

  void residual(FieldView<Scalar, 1, Layout> state, FieldView<Scalar, 1, Layout> out) const
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

    fill_ghost_cells<Scalar, 1, Layout, BoundaryX, Backend>(state, grid, Axis::X, boundary_x);
    if constexpr (y_active) {
      fill_ghost_cells<Scalar, 1, Layout, BoundaryY, Backend>(state, grid, Axis::Y, boundary_y);
    }
    if constexpr (z_active) {
      fill_ghost_cells<Scalar, 1, Layout, BoundaryZ, Backend>(state, grid, Axis::Z, boundary_z);
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
    Backend::run(grid.nx * grid.ny * grid.nz, [=] CFE_HOST_DEVICE(std::size_t linear) mutable {
      const std::size_t local_i = linear % grid.nx;
      const std::size_t local_j = (linear / grid.nx) % grid.ny;
      const std::size_t local_k = linear / (grid.nx * grid.ny);

      const std::size_t i = grid.ngx + local_i;
      const std::size_t j = grid.ngy + local_j;
      const std::size_t k = grid.ngz + local_k;

      Scalar total_flux_difference = detail::axis_flux_difference<Axis::X>(
          state, grid, i, j, k, reconstruction, numerical_flux, field);

      if constexpr (y_active) {
        total_flux_difference += detail::axis_flux_difference<Axis::Y>(
            state, grid, i, j, k, reconstruction, numerical_flux, field);
      }
      if constexpr (z_active) {
        total_flux_difference += detail::axis_flux_difference<Axis::Z>(
            state, grid, i, j, k, reconstruction, numerical_flux, field);
      }

      out(grid.flat_index(i, j, k), 0) = total_flux_difference;
    });
  }

  // How many real cells `ssp_rk2_step` should integrate, and the
  // storage-index mapping it should use to reach them -- pass both as
  // `ssp_rk2_step`'s trailing arguments so its combine kernels visit
  // only real cells, never ghost cells. See detail::CartesianRealCellIndexMap's
  // own doc comment above for why a plain cell count alone isn't enough
  // on a 2D/3D grid.
  std::size_t active_cell_count() const { return grid.nx * grid.ny * grid.nz; }
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
