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
CFE_HOST_DEVICE Scalar axis_flux_difference(FieldViewT state, const CartesianGrid<Scalar>& grid,
                                            std::size_t i, std::size_t j, std::size_t k,
                                            const Reconstruction& reconstruction,
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

// Writes 0 into `out`'s ghost-cell residual entries on one axis (both
// sides), touching only the actual ghost layer -- not the whole padded
// volume. Mirrors PeriodicBoundary::fill_x/y/z's own index enumeration
// (grid/boundary/boundary_condition.hpp) minus the "read from a source
// cell" part, since here every ghost cell just gets the same constant.
// Deliberately NOT a single pass over `grid.n_cells_total()`: measured
// on real hardware, that naive approach re-writes every *real* cell
// twice (once as 0, then again with its actual flux value), roughly
// doubling `out`'s write bandwidth at the large grid sizes this project
// benchmarks at -- a real, measured regression (caught in code review;
// see agent_history.md and docs/performance/0004-...), not a
// theoretical concern traded away for simplicity.
template <class Scalar, class FieldViewT, class Backend>
void zero_ghost_residual_x(FieldViewT out, const CartesianGrid<Scalar>& grid)
{
  if (grid.ngx == 0) return;
  const std::size_t py = grid.padded_ny();
  const std::size_t pz = grid.padded_nz();
  Backend::run(grid.ngx * py * pz, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
    const std::size_t g = idx % grid.ngx;
    const std::size_t rem = idx / grid.ngx;
    const std::size_t j = rem % py;
    const std::size_t k = rem / py;
    out(grid.flat_index(g, j, k), 0) = Scalar(0);
    out(grid.flat_index(grid.ngx + grid.nx + g, j, k), 0) = Scalar(0);
  });
}

template <class Scalar, class FieldViewT, class Backend>
void zero_ghost_residual_y(FieldViewT out, const CartesianGrid<Scalar>& grid)
{
  if (grid.ngy == 0) return;
  const std::size_t px = grid.padded_nx();
  const std::size_t pz = grid.padded_nz();
  Backend::run(grid.ngy * px * pz, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
    const std::size_t g = idx % grid.ngy;
    const std::size_t rem = idx / grid.ngy;
    const std::size_t i = rem % px;
    const std::size_t k = rem / px;
    out(grid.flat_index(i, g, k), 0) = Scalar(0);
    out(grid.flat_index(i, grid.ngy + grid.ny + g, k), 0) = Scalar(0);
  });
}

template <class Scalar, class FieldViewT, class Backend>
void zero_ghost_residual_z(FieldViewT out, const CartesianGrid<Scalar>& grid)
{
  if (grid.ngz == 0) return;
  const std::size_t px = grid.padded_nx();
  const std::size_t py = grid.padded_ny();
  Backend::run(grid.ngz * px * py, [=] CFE_HOST_DEVICE(std::size_t idx) mutable {
    const std::size_t g = idx % grid.ngz;
    const std::size_t rem = idx / grid.ngz;
    const std::size_t i = rem % px;
    const std::size_t j = rem / px;
    out(grid.flat_index(i, j, g), 0) = Scalar(0);
    out(grid.flat_index(i, j, grid.ngz + grid.nz + g), 0) = Scalar(0);
  });
}

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

    // `out` must be fully defined for every *padded* cell, ghost cells
    // included, not just real ones: `ssp_rk2_step`'s combine kernels
    // iterate every cell in `state`'s full storage with no knowledge of
    // which indices are real vs. ghost (see ssp_rk2.hpp -- it is
    // deliberately grid-agnostic). Ghost-cell residual values are
    // physically meaningless (ghost cells are never integrated in time;
    // the next fill_ghost_cells call overwrites them from the interior
    // regardless), but they must be *some* defined value, not whatever
    // the caller's scratch storage happened to contain: on CUDA,
    // freshly cudaMalloc'd storage is uninitialized, and writing only
    // real cells here would leave ghost-cell residual entries
    // permanently unwritten for the process's entire lifetime -- a
    // genuine undefined-behavior read, flagged by `compute-sanitizer
    // --tool initcheck` (caught in code review; see agent_history.md).
    // Zeroing exactly the ghost cells (not the whole padded volume --
    // see zero_ghost_residual_x/y/z's own doc comment for why that
    // matters) keeps this to the same thin-shell cost as one of the
    // fill_ghost_cells calls above.
    detail::zero_ghost_residual_x<Scalar, FieldView<Scalar, 1, Layout>, Backend>(out, grid);
    if constexpr (y_active) {
      detail::zero_ghost_residual_y<Scalar, FieldView<Scalar, 1, Layout>, Backend>(out, grid);
    }
    if constexpr (z_active) {
      detail::zero_ghost_residual_z<Scalar, FieldView<Scalar, 1, Layout>, Backend>(out, grid);
    }

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
