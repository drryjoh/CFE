// Generic explicit FVM solver (task spec item 5, ARCHITECTURE.md #2:
// "Solver" orchestrates Grid + Field(s) + Numerics + Time Integrator; it
// contains no physics itself). Assembles a flux-form residual for a
// single-component conservation law dQ/dt + div(F(Q)) = 0 from:
//   - `Field` -- the physics (task spec item 5's linear scalar advection
//     for Phase 1; a future Burgers/Euler field plugs in here unchanged);
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
// Dimension-agnostic, matching CartesianGrid itself: the residual loop
// runs over however many of X/Y/Z the grid actually has (an axis with
// `n_ghost == 0` is inactive and contributes no flux term), so a 1D, 2D,
// or 3D grid all use this same type. `Field` is equally dimension-generic
// on the physics side: its Calculator methods take the `Axis` the solver
// is currently assembling a flux for (see
// numerics/numerical_flux/upwind.hpp), so ScalarAdvectionField
// (fields/scalar_advection/field.hpp) holds a `Vector<Scalar, Dim>`
// velocity -- one component per active axis -- rather than a single
// scalar speed. A 1D, 2D, or 3D transport problem is the same field type
// at a different `Dim`, not a different type.
//
// Stencil note: computing a cell's full residual along one axis needs
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
CFE_HOST_DEVICE Scalar axis_flux_difference(FieldViewT q, const CartesianGrid<Scalar>& g, std::size_t i,
                                            std::size_t j, std::size_t k,
                                            const Reconstruction& reconstruction,
                                            const NumericalFlux& numerical_flux, const Field& field,
                                            Scalar spacing)
{
  Scalar q_m2, q_m1, q_c, q_p1, q_p2;
  if constexpr (A == Axis::X) {
    q_m2 = q(g.flat_index(i - 2, j, k), 0);
    q_m1 = q(g.flat_index(i - 1, j, k), 0);
    q_c = q(g.flat_index(i, j, k), 0);
    q_p1 = q(g.flat_index(i + 1, j, k), 0);
    q_p2 = q(g.flat_index(i + 2, j, k), 0);
  } else if constexpr (A == Axis::Y) {
    q_m2 = q(g.flat_index(i, j - 2, k), 0);
    q_m1 = q(g.flat_index(i, j - 1, k), 0);
    q_c = q(g.flat_index(i, j, k), 0);
    q_p1 = q(g.flat_index(i, j + 1, k), 0);
    q_p2 = q(g.flat_index(i, j + 2, k), 0);
  } else {
    q_m2 = q(g.flat_index(i, j, k - 2), 0);
    q_m1 = q(g.flat_index(i, j, k - 1), 0);
    q_c = q(g.flat_index(i, j, k), 0);
    q_p1 = q(g.flat_index(i, j, k + 1), 0);
    q_p2 = q(g.flat_index(i, j, k + 2), 0);
  }

  // Right face: left value is this cell's own reconstruction, right value
  // is the next cell's own reconstruction.
  const Scalar q_left_of_right_face = reconstruction.right(q_m1, q_c, q_p1);
  const Scalar q_right_of_right_face = reconstruction.left(q_c, q_p1, q_p2);
  const Scalar flux_right = numerical_flux(q_left_of_right_face, q_right_of_right_face, A, field);

  // Left face: left value is the previous cell's own reconstruction,
  // right value is this cell's own reconstruction.
  const Scalar q_left_of_left_face = reconstruction.right(q_m2, q_m1, q_c);
  const Scalar q_right_of_left_face = reconstruction.left(q_m1, q_c, q_p1);
  const Scalar flux_left = numerical_flux(q_left_of_left_face, q_right_of_left_face, A, field);

  return -(flux_right - flux_left) / spacing;
}

}  // namespace detail

template <class Scalar, class Layout, class Field, class BoundaryX, class BoundaryY = BoundaryX,
          class BoundaryZ = BoundaryX, class Reconstruction = fvm::CentralDifferenceReconstruction,
          class NumericalFlux = UpwindFlux>
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
  // active axis in-place first, so `q` must be mutable storage, not a
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

  void residual(FieldView<Scalar, 1, Layout> q, FieldView<Scalar, 1, Layout> out) const
  {
    assert(grid.ngx >= 2 && "FvmSolver needs at least 2 ghost layers on every active axis");
    if constexpr (y_active) {
      assert(grid.ngy >= 2 && "FvmSolver needs at least 2 ghost layers on every active axis");
    }
    if constexpr (z_active) {
      assert(grid.ngz >= 2 && "FvmSolver needs at least 2 ghost layers on every active axis");
    }

    fill_ghost_cells(q, grid, Axis::X, boundary_x);
    if constexpr (y_active) fill_ghost_cells(q, grid, Axis::Y, boundary_y);
    if constexpr (z_active) fill_ghost_cells(q, grid, Axis::Z, boundary_z);

    const CartesianGrid<Scalar> g = grid;
    // Copied into locals (rather than capturing `this`) so the lambda
    // below holds self-contained, trivially-copyable state -- capturing
    // `this` would capture a host pointer, which breaks once this same
    // residual() body is reused from a CUDA translation unit later.
    const Field field = this->field;
    const Reconstruction reconstruction = this->reconstruction;
    const NumericalFlux numerical_flux = this->numerical_flux;

    cfe::parallel_for(g.nx * g.ny * g.nz, [=](std::size_t linear) mutable {
      const std::size_t local_i = linear % g.nx;
      const std::size_t local_j = (linear / g.nx) % g.ny;
      const std::size_t local_k = linear / (g.nx * g.ny);

      const std::size_t i = g.ngx + local_i;
      const std::size_t j = g.ngy + local_j;
      const std::size_t k = g.ngz + local_k;

      Scalar total_flux_difference = detail::axis_flux_difference<Axis::X>(
          q, g, i, j, k, reconstruction, numerical_flux, field, g.dx);

      if (y_active) {
        total_flux_difference += detail::axis_flux_difference<Axis::Y>(
            q, g, i, j, k, reconstruction, numerical_flux, field, g.dy);
      }
      if (z_active) {
        total_flux_difference += detail::axis_flux_difference<Axis::Z>(
            q, g, i, j, k, reconstruction, numerical_flux, field, g.dz);
      }

      out(g.flat_index(i, j, k), 0) = total_flux_difference;
    });
  }
};

}  // namespace cfe
