// The linear scalar-advection field: dQ/dt + velocity . grad(Q) = 0.
//
// A "Field" per ARCHITECTURE.md #2: it defines the equation being solved
// and provides physics-specific Calculator functions -- here, the
// physical flux and the wave speed used for upwind direction, both
// per-axis. Everything else (grid, boundary conditions, reconstruction,
// the numerical-flux combinator, time integration) is generic and knows
// nothing about this field's physics; it only ever calls into this type,
// passing whichever `Axis` it's currently assembling a flux for. A future
// Burgers or Euler field supplies its own physical_flux()/wave_speed()
// with this same (state, axis)-in shape -- solver/explicit/fvm_solver.hpp
// and numerics/numerical_flux/upwind.hpp would need zero changes.
//
// `Dim` is a compile-time choice (matching this project's compile-time
// state sizing, AGENTS.md #8), made per solver instantiation: a 1D
// problem uses `ScalarAdvectionField<Scalar, 1>` (one velocity
// component), a 2D problem `ScalarAdvectionField<Scalar, 2>`, and so on.
// This is what lets the same field type serve 1D/2D/3D transport
// interchangeably -- the solver's residual loop already runs over
// however many axes the *grid* has active (see fvm_solver.hpp); this is
// the matching per-axis piece on the physics side.
#pragma once

#include <cstddef>

#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"
#include "cfe/math/fixed_array.hpp"

namespace cfe {

template <class Scalar, std::size_t Dim>
struct ScalarAdvectionField
{
  // Exposed so solver code (solver/explicit/fvm_solver.hpp) can select
  // 1D/2D/3D behavior with `if constexpr` at compile time, instead of a
  // per-cell runtime check -- this is the one place a problem's
  // dimensionality is decided, and everything downstream derives from it
  // rather than re-deciding it per thread.
  static constexpr std::size_t dim = Dim;

  Vector<Scalar, Dim> velocity;

  // The physical flux Calculator for one axis: F_axis(q) = velocity[axis] * q.
  CFE_HOST_DEVICE Scalar physical_flux(Scalar q, Axis axis) const
  {
    return velocity[static_cast<std::size_t>(axis)] * q;
  }

  // The wave speed a numerical-flux combinator uses to pick an upwind
  // direction along one axis. Takes both one-sided face values so a
  // future field with a state-dependent characteristic speed (e.g.
  // Burgers, where it's q itself) can use them; this field's wave speed
  // is simply that axis's velocity component.
  CFE_HOST_DEVICE Scalar wave_speed(Scalar /*q_left*/, Scalar /*q_right*/, Axis axis) const
  {
    return velocity[static_cast<std::size_t>(axis)];
  }
};

}  // namespace cfe
