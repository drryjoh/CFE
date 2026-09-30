// The linear scalar-advection field: dQ/dt + velocity . grad(Q) = 0.
//
// A "Field" per ARCHITECTURE.md #2: it defines the equation being solved
// and provides physics-specific Calculator functions -- here, the
// physical flux and the wave speed used for upwind direction, both
// per-axis. Everything else (grid, boundary conditions, reconstruction,
// the numerical-flux combinator, time integration) is generic and knows
// nothing about this field's physics; it only ever calls into this type,
// passing whichever `Axis` it's currently assembling a flux for. A future
// Burgers field -- still single-component, like this one -- can supply
// its own nonlinear `physical_flux()`/`wave_speed()` under this exact
// `(state, axis)`-in shape, with zero interface changes to
// solver/explicit/fvm_solver.hpp. That claim does NOT extend to Euler
// (corrected in code review -- see agent_history.md): Euler's state is
// multi-component (density/momentum/energy), and `FvmSolver` currently
// hardcodes `FieldView<Scalar, 1, Layout>` and component `0` throughout
// -- Euler needs that solver generalized over `NComponents` first, not
// just a new `Field`. Nor does it mean `UpwindFlux` is a complete,
// entropy-correct Burgers solver as-is; see
// numerics/numerical_flux/upwind.hpp's own caveat on that.
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

  // The physical flux Calculator for one axis: F_axis(state) = velocity[axis] * state.
  CFE_HOST_DEVICE Scalar physical_flux(Scalar state, Axis axis) const
  {
    return velocity[static_cast<std::size_t>(axis)] * state;
  }

  // The wave speed a numerical-flux combinator uses to pick an upwind
  // direction along one axis. Takes both one-sided face values so a
  // future field with a state-dependent characteristic speed (e.g.
  // Burgers, where it's state itself) can use them; this field's wave
  // speed is simply that axis's velocity component.
  CFE_HOST_DEVICE Scalar wave_speed(Scalar /*state_left*/, Scalar /*state_right*/, Axis axis) const
  {
    return velocity[static_cast<std::size_t>(axis)];
  }
};

}  // namespace cfe
