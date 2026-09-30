// Numerical flux combinator (task spec item 4: "combine two sides'
// values into a numerical flux"). Physics-agnostic: combines two
// already-computed face values into one flux by upwind selection, using
// only `field.wave_speed(...)` and `field.physical_flux(...)` for the
// given axis -- it never knows what those formulas actually are (per-axis
// velocity * state for linear scalar advection), and no idea whether the
// two face values came from FVM reconstruction (fvm/interface_value.hpp)
// or, later, a DG element's own trace.
//
// This is NOT a general Riemann solver, and swapping in a different
// `Field` does not make it one (caught in code review -- see
// agent_history.md): `UpwindFlux` switches on the sign of a single
// scalar `wave_speed`, which (a) is not entropy-correct for a genuinely
// nonlinear scalar equation like Burgers at a sonic/transonic point,
// where the correct flux needs an explicit entropy fix this scheme does
// not apply, and (b) has no meaning at all for a multi-component system
// like Euler, whose flux depends on the full state vector's eigenstructure,
// not one scalar sign. A future Rusanov/HLLC/AUSM flux is a new type
// under the same `NumericalFlux::operator()(left, right, axis, field)`
// shape -- solver residual code never changes -- but it is a *different*
// algorithm, not this one with a different `Field` plugged in.
#pragma once

#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"

namespace cfe {

template <class Scalar, class Field>
CFE_HOST_DEVICE Scalar upwind_flux(Scalar left_value, Scalar right_value, Axis axis, const Field& field)
{
  const Scalar wave_speed = field.wave_speed(left_value, right_value, axis);
  return wave_speed >= Scalar(0) ? field.physical_flux(left_value, axis)
                                 : field.physical_flux(right_value, axis);
}

// Stateless functor wrapping upwind_flux into the swappable shape solver
// code is generic over: `NumericalFlux::operator()(left, right, axis,
// field)`. A future Rusanov/HLLC/AUSM flux is a new type with this same
// shape, substituted as a template argument; solver residual code never
// changes.
struct UpwindFlux
{
  template <class Scalar, class Field>
  CFE_HOST_DEVICE Scalar operator()(Scalar left_value, Scalar right_value, Axis axis,
                                    const Field& field) const
  {
    return upwind_flux(left_value, right_value, axis, field);
  }
};

}  // namespace cfe
