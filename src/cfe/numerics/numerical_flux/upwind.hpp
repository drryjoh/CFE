// Numerical flux combinator (task spec item 4: "combine two sides'
// values into a numerical flux"). Physics-agnostic: combines two
// already-computed face values into one flux by upwind selection, using
// only `field.wave_speed(...)` and `field.physical_flux(...)` for the
// given axis -- it never knows what those formulas actually are
// (per-axis velocity * q for linear scalar advection; a Burgers or Euler
// field would supply different formulas under this same shape), and no
// idea whether the two face values came from FVM reconstruction
// (fvm/interface_value.hpp) or, later, a DG element's own trace. Keeping
// physics out of this file is what lets Burgers/Euler reuse this exact
// combinator.
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
