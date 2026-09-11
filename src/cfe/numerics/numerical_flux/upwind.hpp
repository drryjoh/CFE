// Numerical flux for linear scalar advection (task spec item 4: "combine
// two sides' values into a numerical flux"). Method-agnostic: this
// function only ever sees two already-computed face values -- it has no
// idea whether they came from FVM reconstruction (fvm/interface_value.hpp)
// or, later, a DG element's own trace -- plus the (fixed-sign) advection
// speed. Upwind selection is the simplest stable choice for linear
// advection with a fixed-sign wave speed.
#pragma once

#include "cfe/core/macros.hpp"

namespace cfe {

template <class Scalar>
CFE_HOST_DEVICE Scalar upwind_flux(Scalar left_value, Scalar right_value, Scalar advection_speed)
{
  return advection_speed >= Scalar(0) ? advection_speed * left_value
                                      : advection_speed * right_value;
}

}  // namespace cfe
