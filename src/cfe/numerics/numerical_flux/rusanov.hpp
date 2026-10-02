// Rusanov (local Lax-Friedrichs) numerical flux (Rusanov, "The
// calculation of the interaction of non-steady shock waves with
// obstacles," 1961; see also Toro, "Riemann Solvers and Numerical
// Methods for Fluid Dynamics," Sec. 10.5, or LeVeque, "Finite Volume
// Methods for Hyperbolic Problems," Sec. 10.3-10.4, for the general
// local-Lax-Friedrichs/Rusanov family):
//
//   F_rusanov(uL, uR) = 0.5*(F(uL) + F(uR)) - 0.5*alpha*(uR - uL)
//
// where `alpha` is a local estimate of the maximum characteristic speed
// across the two states, supplied by `field.wave_speed(uL, uR, axis)`.
// Same physics-agnostic combinator shape as `upwind.hpp`: it only ever
// calls `field.wave_speed(...)`/`field.physical_flux(...)`, never
// assumes any particular formula for either.
//
// Unlike `UpwindFlux` (which switches on the *sign* of a single scalar
// wave speed, and is therefore not entropy-correct for a genuinely
// nonlinear scalar flux at a sonic/transonic point -- see upwind.hpp's
// own caveat), this scheme adds dissipation proportional to the *local
// maximum* characteristic speed regardless of sign, which is entropy-
// satisfying for any convex scalar flux -- this is what makes it valid
// for `BurgersField` specifically, closing the gap `upwind.hpp`
// documents. Like `UpwindFlux`, this is still NOT a complete scheme for
// a multi-component system like Euler: there, `alpha` must be the
// largest eigenvalue magnitude of the flux Jacobian across both states
// (a per-axis `Field::wave_speed(...)` returning one number is still a
// compatible shape for that -- Phase 3's eventual Euler `Field` can
// supply exactly it), and `uL`/`uR`/`F(u)` become vectors, not scalars --
// so this header's single-`Scalar` free function/functor would need a
// vector-valued sibling; the underlying formula generalizes unchanged.
#pragma once

#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"

namespace cfe {

template <class Scalar, class Field>
CFE_HOST_DEVICE
Scalar rusanov_flux(Scalar left_value, Scalar right_value, Axis axis, const Field& field)
{
  const Scalar alpha = field.wave_speed(left_value, right_value, axis);
  const Scalar flux_left = field.physical_flux(left_value, axis);
  const Scalar flux_right = field.physical_flux(right_value, axis);
  return Scalar(0.5) * (flux_left + flux_right) - Scalar(0.5) * alpha * (right_value - left_value);
}

// Stateless functor wrapping rusanov_flux into the same swappable shape
// `UpwindFlux` uses: `NumericalFlux::operator()(left, right, axis,
// field)`. Solver residual code (solver/explicit/fvm_solver.hpp) never
// changes when this is substituted for `UpwindFlux` as a template
// argument.
struct RusanovFlux
{
  template <class Scalar, class Field>
  CFE_HOST_DEVICE
  Scalar operator()(Scalar left_value, Scalar right_value, Axis axis, const Field& field) const
  {
    return rusanov_flux(left_value, right_value, axis, field);
  }
};

}  // namespace cfe
