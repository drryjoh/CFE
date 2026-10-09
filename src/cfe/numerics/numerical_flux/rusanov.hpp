// Rusanov (local Lax-Friedrichs) numerical flux (Rusanov, "The
// calculation of the interaction of non-steady shock waves with
// obstacles," 1961; see also Toro, "Riemann Solvers and Numerical
// Methods for Fluid Dynamics," Sec. 10.5, or LeVeque, "Finite Volume
// Methods for Hyperbolic Problems," Sec. 10.3-10.4, for the general
// local-Lax-Friedrichs/Rusanov family):
//
//   F_rusanov(uL, uR) = 0.5*(F(uL) + F(uR)) - 0.5*alpha*(uR - uL)
//
// where `alpha` is the MAGNITUDE of a local estimate of the maximum
// characteristic speed across the two states -- `|field.wave_speed(uL,
// uR, axis)|`, explicitly absolute-valued below, not the raw return
// value. This matters because `Field::wave_speed(...)` is not
// contractually non-negative: `BurgersField`'s own `wave_speed` already
// returns `max(|uL|,|uR|)` (always non-negative, so the `abs` below is
// a no-op for it), but `ScalarAdvectionField::wave_speed` returns a
// *signed* velocity instead -- designed for `UpwindFlux` to switch on
// its sign, not for use as a dissipation coefficient. Caught in review:
// calling `rusanov_flux` with `ScalarAdvectionField` at a negative
// velocity previously used that signed value directly, flipping the
// dissipation term's sign and silently selecting the wrong upwind state
// (reproducible: velocity=-1, states (uL,uR)=(1,2) gave flux -1 instead
// of the correct -2) -- see `test_rusanov_flux_matches_upwind_flux_for_
// linear_advection_with_negative_velocity` in test_burgers_flux.cpp.
// Taking `|alpha|` makes this correct regardless of a given `Field`'s
// own sign convention for `wave_speed`, with no effect on `BurgersField`
// at all. Same physics-agnostic combinator shape as `upwind.hpp`
// otherwise: it only ever calls `field.wave_speed(...)`/
// `field.physical_flux(...)`, never assumes any particular formula for
// either.
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
  const Scalar raw_wave_speed = field.wave_speed(left_value, right_value, axis);
  const Scalar alpha = raw_wave_speed < Scalar(0) ? -raw_wave_speed : raw_wave_speed;
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
