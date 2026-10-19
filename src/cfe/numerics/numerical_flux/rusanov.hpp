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
// documents.
//
// Vector (system) generalization, below: for a multi-component Field
// like `EulerField` (fields/euler/field.hpp), `alpha` must be a SINGLE
// scalar -- the largest characteristic-speed magnitude across both
// states (`Field::wave_speed(...)` already returns exactly that one
// number, taking full conserved-state vectors instead of bare scalars
// for such a Field) -- shared by every component's dissipation term,
// not a separate value per component. `uL`/`uR`/`F(u)` become
// `State<Scalar, N>` vectors; the underlying formula is otherwise
// unchanged, applied componentwise. This is a genuinely different
// overload (selected by argument type: a bare `Scalar` for N=1 fields
// like `BurgersField`/`ScalarAdvectionField`, a `State<Scalar, N>` for
// multi-component fields), not a replacement for the scalar form above
// -- both are needed since N=1 fields' Calculators take/return bare
// `Scalar`, not `State<Scalar, 1>`.
#pragma once

#include <cstddef>

#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"
#include "cfe/math/fixed_array.hpp"
#include "cfe/math/scalar_ops.hpp"

namespace cfe {

template <class Scalar, class Field>
CFE_HOST_DEVICE
Scalar rusanov_flux(Scalar left_value, Scalar right_value, Axis axis, const Field& field)
{
  const Scalar raw_wave_speed = field.wave_speed(left_value, right_value, axis);
  const Scalar alpha = abs_value(raw_wave_speed);
  const Scalar flux_left = field.physical_flux(left_value, axis);
  const Scalar flux_right = field.physical_flux(right_value, axis);
  return Scalar(0.5) * (flux_left + flux_right) - Scalar(0.5) * alpha * (right_value - left_value);
}

// System (vector-state) form -- see this file's header comment. `alpha`
// is computed ONCE from the full states and reused for every component,
// which is what makes this the correct Rusanov/local-Lax-Friedrichs
// flux for a coupled system rather than N independent scalar problems.
template <class Scalar, std::size_t N, class Field>
CFE_HOST_DEVICE
State<Scalar, N> rusanov_flux(const State<Scalar, N>& left_state, const State<Scalar, N>& right_state,
                              Axis axis, const Field& field)
{
  const Scalar raw_wave_speed = field.wave_speed(left_state, right_state, axis);
  const Scalar alpha = abs_value(raw_wave_speed);
  const State<Scalar, N> flux_left = field.physical_flux(left_state, axis);
  const State<Scalar, N> flux_right = field.physical_flux(right_state, axis);

  State<Scalar, N> flux;
  for (Index c = 0; c < N; ++c) {
    flux[c] =
        Scalar(0.5) * (flux_left[c] + flux_right[c]) - Scalar(0.5) * alpha * (right_state[c] - left_state[c]);
  }
  return flux;
}

// Stateless functor wrapping rusanov_flux into the same swappable shape
// `UpwindFlux` uses: `NumericalFlux::operator()(left, right, axis,
// field)`. Solver residual code (solver/explicit/fvm_solver.hpp) never
// changes when this is substituted for `UpwindFlux` as a template
// argument -- overload resolution on the argument type (bare `Scalar`
// vs. `State<Scalar, N>`) picks the matching `rusanov_flux` above, so
// the SAME `RusanovFlux` type serves both `BurgersField`/
// `ScalarAdvectionField` (N=1, scalar shape) and `EulerField` (N>1,
// vector shape) unchanged.
struct RusanovFlux
{
  template <class Scalar, class Field>
  CFE_HOST_DEVICE
  Scalar operator()(Scalar left_value, Scalar right_value, Axis axis, const Field& field) const
  {
    return rusanov_flux(left_value, right_value, axis, field);
  }

  template <class Scalar, std::size_t N, class Field>
  CFE_HOST_DEVICE
  State<Scalar, N> operator()(const State<Scalar, N>& left_state, const State<Scalar, N>& right_state,
                              Axis axis, const Field& field) const
  {
    return rusanov_flux(left_state, right_state, axis, field);
  }
};

}  // namespace cfe
