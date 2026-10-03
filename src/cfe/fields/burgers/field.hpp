// The inviscid Burgers equation: dQ/dt + div(F(Q)) = 0, F(Q) = Q^2/2.
// The standard canonical scalar conservation law used to test
// shock-forming/shock-capturing numerics (see e.g. LeVeque, "Finite
// Volume Methods for Hyperbolic Problems," Ch. 11) -- its flux is
// nonlinear and convex, so smooth initial data can steepen into a
// genuine discontinuity in finite time (unlike linear scalar advection,
// which only ever translates its initial profile unchanged).
//
// A "Field" per ARCHITECTURE.md #2, same exact Calculator shape as
// `fields/scalar_advection/field.hpp` -- `physical_flux(state, axis)` /
// `wave_speed(state_left, state_right, axis)`, both `(state, axis)`-in --
// so `solver/explicit/fvm_solver.hpp` requires zero interface changes to
// host this field; only the formulas differ. Unlike
// `ScalarAdvectionField`, there is no `velocity` member: Burgers is
// self-advecting (the characteristic speed is the state itself,
// F'(Q) = Q), so `physical_flux`/`wave_speed` depend only on the state
// values passed in, not on any stored parameter.
//
// `wave_speed` returns max(|state_left|, |state_right|) -- the local
// maximum characteristic speed across the two one-sided face values,
// which is exactly the dissipation coefficient a Rusanov/local
// Lax-Friedrichs numerical flux needs (numerics/numerical_flux/
// rusanov.hpp). This is a different role than `ScalarAdvectionField`'s
// own `wave_speed` (there, a signed value selecting an upwind direction
// for `UpwindFlux`): `UpwindFlux` is not entropy-correct for a genuinely
// nonlinear flux like this one at a sonic/transonic point (see
// upwind.hpp's own caveat), so `BurgersField` is meant to be paired with
// `RusanovFlux`, not `UpwindFlux`.
//
// Same `Dim` compile-time-choice pattern as `ScalarAdvectionField`
// (AGENTS.md #8): `BurgersField<Scalar, 1>` for 1D, etc. -- lets the same
// field type serve 1D/2D/3D interchangeably, matching the grid/solver's
// own per-axis dispatch.
#pragma once

#include <cstddef>

#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"

namespace cfe {

template <class Scalar, std::size_t Dim>
struct BurgersField
{
  static constexpr std::size_t dim = Dim;

  // The physical flux Calculator for one axis: F_axis(state) = state^2 / 2.
  // Identical along every axis -- Burgers has no directional parameter --
  // `axis` is accepted only to match the shape every other Field/
  // NumericalFlux/Reconstruction type in this codebase is generic over.
  CFE_HOST_DEVICE
  Scalar physical_flux(Scalar state, Axis /*axis*/) const
  {
    return Scalar(0.5) * state * state;
  }

  // The local maximum characteristic speed |F'(Q)| = |Q| across the two
  // one-sided face values -- the dissipation coefficient `RusanovFlux`
  // needs. See this type's own header comment for why this is a
  // different quantity than `ScalarAdvectionField::wave_speed`'s signed
  // upwind-direction value.
  CFE_HOST_DEVICE
  Scalar wave_speed(Scalar state_left, Scalar state_right, Axis /*axis*/) const
  {
    const Scalar abs_left = state_left < Scalar(0) ? -state_left : state_left;
    const Scalar abs_right = state_right < Scalar(0) ? -state_right : state_right;
    return abs_left > abs_right ? abs_left : abs_right;
  }
};

}  // namespace cfe
