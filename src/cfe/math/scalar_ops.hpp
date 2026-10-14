// Small scalar utilities shared across Fields/BoundaryConditions (PR #6
// cleanup task -- Amelia's consult identified these as the genuine small
// duplications worth collecting, see agent_history.md and
// tasks/0014-pr6-cleanup.md). NOT a general "common math" dumping ground
// (AGENTS.md #6): each addition here must already be an independently
// repeated, host/device-callable scalar primitive, not a speculative
// abstraction added ahead of a second use site.
//
// - `abs_value`: the `x < Scalar(0) ? -x : x` ternary, previously
//   hand-rolled independently in `BurgersField::wave_speed`
//   (fields/burgers/field.hpp), `EulerField::wave_speed`
//   (fields/euler/field.hpp), both `rusanov_flux` overloads
//   (numerics/numerical_flux/rusanov.hpp), and
//   `BlockInterfaceBoundary::detail::close` (grid/boundary/
//   block_interface_boundary.hpp) -- five independent copies of the
//   exact same primitive across four files, not a coincidence worth
//   leaving duplicated. Named `abs_value`, not `abs`, to avoid any
//   ambiguity with `std::abs`/`::abs`'s own mixed integer/floating-point
//   overload set (this always takes and returns the same `Scalar`).
//
// Transcendental functions (`sqrt`, `pow`) are deliberately NOT wrapped
// here, and this is a decision, not an oversight: see
// `fields/euler/field.hpp`'s `sound_speed` and
// `docs/adr/0010-euler-field-and-shock-capturing.md`'s addendum for the
// now-settled convention (call them unqualified, not `std::`-qualified)
// closing the "noted, not acted on" item that ADR's own review left
// open.
#pragma once

#include "cfe/core/macros.hpp"

namespace cfe {

template <class Scalar>
CFE_HOST_DEVICE
CFE_FORCEINLINE
Scalar abs_value(const Scalar& x)
{
  return x < Scalar(0) ? -x : x;
}

}  // namespace cfe
