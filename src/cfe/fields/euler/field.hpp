// The compressible Euler equations for a calorically perfect, gamma-law
// gas: dQ/dt + div(F(Q)) = 0, Q = (density, momentum_x, [momentum_y,
// [momentum_z,]] total_energy), F a function of Q and an equation of
// state p = (gamma-1)*(E - 0.5*|momentum|^2/density) (see e.g. Toro,
// "Riemann Solvers and Numerical Methods for Fluid Dynamics," Ch. 3, or
// LeVeque, "Finite Volume Methods for Hyperbolic Problems," Ch. 14).
// gamma defaults to 1.4 (diatomic ideal gas, e.g. air) but is a runtime
// member, not a compile-time constant -- the same "physical parameter
// as a plain data member" shape `ScalarAdvectionField::velocity` already
// uses, so a case can choose a different gas without recompiling.
//
// Units: this Field is unit-system-agnostic -- it just needs `density`,
// `momentum`/`velocity`, `pressure`, and `total_energy` to be mutually
// consistent in WHATEVER unit system the caller picks (e.g. plain SI, or
// a nondimensionalized system where density/velocity/pressure are each
// scaled by their own reference value). Every canonical compressible-
// flow test case in CFD, including the Sod shock tube this Field is
// first verified against (tests/verification/test_euler_sod_shock_tube.cpp),
// is conventionally stated in nondimensional units already (e.g.
// density=1.0, pressure=1.0 is not "one kg per cubic meter at one
// pascal" -- it is "the reference density/pressure," by construction)
// -- so no unit conversion is needed to reproduce it, but a case
// intending to model a REAL physical flow (e.g. actual air at actual
// atmospheric conditions) is responsible for its own consistent unit
// choice; this Field performs no unit checking or conversion itself.
//
// A "Field" per ARCHITECTURE.md #2, but UNLIKE every previous field in
// this codebase (`ScalarAdvectionField`, `BurgersField`, both
// single-component, N=1): this is CFE's first genuinely multi-component
// Field, with `n_components = Dim + 2`. That is exactly the
// generalization `fields/scalar_advection/field.hpp`'s own header
// comment and `tasks/0003-...md` flagged as still needed before Euler
// could be added -- see `solver/explicit/fvm_solver.hpp`'s header
// comment for how `FvmSolver` was generalized over `Field::n_components`
// to host this type. `physical_flux`/`wave_speed` therefore operate on
// `State<Scalar, n_components>` (a full conserved-state vector), not a
// bare `Scalar` -- a different Calculator shape than the N=1 fields'
// `Scalar`-in/`Scalar`-out one, reflecting that Euler's flux for any one
// component genuinely depends on the whole state (e.g. the mass flux
// `rho*u` needs `u = momentum/density`, not just `rho` alone). See
// `numerics/numerical_flux/rusanov.hpp` for how `RusanovFlux` is
// overloaded to serve both shapes from the same functor type.
//
// Conserved-state component layout (index 0..n_components-1):
//   0                  : density
//   1 .. Dim           : momentum (component `1 + axis_index`)
//   Dim + 1            : total energy
// This layout is a property of THIS Field, not something generic
// boundary/solver code should hardcode -- `momentum_index(Axis)` is
// `static constexpr` specifically so other code (the slip-wall boundary
// condition, grid/boundary/slip_wall_boundary.hpp) can ask this Field
// which raw component is "momentum normal to a given axis" instead of
// assuming index `1 + axis` applies to every possible multi-component
// Field. This is the "Field-aware" boundary-condition shape
// ARCHITECTURE.md #11 describes (a BoundaryCondition takes a Field type
// and calls into it rather than only ever writing/copying raw component
// values) -- a lighter-weight instance of it than the characteristic-
// based partial-specification subsonic inflow/outflow BC that same
// section describes (no equation-of-state or wave-speed evaluation
// needed here, just a static component-index lookup plus a sign flip),
// confirmed against that distinction with Amelia/Violet before
// implementation (see docs/adr/0010-euler-field-and-shock-capturing.md).
//
// Same `Dim` compile-time-choice pattern as `ScalarAdvectionField`/
// `BurgersField` (AGENTS.md #8): `EulerField<Scalar, 1>` for 1D (3
// components), `EulerField<Scalar, 2>` for 2D (4 components),
// `EulerField<Scalar, 3>` for 3D (5 components).
#pragma once

#include <cmath>
#include <cstddef>

#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"
#include "cfe/math/fixed_array.hpp"

namespace cfe {

template <class Scalar, std::size_t Dim>
struct EulerField
{
  static_assert(Dim >= 1 && Dim <= 3, "EulerField supports Dim = 1, 2, or 3");

  static constexpr std::size_t dim = Dim;
  static constexpr std::size_t n_components = Dim + 2;

  using ConservedState = State<Scalar, n_components>;

  // Ratio of specific heats. 1.4 (diatomic ideal gas, e.g. air) unless a
  // case sets a different gas -- a runtime member (not a template
  // parameter) since nothing here depends on it at compile time, and a
  // future case varying it should not need a recompile.
  Scalar gamma = Scalar(1.4);

  // Which raw conserved-component index holds momentum along `axis`.
  // `static constexpr` (no `this` needed) so boundary conditions and
  // other non-Calculator code can call it directly off the type, e.g.
  // `EulerField<Scalar, Dim>::momentum_index(Axis::X)`, without
  // constructing a Field instance -- see this file's header comment and
  // grid/boundary/slip_wall_boundary.hpp.
  CFE_HOST_DEVICE
  static constexpr Index momentum_index(Axis axis) { return 1 + static_cast<Index>(axis); }

  CFE_HOST_DEVICE
  static constexpr Index energy_index() { return Dim + 1; }

  // --- semantic accessors (AGENTS.md #12) ---------------------------------

  CFE_HOST_DEVICE
  Scalar density(const ConservedState& q) const { return q[0]; }

  CFE_HOST_DEVICE
  Scalar momentum(const ConservedState& q, Axis axis) const { return q[momentum_index(axis)]; }

  CFE_HOST_DEVICE
  Scalar total_energy(const ConservedState& q) const { return q[energy_index()]; }

  CFE_HOST_DEVICE
  Scalar velocity_component(const ConservedState& q, Axis axis) const
  {
    return momentum(q, axis) / density(q);
  }

  // 0.5 * |momentum|^2 / density = 0.5 * density * |velocity|^2.
  CFE_HOST_DEVICE
  Scalar kinetic_energy(const ConservedState& q) const
  {
    Scalar momentum_squared_sum = Scalar(0);
    for (Index d = 0; d < Dim; ++d) {
      const Scalar m = q[1 + d];
      momentum_squared_sum += m * m;
    }
    return Scalar(0.5) * momentum_squared_sum / density(q);
  }

  // Ideal-gas equation of state: p = (gamma - 1) * (E - kinetic_energy).
  CFE_HOST_DEVICE
  Scalar pressure(const ConservedState& q) const
  {
    return (gamma - Scalar(1)) * (total_energy(q) - kinetic_energy(q));
  }

  CFE_HOST_DEVICE
  Scalar sound_speed(const ConservedState& q) const
  {
    return sqrt(gamma * pressure(q) / density(q));
  }

  // --- Calculators (ARCHITECTURE.md #2) -----------------------------------

  // The physical flux Calculator for one axis:
  //   F_axis(Q) = (rho*u_axis,
  //                momentum_d * u_axis + delta(d,axis)*p  for each d,
  //                (E + p) * u_axis)
  // where u_axis = velocity_component(Q, axis).
  CFE_HOST_DEVICE
  ConservedState physical_flux(const ConservedState& q, Axis axis) const
  {
    const Scalar u_axis = velocity_component(q, axis);
    const Scalar p = pressure(q);
    const Index normal_component = momentum_index(axis);

    ConservedState flux;
    flux[0] = density(q) * u_axis;
    for (Index d = 0; d < Dim; ++d) {
      const Index component = 1 + d;
      flux[component] = q[component] * u_axis + (component == normal_component ? p : Scalar(0));
    }
    flux[energy_index()] = (total_energy(q) + p) * u_axis;
    return flux;
  }

  // The local maximum characteristic speed |u_axis| + c, taken as the
  // larger of the two one-sided states -- the single scalar dissipation
  // coefficient a Rusanov/local-Lax-Friedrichs numerical flux needs,
  // shared across every component of the system (see
  // numerics/numerical_flux/rusanov.hpp's vector overload). This is
  // DIFFERENT in kind from `BurgersField::wave_speed` only in that it
  // takes full conserved-state vectors rather than bare scalars -- the
  // role (a single non-negative dissipation magnitude) is identical.
  CFE_HOST_DEVICE
  Scalar wave_speed(const ConservedState& state_left, const ConservedState& state_right, Axis axis) const
  {
    const Scalar u_left = velocity_component(state_left, axis);
    const Scalar c_left = sound_speed(state_left);
    const Scalar u_right = velocity_component(state_right, axis);
    const Scalar c_right = sound_speed(state_right);

    const Scalar abs_u_left = u_left < Scalar(0) ? -u_left : u_left;
    const Scalar abs_u_right = u_right < Scalar(0) ? -u_right : u_right;
    const Scalar speed_left = abs_u_left + c_left;
    const Scalar speed_right = abs_u_right + c_right;
    return speed_left > speed_right ? speed_left : speed_right;
  }

  // --- initial-condition / boundary-value convenience ---------------------

  // Builds a conserved state from primitive variables (density, velocity
  // vector, pressure) and this field's own `gamma` -- host-only
  // convenience for setting initial/boundary conditions, which are
  // naturally specified in primitive form (ARCHITECTURE.md #2: a
  // Calculator provides "primitive/conserved conversion").
  ConservedState conserved_from_primitive(Scalar density_value, const Vector<Scalar, Dim>& velocity,
                                           Scalar pressure_value) const
  {
    ConservedState q;
    q[0] = density_value;
    Scalar momentum_squared_sum = Scalar(0);
    for (Index d = 0; d < Dim; ++d) {
      const Scalar m = density_value * velocity[d];
      q[1 + d] = m;
      momentum_squared_sum += m * m;
    }
    const Scalar kinetic = Scalar(0.5) * momentum_squared_sum / density_value;
    q[energy_index()] = pressure_value / (gamma - Scalar(1)) + kinetic;
    return q;
  }
};

}  // namespace cfe
