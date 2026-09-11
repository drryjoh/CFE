// FVM interface-value reconstruction (task spec item 4): produces the
// state ONE cell contributes to ONE of its faces, using only that cell's
// own immediate ("1-ring") neighbors -- central-difference-style, as the
// task spec requires.
//
// This is the FVM implementation of a method-agnostic contract: "given my
// own local representation, produce my value at a face." A future DG
// element (evaluating its own internal degrees of freedom, no neighbor
// cell data needed at all) could implement the same contract without
// numerics/numerical_flux/upwind.hpp changing at all -- that stage only
// ever sees two already-computed face values, never cares how either side
// produced them. See docs/adr/0007-interface-flux-and-time-integration.md.
#pragma once

#include "cfe/core/macros.hpp"

namespace cfe {
namespace fvm {

// This cell's value at its RIGHT (+axis) face: linear extrapolation using
// a central-difference slope estimate from its own left/right neighbors.
//   Q_face = Q_self + (Q_right_neighbor - Q_left_neighbor) / 4
template <class Scalar>
CFE_HOST_DEVICE Scalar interface_value_right(Scalar q_left_neighbor, Scalar q_self,
                                              Scalar q_right_neighbor)
{
  return q_self + (q_right_neighbor - q_left_neighbor) * Scalar(0.25);
}

// This cell's value at its LEFT (-axis) face -- the mirror image of
// interface_value_right, same 3-point stencil:
//   Q_face = Q_self - (Q_right_neighbor - Q_left_neighbor) / 4
template <class Scalar>
CFE_HOST_DEVICE Scalar interface_value_left(Scalar q_left_neighbor, Scalar q_self,
                                             Scalar q_right_neighbor)
{
  return q_self - (q_right_neighbor - q_left_neighbor) * Scalar(0.25);
}

// Stateless functor wrapping the two free functions above into the
// swappable shape solver code (solver/scalar_advection/solver.hpp) is
// generic over: `Reconstruction::right(...)` / `::left(...)`. A future
// MUSCL/PPM/WENO reconstruction -- or a DG element's own trace evaluation
// -- is a new type with this same two-method shape, substituted as a
// template argument; solver residual code never changes.
struct CentralDifferenceReconstruction
{
  template <class Scalar>
  CFE_HOST_DEVICE Scalar right(Scalar q_left_neighbor, Scalar q_self, Scalar q_right_neighbor) const
  {
    return interface_value_right(q_left_neighbor, q_self, q_right_neighbor);
  }

  template <class Scalar>
  CFE_HOST_DEVICE Scalar left(Scalar q_left_neighbor, Scalar q_self, Scalar q_right_neighbor) const
  {
    return interface_value_left(q_left_neighbor, q_self, q_right_neighbor);
  }
};

}  // namespace fvm
}  // namespace cfe
