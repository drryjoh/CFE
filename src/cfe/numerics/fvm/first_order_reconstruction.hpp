// First-order (piecewise-constant) FVM reconstruction: a cell's value at
// either face is just its own cell average, with no slope at all. The
// simplest possible member of the `Reconstruction` family -- useful here
// specifically as the baseline a limited second-order scheme
// (numerics/fvm/muscl_minmod.hpp) is compared against, not because it is
// ever the preferred production scheme (it is the most diffusive
// possible choice, smearing a shock over many more cells than a limited
// scheme at the same resolution; see e.g. LeVeque, "Finite Volume
// Methods for Hyperbolic Problems," Ch. 4, for Godunov's original
// first-order scheme this corresponds to when paired with an exact or
// approximate Riemann solver).
//
// Same `Reconstruction::right(...)/left(...)` two-method shape, same
// 3-point (left_neighbor, self, right_neighbor) stencil
// `CentralDifferenceReconstruction`/`MusclMinmodReconstruction` already
// use -- solver residual code (solver/explicit/fvm_solver.hpp) requires
// zero changes to use this reconstruction instead; `left_neighbor`/
// `right_neighbor` are accepted only to match that shared shape, unused
// here.
#pragma once

#include "cfe/core/macros.hpp"

namespace cfe {
namespace fvm {

struct FirstOrderReconstruction
{
  template <class Scalar>
  CFE_HOST_DEVICE
  Scalar right(Scalar /*state_left_neighbor*/, Scalar state_self, Scalar /*state_right_neighbor*/) const
  {
    return state_self;
  }

  template <class Scalar>
  CFE_HOST_DEVICE
  Scalar left(Scalar /*state_left_neighbor*/, Scalar state_self, Scalar /*state_right_neighbor*/) const
  {
    return state_self;
  }
};

}  // namespace fvm
}  // namespace cfe
