// MUSCL reconstruction (van Leer, "Towards the ultimate conservative
// difference scheme. V," J. Comput. Phys. 32, 1979) with the minmod
// slope limiter (see e.g. LeVeque, "Finite Volume Methods for
// Hyperbolic Problems," Sec. 9.3, or Toro, "Riemann Solvers and
// Numerical Methods for Fluid Dynamics," Sec. 13.3, for the general
// slope-limiter family): a shock-capturing alternative to
// `interface_value.hpp`'s `CentralDifferenceReconstruction` that does
// not oscillate at a discontinuity.
//
// `CentralDifferenceReconstruction`'s unlimited central-difference slope
// overshoots/undershoots next to a genuine discontinuity (Gibbs-type
// oscillation) -- exactly what `BurgersField`'s shock-formation test
// needs to avoid. A limited slope avoids this by construction: the
// minmod limiter picks whichever of the two one-sided differences is
// smaller in magnitude when they agree in sign, and clips the slope to
// exactly zero when they disagree (i.e. at a local extremum or a
// discontinuity) -- this is what makes the scheme TVD (total-variation-
// diminishing; Harten, 1983) by construction, at the cost of being the
// most diffusive of the standard limiters (superbee, van Leer, monotonized
// central (MC), ...) and of reducing formal accuracy to 1st order at
// smooth extrema, where the limiter clips even though the true solution
// has no discontinuity there -- a known, accepted property of TVD
// limiters (see e.g. Sweby, "High resolution schemes using flux limiters
// for hyperbolic conservation laws," SIAM J. Numer. Anal. 21, 1984), not
// a bug. minmod is chosen here as the simplest, most robust member of
// this family for Burgers's first shock-capturing scheme; sharper
// limiters are named future work (docs/adr/0008-...).
//
//   minmod(a, b) = 0                       if a, b disagree in sign, or
//                                           either is exactly zero
//                = sign(a) * min(|a|, |b|) otherwise
//   slope(l, c, r) = minmod(c - l, r - c)
//   right(l, c, r) = c + slope(l, c, r) / 2   -- this cell's value at its
//                                                 right (+axis) face
//   left(l, c, r)  = c - slope(l, c, r) / 2   -- this cell's value at its
//                                                 left (-axis) face
//
// Same `Reconstruction::right(...)/left(...)` two-method shape, same
// 3-point (left_neighbor, self, right_neighbor) stencil
// `CentralDifferenceReconstruction` already uses and
// `solver/explicit/fvm_solver.hpp` already calls it with -- solver
// residual code requires zero changes to use this reconstruction instead.
#pragma once

#include "cfe/core/macros.hpp"

namespace cfe {
namespace fvm {

// The minmod slope limiter: 0 if `a`/`b` disagree in sign (or either is
// exactly zero), otherwise whichever of the two is smaller in magnitude
// (with their shared sign).
//
// Deliberately does NOT use the textbook `a*b <= 0` sign test (caught in
// review): that product underflows to exactly 0 for small-but-equal-
// magnitude, same-sign `a`/`b` well before either operand itself
// underflows -- e.g. `a=b=1e-200` (double) gives `a*b=1e-400`, which
// flushes to `0.0` (doubles bottom out around `1e-308`), wrongly taking
// the `<=0` branch and returning `0` instead of the correct `1e-200`.
// The same failure mode hits `float` far sooner (`1e-25f` is a perfectly
// representable `float`, but `(1e-25f)*(1e-25f)=1e-50f` underflows,
// `float`'s smallest subnormal being around `1e-45`). Direct sign
// comparisons below never multiply `a` and `b` together, so this
// failure mode cannot occur at any representable magnitude -- see
// `test_minmod_handles_tiny_representable_slopes_without_underflow` in
// test_muscl_reconstruction.cpp for the regression coverage.
template <class Scalar>
CFE_HOST_DEVICE
Scalar minmod(Scalar a, Scalar b)
{
  if (a > Scalar(0) && b > Scalar(0)) return a < b ? a : b;
  if (a < Scalar(0) && b < Scalar(0)) return a > b ? a : b;
  return Scalar(0);
}

// This cell's limited slope, from its own immediate ("1-ring")
// neighbors: minmod of the left-biased and right-biased one-sided
// differences.
template <class Scalar>
CFE_HOST_DEVICE
Scalar muscl_minmod_slope(Scalar state_left_neighbor, Scalar state_self, Scalar state_right_neighbor)
{
  return minmod(state_self - state_left_neighbor, state_right_neighbor - state_self);
}

// This cell's value at its RIGHT (+axis) face: its own value plus half
// its limited slope.
template <class Scalar>
CFE_HOST_DEVICE
Scalar muscl_minmod_value_right(Scalar state_left_neighbor, Scalar state_self,
                                 Scalar state_right_neighbor)
{
  return state_self +
         Scalar(0.5) * muscl_minmod_slope(state_left_neighbor, state_self, state_right_neighbor);
}

// This cell's value at its LEFT (-axis) face: its own value minus half
// its limited slope (same slope as the right face -- a single linear
// reconstruction per cell, evaluated at both its faces).
template <class Scalar>
CFE_HOST_DEVICE
Scalar muscl_minmod_value_left(Scalar state_left_neighbor, Scalar state_self,
                                Scalar state_right_neighbor)
{
  return state_self -
         Scalar(0.5) * muscl_minmod_slope(state_left_neighbor, state_self, state_right_neighbor);
}

// Stateless functor wrapping the free functions above into the same
// swappable shape `CentralDifferenceReconstruction` uses:
// `Reconstruction::right(...)/left(...)`. Solver residual code
// (solver/explicit/fvm_solver.hpp) never changes when this is
// substituted for `CentralDifferenceReconstruction` as a template
// argument.
struct MusclMinmodReconstruction
{
  template <class Scalar>
  CFE_HOST_DEVICE
  Scalar right(Scalar state_left_neighbor, Scalar state_self, Scalar state_right_neighbor) const
  {
    return muscl_minmod_value_right(state_left_neighbor, state_self, state_right_neighbor);
  }

  template <class Scalar>
  CFE_HOST_DEVICE
  Scalar left(Scalar state_left_neighbor, Scalar state_self, Scalar state_right_neighbor) const
  {
    return muscl_minmod_value_left(state_left_neighbor, state_self, state_right_neighbor);
  }
};

}  // namespace fvm
}  // namespace cfe
