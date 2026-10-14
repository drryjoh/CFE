// Standard normal-shock (Rankine-Hugoniot) jump conditions for an ideal
// gas, ahead state at rest -- the shared closed-form algebra behind
// BOTH task 0012's Shu-Osher left (post-shock) state and task 0013's
// standalone propagating-shock IC (shu_osher_setup.hpp,
// propagating_shock_setup.hpp). Pulled out to its own header (Snorlax/
// Sally review finding on tasks 0012/0013: shu_osher_setup.hpp
// originally hardcoded its own Mach-3 jump as three separately-derived
// literal constants instead of calling this same formula, leaving a
// student wanting a different Mach number with no single knob to turn)
// -- one formula, one place, reused by both canonical cases.
//
// Not the general iterative Riemann solver sod_shock_tube_exact.hpp
// implements (that handles an ARBITRARY left/right state pair with any
// mix of shock/rarefaction/contact, via Newton iteration); this is pure
// closed-form algebra for exactly one normal-shock jump at a chosen
// Mach number, valid only when the ahead state is at rest
// (`u_ahead=0`) -- e.g. Anderson, "Modern Compressible Flow," Ch. 3, or
// any standard gas-dynamics reference.
#pragma once

#include <cmath>

#include "sod_shock_tube_exact.hpp"  // for PrimitiveState (shared shape, not its Riemann solver)

namespace cfe_verification {

struct NormalShockJump
{
  PrimitiveState ahead;
  PrimitiveState behind;
  double shock_speed;  // lab-frame speed of the shock itself (rightward, positive)
};

inline NormalShockJump normal_shock_jump(double gamma, double rho_ahead, double p_ahead, double mach)
{
  const double c_ahead = std::sqrt(gamma * p_ahead / rho_ahead);
  const double rho_ratio = (gamma + 1.0) * mach * mach / ((gamma - 1.0) * mach * mach + 2.0);
  const double p_ratio = (2.0 * gamma * mach * mach - (gamma - 1.0)) / (gamma + 1.0);
  const double u_behind = (2.0 / (gamma + 1.0)) * (mach - 1.0 / mach) * c_ahead;

  NormalShockJump jump;
  jump.ahead = PrimitiveState{rho_ahead, 0.0, p_ahead};
  jump.behind = PrimitiveState{rho_ahead * rho_ratio, u_behind, p_ahead * p_ratio};
  jump.shock_speed = mach * c_ahead;
  return jump;
}

}  // namespace cfe_verification
