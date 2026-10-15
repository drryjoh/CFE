// Exact Riemann solver for a calorically-perfect ideal gas (Toro,
// "Riemann Solvers and Numerical Methods for Fluid Dynamics," Ch. 4:
// Sec. 4.3's iterative pressure solution, Sec. 4.5's "Sampling the
// solution" procedure). General (any left/right primitive state, any
// gamma), not specific to the Sod tube -- it is named after Sod's
// problem only because that is the one canonical case this codebase
// currently verifies against (tests/verification/test_euler_sod_shock_tube.cpp,
// tutorials/euler_1d_sod_shock_tube/). Reused, not reimplemented, by
// both.
//
// This lives under tests/verification/ (ARCHITECTURE.md #3's proposed
// `tests/verification/` category), not src/cfe/: it is a reference
// solution for validating the solver, not part of the physics/numerics
// the solver itself ships.
//
// Every value here is plain `double`, regardless of what `Scalar` the
// solver/field UNDER TEST happens to be instantiated at (`float` or
// `double` -- see `AGENTS.md` #11, `docs/adr/0014-dual-precision-
// verification-requirement.md`): this is a reference answer, not
// simulation state, so it is computed once, at the tightest precision
// readily available, and compared against regardless of which
// precision is being verified. See any `tests/verification/test_euler_*
// .cpp` file's own header comment for a worked example of this
// convention at a call site.
//
// The constants this solver produces for the classic Sod parameters
// (rho_L=1, u_L=0, p_L=1, rho_R=0.125, u_R=0, p_R=0.1, gamma=1.4) were
// independently cross-checked with a standalone Python re-implementation
// of the same Newton-Raphson iteration before this header was trusted as
// a test oracle: p*=0.303130, u*=0.927453, rho*_L=0.426319,
// rho*_R=0.265574, left rarefaction head/tail speeds -1.183216/-0.070273,
// right shock speed 1.752156 -- all of which also match the commonly
// published values for this exact problem (e.g. Toro Table 4.1's own
// worked Sod-problem example).
#pragma once

#include <cmath>

namespace cfe_verification {

struct PrimitiveState
{
  double rho;
  double u;
  double p;
};

namespace detail {

inline double A_coeff(double gamma, double rho_k) { return 2.0 / ((gamma + 1.0) * rho_k); }
inline double B_coeff(double gamma, double p_k) { return (gamma - 1.0) / (gamma + 1.0) * p_k; }

// Toro eq. 4.6/4.7: the pressure function for one side and its
// derivative, branching on shock (p > p_k) vs. rarefaction (p <= p_k).
inline double pressure_function(double gamma, double p, double rho_k, double p_k, double c_k)
{
  if (p > p_k) {
    const double a_k = A_coeff(gamma, rho_k);
    const double b_k = B_coeff(gamma, p_k);
    return (p - p_k) * std::sqrt(a_k / (p + b_k));
  }
  return (2.0 * c_k / (gamma - 1.0)) * (std::pow(p / p_k, (gamma - 1.0) / (2.0 * gamma)) - 1.0);
}

inline double pressure_function_derivative(double gamma, double p, double rho_k, double p_k, double c_k)
{
  if (p > p_k) {
    const double a_k = A_coeff(gamma, rho_k);
    const double b_k = B_coeff(gamma, p_k);
    return std::sqrt(a_k / (b_k + p)) * (1.0 - (p - p_k) / (2.0 * (b_k + p)));
  }
  return (1.0 / (rho_k * c_k)) * std::pow(p / p_k, -(gamma + 1.0) / (2.0 * gamma));
}

}  // namespace detail

// Solves for the star-region pressure p* and velocity u* via
// Newton-Raphson (Toro Sec. 4.3), starting from the simple two-shock-ish
// initial guess `0.5*(p_l+p_r)` -- adequate for the Sod problem this
// header is exercised against; a stronger initial guess (e.g. Toro's
// two-rarefaction/two-shock approximations) would be needed for more
// extreme pressure ratios, out of scope here.
struct StarRegion
{
  double p_star;
  double u_star;
};

inline StarRegion solve_star_region(double gamma, const PrimitiveState& left, const PrimitiveState& right,
                                     int max_iterations = 100, double tolerance = 1e-14)
{
  const double c_l = std::sqrt(gamma * left.p / left.rho);
  const double c_r = std::sqrt(gamma * right.p / right.rho);

  auto f = [&](double p) {
    return detail::pressure_function(gamma, p, left.rho, left.p, c_l) +
           detail::pressure_function(gamma, p, right.rho, right.p, c_r) + (right.u - left.u);
  };
  auto f_prime = [&](double p) {
    return detail::pressure_function_derivative(gamma, p, left.rho, left.p, c_l) +
           detail::pressure_function_derivative(gamma, p, right.rho, right.p, c_r);
  };

  double p = 0.5 * (left.p + right.p);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    const double f_p = f(p);
    const double f_prime_p = f_prime(p);
    double p_new = p - f_p / f_prime_p;
    if (p_new <= 0.0) p_new = 1e-6;
    const bool converged = std::fabs(p_new - p) < tolerance;
    p = p_new;
    if (converged) break;
  }

  const double u_star = 0.5 * (left.u + right.u) +
                         0.5 * (detail::pressure_function(gamma, p, right.rho, right.p, c_r) -
                                detail::pressure_function(gamma, p, left.rho, left.p, c_l));
  return StarRegion{p, u_star};
}

// Samples the exact solution at similarity variable s = (x - x0) / t
// (t > 0), given the already-solved star region -- Toro Sec. 4.5.
inline PrimitiveState sample_exact_solution(double gamma, const PrimitiveState& left,
                                             const PrimitiveState& right, const StarRegion& star, double s)
{
  const double c_l = std::sqrt(gamma * left.p / left.rho);
  const double c_r = std::sqrt(gamma * right.p / right.rho);
  const double p_star = star.p_star;
  const double u_star = star.u_star;

  if (s <= u_star) {
    // Left of the contact: left state, left star state, or inside a
    // left-facing wave, depending on wave type and whether `s` is
    // inside a rarefaction fan.
    //
    // Physically: if the star-region pressure p_star is LOWER than
    // this side's own initial pressure, the gas on this side must
    // EXPAND to reach it -- an expansion is smooth and isentropic, a
    // rarefaction fan, never a shock (a real gas cannot shock into a
    // lower-pressure state). If p_star is HIGHER instead, the gas must
    // be compressed, which happens discontinuously -- a shock. This
    // same left.p/right.p comparison against p_star is what selects
    // the branch on both sides below.
    if (p_star <= left.p) {
      // Left rarefaction.
      const double head_speed = left.u - c_l;
      if (s <= head_speed) return left;
      const double c_star_l = c_l * std::pow(p_star / left.p, (gamma - 1.0) / (2.0 * gamma));
      const double tail_speed = u_star - c_star_l;
      if (s >= tail_speed) {
        const double rho_star_l = left.rho * std::pow(p_star / left.p, 1.0 / gamma);
        return PrimitiveState{rho_star_l, u_star, p_star};
      }
      // Inside the fan itself (Toro eq. 4.56).
      const double c = (2.0 / (gamma + 1.0)) * c_l + ((gamma - 1.0) / (gamma + 1.0)) * (left.u - s);
      const double u = (2.0 / (gamma + 1.0)) * (c_l + (gamma - 1.0) / 2.0 * left.u + s);
      const double rho = left.rho * std::pow(c / c_l, 2.0 / (gamma - 1.0));
      const double p = left.p * std::pow(c / c_l, 2.0 * gamma / (gamma - 1.0));
      return PrimitiveState{rho, u, p};
    } else {
      // Left shock.
      const double shock_speed =
          left.u - c_l * std::sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / left.p) +
                                    (gamma - 1.0) / (2.0 * gamma));
      if (s <= shock_speed) return left;
      const double ratio = p_star / left.p;
      const double rho_star_l =
          left.rho * ((ratio + (gamma - 1.0) / (gamma + 1.0)) /
                       ((gamma - 1.0) / (gamma + 1.0) * ratio + 1.0));
      return PrimitiveState{rho_star_l, u_star, p_star};
    }
  } else {
    // Right of the contact.
    if (p_star > right.p) {
      // Right shock.
      const double shock_speed =
          right.u + c_r * std::sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / right.p) +
                                     (gamma - 1.0) / (2.0 * gamma));
      if (s >= shock_speed) return right;
      const double ratio = p_star / right.p;
      const double rho_star_r =
          right.rho * ((ratio + (gamma - 1.0) / (gamma + 1.0)) /
                        ((gamma - 1.0) / (gamma + 1.0) * ratio + 1.0));
      return PrimitiveState{rho_star_r, u_star, p_star};
    } else {
      // Right rarefaction.
      const double head_speed = right.u + c_r;
      if (s >= head_speed) return right;
      const double c_star_r = c_r * std::pow(p_star / right.p, (gamma - 1.0) / (2.0 * gamma));
      const double tail_speed = u_star + c_star_r;
      if (s <= tail_speed) {
        const double rho_star_r = right.rho * std::pow(p_star / right.p, 1.0 / gamma);
        return PrimitiveState{rho_star_r, u_star, p_star};
      }
      // Inside the fan itself (mirrored Toro eq. 4.63).
      const double c = (2.0 / (gamma + 1.0)) * c_r - ((gamma - 1.0) / (gamma + 1.0)) * (right.u - s);
      const double u = (2.0 / (gamma + 1.0)) * (-c_r + (gamma - 1.0) / 2.0 * right.u + s);
      const double rho = right.rho * std::pow(c / c_r, 2.0 / (gamma - 1.0));
      const double p = right.p * std::pow(c / c_r, 2.0 * gamma / (gamma - 1.0));
      return PrimitiveState{rho, u, p};
    }
  }
}

}  // namespace cfe_verification
