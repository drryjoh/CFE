// Shock-feature-tracking utilities: locating a shock's position from a
// density profile, and estimating a propagation speed from tracked
// positions over time (ROADMAP.md Phase 4's "feature tracking and
// periodic propagation-speed estimate",
// tasks/0013-phase4-propagating-shock-tracking.md).
//
// Deliberately generic over the caller's profile representation (two
// parallel std::vector<Scalar>, x and rho) rather than tied to any one
// test's own snapshot struct, so this is directly reusable by a future
// moving-frame/grid-recycling prototype (ARCHITECTURE.md #12's own
// step 1-2, "track a feature" / "estimate propagation speed
// periodically", is exactly this capability).
#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "cfe/core/types.hpp"

namespace cfe_verification {

// Locates the x-position where a monotonic density profile crosses the
// midpoint value between two known plateaus (`rho_behind` on the side
// the shock has already passed, `rho_ahead` on the side it has not),
// via linear interpolation between the two bracketing cells.
//
// Chosen over steepest-gradient (argmax|drho/dx|) specifically because
// a midpoint-density crossing is, to first order, insensitive to the
// WIDTH/SHAPE of the numerically smeared transition (as long as the
// profile is monotonic through it) -- RusanovFlux and HllcFlux smear a
// shock by different amounts and with different profile shapes, so a
// shape-sensitive detector like argmax-gradient risks reporting
// different "positions" for the two fluxes even when the physical
// shock has not moved, exactly the spurious-speed-difference risk this
// file exists to avoid introducing itself (Violet's pre-implementation
// consult, tasks/0013-....md).
//
// Scans left-to-right for the first crossing of the midpoint in the
// direction implied by which plateau is larger (handles both a
// right-moving shock with higher density behind it, the common case
// here, and the mirrored left-moving case) and throws if no crossing is
// found (a caller bug -- e.g. a snapshot taken before the shock entered
// the domain, or after it exited -- not a case to silently return a
// nonsense position for).
template <class Scalar>
double locate_shock_position(const std::vector<Scalar>& x, const std::vector<Scalar>& rho, double rho_behind,
                              double rho_ahead)
{
  if (x.size() != rho.size() || x.size() < 2) {
    throw std::invalid_argument("locate_shock_position: x/rho must be parallel vectors of size >= 2");
  }

  const double mid = 0.5 * (rho_behind + rho_ahead);
  const bool behind_is_higher = rho_behind > rho_ahead;

  for (cfe::Index i = 0; i + 1 < rho.size(); ++i) {
    const double a = static_cast<double>(rho[i]);
    const double b = static_cast<double>(rho[i + 1]);
    const bool crosses = behind_is_higher ? (a >= mid && b < mid) : (a <= mid && b > mid);
    if (crosses) {
      const double frac = (mid - a) / (b - a);
      return static_cast<double>(x[i]) + frac * (static_cast<double>(x[i + 1]) - static_cast<double>(x[i]));
    }
  }
  throw std::runtime_error("locate_shock_position: no midpoint crossing found in profile");
}

// Ordinary least-squares slope of `positions` against `times` (both
// size >= 2) -- the propagation-speed estimate from several tracked
// snapshots, averaging down per-snapshot position noise roughly as
// 1/sqrt(N) (Violet's consult) rather than a two-point endpoint
// difference, which uses only the first/last snapshot and discards
// every point in between.
inline double least_squares_slope(const std::vector<double>& times, const std::vector<double>& positions)
{
  if (times.size() != positions.size() || times.size() < 2) {
    throw std::invalid_argument("least_squares_slope: times/positions must be parallel vectors of size >= 2");
  }
  const cfe::Index n = times.size();
  double sum_t = 0.0, sum_x = 0.0, sum_tt = 0.0, sum_tx = 0.0;
  for (cfe::Index i = 0; i < n; ++i) {
    sum_t += times[i];
    sum_x += positions[i];
    sum_tt += times[i] * times[i];
    sum_tx += times[i] * positions[i];
  }
  const double n_d = static_cast<double>(n);
  const double denominator = n_d * sum_tt - sum_t * sum_t;
  return (n_d * sum_tx - sum_t * sum_x) / denominator;
}

}  // namespace cfe_verification
