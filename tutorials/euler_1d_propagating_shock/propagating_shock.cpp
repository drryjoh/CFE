// 1D compressible Euler tutorial: a standalone propagating shock, with
// feature tracking and a periodic propagation-speed estimate
// (ROADMAP.md Phase 4, tasks/0013-phase4-propagating-shock-tracking.md).
// Same EulerField + MusclMinmodReconstruction + SSP-RK2 stack
// tests/verification/test_euler_propagating_shock.cpp verifies -- this
// tutorial's own job is to run it with both numerical fluxes, write the
// tracked shock position at each snapshot, fit a propagation-speed
// estimate, and report it against the analytic Rankine-Hugoniot speed.
//
// A Mach-2 right-moving shock into otherwise-uniform, at-rest gas
// (gamma=1.4) -- deliberately NOT a reuse of Sod's own shock (adjacent
// to a contact that smears differently between fluxes, which would
// contaminate a position measurement; see propagating_shock_setup.hpp's
// header comment for the full rationale).
//
// Field/grid/solver precision is `cfe::scalar`, matching every other
// tutorial's convention; the RH jump/analytic speed and reported
// metrics stay `double`.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "cfe/core/types.hpp"
#include "cfe/numerics/numerical_flux/hllc.hpp"
#include "propagating_shock_setup.hpp"
#include "shock_feature_tracking.hpp"

namespace fs = std::filesystem;
using Scalar = cfe::scalar;

namespace {

constexpr Scalar kGamma = Scalar(1.4);
constexpr Scalar kMach = Scalar(2.0);
constexpr Scalar kX0 = Scalar(0.2);
constexpr Scalar kCfl = Scalar(0.4);
const std::vector<std::size_t> kGrids = {100, 200, 400, 800};

// Picked so the shock (speed ~2.3664 for the default Mach 2 below)
// travels comfortably inside [0,1] for the whole run -- if you change
// kMach/kX0, re-check that the LAST output time here still keeps
// `x0 + shock_speed*t_final` inside (0,1) with margin. If the shock
// exits the domain before a snapshot is taken,
// `locate_shock_position` (shock_feature_tracking.hpp) finds no
// midpoint-density crossing in that snapshot's profile and throws
// `std::runtime_error` rather than silently returning a wrong position.
std::vector<Scalar> output_times()
{
  std::vector<Scalar> times;
  for (int i = 1; i <= 8; ++i) times.push_back(Scalar(0.025) * Scalar(i));
  return times;
}

template <class NumericalFlux>
void run_case(const std::string& flux_name, std::size_t nx, const cfe_verification::NormalShockJump& jump,
              std::ofstream& summary, std::ofstream& tracks)
{
  const auto snapshots =
      cfe_verification::run_propagating_shock<Scalar, NumericalFlux>(nx, kGamma, jump, kX0, output_times(), kCfl);

  std::vector<double> times, positions;
  for (const auto& snap : snapshots) {
    const double t = static_cast<double>(snap.t);
    const double x = cfe_verification::locate_shock_position(snap.x, snap.rho, jump.behind.rho, jump.ahead.rho);
    times.push_back(t);
    positions.push_back(x);
    tracks << nx << ',' << flux_name << ',' << t << ',' << x << '\n';
  }

  const double speed = cfe_verification::least_squares_slope(times, positions);
  const double error = std::fabs(speed - jump.shock_speed);
  summary << nx << ',' << flux_name << ',' << speed << ',' << jump.shock_speed << ',' << error << '\n';
  std::printf("  [nx=%4zu, %-7s] measured speed=%.6f  analytic=%.6f  |error|=%.6f\n", nx, flux_name.c_str(), speed,
              jump.shock_speed, error);
}

}  // namespace

int main()
{
  const cfe_verification::NormalShockJump jump =
      cfe_verification::normal_shock_jump(static_cast<double>(kGamma), 1.0, 1.0, static_cast<double>(kMach));

  std::printf("=== 1D propagating shock (Mach %.1f, gamma=1.4): feature tracking, Rusanov vs. HLLC ===\n",
              static_cast<double>(kMach));
  std::printf("Behind state: rho=%.6f u=%.6f p=%.6f | shock speed (analytic) = %.6f\n", jump.behind.rho,
              jump.behind.u, jump.behind.p, jump.shock_speed);

  const fs::path data_dir = "data";
  fs::create_directories(data_dir);
  std::ofstream summary(data_dir / "summary.csv");
  summary << "nx,flux,measured_speed,analytic_speed,abs_error\n";
  std::ofstream tracks(data_dir / "tracked_positions.csv");
  tracks << "nx,flux,t,x_shock\n";

  for (std::size_t nx : kGrids) {
    run_case<cfe::RusanovFlux>("rusanov", nx, jump, summary, tracks);
    run_case<cfe::HllcFlux>("hllc", nx, jump, summary, tracks);
  }
  summary.close();
  tracks.close();

  std::printf(
      "\nWrote %s/summary.csv (measured vs analytic speed) and %s/tracked_positions.csv (shock position per "
      "snapshot). Run plot_results.py to generate figures.\n",
      data_dir.string().c_str(), data_dir.string().c_str());
  return 0;
}
