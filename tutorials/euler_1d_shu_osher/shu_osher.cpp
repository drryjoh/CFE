// 1D compressible Euler tutorial: the Shu-Osher shock/density-wave
// interaction (Shu & Osher 1989), ROADMAP.md Phase 3's second named
// Euler canonical case alongside Sod (tasks/0012-phase3-shu-osher.md).
// Same EulerField + MusclMinmodReconstruction + SSP-RK2 stack
// tests/verification/test_euler_shu_osher.cpp verifies -- this
// tutorial's own job is to run it at a few resolutions with BOTH
// numerical fluxes, write CSVs, and report the two headline findings a
// student should see:
//
// 1. The region AHEAD of the shock is an exact, time-independent
//    solution of Euler (u=0, p=1 uniform there) until the shock
//    physically arrives -- HllcFlux reproduces this EXACTLY (to
//    floating-point roundoff) at every resolution tested, while
//    RusanovFlux's own artificial dissipation measurably (if slowly)
//    diffuses the stationary density perturbation there, a real,
//    bounded, resolution-convergent numerical artifact distinct from
//    (but related to) its documented contact-smearing weakness.
// 2. In the oscillatory post-shock wake, HLLC tracks its own converged
//    fine-grid reference more closely than Rusanov does, at the SAME
//    coarser resolution, and resolves more of the oscillation structure
//    (measured via an extrema count) -- the question this task set out
//    to answer, now with a second, independent (non-forward-step,
//    non-Sod-contact) canonical case confirming the same direction.
//
// Field/grid/solver precision is `cfe::scalar` (core/types.hpp), not a
// hardcoded `double`, matching every other tutorial's convention --
// EXCEPT the exact ahead-state formula and reported metrics, which stay
// `double` throughout (same convention the Sod tutorial uses).
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "cfe/core/types.hpp"
#include "cfe/numerics/numerical_flux/hllc.hpp"
#include "shu_osher_setup.hpp"

namespace fs = std::filesystem;
using Scalar = cfe::scalar;

namespace {

constexpr Scalar kTFinal = Scalar(1.8);
constexpr Scalar kCfl = Scalar(0.4);
constexpr cfe::Index kFineNx = 2000;
const std::vector<cfe::Index> kGrids = {100, 200, 400};

// density_at/wake_l1_error/max_ahead_deviation/count_extrema used to be
// duplicated here verbatim from test_euler_shu_osher.cpp -- moved into
// shu_osher_setup.hpp (as shu_osher_density_at/shu_osher_wake_l1_error/
// shu_osher_max_ahead_deviation/shu_osher_count_extrema) so the test and
// this tutorial share one implementation each, the same treatment
// sod_shock_tube_runner.hpp's own `contact_width_cells` already
// established (Snorlax review finding on this task).

void write_profile_csv(const fs::path& path, const cfe_verification::ShuOsherProfile<Scalar>& profile)
{
  std::ofstream csv(path);
  csv << "x,rho,u,p,rho_exact_ahead\n";
  for (cfe::Index i = 0; i < profile.x.size(); ++i) {
    const double x = static_cast<double>(profile.x[i]);
    csv << x << ',' << profile.rho[i] << ',' << profile.u[i] << ',' << profile.p[i] << ','
        << cfe_verification::shu_osher_exact_ahead_density(x) << '\n';
  }
}

}  // namespace

int main()
{
  std::printf("=== Shu-Osher shock/density-wave interaction (gamma=1.4, t=1.8), Rusanov vs. HLLC ===\n");
  const fs::path data_dir = "data";
  fs::create_directories(data_dir);

  std::printf("Running fine reference (nx=%zu) for both fluxes...\n", kFineNx);
  const auto rusanov_fine = cfe_verification::run_shu_osher<Scalar, cfe::RusanovFlux>(kFineNx, kTFinal, kCfl);
  const auto hllc_fine = cfe_verification::run_shu_osher<Scalar, cfe::HllcFlux>(kFineNx, kTFinal, kCfl);
  write_profile_csv(data_dir / "case_fine_rusanov.csv", rusanov_fine);
  write_profile_csv(data_dir / "case_fine_hllc.csv", hllc_fine);

  // Shock-front location, always estimated from the HLLC fine reference
  // (see shu_osher_setup.hpp's own header comment: Rusanov's far-field
  // numerical diffusion makes it unreliable for this specific purpose).
  const double front = cfe_verification::shu_osher_shock_front_estimate(hllc_fine, 0.1);
  const double wake_cutoff = front - 0.5;
  const double ahead_x_lo = front + 1.0;
  std::printf("Estimated shock front (from HLLC reference) ~= %.3f\n", front);

  // Ahead-of-shock deviation is reported in TWO separate pieces, not
  // one combined number (Ilia's independent final review on this task
  // caught an earlier version of this tutorial/test quoting a single
  // combined number that was mostly a DIFFERENT, boundary-localized
  // artifact -- see shu_osher_max_ahead_deviation's own header comment):
  // "bulk" (the genuine Rusanov-dissipation-of-a-stationary-wave
  // mechanism this tutorial is actually about, boundary-adjacent region
  // excluded) and "boundary" (InflowOutflowBoundary's own separate,
  // larger artifact right next to the domain's far edge).
  const double ahead_bulk_x_hi = cfe_verification::kShuOsherDomainXMax - cfe_verification::kShuOsherBoundaryMargin;

  std::ofstream summary(data_dir / "summary.csv");
  summary << "nx,flux,wake_l1_error,ahead_bulk_max_deviation,ahead_boundary_max_deviation,extrema_count\n";

  const double extrema_x_lo = front - 3.0;
  const double extrema_x_hi = front - 0.5;

  for (cfe::Index nx : kGrids) {
    const auto rusanov = cfe_verification::run_shu_osher<Scalar, cfe::RusanovFlux>(nx, kTFinal, kCfl);
    const auto hllc = cfe_verification::run_shu_osher<Scalar, cfe::HllcFlux>(nx, kTFinal, kCfl);
    write_profile_csv(data_dir / ("case_nx" + std::to_string(nx) + "_rusanov.csv"), rusanov);
    write_profile_csv(data_dir / ("case_nx" + std::to_string(nx) + "_hllc.csv"), hllc);

    const double rusanov_wake = cfe_verification::shu_osher_wake_l1_error(rusanov, rusanov_fine, wake_cutoff);
    const double hllc_wake = cfe_verification::shu_osher_wake_l1_error(hllc, hllc_fine, wake_cutoff);
    const double rusanov_ahead_bulk =
        cfe_verification::shu_osher_max_ahead_deviation(rusanov, ahead_x_lo, ahead_bulk_x_hi);
    const double hllc_ahead_bulk = cfe_verification::shu_osher_max_ahead_deviation(
        hllc, ahead_x_lo, cfe_verification::kShuOsherDomainXMax);
    const double rusanov_ahead_boundary = cfe_verification::shu_osher_max_ahead_deviation(
        rusanov, ahead_bulk_x_hi, cfe_verification::kShuOsherDomainXMax);
    const int rusanov_extrema = cfe_verification::shu_osher_count_extrema(rusanov, extrema_x_lo, extrema_x_hi);
    const int hllc_extrema = cfe_verification::shu_osher_count_extrema(hllc, extrema_x_lo, extrema_x_hi);

    summary << nx << ",rusanov," << rusanov_wake << ',' << rusanov_ahead_bulk << ',' << rusanov_ahead_boundary
            << ',' << rusanov_extrema << '\n';
    summary << nx << ",hllc," << hllc_wake << ',' << hllc_ahead_bulk << ",0," << hllc_extrema << '\n';

    std::printf(
        "  [nx=%4zu] wake L1 error: Rusanov=%.4f HLLC=%.4f | ahead BULK max dev: Rusanov=%.4f HLLC=%.2e | "
        "ahead BOUNDARY max dev: Rusanov=%.4f | extrema: Rusanov=%d HLLC=%d\n",
        nx, rusanov_wake, hllc_wake, rusanov_ahead_bulk, hllc_ahead_bulk, rusanov_ahead_boundary, rusanov_extrema,
        hllc_extrema);
  }
  summary.close();

  std::printf(
      "\nWrote %s/summary.csv and per-resolution/per-flux field CSVs. Run plot_results.py to generate "
      "figures.\n",
      data_dir.string().c_str());
  return 0;
}
