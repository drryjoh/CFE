// 1D compressible Euler tutorial: the Sod (1978) shock tube, Phase 3's
// companion 1D verification case (ROADMAP.md Phase 3, "Sod shock
// tube"). Same EulerField + MusclMinmodReconstruction + SSP-RK2 stack
// tests/verification/test_euler_sod_shock_tube.cpp (Rusanov) and
// test_euler_sod_shock_tube_hllc.cpp (HLLC) verify -- in fact the actual
// simulation driver (cfe_verification::run_sod_shock_tube,
// tests/verification/sod_shock_tube_runner.hpp) is the SAME code both
// unit tests and this tutorial call, not a second independent
// implementation; this tutorial's own job is just to run it at a few
// resolutions with BOTH numerical fluxes, compare against the exact
// solution, measure/report each flux's contact-discontinuity width, and
// write CSVs/plots for a student to look at.
//
// Classic parameters: gamma=1.4, left (rho,u,p)=(1,0,1), right
// (rho,u,p)=(0.125,0,0.1), diaphragm at x=0.5 in a [0,1] tube, 3
// resolutions (100/200/400 cells) run to t=0.2 and compared against the
// exact Riemann solution (tests/verification/sod_shock_tube_exact.hpp).
//
// Task 0011 (docs/adr/0012-hllc-numerical-flux.md) added HLLC alongside
// the original Rusanov run specifically to give a student a direct,
// quantitative look at HLLC's headline claim (sharper contact
// resolution) rather than just a second set of numbers: each resolution
// now reports BOTH fluxes' measured contact width (cells spanned
// crossing from 5% to 95% of the density jump across the contact), the
// same metric test_euler_sod_shock_tube_hllc.cpp's own verification
// test uses.
//
// Field/grid/solver precision is `cfe::scalar` (core/types.hpp), not a
// hardcoded `double`, matching every other tutorial's convention
// (`-DCFE_SCALAR_TYPE=float` switches this tutorial's actual simulation
// precision too) -- EXCEPT the exact-solution oracle and reported
// metrics, which stay `double` throughout (same convention
// tests/unit/test_burgers_*.cpp and that tutorial use), so "scheme
// behavior at reduced storage precision" is never conflated with
// "reporting/accumulation roundoff".
//
// Output: data/summary.csv (L1 errors AND contact width, one row per
// resolution/flux) plus data/case_nx*_<flux>.csv (x, rho/u/p numerical
// and exact at t=0.2) for plot_results.py.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "cfe/core/types.hpp"
#include "cfe/numerics/numerical_flux/hllc.hpp"
#include "sod_shock_tube_exact.hpp"
#include "sod_shock_tube_runner.hpp"

namespace fs = std::filesystem;
using Scalar = cfe::scalar;

namespace {

constexpr Scalar kGamma = Scalar(1.4);
constexpr Scalar kX0 = Scalar(0.5);
constexpr Scalar kTFinal = Scalar(0.2);
constexpr Scalar kCfl = Scalar(0.4);
const std::vector<std::size_t> kGrids = {100, 200, 400};

const cfe_verification::PrimitiveState kLeft{1.0, 0.0, 1.0};
const cfe_verification::PrimitiveState kRight{0.125, 0.0, 0.1};

// Number of cells spanned crossing from 5% to 95% of the density jump
// across the contact -- cfe_verification::contact_width_cells
// (sod_shock_tube_runner.hpp), shared with
// test_euler_sod_shock_tube_hllc.cpp's own verification test (Snorlax
// review finding: this used to be duplicated verbatim between the two).

template <class NumericalFlux>
void run_case(const std::string& flux_name, std::size_t nx, std::ofstream& summary, const fs::path& data_dir)
{
  const cfe_verification::SodProfile<Scalar> profile =
      cfe_verification::run_sod_shock_tube<Scalar, NumericalFlux>(nx, kGamma, kLeft, kRight, kX0, kTFinal, kCfl);

  const auto star = cfe_verification::solve_star_region(static_cast<double>(kGamma), kLeft, kRight);

  std::ofstream field_csv(data_dir / ("case_nx" + std::to_string(nx) + "_" + flux_name + ".csv"));
  field_csv << "x,rho_numerical,u_numerical,p_numerical,rho_exact,u_exact,p_exact\n";

  double l1_rho = 0.0, l1_u = 0.0, l1_p = 0.0;
  const double dx_d = 1.0 / static_cast<double>(nx);
  for (std::size_t i = 0; i < profile.x.size(); ++i) {
    const double x = static_cast<double>(profile.x[i]);
    const double rho_num = static_cast<double>(profile.rho[i]);
    const double u_num = static_cast<double>(profile.u[i]);
    const double p_num = static_cast<double>(profile.p[i]);

    const double s = (x - static_cast<double>(kX0)) / static_cast<double>(kTFinal);
    const auto exact = cfe_verification::sample_exact_solution(static_cast<double>(kGamma), kLeft,
                                                                 kRight, star, s);

    l1_rho += std::fabs(rho_num - exact.rho) * dx_d;
    l1_u += std::fabs(u_num - exact.u) * dx_d;
    l1_p += std::fabs(p_num - exact.p) * dx_d;

    field_csv << x << ',' << rho_num << ',' << u_num << ',' << p_num << ',' << exact.rho << ','
              << exact.u << ',' << exact.p << '\n';
  }

  const double x_contact = static_cast<double>(kX0) + star.u_star * static_cast<double>(kTFinal);
  const double s_contact_minus = (x_contact - 0.01 - static_cast<double>(kX0)) / static_cast<double>(kTFinal);
  const double s_contact_plus = (x_contact + 0.01 - static_cast<double>(kX0)) / static_cast<double>(kTFinal);
  const double rho_star_left =
      cfe_verification::sample_exact_solution(static_cast<double>(kGamma), kLeft, kRight, star, s_contact_minus).rho;
  const double rho_star_right =
      cfe_verification::sample_exact_solution(static_cast<double>(kGamma), kLeft, kRight, star, s_contact_plus).rho;
  const std::size_t width = cfe_verification::contact_width_cells(profile, x_contact, rho_star_left, rho_star_right);

  summary << nx << ',' << flux_name << ',' << l1_rho << ',' << l1_u << ',' << l1_p << ',' << width << '\n';
  std::printf("  [nx=%4zu, %-7s] L1(rho)=%.4e  L1(u)=%.4e  L1(p)=%.4e  contact_width=%zu cells\n", nx,
              flux_name.c_str(), l1_rho, l1_u, l1_p, width);
}

}  // namespace

int main()
{
  std::printf("=== Sod shock tube (gamma=1.4, t=0.2), Rusanov vs. HLLC ===\n");
  const fs::path data_dir = "data";
  fs::create_directories(data_dir);
  std::ofstream summary(data_dir / "summary.csv");
  summary << "nx,flux,l1_rho,l1_u,l1_p,contact_width_cells\n";
  for (std::size_t nx : kGrids) {
    run_case<cfe::RusanovFlux>("rusanov", nx, summary, data_dir);
    run_case<cfe::HllcFlux>("hllc", nx, summary, data_dir);
  }
  summary.close();
  std::printf("\nWrote %s/summary.csv and per-resolution/per-flux field CSVs. Run plot_results.py to "
              "generate figures.\n",
              data_dir.string().c_str());
  return 0;
}
