// 1D inviscid Burgers tutorial: three selectable cases, all using the
// same FvmSolver + BurgersField + RusanovFlux + SSP-RK2 stack PR #3
// verified, with ghost cells refreshed by solver.residual() before
// every single residual evaluation (the default behavior that stack
// already has -- none of these cases' boundary values depend on
// wall-clock time, so the existing, unmodified cfe::ssp_rk2_step is
// used directly; see the 2D tutorial for the case where per-RK-stage
// time threading is actually needed).
//
//   Case A ("shock"): a moving step (u_left=1 > u_right=0) with fixed
//   inflow at the left and zero-order extrapolation at the right
//   (InflowOutflowBoundary) -- compares FirstOrderReconstruction against
//   MusclMinmodReconstruction at three grid resolutions, against the
//   exact Rankine-Hugoniot cell-average reference (including cells the
//   shock itself straddles).
//
//   Case B ("steepening"): a smooth periodic sine profile that steepens
//   into a shock in finite time -- shows the transition directly,
//   against the exact method-of-characteristics reference before the
//   analytic breaking time and labeled as numerical-only after it (no
//   multivalued-characteristics reference is ever plotted).
//
//   Case C ("rarefaction"): a TRANSONIC rarefaction fan (u_left=-1 <
//   u_right=1, fixed far-field StaticBoundary on both ends) -- the
//   opposite ordering from Case A, so the Lax entropy condition selects
//   a smooth fan instead of a shock, and the fan's own center is exactly
//   where the characteristic speed crosses zero. This is the direct,
//   end-to-end check of the specific design claim
//   numerics/numerical_flux/rusanov.hpp's own header comment makes:
//   `RusanovFlux` (not `UpwindFlux`) is required for Burgers because
//   `UpwindFlux` has no well-defined upwind side exactly at a sonic
//   point like this one.
//
// Run with no arguments to do all three cases (what "run everything
// before completing the work" needs in one invocation), or
// `--case=shock` / `--case=steepening` / `--case=rarefaction` to run
// just one.
//
// Field/grid/solver precision is `cfe::scalar` (core/types.hpp,
// configured project-wide via the `CFE_SCALAR_TYPE` CMake cache
// variable, `double` by default) -- NOT a hardcoded `double` -- so
// `-DCFE_SCALAR_TYPE=float` switches this tutorial's actual simulation
// precision too, matching AGENTS.md #11's project-wide convention
// (`tutorials/hello_parallel_for/hello_parallel_for.cpp` is the other
// existing example). Reported metrics (L1 error, overshoot/undershoot,
// domain mean/integral) are deliberately still accumulated in `double`
// regardless of `cfe::scalar`, the same convention
// tests/unit/test_burgers_*.cpp's dual-precision tests use -- this
// keeps the reporting itself numerically stable and avoids conflating
// "accumulation roundoff" with "scheme behavior at reduced storage
// precision" when `cfe::scalar` is `float`.
//
// Output: data/summary.csv (one row per case/grid/reconstruction/time,
// the single source of truth for every reported number) plus a handful
// of representative data/case_*.csv field dumps (x, u_numerical,
// u_exact) for plot_results.py to render -- see this directory's
// README.md for exactly which combinations get a field dump and why.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/burgers/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/numerics/fvm/first_order_reconstruction.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

// An anonymous namespace: everything below, until the matching `}` near
// the bottom of this file, is private to this file only. Closest Python
// analogy: names that are never exported from a module.
namespace {

namespace fs = std::filesystem;

// A short, local name standing in for whichever type `cfe::scalar`
// currently is (`float` or `double` -- see this file's header comment).
using Scalar = cfe::scalar;

constexpr Scalar kPi = Scalar(3.14159265358979323846);
constexpr Scalar kCfl = Scalar(0.4);
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------

// Finds where a cell-centered profile crosses `level` on its way down
// (the last index where values[i] >= level and values[i+1] < level),
// linearly interpolating between the two cell centers. NaN if the
// profile never crosses `level` at all (should not happen for either
// case here, but reported honestly rather than silently returning 0).
// Returns `double` (a reported metric, not a state value) even though
// `values` is `Scalar`-precision.
double find_falling_crossing(const std::vector<Scalar>& values, Scalar x0, Scalar dx, Scalar level)
{
  for (std::size_t i = 0; i + 1 < values.size(); ++i) {
    if (values[i] >= level && values[i + 1] < level) {
      const Scalar x_i = x0 + static_cast<Scalar>(i) * dx;
      const Scalar x_ip1 = x0 + static_cast<Scalar>(i + 1) * dx;
      const Scalar frac = (values[i] - level) / (values[i] - values[i + 1]);
      return static_cast<double>(x_i + frac * (x_ip1 - x_i));
    }
  }
  return kNan;
}

std::ofstream open_summary_csv(const fs::path& data_dir)
{
  std::ofstream out(data_dir / "summary.csv");
  out << "case,grid,reconstruction,time,l1_error,shock_position_exact,shock_position_numerical,"
         "overshoot,undershoot,domain_mean\n";
  return out;
}

void append_summary_row(std::ofstream& out, const std::string& case_name, std::size_t nx,
                         const std::string& reconstruction_name, double t, double l1_error,
                         double shock_exact, double shock_numerical, double overshoot, double undershoot,
                         double domain_mean)
{
  out << case_name << ',' << nx << ',' << reconstruction_name << ',' << t << ',' << l1_error << ','
      << shock_exact << ',' << shock_numerical << ',' << overshoot << ',' << undershoot << ','
      << domain_mean << '\n';
}

void write_field_csv(const fs::path& path, const std::vector<Scalar>& x, const std::vector<Scalar>& u_num,
                      const std::vector<Scalar>& u_exact)
{
  std::ofstream out(path);
  out << "x,u_numerical,u_exact\n";
  for (std::size_t i = 0; i < x.size(); ++i) {
    out << x[i] << ',' << u_num[i] << ',';
    if (i < u_exact.size()) {
      out << u_exact[i];
    } else {
      out << "nan";
    }
    out << '\n';
  }
}

// ---------------------------------------------------------------------
// Case A: moving shock
// ---------------------------------------------------------------------

constexpr Scalar kShockULeft = Scalar(1.0);
constexpr Scalar kShockURight = Scalar(0.0);
constexpr Scalar kShockOrigin = Scalar(0.25);
constexpr Scalar kShockSpeed = Scalar(0.5) * (kShockULeft + kShockURight);  // Rankine-Hugoniot, exact for Burgers
const std::vector<std::size_t> kShockGrids = {100, 200, 400};
const std::vector<double> kShockOutputTimes = {0.0, 0.25, 0.5, 1.0};

// Exact cell average of the moving step over [x_lo, x_hi] at time t,
// correct (a weighted blend, not a 0/1 pick) even for a cell the shock
// itself currently straddles.
Scalar shock_exact_cell_average(Scalar x_lo, Scalar x_hi, Scalar t)
{
  const Scalar xs = kShockOrigin + kShockSpeed * t;
  if (xs <= x_lo) return kShockURight;
  if (xs >= x_hi) return kShockULeft;
  const Scalar frac_left = (xs - x_lo) / (x_hi - x_lo);
  return frac_left * kShockULeft + (Scalar(1.0) - frac_left) * kShockURight;
}

template <class Reconstruction>
void run_shock_case(std::size_t nx, const std::string& reconstruction_name, bool write_fields,
                     std::ofstream& summary, const fs::path& data_dir)
{
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = nx;
  grid.ngx = 2;
  grid.dx = Scalar(1.0) / static_cast<Scalar>(nx);

  // Three arrays of `nx` numbers each (plus ghost cells), the same shape
  // as a 1D numpy array: `state` holds the simulation's current values;
  // `stage1` and `scratch` are scratch space the time-stepping algorithm
  // below uses internally.
  cfe::Field<Scalar, 1> state(grid.n_cells_total());
  cfe::Field<Scalar, 1> stage1(grid.n_cells_total());
  cfe::Field<Scalar, 1> scratch(grid.n_cells_total());

  // Initialize finite-volume states using (exact) cell averages, not a
  // cell-center 0/1 sample -- the IC already contains the discontinuity
  // at x=0.25, so this uses the exact same cut-cell formula the
  // reference solution at later times uses.
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const Scalar x_lo = static_cast<Scalar>(i) * grid.dx;
    const Scalar x_hi = x_lo + grid.dx;
    state(grid.flat_index(grid.ngx + i, 0, 0), 0) = shock_exact_cell_average(x_lo, x_hi, Scalar(0.0));
  }

  // `field` picks WHICH equation is being solved (Burgers' equation).
  // `boundary` fixes the inflow value (`kShockULeft`) at the left edge
  // and copies the nearest interior cell at the right edge (outflow).
  // `solver` bundles the equation, grid, boundary, and the
  // `Reconstruction`/flux scheme this function was called with.
  cfe::BurgersField<Scalar, 1> field{};
  cfe::InflowOutflowBoundary<Scalar, 1> boundary{cfe::State<Scalar, 1>(kShockULeft)};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::BurgersField<Scalar, 1>,
                 cfe::InflowOutflowBoundary<Scalar, 1>, cfe::InflowOutflowBoundary<Scalar, 1>,
                 cfe::InflowOutflowBoundary<Scalar, 1>, Reconstruction, cfe::RusanovFlux>
      solver{grid, field, boundary};
  // `residual` is a small function: given the current values (`in`), it
  // writes the instantaneous rate of change into `out` -- the
  // right-hand side `f(u)` of the ODE `du/dt = f(u)` the time-stepper
  // below integrates.
  auto residual = [&](cfe::FieldView<Scalar, 1> in, cfe::FieldView<Scalar, 1> out) {
    solver.residual(in, out);
  };

  auto report = [&](Scalar t) {
    std::vector<Scalar> x(grid.nx), u_num(grid.nx), u_exact(grid.nx);
    double sum_abs_error = 0.0;
    double max_val = static_cast<double>(state(grid.flat_index(grid.ngx, 0, 0), 0));
    double min_val = max_val;
    for (std::size_t i = 0; i < grid.nx; ++i) {
      const Scalar x_lo = static_cast<Scalar>(i) * grid.dx;
      const Scalar x_hi = x_lo + grid.dx;
      const Scalar value = state(grid.flat_index(grid.ngx + i, 0, 0), 0);
      const Scalar exact = shock_exact_cell_average(x_lo, x_hi, t);
      x[i] = grid.x_center(grid.ngx + i);
      u_num[i] = value;
      u_exact[i] = exact;
      sum_abs_error += static_cast<double>(std::abs(value - exact)) * static_cast<double>(grid.dx);
      max_val = std::max(max_val, static_cast<double>(value));
      min_val = std::min(min_val, static_cast<double>(value));
    }
    const double shock_numerical =
        find_falling_crossing(u_num, x[0], grid.dx, Scalar(0.5) * (kShockULeft + kShockURight));
    const double shock_exact = static_cast<double>(kShockOrigin + kShockSpeed * t);
    const double overshoot = std::max(0.0, max_val - static_cast<double>(kShockULeft));
    const double undershoot = std::max(0.0, static_cast<double>(kShockURight) - min_val);
    double domain_mean = 0.0;
    for (Scalar v : u_num) domain_mean += static_cast<double>(v);
    domain_mean *= static_cast<double>(grid.dx);

    const double t_d = static_cast<double>(t);
    append_summary_row(summary, "shock", grid.nx, reconstruction_name, t_d, sum_abs_error, shock_exact,
                        shock_numerical, overshoot, undershoot, domain_mean);
    if (write_fields) {
      char name[160];
      std::snprintf(name, sizeof(name), "case_shock_nx%04zu_%s_t%.2f.csv", grid.nx,
                    reconstruction_name.c_str(), t_d);
      write_field_csv(data_dir / name, x, u_num, u_exact);
    }
    std::printf(
        "  [shock nx=%4zu %-20s] t=%.2f  L1=%.3e  shock(exact=%.4f, num=%.4f)  overshoot=%.2e  "
        "undershoot=%.2e\n",
        grid.nx, reconstruction_name.c_str(), t_d, sum_abs_error, shock_exact, shock_numerical, overshoot,
        undershoot);
  };

  report(Scalar(0.0));
  Scalar t = Scalar(0.0);
  for (std::size_t seg = 1; seg < kShockOutputTimes.size(); ++seg) {
    const Scalar t_target = static_cast<Scalar>(kShockOutputTimes[seg]);
    const Scalar dt_target = kCfl * grid.dx / kShockULeft;
    const int n_steps = static_cast<int>(std::ceil(static_cast<double>((t_target - t) / dt_target)));
    const Scalar dt = (t_target - t) / static_cast<Scalar>(n_steps);
    for (int step = 0; step < n_steps; ++step) {
      cfe::ssp_rk2_step<Scalar>(state.view(), stage1.view(), scratch.view(), dt, residual,
                                 solver.active_cell_count(), solver.active_cell_index_map());
    }
    t = t_target;
    report(t);
  }
}

void run_case_shock(std::ofstream& summary, const fs::path& data_dir)
{
  std::printf("=== Case A: moving shock ===\n");
  for (std::size_t nx : kShockGrids) {
    // Finest grid only -- written for BOTH reconstructions (not just
    // second-order) specifically so plot_results.py's shock-capturing
    // zoom-in can compare them directly at matching resolution.
    const bool write_representative = (nx == kShockGrids.back());
    run_shock_case<cfe::fvm::FirstOrderReconstruction>(nx, "first_order", write_representative, summary,
                                                        data_dir);
    run_shock_case<cfe::fvm::MusclMinmodReconstruction>(nx, "second_order_limited", write_representative,
                                                         summary, data_dir);
  }
}

// ---------------------------------------------------------------------
// Case B: sinusoidal steepening
// ---------------------------------------------------------------------

constexpr Scalar kSteepeningBackground = Scalar(1.0);
constexpr Scalar kSteepeningAmplitude = Scalar(0.5);
constexpr Scalar kSteepeningDomainLength = Scalar(1.0);
// t_s = -1/min(u0') ; u0'(x) = amplitude*2*pi*cos(2*pi*x), min = -amplitude*2*pi
// -> t_s = 1/(2*pi*amplitude) = 1/pi for amplitude=0.5, matching the spec's own stated value.
constexpr Scalar kSteepeningBreakTime = Scalar(1.0) / (Scalar(2.0) * kPi * kSteepeningAmplitude);
const std::vector<double> kSteepeningOutputTimes = {0.0, 0.15, 0.30, 0.40, 0.60};
constexpr std::size_t kSteepeningGrid = 400;

Scalar steepening_ic(Scalar x) { return kSteepeningBackground + kSteepeningAmplitude * std::sin(Scalar(2.0) * kPi * x); }

// Exact method-of-characteristics solution (x = xi + u0(xi)*t, u = u0(xi))
// -- only valid, and only used, strictly before kSteepeningBreakTime;
// see this file's header comment for why.
//
// Solved by bisection, not Newton's method: g(xi) = xi + u0(xi)*t - x is
// strictly increasing in xi for any t < kSteepeningBreakTime
// (g'(xi) = 1 + u0'(xi)*t > 0 everywhere below the breaking time by
// that time's own definition), so a bracketed bisection is
// unconditionally robust here. Plain Newton's method (what
// tests/unit/test_burgers_convergence.cpp uses) is fine at that test's
// own, comfortably-sub-breaking final time, but this tutorial's own
// output times run deliberately close to kSteepeningBreakTime (t=0.30
// vs. t_s=0.318) specifically to show the steepening right up to the
// edge -- and right at that edge, g' gets small for some x, which made
// plain Newton's correction step overshoot wildly for those points
// (caught by inspecting the generated plot: a vertical scribble cutting
// through an otherwise-smooth exact-reference curve, not a subtle
// numerical error). The characteristic displacement u0(xi)*t is bounded
// by the IC's own amplitude times the elapsed time, so [x-2, x+2] safely
// brackets the root for every (x, t) this tutorial ever calls this
// with.
Scalar steepening_exact_presolve(Scalar x, Scalar t)
{
  auto g = [&](Scalar xi) { return xi + steepening_ic(xi) * t - x; };
  Scalar lo = x - Scalar(2.0);
  Scalar hi = x + Scalar(2.0);
  Scalar g_hi = g(hi);
  for (int iteration = 0; iteration < 100; ++iteration) {
    const Scalar mid = Scalar(0.5) * (lo + hi);
    const Scalar g_mid = g(mid);
    if (std::abs(g_mid) < Scalar(1e-13) || (hi - lo) < Scalar(1e-13)) return steepening_ic(mid);
    if ((g_mid > Scalar(0.0)) == (g_hi > Scalar(0.0))) {
      hi = mid;
      g_hi = g_mid;
    } else {
      lo = mid;
    }
  }
  return steepening_ic(Scalar(0.5) * (lo + hi));
}

void run_case_steepening(std::ofstream& summary, const fs::path& data_dir)
{
  std::printf("=== Case B: sinusoidal steepening (breaking time t_s=%.5f) ===\n",
              static_cast<double>(kSteepeningBreakTime));

  cfe::CartesianGrid<Scalar> grid;
  grid.nx = kSteepeningGrid;
  grid.ngx = 2;
  grid.dx = kSteepeningDomainLength / static_cast<Scalar>(kSteepeningGrid);

  cfe::Field<Scalar, 1> state(grid.n_cells_total());
  cfe::Field<Scalar, 1> stage1(grid.n_cells_total());
  cfe::Field<Scalar, 1> scratch(grid.n_cells_total());

  Scalar max_abs_u0 = Scalar(0.0);
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const Scalar x = grid.x_center(grid.ngx + i);
    const Scalar value = steepening_ic(x);
    state(grid.flat_index(grid.ngx + i, 0, 0), 0) = value;
    max_abs_u0 = std::max(max_abs_u0, std::abs(value));
  }

  // Same solver-assembly pattern as Case A above, but with
  // `PeriodicBoundary` (the right edge wraps around to the left) since
  // this case's sine wave has no fixed inflow/outflow.
  cfe::BurgersField<Scalar, 1> field{};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::BurgersField<Scalar, 1>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux>
      solver{grid, field, cfe::PeriodicBoundary{}};
  auto residual = [&](cfe::FieldView<Scalar, 1> in, cfe::FieldView<Scalar, 1> out) {
    solver.residual(in, out);
  };

  const double bound_low = static_cast<double>(kSteepeningBackground - kSteepeningAmplitude);
  const double bound_high = static_cast<double>(kSteepeningBackground + kSteepeningAmplitude);

  auto report = [&](Scalar t) {
    const bool pre_shock = t < kSteepeningBreakTime;
    std::vector<Scalar> x(grid.nx), u_num(grid.nx), u_exact;
    if (pre_shock) u_exact.resize(grid.nx);
    double sum_abs_error = 0.0;
    double max_val = static_cast<double>(state(grid.flat_index(grid.ngx, 0, 0), 0));
    double min_val = max_val;
    double domain_integral = 0.0;
    for (std::size_t i = 0; i < grid.nx; ++i) {
      const Scalar value = state(grid.flat_index(grid.ngx + i, 0, 0), 0);
      x[i] = grid.x_center(grid.ngx + i);
      u_num[i] = value;
      domain_integral += static_cast<double>(value) * static_cast<double>(grid.dx);
      max_val = std::max(max_val, static_cast<double>(value));
      min_val = std::min(min_val, static_cast<double>(value));
      if (pre_shock) {
        const Scalar exact = steepening_exact_presolve(x[i], t);
        u_exact[i] = exact;
        sum_abs_error += static_cast<double>(std::abs(value - exact)) * static_cast<double>(grid.dx);
      }
    }
    const double l1_error = pre_shock ? sum_abs_error : kNan;
    const double overshoot = std::max(0.0, max_val - bound_high);
    const double undershoot = std::max(0.0, bound_low - min_val);

    const double t_d = static_cast<double>(t);
    append_summary_row(summary, "steepening", grid.nx, "second_order_limited", t_d, l1_error, kNan, kNan,
                        overshoot, undershoot, domain_integral);
    char name[160];
    std::snprintf(name, sizeof(name), "case_steepening_t%.2f.csv", t_d);
    write_field_csv(data_dir / name, x, u_num, u_exact);
    std::printf(
        "  [steepening %s] t=%.2f  %s  domain_mean=%.6f  overshoot=%.2e  undershoot=%.2e\n",
        pre_shock ? "pre-shock " : "post-shock", t_d, pre_shock ? "(exact reference available)"
                                                               : "(NUMERICAL ONLY -- no exact reference)",
        domain_integral, overshoot, undershoot);
  };

  report(Scalar(0.0));
  Scalar t = Scalar(0.0);
  for (std::size_t seg = 1; seg < kSteepeningOutputTimes.size(); ++seg) {
    const Scalar t_target = static_cast<Scalar>(kSteepeningOutputTimes[seg]);
    const Scalar dt_target = kCfl * grid.dx / max_abs_u0;
    const int n_steps = static_cast<int>(std::ceil(static_cast<double>((t_target - t) / dt_target)));
    const Scalar dt = (t_target - t) / static_cast<Scalar>(n_steps);
    for (int step = 0; step < n_steps; ++step) {
      cfe::ssp_rk2_step<Scalar>(state.view(), stage1.view(), scratch.view(), dt, residual,
                                 solver.active_cell_count(), solver.active_cell_index_map());
    }
    t = t_target;
    report(t);
  }
}

// ---------------------------------------------------------------------
// Case C: transonic rarefaction
// ---------------------------------------------------------------------

constexpr Scalar kRarefactionULeft = Scalar(-1.0);
constexpr Scalar kRarefactionURight = Scalar(1.0);
constexpr Scalar kRarefactionOrigin = Scalar(5.0);
constexpr Scalar kRarefactionDomainLength = Scalar(10.0);
constexpr std::size_t kRarefactionGrid = 400;
const std::vector<double> kRarefactionOutputTimes = {0.0, 1.0, 2.0};  // fan spans [3,7] at t=2.0 --
                                                                       // 3.0 of margin either side of [0,10]

// Exact self-similar rarefaction-fan cell average (same construction as
// tests/unit/test_burgers_rarefaction.cpp's own helper -- see that
// file's header comment for the full derivation). u_left < u_right
// selects a fan (Lax entropy condition), not a shock.
Scalar rarefaction_exact_cell_average(Scalar x_lo, Scalar x_hi, Scalar x0, Scalar t, Scalar u_left,
                                       Scalar u_right)
{
  if (t == Scalar(0.0)) {
    if (x_hi <= x0) return u_left;
    if (x_lo >= x0) return u_right;
    const Scalar frac_left = (x0 - x_lo) / (x_hi - x_lo);
    return frac_left * u_left + (Scalar(1.0) - frac_left) * u_right;
  }
  auto antideriv = [&](Scalar s) {
    if (s <= u_left * t) return u_left * s - Scalar(0.5) * u_left * u_left * t;
    if (s >= u_right * t) return u_right * s - Scalar(0.5) * u_right * u_right * t;
    return s * s / (Scalar(2.0) * t);
  };
  return (antideriv(x_hi - x0) - antideriv(x_lo - x0)) / (x_hi - x_lo);
}

void run_case_rarefaction(std::ofstream& summary, const fs::path& data_dir)
{
  std::printf("=== Case C: transonic rarefaction ===\n");

  cfe::CartesianGrid<Scalar> grid;
  grid.nx = kRarefactionGrid;
  grid.ngx = 2;
  grid.dx = kRarefactionDomainLength / static_cast<Scalar>(kRarefactionGrid);

  cfe::Field<Scalar, 1> state(grid.n_cells_total());
  cfe::Field<Scalar, 1> stage1(grid.n_cells_total());
  cfe::Field<Scalar, 1> scratch(grid.n_cells_total());

  for (std::size_t i = 0; i < grid.nx; ++i) {
    const Scalar x_lo = static_cast<Scalar>(i) * grid.dx;
    const Scalar x_hi = x_lo + grid.dx;
    state(grid.flat_index(grid.ngx + i, 0, 0), 0) = rarefaction_exact_cell_average(
        x_lo, x_hi, kRarefactionOrigin, Scalar(0.0), kRarefactionULeft, kRarefactionURight);
  }

  cfe::BurgersField<Scalar, 1> field{};
  // Fixed far-field values on BOTH ends -- see test_burgers_rarefaction.cpp's
  // own comment for why StaticBoundary (not InflowOutflowBoundary) is
  // the faithful choice for a rarefaction, unlike Case A's moving shock.
  cfe::StaticBoundary<Scalar, 1> boundary{cfe::State<Scalar, 1>(kRarefactionULeft),
                                          cfe::State<Scalar, 1>(kRarefactionURight)};
  // Same solver-assembly pattern as the other two cases, with
  // `StaticBoundary` (each end fixed to its own far-field value, never
  // changing) in place of Case A's `InflowOutflowBoundary`.
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::BurgersField<Scalar, 1>, cfe::StaticBoundary<Scalar, 1>,
                 cfe::StaticBoundary<Scalar, 1>, cfe::StaticBoundary<Scalar, 1>,
                 cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux>
      solver{grid, field, boundary};
  auto residual = [&](cfe::FieldView<Scalar, 1> in, cfe::FieldView<Scalar, 1> out) {
    solver.residual(in, out);
  };

  // dt sized from max(|u_left|,|u_right|) -- the KNOWN, fixed far-field
  // magnitudes -- not from a signed state value: u_left is negative
  // here, so Case A's own `cfl*dx/u_left` pattern (safe only because
  // Case A's u_left happens to be positive) would give a negative dt.
  const Scalar max_speed = std::max(std::abs(kRarefactionULeft), std::abs(kRarefactionURight));

  auto report = [&](Scalar t) {
    std::vector<Scalar> x(grid.nx), u_num(grid.nx), u_exact(grid.nx);
    double sum_abs_error = 0.0;
    double max_val = static_cast<double>(state(grid.flat_index(grid.ngx, 0, 0), 0));
    double min_val = max_val;
    double domain_integral = 0.0;
    for (std::size_t i = 0; i < grid.nx; ++i) {
      const Scalar x_lo = static_cast<Scalar>(i) * grid.dx;
      const Scalar x_hi = x_lo + grid.dx;
      const Scalar value = state(grid.flat_index(grid.ngx + i, 0, 0), 0);
      const Scalar exact =
          rarefaction_exact_cell_average(x_lo, x_hi, kRarefactionOrigin, t, kRarefactionULeft,
                                          kRarefactionURight);
      x[i] = grid.x_center(grid.ngx + i);
      u_num[i] = value;
      u_exact[i] = exact;
      sum_abs_error += static_cast<double>(std::abs(value - exact)) * static_cast<double>(grid.dx);
      max_val = std::max(max_val, static_cast<double>(value));
      min_val = std::min(min_val, static_cast<double>(value));
      domain_integral += static_cast<double>(value) * static_cast<double>(grid.dx);
    }
    const double overshoot = std::max(0.0, max_val - static_cast<double>(kRarefactionURight));
    const double undershoot = std::max(0.0, static_cast<double>(kRarefactionULeft) - min_val);

    const double t_d = static_cast<double>(t);
    append_summary_row(summary, "rarefaction", grid.nx, "second_order_limited", t_d, sum_abs_error, kNan,
                        kNan, overshoot, undershoot, domain_integral);
    char name[160];
    std::snprintf(name, sizeof(name), "case_rarefaction_t%.2f.csv", t_d);
    write_field_csv(data_dir / name, x, u_num, u_exact);
    std::printf("  [rarefaction] t=%.2f  L1=%.3e  overshoot=%.2e  undershoot=%.2e  domain_integral=%.6f\n",
                t_d, sum_abs_error, overshoot, undershoot, domain_integral);
  };

  report(Scalar(0.0));
  Scalar t = Scalar(0.0);
  for (std::size_t seg = 1; seg < kRarefactionOutputTimes.size(); ++seg) {
    const Scalar t_target = static_cast<Scalar>(kRarefactionOutputTimes[seg]);
    const Scalar dt_target = kCfl * grid.dx / max_speed;
    const int n_steps = static_cast<int>(std::ceil(static_cast<double>((t_target - t) / dt_target)));
    const Scalar dt = (t_target - t) / static_cast<Scalar>(n_steps);
    for (int step = 0; step < n_steps; ++step) {
      cfe::ssp_rk2_step<Scalar>(state.view(), stage1.view(), scratch.view(), dt, residual,
                                 solver.active_cell_count(), solver.active_cell_index_map());
    }
    t = t_target;
    report(t);
  }
}

}  // namespace

// Entry point: parses an optional `--case=` command-line flag, then
// runs whichever case(s) it selects (all three by default), writing one
// shared summary CSV plus each case's own field CSVs.
int main(int argc, char** argv)
{
  bool run_shock = true;
  bool run_steepening = true;
  bool run_rarefaction = true;
  if (argc > 1) {
    const std::string arg = argv[1];
    if (arg == "--case=shock") {
      run_steepening = false;
      run_rarefaction = false;
    } else if (arg == "--case=steepening") {
      run_shock = false;
      run_rarefaction = false;
    } else if (arg == "--case=rarefaction") {
      run_shock = false;
      run_steepening = false;
    } else {
      std::printf(
          "Unknown argument '%s' -- expected --case=shock, --case=steepening, or "
          "--case=rarefaction. Running all three.\n",
          arg.c_str());
    }
  }

  const fs::path data_dir = "data";
  fs::create_directories(data_dir);
  std::ofstream summary = open_summary_csv(data_dir);

  if (run_shock) run_case_shock(summary, data_dir);
  if (run_steepening) run_case_steepening(summary, data_dir);
  if (run_rarefaction) run_case_rarefaction(summary, data_dir);

  summary.close();
  std::printf("\nWrote %s/summary.csv and field CSVs. Run plot_results.py to generate figures.\n",
              data_dir.string().c_str());
  return 0;
}
