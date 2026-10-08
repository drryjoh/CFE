// 2D inviscid Burgers tutorial: a diagonal moving shock
// (u=1 where x+y<0.5+t, u=0 elsewhere), run on the same FvmSolver +
// BurgersField<Scalar,2> + RusanovFlux + SSP-RK2 stack as every other
// Burgers tutorial in this repo.
//
// This is the one tutorial that genuinely needs per-RK-stage,
// time-dependent ghost fills: ghost cells are populated directly from
// the known exact solution (not from neighboring interior state), and
// that exact solution moves with t. `cfe::ssp_rk2_step` has no time
// parameter and is not modified here (zero blast radius on that shared,
// already-reviewed helper) -- instead, Heun's method (the same two
// stages ssp_rk2_step already encodes) is written out explicitly below,
// setting the boundary's own `time` member to the correct stage time
// (t_n for the first residual evaluation, t_n+dt for the second) before
// each of the two `solver.residual(...)` calls.
//
// Initial condition, every exact reference cell average, AND every
// ghost cell `DiagonalShockExactBoundary` fills all use the SAME
// `cut_cell_fraction` helper: the closed-form area of a square cell cut
// by the diagonal line x+y=c, evaluated at c=0.5 for t=0 and c=0.5+t for
// every later reference/ghost fill -- so a cell the shock straddles
// (ghost cells included) gets its true fractional coverage, not a
// cell-center 0/1 guess. The ghost-cell case was a real bug caught in
// review (PR #3, commit 1d3654f): ghost cells previously used a
// cell-center point sample, inconsistent with the cell-average
// formulation everywhere else in this file, and measurably wrong for
// the handful of ghost cells the shock actually passes through at any
// given time (fixed below; see `cut_cell_fraction`'s own doc comment).
//
// Driver precision is `cfe::scalar` (core/types.hpp, project-wide via
// the `CFE_SCALAR_TYPE` CMake cache variable, `double` by default) --
// NOT a hardcoded `double` -- matching AGENTS.md #11. `cut_cell_fraction`
// and `cell_lower_edge` are templated on `Scalar` (not fixed to
// `cfe::scalar`) specifically so `DiagonalShockExactBoundary`'s own
// `fill_x`/`fill_y`/`fill_z` -- already generic, following the same
// convention `StaticBoundary`/`PeriodicBoundary` use -- can keep calling
// them with whatever `Scalar` their `FieldView` argument actually has,
// with no cast. Reported metrics (L1 error, overshoot/undershoot,
// domain integral) are deliberately still accumulated in `double`
// regardless of `cfe::scalar`, the same convention every other Burgers
// tutorial/test in this repo uses.
//
// Output: data/summary.csv (every grid/reconstruction/time row) plus,
// for one representative combination (the smallest grid, 100^2 --
// chosen specifically to keep committed file size down, since a full
// field dump grows with N^2 unlike the 1D tutorial's N; see README.md),
// a full-grid field CSV and a diagonal (x=y) profile CSV at every
// output time.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "cfe/backend/parallel_for.hpp"
#include "cfe/core/macros.hpp"
#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/burgers/field.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/numerics/fvm/first_order_reconstruction.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"

// An anonymous namespace: everything below, until the matching `}` near
// the bottom of this file, is private to this file only. Closest Python
// analogy: names that are never exported from a module. This file does
// not use a local `Scalar` alias like the other tutorials -- `cfe::scalar`
// is spelled out everywhere, since `cut_cell_fraction`/`cell_lower_edge`
// below are genuine templates whose own parameter is already named
// `Scalar`, and `DiagonalShockExactBoundary`'s member functions follow
// that same convention.
namespace {

namespace fs = std::filesystem;

constexpr cfe::scalar kCfl = cfe::scalar(0.4);
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr cfe::scalar kShockOrigin = cfe::scalar(0.5);
const std::vector<std::size_t> kGrids = {100, 200, 400};
const std::vector<double> kOutputTimes = {0.0, 0.25, 0.5};
constexpr std::size_t kRepresentativeGrid = 100;  // smallest -- see file header comment

// Exact area fraction of a [x_lo,x_lo+h]x[y_lo,y_lo+h] cell lying in
// {x+y < c} -- the standard closed form for a square clipped by a
// slope-(-1) line (assumes a square cell: dx==dy==h, true everywhere
// this tutorial uses it). Used for the t=0 initial condition (c=0.5),
// every later exact reference cell average (c=0.5+t), AND -- below --
// every ghost cell `DiagonalShockExactBoundary` fills: a ghost cell the
// shock line currently cuts through must get the same true fractional
// coverage a real cell would, not a cell-center 0/1 sample (caught in
// review: a point sample at the ghost cell's center is inconsistent
// with the cell-average formulation everywhere else in this file, and
// measurably wrong for exactly the ghost cells the shock passes
// through). Defined before `DiagonalShockExactBoundary` specifically so
// that type can call it. Templated (not fixed to `cfe::scalar`) so
// `DiagonalShockExactBoundary`'s own generic `fill_x`/`fill_y`/`fill_z`
// can call it with whatever `Scalar` their `FieldView` argument has.
template <class Scalar>
Scalar cut_cell_fraction(Scalar x_lo, Scalar y_lo, Scalar h, Scalar c)
{
  const Scalar c_local = c - x_lo - y_lo;
  if (c_local <= Scalar(0.0)) return Scalar(0.0);
  if (c_local >= Scalar(2.0) * h) return Scalar(1.0);
  if (c_local <= h) return (c_local * c_local) / (Scalar(2.0) * h * h);
  const Scalar d = Scalar(2.0) * h - c_local;
  return Scalar(1.0) - (d * d) / (Scalar(2.0) * h * h);
}

// The lower edge of the cell at padded axis index `i_pad` (ghost or
// real alike): `grid.x_center`/`y_center`'s own formula minus half a
// cell, but computed directly in `Scalar` rather than by subtracting
// `std::size_t`s (which underflows for any ghost index below the
// origin, e.g. i_pad=0 with ngx=2).
template <class Scalar>
Scalar cell_lower_edge(std::size_t i_pad, std::size_t n_ghost, Scalar h)
{
  return (static_cast<Scalar>(i_pad) - static_cast<Scalar>(n_ghost)) * h;
}

// ---------------------------------------------------------------------
// Problem-specific boundary condition: deliberately tutorial-LOCAL, not
// added to src/cfe/grid/boundary/ -- it hardcodes this one problem's
// exact solution formula, so it belongs with the application, not the
// generic boundary-condition library (same physics/generic-numerics
// separation already established for Reconstruction/NumericalFlux
// types). `time` is mutable and public specifically so the driver below
// can set it directly on `solver.boundary_x`/`solver.boundary_y` between
// RK stages -- it is `cfe::scalar` (the driver's own concrete choice),
// not a template parameter, since it is always set from this file's own
// driver code. Implements fill_x/fill_y/fill_z for the same reason
// `StaticBoundary`/`PeriodicBoundary` do: `grid/ghost/ghost_fill.hpp`'s
// `fill_ghost_cells` is one function template with a runtime `switch`
// over all three axes, so instantiating it for any one axis (X is
// always active, called unconditionally by `FvmSolver::residual()`)
// requires `fill_z` to exist and compile even though Axis::Z is never
// actually reached at runtime for a 2D field -- `if constexpr
// (z_active)` guards the *call site* in `fvm_solver.hpp`, not this
// switch, so it does not exempt this type from needing all three.
struct DiagonalShockExactBoundary
{
  cfe::scalar time = cfe::scalar(0.0);

  template <class Backend = cfe::CpuParallelFor, class Scalar, std::size_t N, class Layout>
  void fill_x(cfe::FieldView<Scalar, N, Layout> field, const cfe::CartesianGrid<Scalar> grid) const
  {
    if (grid.ngx == 0) return;
    const Scalar c = kShockOrigin + static_cast<Scalar>(time);
    const Scalar h = grid.dx;
    for (std::size_t j = 0; j < grid.padded_ny(); ++j) {
      const Scalar y_lo = cell_lower_edge(j, grid.ngy, grid.dy);
      for (std::size_t g = 0; g < grid.ngx; ++g) {
        const std::size_t i_low = grid.ngx - 1 - g;
        const std::size_t i_high = grid.ngx + grid.nx + g;
        field(grid.flat_index(i_low, j, 0), 0) = cut_cell_fraction(cell_lower_edge(i_low, grid.ngx, h),
                                                                    y_lo, h, c);
        field(grid.flat_index(i_high, j, 0), 0) = cut_cell_fraction(cell_lower_edge(i_high, grid.ngx, h),
                                                                     y_lo, h, c);
      }
    }
  }

  template <class Backend = cfe::CpuParallelFor, class Scalar, std::size_t N, class Layout>
  void fill_y(cfe::FieldView<Scalar, N, Layout> field, const cfe::CartesianGrid<Scalar> grid) const
  {
    if (grid.ngy == 0) return;
    const Scalar c = kShockOrigin + static_cast<Scalar>(time);
    const Scalar h = grid.dy;
    for (std::size_t i = 0; i < grid.padded_nx(); ++i) {
      const Scalar x_lo = cell_lower_edge(i, grid.ngx, grid.dx);
      for (std::size_t g = 0; g < grid.ngy; ++g) {
        const std::size_t j_low = grid.ngy - 1 - g;
        const std::size_t j_high = grid.ngy + grid.ny + g;
        field(grid.flat_index(i, j_low, 0), 0) = cut_cell_fraction(x_lo, cell_lower_edge(j_low, grid.ngy, h),
                                                                    h, c);
        field(grid.flat_index(i, j_high, 0), 0) =
            cut_cell_fraction(x_lo, cell_lower_edge(j_high, grid.ngy, h), h, c);
      }
    }
  }

  // Never actually reached at runtime for a 2D field (see this type's
  // header comment) -- `grid.ngz == 0` makes this an immediate no-op in
  // that case, same early-return every other axis's fill already uses.
  template <class Backend = cfe::CpuParallelFor, class Scalar, std::size_t N, class Layout>
  void fill_z(cfe::FieldView<Scalar, N, Layout> field, const cfe::CartesianGrid<Scalar> grid) const
  {
    if (grid.ngz == 0) return;
    const Scalar c = kShockOrigin + static_cast<Scalar>(time);
    const Scalar h = grid.dx;
    for (std::size_t j = 0; j < grid.padded_ny(); ++j) {
      const Scalar y_lo = cell_lower_edge(j, grid.ngy, grid.dy);
      for (std::size_t i = 0; i < grid.padded_nx(); ++i) {
        const Scalar x_lo = cell_lower_edge(i, grid.ngx, grid.dx);
        const Scalar value = cut_cell_fraction(x_lo, y_lo, h, c);
        for (std::size_t g = 0; g < grid.ngz; ++g) {
          const std::size_t k_low = grid.ngz - 1 - g;
          const std::size_t k_high = grid.ngz + grid.nz + g;
          field(grid.flat_index(i, j, k_low), 0) = value;
          field(grid.flat_index(i, j, k_high), 0) = value;
        }
      }
    }
  }
};

// ---------------------------------------------------------------------
// Hand-rolled SSP-RK2 (Heun's method) -- see file header comment for why
// this is not `cfe::ssp_rk2_step`. `solver` is mutable because its
// boundary members' `.time` fields are set here.
// ---------------------------------------------------------------------

template <class Solver>
void step_once(Solver& solver, cfe::Field<cfe::scalar, 1>& state, cfe::Field<cfe::scalar, 1>& stage1,
               cfe::Field<cfe::scalar, 1>& scratch, cfe::scalar dt, cfe::scalar t_n)
{
  const std::size_t n_active = solver.active_cell_count();
  const auto index_map = solver.active_cell_index_map();

  solver.boundary_x.time = t_n;
  solver.boundary_y.time = t_n;
  solver.residual(state.view(), scratch.view());
  for (std::size_t r = 0; r < n_active; ++r) {
    const std::size_t cell = index_map(r);
    stage1(cell, 0) = state(cell, 0) + dt * scratch(cell, 0);
  }

  solver.boundary_x.time = t_n + dt;
  solver.boundary_y.time = t_n + dt;
  solver.residual(stage1.view(), scratch.view());
  for (std::size_t r = 0; r < n_active; ++r) {
    const std::size_t cell = index_map(r);
    state(cell, 0) = cfe::scalar(0.5) * state(cell, 0) + cfe::scalar(0.5) * (stage1(cell, 0) + dt * scratch(cell, 0));
  }
}

// ---------------------------------------------------------------------
// Data output
// ---------------------------------------------------------------------

std::ofstream open_summary_csv(const fs::path& data_dir)
{
  std::ofstream out(data_dir / "summary.csv");
  out << "grid,reconstruction,time,l1_error,shock_position_exact,shock_position_numerical,overshoot,"
         "undershoot,domain_mean\n";
  return out;
}

void write_field_csv(const fs::path& path, const cfe::CartesianGrid<cfe::scalar>& grid,
                      cfe::FieldView<cfe::scalar, 1> state, cfe::scalar t)
{
  std::ofstream out(path);
  out << "x,y,u_numerical,u_exact\n";
  const cfe::scalar h = grid.dx;
  for (std::size_t j = 0; j < grid.ny; ++j) {
    const cfe::scalar y_lo = static_cast<cfe::scalar>(j) * h;
    for (std::size_t i = 0; i < grid.nx; ++i) {
      const cfe::scalar x_lo = static_cast<cfe::scalar>(i) * h;
      const cfe::scalar value = state(grid.flat_index(grid.ngx + i, grid.ngy + j, 0), 0);
      const cfe::scalar exact = cut_cell_fraction(x_lo, y_lo, h, kShockOrigin + t);
      out << grid.x_center(grid.ngx + i) << ',' << grid.y_center(grid.ngy + j) << ',' << value << ','
          << exact << '\n';
    }
  }
}

void write_diagonal_profile_csv(const fs::path& path, const cfe::CartesianGrid<cfe::scalar>& grid,
                                 cfe::FieldView<cfe::scalar, 1> state, cfe::scalar t)
{
  std::ofstream out(path);
  out << "s,u_numerical,u_exact\n";
  const cfe::scalar h = grid.dx;
  for (std::size_t i = 0; i < grid.nx; ++i) {
    const cfe::scalar s = grid.x_center(grid.ngx + i);  // x==y along the diagonal, nx==ny here
    const cfe::scalar x_lo = static_cast<cfe::scalar>(i) * h;
    const cfe::scalar value = state(grid.flat_index(grid.ngx + i, grid.ngy + i, 0), 0);
    const cfe::scalar exact = cut_cell_fraction(x_lo, x_lo, h, kShockOrigin + t);
    out << s << ',' << value << ',' << exact << '\n';
  }
}

// ---------------------------------------------------------------------
// One full run
// ---------------------------------------------------------------------

template <class Reconstruction>
void run_diagonal_case(std::size_t n, const std::string& reconstruction_name, bool write_fields,
                       std::ofstream& summary, const fs::path& data_dir)
{
  cfe::CartesianGrid<cfe::scalar> grid;
  grid.nx = n;
  grid.ny = n;
  grid.ngx = 2;
  grid.ngy = 2;
  grid.dx = cfe::scalar(1.0) / static_cast<cfe::scalar>(n);
  grid.dy = grid.dx;

  // Three arrays of `n*n` numbers each (plus ghost cells), the same
  // shape as a 2D numpy array flattened to 1D: `state` holds the
  // simulation's current values; `stage1` and `scratch` are scratch
  // space `step_once` below uses internally.
  cfe::Field<cfe::scalar, 1> state(grid.n_cells_total());
  cfe::Field<cfe::scalar, 1> stage1(grid.n_cells_total());
  cfe::Field<cfe::scalar, 1> scratch(grid.n_cells_total());

  const cfe::scalar h = grid.dx;
  for (std::size_t j = 0; j < grid.ny; ++j) {
    const cfe::scalar y_lo = static_cast<cfe::scalar>(j) * h;
    for (std::size_t i = 0; i < grid.nx; ++i) {
      const cfe::scalar x_lo = static_cast<cfe::scalar>(i) * h;
      state(grid.flat_index(grid.ngx + i, grid.ngy + j, 0), 0) =
          cut_cell_fraction(x_lo, y_lo, h, kShockOrigin);
    }
  }

  // `field` picks WHICH equation is being solved (2D Burgers' equation).
  // `solver` bundles that equation with the grid, the problem-specific
  // exact-solution boundary on both the X and Y faces, and the
  // `Reconstruction` scheme this function was called with.
  cfe::BurgersField<cfe::scalar, 2> field{};
  DiagonalShockExactBoundary boundary_x{};
  DiagonalShockExactBoundary boundary_y{};
  cfe::FvmSolver<cfe::scalar, cfe::AoSLayout, cfe::BurgersField<cfe::scalar, 2>, DiagonalShockExactBoundary,
                 DiagonalShockExactBoundary, DiagonalShockExactBoundary, Reconstruction, cfe::RusanovFlux>
      solver{grid, field, boundary_x, boundary_y};

  auto report = [&](cfe::scalar t) {
    double sum_abs_error = 0.0;
    double max_val = static_cast<double>(state(grid.flat_index(grid.ngx, grid.ngy, 0), 0));
    double min_val = max_val;
    double domain_integral = 0.0;
    const double cell_area = static_cast<double>(grid.dx) * static_cast<double>(grid.dy);
    for (std::size_t j = 0; j < grid.ny; ++j) {
      const cfe::scalar y_lo = static_cast<cfe::scalar>(j) * h;
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const cfe::scalar x_lo = static_cast<cfe::scalar>(i) * h;
        const cfe::scalar value = state(grid.flat_index(grid.ngx + i, grid.ngy + j, 0), 0);
        const cfe::scalar exact = cut_cell_fraction(x_lo, y_lo, h, kShockOrigin + t);
        sum_abs_error += static_cast<double>(std::abs(value - exact)) * cell_area;
        max_val = std::max(max_val, static_cast<double>(value));
        min_val = std::min(min_val, static_cast<double>(value));
        domain_integral += static_cast<double>(value) * cell_area;
      }
    }

    // Shock straightness: find where the diagonal (x=y) profile crosses
    // 0.5, compare against the exact diagonal crossing (0.5+t)/2.
    double shock_numerical = kNan;
    for (std::size_t i = 0; i + 1 < grid.nx; ++i) {
      const double v0 = static_cast<double>(state(grid.flat_index(grid.ngx + i, grid.ngy + i, 0), 0));
      const double v1 =
          static_cast<double>(state(grid.flat_index(grid.ngx + i + 1, grid.ngy + i + 1, 0), 0));
      if (v0 >= 0.5 && v1 < 0.5) {
        const double s0 = static_cast<double>(grid.x_center(grid.ngx + i));
        const double s1 = static_cast<double>(grid.x_center(grid.ngx + i + 1));
        const double frac = (v0 - 0.5) / (v0 - v1);
        shock_numerical = s0 + frac * (s1 - s0);
        break;
      }
    }
    const double shock_exact = 0.5 * (static_cast<double>(kShockOrigin) + static_cast<double>(t));
    const double overshoot = std::max(0.0, max_val - 1.0);
    const double undershoot = std::max(0.0, 0.0 - min_val);

    const double t_d = static_cast<double>(t);
    summary << n << ',' << reconstruction_name << ',' << t_d << ',' << sum_abs_error << ',' << shock_exact
            << ',' << shock_numerical << ',' << overshoot << ',' << undershoot << ',' << domain_integral
            << '\n';
    std::printf(
        "  [2D n=%4zu %-20s] t=%.2f  L1=%.3e  diag_crossing(exact=%.4f, num=%.4f)  overshoot=%.2e  "
        "undershoot=%.2e\n",
        n, reconstruction_name.c_str(), t_d, sum_abs_error, shock_exact, shock_numerical, overshoot,
        undershoot);

    if (write_fields) {
      char field_name[160];
      std::snprintf(field_name, sizeof(field_name), "field_nx%04zu_%s_t%.2f.csv", n,
                    reconstruction_name.c_str(), t_d);
      write_field_csv(data_dir / field_name, grid, state.view(), t);
      char diag_name[160];
      std::snprintf(diag_name, sizeof(diag_name), "diagonal_nx%04zu_%s_t%.2f.csv", n,
                    reconstruction_name.c_str(), t_d);
      write_diagonal_profile_csv(data_dir / diag_name, grid, state.view(), t);
    }
  };

  report(cfe::scalar(0.0));
  cfe::scalar t = cfe::scalar(0.0);
  for (std::size_t seg = 1; seg < kOutputTimes.size(); ++seg) {
    const cfe::scalar t_target = static_cast<cfe::scalar>(kOutputTimes[seg]);
    const cfe::scalar dt_target = kCfl / (cfe::scalar(1.0) * (cfe::scalar(1.0) / grid.dx + cfe::scalar(1.0) / grid.dy));
    const int n_steps = static_cast<int>(std::ceil(static_cast<double>((t_target - t) / dt_target)));
    const cfe::scalar dt = (t_target - t) / static_cast<cfe::scalar>(n_steps);
    for (int step = 0; step < n_steps; ++step) {
      step_once(solver, state, stage1, scratch, dt, t);
      t += dt;
    }
    t = t_target;
    report(t);
  }
}

}  // namespace

// Entry point: for each grid size, runs both reconstructions and writes
// one shared summary CSV, plus a full field + diagonal-profile CSV for
// one representative (grid, reconstruction) combination.
int main()
{
  const fs::path data_dir = "data";
  fs::create_directories(data_dir);
  std::ofstream summary = open_summary_csv(data_dir);

  std::printf("=== 2D diagonal moving shock ===\n");
  for (std::size_t n : kGrids) {
    const bool write_representative = (n == kRepresentativeGrid);
    run_diagonal_case<cfe::fvm::FirstOrderReconstruction>(n, "first_order", false, summary, data_dir);
    run_diagonal_case<cfe::fvm::MusclMinmodReconstruction>(n, "second_order_limited", write_representative,
                                                           summary, data_dir);
  }

  summary.close();
  std::printf("\nWrote %s/summary.csv and field CSVs. Run plot_results.py to generate figures.\n",
              data_dir.string().c_str());
  return 0;
}
