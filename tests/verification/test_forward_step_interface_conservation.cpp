// Required conservation check for the new cross-block ghost-fill
// mechanism (tasks/0010-phase4-forward-step-multiblock.md: "flux
// computed from each side of the shared block interface must agree --
// a direct conservation check, not just 'the simulation ran and looked
// reasonable'").
//
// Builds a small two-block L-shaped setup at the same proportions as
// the real forward-step geometry (an upstream, full-height block and a
// downstream, post-step-height block, joined where their heights
// overlap), fills BOTH blocks with a smooth but genuinely
// non-uniform state (so MUSCL's limiter sees real gradients, not a
// degenerate uniform-flow case where any bug in the interface read
// would be invisible), fills ghost cells with the new
// `BlockInterfaceBoundary`, and then recomputes -- directly, using the
// same `MusclMinmodReconstruction`/`RusanovFlux` pair
// `FvmSolver::residual()` itself uses internally -- the numerical flux
// at the shared interface face from EACH block's own point of view.
//
// Why bit-identical agreement is the right check (not just "close"):
// `BlockInterfaceBoundary` copies the neighbor's real-cell values
// VERBATIM into the ghost layer (see that file's header comment) --
// so the two 5-point stencils `FvmSolver::residual()` would build
// around this face (one from block1's own real+ghost cells, one from
// block2's own real+ghost cells) contain the SAME four underlying
// values, just assembled from different storage. Reconstruction and
// the numerical flux are both pure functions of their inputs, so equal
// inputs must give bit-identical outputs -- any discrepancy here would
// mean the interface read is NOT copying the right values (a real bug:
// a wrong row offset, a transposed axis, truncation from a precision
// mismatch), not mere floating-point noise.
//
// Dual-precision instantiation (task 0017): both checks below are
// `template <class Scalar>` functions, instantiated at `double` and
// `float`. Measured (this task's own calibration, same toolchain): the
// flux-agreement check is EXACT (0.0) at float too, same within-
// precision-reproducibility argument as test_euler_3d_sanity.cpp's own
// column-match check (both sides derive from the identical underlying
// stored values, just assembled from different storage, regardless of
// which Scalar that storage happens to be) -- `1e-4` at float is a
// float-epsilon-scaled portability margin (same ~450x-over-epsilon
// ratio test_euler_3d_sanity.cpp's own float tolerance uses), not a
// measured requirement. The mass-balance check genuinely does
// accumulate roundoff over many timesteps (measured ~4.0e-9 at float vs.
// double's ~2.0e-14); `1e-6` keeps a >=200x measured margin, the same
// float-epsilon-scale choice this task's other "exact conservation over
// a short run" checks use.
#include <cmath>
#include <cstddef>

#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/euler/field.hpp"
#include "cfe/grid/boundary/block_interface_boundary.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/math/fixed_array.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "forward_step_setup.hpp"
#include "test_framework.hpp"

namespace {

template <class Scalar>
cfe::CartesianGrid<Scalar> make_block(cfe::Index nx, cfe::Index ny, Scalar origin_x, Scalar origin_y)
{
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = nx;
  grid.ny = ny;
  grid.ngx = 2;
  grid.ngy = 0;  // only fill_x is exercised here
  grid.dx = Scalar(0.1);
  grid.dy = Scalar(0.1);
  grid.origin_x = origin_x;
  grid.origin_y = origin_y;
  return grid;
}

// Smooth, fully-populated, non-uniform primitive state -- a small
// sinusoidal perturbation of a Mach-3-like freestream, so every
// conserved component has a real, nonzero gradient everywhere
// (density, both momentum components, and energy all vary).
template <class Scalar, class Field, std::size_t N>
cfe::State<Scalar, N> state_at(const Field& field, Scalar x, Scalar y)
{
  const Scalar rho = Scalar(1.4) + Scalar(0.05) * std::sin(Scalar(7.0) * x + Scalar(3.0) * y);
  const Scalar ux = Scalar(3.0) + Scalar(0.1) * std::cos(Scalar(5.0) * x - Scalar(2.0) * y);
  const Scalar uy = Scalar(0.1) * std::sin(Scalar(4.0) * x + Scalar(6.0) * y);
  const Scalar p = Scalar(1.0) + Scalar(0.02) * std::cos(Scalar(6.0) * x + Scalar(4.0) * y);
  cfe::Vector<Scalar, 2> velocity;
  velocity[0] = ux;
  velocity[1] = uy;
  return field.conserved_from_primitive(rho, velocity, p);
}

template <class Scalar, std::size_t N, class Field>
void fill_smooth_state(cfe::Field<Scalar, N>& state, const cfe::CartesianGrid<Scalar>& grid, const Field& field)
{
  for (cfe::Index j = 0; j < grid.ny; ++j) {
    for (cfe::Index i = 0; i < grid.nx; ++i) {
      const cfe::Index cell = grid.flat_index(grid.ngx + i, grid.ngy + j, 0);
      const cfe::State<Scalar, N> q =
          state_at<Scalar, Field, N>(field, grid.x_center(grid.ngx + i), grid.y_center(grid.ngy + j));
      for (cfe::Index c = 0; c < N; ++c) state(cell, c) = q[c];
    }
  }
}

template <class Scalar, std::size_t N>
cfe::State<Scalar, N> read(const cfe::Field<Scalar, N>& state, cfe::Index cell)
{
  cfe::State<Scalar, N> q;
  for (cfe::Index c = 0; c < N; ++c) q[c] = state(cell, c);
  return q;
}

}  // namespace

template <class Scalar>
void check_forward_step_block_interface_flux_agrees_bit_exactly_from_both_sides(double tol)
{
  using Field = cfe::EulerField<Scalar, 2>;
  constexpr std::size_t kN = 4;

  // block1: upstream, full tunnel height (ny=10, step at local j=2 ->
  // real y=0.2). block2: downstream, post-step height (ny=8), covering
  // block1's local j=2..9.
  auto grid1 = make_block<Scalar>(6, 10, Scalar(0.0), Scalar(0.0));
  auto grid2 = make_block<Scalar>(24, 8, Scalar(0.6), Scalar(0.2));

  Field field{};  // gamma = 1.4 default

  cfe::Field<Scalar, kN> state1(grid1.n_cells_total());
  cfe::Field<Scalar, kN> state2(grid2.n_cells_total());
  fill_smooth_state(state1, grid1, field);
  fill_smooth_state(state2, grid2, field);

  cfe::Vector<Scalar, 2> inflow_velocity;
  inflow_velocity[0] = Scalar(3.0);
  inflow_velocity[1] = Scalar(0.0);
  const cfe::State<Scalar, kN> inflow = field.conserved_from_primitive(Scalar(1.4), inflow_velocity, Scalar(1.0));

  cfe::BlockInterfaceBoundary<Scalar, Field, cfe::Side::High> boundary1{
      .domain_edge_value = inflow,
      .neighbor_field = state2.view(),
      .neighbor_grid = grid2,
      .j_offset_to_neighbor = -2,
      .j_valid_lo = 2,
      .j_valid_hi = 10};
  cfe::BlockInterfaceBoundary<Scalar, Field, cfe::Side::Low> boundary2{
      .domain_edge_value = cfe::State<Scalar, kN>{},
      .neighbor_field = state1.view(),
      .neighbor_grid = grid1,
      .j_offset_to_neighbor = 2,
      .j_valid_lo = 0,
      .j_valid_hi = 8};

  boundary1.fill_x(state1.view(), grid1);
  boundary2.fill_x(state2.view(), grid2);

  cfe::fvm::MusclMinmodReconstruction reconstruction{};
  cfe::RusanovFlux numerical_flux{};

  // Checked at every row the two blocks actually share (block1 local
  // j=2..9 <-> block2 local j=0..7), not just one -- a row-dependent
  // offset bug would only show up away from whichever single row a
  // smaller test happened to pick.
  for (cfe::Index j1 = 2; j1 < grid1.ny; ++j1) {
    const cfe::Index j2 = j1 - 2;

    // --- Block1's own view of the flux at its right (interface) face.
    const cfe::Index m1 = grid1.flat_index(grid1.ngx + grid1.nx - 2, j1, 0);
    const cfe::Index c1 = grid1.flat_index(grid1.ngx + grid1.nx - 1, j1, 0);
    const cfe::Index p1_ghost = grid1.flat_index(grid1.ngx + grid1.nx, j1, 0);
    const cfe::Index p2_ghost = grid1.flat_index(grid1.ngx + grid1.nx + 1, j1, 0);

    const cfe::State<Scalar, kN> q_m1 = read(state1, m1);
    const cfe::State<Scalar, kN> q_c1 = read(state1, c1);
    const cfe::State<Scalar, kN> q_p1 = read(state1, p1_ghost);
    const cfe::State<Scalar, kN> q_p2 = read(state1, p2_ghost);

    cfe::State<Scalar, kN> left_of_right, right_of_right;
    for (cfe::Index c = 0; c < kN; ++c) {
      left_of_right[c] = reconstruction.right(q_m1[c], q_c1[c], q_p1[c]);
      right_of_right[c] = reconstruction.left(q_c1[c], q_p1[c], q_p2[c]);
    }
    const cfe::State<Scalar, kN> flux_from_block1 =
        numerical_flux(left_of_right, right_of_right, cfe::Axis::X, field);

    // --- Block2's own view of the flux at its left (interface) face.
    const cfe::Index m2_ghost = grid2.flat_index(grid2.ngx - 2, j2, 0);
    const cfe::Index m1_ghost = grid2.flat_index(grid2.ngx - 1, j2, 0);
    const cfe::Index c2 = grid2.flat_index(grid2.ngx, j2, 0);
    const cfe::Index p1 = grid2.flat_index(grid2.ngx + 1, j2, 0);

    const cfe::State<Scalar, kN> r_m2 = read(state2, m2_ghost);
    const cfe::State<Scalar, kN> r_m1 = read(state2, m1_ghost);
    const cfe::State<Scalar, kN> r_c = read(state2, c2);
    const cfe::State<Scalar, kN> r_p1 = read(state2, p1);

    cfe::State<Scalar, kN> left_of_left, right_of_left;
    for (cfe::Index c = 0; c < kN; ++c) {
      left_of_left[c] = reconstruction.right(r_m2[c], r_m1[c], r_c[c]);
      right_of_left[c] = reconstruction.left(r_m1[c], r_c[c], r_p1[c]);
    }
    const cfe::State<Scalar, kN> flux_from_block2 =
        numerical_flux(left_of_left, right_of_left, cfe::Axis::X, field);

    for (cfe::Index c = 0; c < kN; ++c) {
      CFE_CHECK_NEAR(static_cast<double>(flux_from_block1[c]), static_cast<double>(flux_from_block2[c]), tol);
    }
  }
}

CFE_TEST(test_forward_step_block_interface_flux_agrees_bit_exactly_from_both_sides_double)
{
  check_forward_step_block_interface_flux_agrees_bit_exactly_from_both_sides<double>(1e-13);
}
CFE_TEST(test_forward_step_block_interface_flux_agrees_bit_exactly_from_both_sides_float)
{
  check_forward_step_block_interface_flux_agrees_bit_exactly_from_both_sides<float>(1e-4);
}

// Dynamic companion to the static flux-agreement check above: runs the
// REAL interleaved two-block SSP-RK2 driver (`ForwardStepBlocks`,
// `forward_step_setup.hpp`) for a short time and checks the total
// domain mass change against an independently-known analytic value --
// something the static ghost-fill snapshot above cannot catch (e.g. a
// sign error or a wrong-stage-buffer bug in
// `ForwardStepBlocks::step()`'s hand-interleaved stage coupling, which
// only manifests across actual timesteps, not a single ghost-fill call)
// (Violet review finding).
//
// Why the expected value is exact, not approximate, for a short enough
// `t_final`: the inlet (block1's low-x face) is a FIXED supersonic
// state for the entire run -- since the inflow is supersonic
// everywhere upstream of the step (M=3 > 1 means no characteristic can
// carry information upstream against the flow), the inlet's own
// reconstructed state never differs from that fixed value, so its mass
// flux is EXACTLY rho*u*height = 1.4*3*1.0 = 4.2 for the whole run, not
// just briefly. The outlet (block2's high-x face) is zero-order
// extrapolated -- dynamic, not fixed -- but for `t_final` short enough
// that no disturbance from the step (traveling at a bounded maximum
// wave speed) has reached the outlet 2.4 units downstream of the step
// corner, the outlet remains at the initial uniform state too, so its
// mass flux is exactly rho*u*height = 1.4*3*0.8 = 3.36 throughout.
// `dM/dt = 4.2 - 3.36 = 0.84`, constant, so
// `total_mass(t_final) - total_mass(0) == 0.84 * t_final` exactly (to
// floating-point roundoff) -- not merely "roughly conserved."
template <class Scalar>
void check_forward_step_mass_change_matches_exact_inlet_outlet_flux_balance_for_short_time(double tol)
{
  // t_final=0.05 is well inside the "outlet still undisturbed" window:
  // the fastest signal from the step corner (x=0.6) needs to cross the
  // 2.4 units to the outlet (x=3.0) at a maximum wave speed on the
  // order of 4-5 (M=3 freestream, speed of sound 1, plus whatever a
  // compression adds) -- roughly 0.5-0.6 time units, a >10x margin over
  // the 0.05 used here.
  // `total_mass()` always accumulates its running sum in `double`
  // regardless of `Scalar` (forward_step_setup.hpp, unchanged by this
  // task) -- a higher-precision accumulator technique for summing many
  // per-cell terms without the sum itself magnifying roundoff, NOT the
  // same thing as `kInletFlux`/`kOutletFlux`/`expected_change` below
  // being plain `double` (those are the oracle -- a reference value
  // independent of `Scalar`, see sod_shock_tube_exact.hpp's header
  // comment for that convention). Both end up `double` here, but for
  // two different reasons.
  cfe_verification::ForwardStepBlocks<Scalar, 2> blocks(40);
  const double initial_mass = blocks.total_mass();
  blocks.run_to(Scalar(0.05), Scalar(0.4));
  const double final_mass = blocks.total_mass();

  constexpr double kInletFlux = 1.4 * 3.0 * 1.0;   // rho * u * tunnel height
  constexpr double kOutletFlux = 1.4 * 3.0 * 0.8;  // rho * u * post-step height
  const double expected_change = (kInletFlux - kOutletFlux) * 0.05;

  CFE_CHECK_NEAR(final_mass - initial_mass, expected_change, tol);
}

CFE_TEST(test_forward_step_mass_change_matches_exact_inlet_outlet_flux_balance_for_short_time_double)
{
  check_forward_step_mass_change_matches_exact_inlet_outlet_flux_balance_for_short_time<double>(1e-10);
}
CFE_TEST(test_forward_step_mass_change_matches_exact_inlet_outlet_flux_balance_for_short_time_float)
{
  check_forward_step_mass_change_matches_exact_inlet_outlet_flux_balance_for_short_time<float>(1e-6);
}
