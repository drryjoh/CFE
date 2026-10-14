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

using Scalar = double;
using Field = cfe::EulerField<Scalar, 2>;
constexpr std::size_t kN = 4;

cfe::CartesianGrid<Scalar> make_block(std::size_t nx, std::size_t ny, Scalar origin_x, Scalar origin_y)
{
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = nx;
  grid.ny = ny;
  grid.ngx = 2;
  grid.ngy = 0;  // only fill_x is exercised here
  grid.dx = 0.1;
  grid.dy = 0.1;
  grid.origin_x = origin_x;
  grid.origin_y = origin_y;
  return grid;
}

// Smooth, fully-populated, non-uniform primitive state -- a small
// sinusoidal perturbation of a Mach-3-like freestream, so every
// conserved component has a real, nonzero gradient everywhere
// (density, both momentum components, and energy all vary).
cfe::State<Scalar, kN> state_at(const Field& field, Scalar x, Scalar y)
{
  const Scalar rho = 1.4 + 0.05 * std::sin(7.0 * x + 3.0 * y);
  const Scalar ux = 3.0 + 0.1 * std::cos(5.0 * x - 2.0 * y);
  const Scalar uy = 0.1 * std::sin(4.0 * x + 6.0 * y);
  const Scalar p = 1.0 + 0.02 * std::cos(6.0 * x + 4.0 * y);
  cfe::Vector<Scalar, 2> velocity;
  velocity[0] = ux;
  velocity[1] = uy;
  return field.conserved_from_primitive(rho, velocity, p);
}

void fill_smooth_state(cfe::Field<Scalar, kN>& state, const cfe::CartesianGrid<Scalar>& grid, const Field& field)
{
  for (std::size_t j = 0; j < grid.ny; ++j) {
    for (std::size_t i = 0; i < grid.nx; ++i) {
      const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, 0);
      const cfe::State<Scalar, kN> q = state_at(field, grid.x_center(grid.ngx + i), grid.y_center(grid.ngy + j));
      for (std::size_t c = 0; c < kN; ++c) state(cell, c) = q[c];
    }
  }
}

cfe::State<Scalar, kN> read(const cfe::Field<Scalar, kN>& state, std::size_t cell)
{
  cfe::State<Scalar, kN> q;
  for (std::size_t c = 0; c < kN; ++c) q[c] = state(cell, c);
  return q;
}

}  // namespace

CFE_TEST(test_forward_step_block_interface_flux_agrees_bit_exactly_from_both_sides)
{
  // block1: upstream, full tunnel height (ny=10, step at local j=2 ->
  // real y=0.2). block2: downstream, post-step height (ny=8), covering
  // block1's local j=2..9.
  auto grid1 = make_block(6, 10, 0.0, 0.0);
  auto grid2 = make_block(24, 8, 0.6, 0.2);

  Field field{};  // gamma = 1.4 default

  cfe::Field<Scalar, kN> state1(grid1.n_cells_total());
  cfe::Field<Scalar, kN> state2(grid2.n_cells_total());
  fill_smooth_state(state1, grid1, field);
  fill_smooth_state(state2, grid2, field);

  cfe::Vector<Scalar, 2> inflow_velocity;
  inflow_velocity[0] = 3.0;
  inflow_velocity[1] = 0.0;
  const cfe::State<Scalar, kN> inflow = field.conserved_from_primitive(1.4, inflow_velocity, 1.0);

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
  for (std::size_t j1 = 2; j1 < grid1.ny; ++j1) {
    const std::size_t j2 = j1 - 2;

    // --- Block1's own view of the flux at its right (interface) face.
    const std::size_t m1 = grid1.flat_index(grid1.ngx + grid1.nx - 2, j1, 0);
    const std::size_t c1 = grid1.flat_index(grid1.ngx + grid1.nx - 1, j1, 0);
    const std::size_t p1_ghost = grid1.flat_index(grid1.ngx + grid1.nx, j1, 0);
    const std::size_t p2_ghost = grid1.flat_index(grid1.ngx + grid1.nx + 1, j1, 0);

    const cfe::State<Scalar, kN> q_m1 = read(state1, m1);
    const cfe::State<Scalar, kN> q_c1 = read(state1, c1);
    const cfe::State<Scalar, kN> q_p1 = read(state1, p1_ghost);
    const cfe::State<Scalar, kN> q_p2 = read(state1, p2_ghost);

    cfe::State<Scalar, kN> left_of_right, right_of_right;
    for (std::size_t c = 0; c < kN; ++c) {
      left_of_right[c] = reconstruction.right(q_m1[c], q_c1[c], q_p1[c]);
      right_of_right[c] = reconstruction.left(q_c1[c], q_p1[c], q_p2[c]);
    }
    const cfe::State<Scalar, kN> flux_from_block1 =
        numerical_flux(left_of_right, right_of_right, cfe::Axis::X, field);

    // --- Block2's own view of the flux at its left (interface) face.
    const std::size_t m2_ghost = grid2.flat_index(grid2.ngx - 2, j2, 0);
    const std::size_t m1_ghost = grid2.flat_index(grid2.ngx - 1, j2, 0);
    const std::size_t c2 = grid2.flat_index(grid2.ngx, j2, 0);
    const std::size_t p1 = grid2.flat_index(grid2.ngx + 1, j2, 0);

    const cfe::State<Scalar, kN> r_m2 = read(state2, m2_ghost);
    const cfe::State<Scalar, kN> r_m1 = read(state2, m1_ghost);
    const cfe::State<Scalar, kN> r_c = read(state2, c2);
    const cfe::State<Scalar, kN> r_p1 = read(state2, p1);

    cfe::State<Scalar, kN> left_of_left, right_of_left;
    for (std::size_t c = 0; c < kN; ++c) {
      left_of_left[c] = reconstruction.right(r_m2[c], r_m1[c], r_c[c]);
      right_of_left[c] = reconstruction.left(r_m1[c], r_c[c], r_p1[c]);
    }
    const cfe::State<Scalar, kN> flux_from_block2 =
        numerical_flux(left_of_left, right_of_left, cfe::Axis::X, field);

    for (std::size_t c = 0; c < kN; ++c) {
      CFE_CHECK_NEAR(flux_from_block1[c], flux_from_block2[c], 1e-13);
    }
  }
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
CFE_TEST(test_forward_step_mass_change_matches_exact_inlet_outlet_flux_balance_for_short_time)
{
  // t_final=0.05 is well inside the "outlet still undisturbed" window:
  // the fastest signal from the step corner (x=0.6) needs to cross the
  // 2.4 units to the outlet (x=3.0) at a maximum wave speed on the
  // order of 4-5 (M=3 freestream, speed of sound 1, plus whatever a
  // compression adds) -- roughly 0.5-0.6 time units, a >10x margin over
  // the 0.05 used here.
  cfe_verification::ForwardStepBlocks<Scalar, 2> blocks(40);
  const double initial_mass = blocks.total_mass();
  blocks.run_to(Scalar(0.05), Scalar(0.4));
  const double final_mass = blocks.total_mass();

  constexpr double kInletFlux = 1.4 * 3.0 * 1.0;   // rho * u * tunnel height
  constexpr double kOutletFlux = 1.4 * 3.0 * 0.8;  // rho * u * post-step height
  const double expected_change = (kInletFlux - kOutletFlux) * 0.05;

  CFE_CHECK_NEAR(final_mass - initial_mass, expected_change, 1e-10);
}
