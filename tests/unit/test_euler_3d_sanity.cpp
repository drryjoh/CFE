// 3D CPU correctness test for EulerField (closes the gap flagged when
// task 0009 was otherwise ready to commit: "is there a 3D Sod shock
// tube?" -- there wasn't one). Same philosophy as
// test_scalar_advection_2d_sanity.cpp/test_burgers_3d_sanity.cpp: the
// Sod shock tube (sod_shock_tube_runner.hpp, shared with the production
// verification test tests/verification/test_euler_sod_shock_tube.cpp and
// the tutorial) varies only in X; made uniform (and periodic, hence
// inert -- zero gradient means every face sees equal left/right states,
// so the Rusanov flux there reduces to the plain physical flux with no
// dissipation term, and the flux DIFFERENCE across that face is exactly
// zero) in Y and Z, so every (j,k) column is an independent copy of the
// same 1D problem. A 3D solve should therefore match the shared 1D
// reference column for column, including: the adaptive-CFL dt sequence
// itself, since the per-step max-wave-speed scan over the FULL 3D domain
// equals the 1D scan's value at every step (every column is identical
// and Y/Z momentum/velocity stays exactly zero throughout -- argued
// below, not just asserted at t=0).
//
// This directly exercises the Y and Z `axis_flux_difference` branches
// with EulerField/RusanovFlux/MusclMinmodReconstruction -- the
// production verification test only ever runs `EulerField<Scalar,1>`,
// so it cannot distinguish "Y/Z transport correct for a multi-component
// Field" from "silently broken" (an axis mixup, e.g. reading
// `momentum_index(Axis::Y)` where `Axis::Z` was meant, would not show up
// there at all).
#include <cmath>
#include <cstddef>

#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/euler/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/math/fixed_array.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"
#include "sod_shock_tube_runner.hpp"
#include "test_framework.hpp"

namespace {

using Scalar = double;
constexpr Scalar kGamma = Scalar(1.4);
constexpr Scalar kX0 = Scalar(0.5);
constexpr Scalar kTFinal = Scalar(0.2);
constexpr Scalar kCfl = Scalar(0.4);
constexpr std::size_t kNx = 100;
constexpr std::size_t kN3 = 5;  // EulerField<Scalar,3>::n_components

const cfe_verification::PrimitiveState kLeft{1.0, 0.0, 1.0};
const cfe_verification::PrimitiveState kRight{0.125, 0.0, 0.1};

}  // namespace

CFE_TEST(test_euler_3d_sod_shock_tube_matches_1d_reference_column_for_column_when_uniform_in_y_and_z)
{
  // --- 1D reference: the exact same shared driver the production Sod
  // verification test and tutorial use, at the classic parameters.
  const cfe_verification::SodProfile<Scalar> reference =
      cfe_verification::run_sod_shock_tube<Scalar>(kNx, kGamma, kLeft, kRight, kX0, kTFinal, kCfl);

  // --- 3D solve: same X profile repeated across every (j,k) column,
  // periodic (inert) Y/Z.
  cfe::CartesianGrid<Scalar> grid;
  grid.nx = kNx;
  grid.ny = 5;
  grid.nz = 3;
  grid.ngx = 2;
  grid.ngy = 2;
  grid.ngz = 2;
  grid.dx = Scalar(1.0) / static_cast<Scalar>(kNx);
  grid.dy = Scalar(0.3);  // deliberately different from dx/dz -- should have no effect
  grid.dz = Scalar(0.7);

  cfe::Field<Scalar, kN3> state(grid.n_cells_total());
  cfe::Field<Scalar, kN3> stage1(grid.n_cells_total());
  cfe::Field<Scalar, kN3> scratch(grid.n_cells_total());

  cfe::EulerField<Scalar, 3> field{};
  field.gamma = kGamma;

  // Zero-initialized (FixedArray's default constructor zero-fills), so
  // this is exactly a zero velocity vector in all 3 directions.
  const cfe::Vector<Scalar, 3> zero_velocity{};
  const cfe::State<Scalar, kN3> left_q =
      field.conserved_from_primitive(Scalar(kLeft.rho), zero_velocity, Scalar(kLeft.p));
  const cfe::State<Scalar, kN3> right_q =
      field.conserved_from_primitive(Scalar(kRight.rho), zero_velocity, Scalar(kRight.p));

  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const Scalar x = grid.x_center(grid.ngx + i);
        const cfe::State<Scalar, kN3>& q = (x < kX0) ? left_q : right_q;
        const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
        for (std::size_t c = 0; c < kN3; ++c) state(cell, c) = q[c];
      }
    }
  }

  cfe::StaticBoundary<Scalar, kN3> boundary_x{left_q, right_q};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::EulerField<Scalar, 3>, cfe::StaticBoundary<Scalar, kN3>,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux>
      solver{grid, field, boundary_x};
  auto residual = [&](cfe::FieldView<Scalar, kN3> in, cfe::FieldView<Scalar, kN3> out) {
    solver.residual(in, out);
  };

  // Same adaptive-CFL stepping scheme as sod_shock_tube_runner.hpp's 1D
  // loop, but scanning the FULL 3D domain for the current max wave
  // speed -- this must land on exactly the same dt sequence as the 1D
  // reference: every column holds an identical state to the 1D profile
  // at every step (Y/Z start uniform and periodic Y/Z ghost-fill keeps
  // them uniform; the Y/Z flux difference is identically zero at every
  // face, as this file's header comment argues, so nothing ever
  // introduces column-to-column variation), and Y/Z velocity is zero
  // throughout, so `|velocity_component(q, Axis::X)| + sound_speed(q)`
  // evaluated at any cell equals the 1D driver's own per-cell value.
  Scalar t = Scalar(0.0);
  while (t < kTFinal) {
    Scalar max_speed = Scalar(0.0);
    for (std::size_t k = 0; k < grid.nz; ++k) {
      for (std::size_t j = 0; j < grid.ny; ++j) {
        for (std::size_t i = 0; i < grid.nx; ++i) {
          cfe::State<Scalar, kN3> q;
          const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
          for (std::size_t c = 0; c < kN3; ++c) q[c] = state(cell, c);
          const Scalar speed = std::abs(field.velocity_component(q, cfe::Axis::X)) + field.sound_speed(q);
          max_speed = speed > max_speed ? speed : max_speed;
        }
      }
    }
    Scalar dt = kCfl * grid.dx / max_speed;
    if (t + dt > kTFinal) dt = kTFinal - t;
    cfe::ssp_rk2_step<Scalar>(state.view(), stage1.view(), scratch.view(), dt, residual,
                               solver.active_cell_count(), solver.active_cell_index_map());
    t += dt;
  }

  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        cfe::State<Scalar, kN3> q;
        const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
        for (std::size_t c = 0; c < kN3; ++c) q[c] = state(cell, c);

        // This file's header comment argues every (j,k) column runs the
        // IDENTICAL sequence of floating-point operations as the 1D
        // reference (same dt sequence, same per-cell arithmetic, Y/Z
        // flux differences identically zero). On this development
        // machine (arm64/clang) that reasoning holds to the bit --
        // EXACT (0.0) tolerance passes here, deterministically, across
        // repeated runs. It does NOT reproduce bit-exact on PSC
        // Bridges-2 (x86-64/gcc 13.3.1): the same reasoning/code
        // produces a nonzero (sub-1e-13, so not a real discrepancy --
        // ordinary cross-platform/cross-compiler floating-point
        // reassociation, e.g. differing FMA contraction) difference
        // there instead. `1e-13` (not `0.0`) is used for exactly this
        // reason -- still ~10^3 tighter than the original `1e-10` this
        // tolerance was tightened from, for real regression protection,
        // while staying portable across the toolchains this project
        // actually builds on (AGENTS.md #9: x86-64, Apple Silicon,
        // CUDA).
        CFE_CHECK_NEAR(field.density(q), reference.rho[i], 1e-13);
        CFE_CHECK_NEAR(field.velocity_component(q, cfe::Axis::X), reference.u[i], 1e-13);
        CFE_CHECK_NEAR(field.pressure(q), reference.p[i], 1e-13);
        // Y/Z momentum (components 2 and 3 of EulerField<Scalar,3>'s
        // conserved-state layout): must stay exactly zero -- the
        // problem never varies in Y/Z and starts at exactly zero there.
        CFE_CHECK_NEAR(q[2], Scalar(0), 1e-13);
        CFE_CHECK_NEAR(q[3], Scalar(0), 1e-13);
      }
    }
  }
}
