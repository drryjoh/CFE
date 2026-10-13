// Shared forward-facing-step (Woodward & Colella, 1984, JCP 54,
// 115-173, Sec. IV.b) two-block assembly, used by both
// tests/verification/test_forward_step_interface_conservation.cpp's
// bigger sibling checks (if any) and the benchmark tutorials
// (tutorials/euler_2d_forward_step/, tutorials/
// euler_3d_forward_step_extrusion/) -- the same "one driver, shared by
// a test and a tutorial" shape tests/verification/
// sod_shock_tube_runner.hpp already established for the Sod tube.
//
// Geometry (paper's own Sec. IV.b, Mach 3 case): a 1-unit-wide,
// 3-unit-long tunnel with a 0.2-unit step 0.6 units from the inlet.
// Two `CartesianGrid` blocks (task 0010's agreed design -- see
// docs/adr/0011-forward-step-block-interface.md and
// tasks/0010-phase4-forward-step-multiblock.md):
//   - block 1 ("upstream"): x in [0, 0.6), full tunnel height [0, 1.0).
//   - block 2 ("downstream"): x in [0.6, 3.0), post-step height
//     [0.2, 1.0).
// joined by `BlockInterfaceBoundary` (grid/boundary/
// block_interface_boundary.hpp) where they share height
// (y >= 0.2) and a local slip-wall reflection where they don't
// (y < 0.2, the step's riser face).
//
// Boundary conditions:
//   - block 1 low-x (tunnel inlet): fixed supersonic inflow
//     (rho=1.4, u=3, v=w=0, p=1.0), continually fed.
//   - block 2 high-x (tunnel outlet): zero-order-extrapolated
//     supersonic outflow.
//   - every tunnel-wall/step-top face on both blocks (low-y and high-y
//     on both blocks): `SlipWallBoundary` (reflecting).
//   - (3D only) low-z/high-z on both blocks: `PeriodicBoundary` -- a
//     simple extrusion with no 3D physics introduced (symmetric by
//     construction: the IC and every other BC are uniform in Z, so
//     Z-momentum never develops; see this header's `run_to` for the
//     argument this stays exactly true throughout the run, mirroring
//     tests/unit/test_euler_3d_sanity.cpp's own column-for-column
//     argument).
//
// Time stepping: SSP-RK2, adaptive CFL `dt` recomputed every step from
// the CURRENT max wave speed over BOTH blocks' real cells (not a fixed
// a priori bound) -- same policy sod_shock_tube_runner.hpp uses.
//
// Why this does NOT just call `cfe::ssp_rk2_step` once per block: that
// helper's own signature bundles a stage's residual-then-combine
// together internally, assuming nothing else needs to happen between
// its two internal residual() calls. Here something does: block 1's
// stage-2 residual (evaluated on its OWN stage-1 buffer) must read
// block 2's stage-1 buffer through the ghost layer, not block 2's
// current/stage-0 state -- exactly the same "every rank must be in the
// same stage when its neighbor reads it" requirement MPI-decomposed
// solves satisfy via lockstep blocking sends/receives (see
// grid/boundary/mpi_halo_boundary.hpp), but here there is no second
// process/rank to synchronize with -- it is one process doing both
// blocks sequentially, so the two blocks' stages must be interleaved
// BY HAND: compute both blocks' stage-1 residuals (reading each
// other's stage-0 state), combine into both stage-1 buffers, re-point
// each block's interface boundary at the OTHER block's stage-1 buffer,
// then compute both blocks' stage-2 residuals and combine. This file's
// `step()` does exactly that, calling
// `cfe::ssp_rk2_combine_stage1`/`cfe::ssp_rk2_combine_stage2`
// (`solver/time_integration/ssp_rk2.hpp`) directly -- the same two
// formulas `ssp_rk2_step` itself uses internally, pulled out as their
// own reusable functions specifically for this kind of multi-solver
// caller (see that header's own comment).
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>

#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/euler/field.hpp"
#include "cfe/grid/boundary/block_interface_boundary.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/boundary/slip_wall_boundary.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/io/vtk_writer.hpp"
#include "cfe/math/fixed_array.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace cfe_verification {

template <class Scalar, std::size_t Dim>
struct ForwardStepBlocks
{
  using Field = cfe::EulerField<Scalar, Dim>;
  static constexpr std::size_t N = Field::n_components;
  using Storage = cfe::Field<Scalar, N>;
  using View = cfe::FieldView<Scalar, N, cfe::AoSLayout>;

  using BoundaryY = cfe::SlipWallBoundary<Field>;
  using BoundaryZ = cfe::PeriodicBoundary;  // inert (never invoked) when Dim == 2
  using BoundaryX1 = cfe::BlockInterfaceBoundary<Scalar, Field, cfe::Side::High>;
  using BoundaryX2 = cfe::BlockInterfaceBoundary<Scalar, Field, cfe::Side::Low>;

  using Solver1 = cfe::FvmSolver<Scalar, cfe::AoSLayout, Field, BoundaryX1, BoundaryY, BoundaryZ,
                                  cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux>;
  using Solver2 = cfe::FvmSolver<Scalar, cfe::AoSLayout, Field, BoundaryX2, BoundaryY, BoundaryZ,
                                  cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux>;

  static constexpr Scalar kGamma = Scalar(1.4);
  static constexpr Scalar kTunnelHeight = Scalar(1.0);
  static constexpr Scalar kTunnelLength = Scalar(3.0);
  static constexpr Scalar kStepHeight = Scalar(0.2);
  static constexpr Scalar kStepPosition = Scalar(0.6);
  static constexpr Scalar kInflowDensity = Scalar(1.4);
  static constexpr Scalar kInflowPressure = Scalar(1.0);
  static constexpr Scalar kInflowVelocity = Scalar(3.0);

  Field field{};
  cfe::CartesianGrid<Scalar> grid1, grid2;
  // `Field<Scalar,N>` (this struct's `Storage` alias) has no default
  // constructor (its one constructor takes `n_cells`) -- these must be
  // built from `grid1`/`grid2` in the member-initializer list below, in
  // DECLARATION order, not reassigned in the constructor body.
  Storage state1, stage1_1, scratch1;
  Storage state2, stage1_2, scratch2;
  Solver1 solver1;
  Solver2 solver2;

  static cfe::CartesianGrid<Scalar> make_grid1(std::size_t cells_per_unit_height, std::size_t nz_cells)
  {
    const Scalar dxy = kTunnelHeight / static_cast<Scalar>(cells_per_unit_height);
    constexpr bool kDim3 = (Dim == 3);
    cfe::CartesianGrid<Scalar> grid;
    grid.nx = static_cast<std::size_t>(
        std::lround(static_cast<double>(kStepPosition) / static_cast<double>(dxy)));
    grid.ny = cells_per_unit_height;
    grid.nz = kDim3 ? nz_cells : 1;
    grid.ngx = 2;
    grid.ngy = 2;
    grid.ngz = kDim3 ? 2 : 0;
    grid.dx = dxy;
    grid.dy = dxy;
    // `CartesianGrid::dz` is documented as "meaningless (and unused) on
    // an axis with n_cells == 1" -- true for the solver/reconstruction
    // path, but NOT for this file's own `total_mass()`, which multiplies
    // `dx*dy*dz` to get a cell volume regardless of dimensionality.
    // Left at its CartesianGrid-default `Scalar(1)` here for the 2D case
    // (matching a genuinely inactive axis's neutral/no-op value);
    // overwritten below by the constructor, to the real spanwise
    // spacing, once `span_z` is known, for Dim==3.
    grid.origin_x = Scalar(0);
    grid.origin_y = Scalar(0);
    grid.origin_z = Scalar(0);
    return grid;
  }

  static cfe::CartesianGrid<Scalar> make_grid2(std::size_t cells_per_unit_height, std::size_t nz_cells)
  {
    const Scalar dxy = kTunnelHeight / static_cast<Scalar>(cells_per_unit_height);
    const std::size_t step_rows = static_cast<std::size_t>(
        std::lround(static_cast<double>(kStepHeight) / static_cast<double>(dxy)));
    constexpr bool kDim3 = (Dim == 3);
    cfe::CartesianGrid<Scalar> grid;
    grid.nx = static_cast<std::size_t>(
        std::lround(static_cast<double>(kTunnelLength - kStepPosition) / static_cast<double>(dxy)));
    grid.ny = cells_per_unit_height - step_rows;
    grid.nz = kDim3 ? nz_cells : 1;
    grid.ngx = 2;
    grid.ngy = 2;
    grid.ngz = kDim3 ? 2 : 0;
    grid.dx = dxy;
    grid.dy = dxy;
    // See make_grid1's comment above: left at the CartesianGrid default
    // (Scalar(1)) for 2D, overwritten by the constructor for Dim==3.
    grid.origin_x = kStepPosition;
    grid.origin_y = kStepHeight;
    grid.origin_z = Scalar(0);
    return grid;
  }

  // `cells_per_unit_height`: the paper's own "80x240" naming is cells
  // per unit height (dx=dy=1/80 there); pass 20/40/80 for the paper's
  // three reference resolutions. `nz_cells`/`span_z` matter only when
  // Dim == 3 (a thin spanwise extrusion -- "see it work," not a
  // physically meaningful depth).
  ForwardStepBlocks(std::size_t cells_per_unit_height, std::size_t nz_cells = 1, Scalar span_z = Scalar(1))
      : grid1(make_grid1(cells_per_unit_height, nz_cells)),
        grid2(make_grid2(cells_per_unit_height, nz_cells)),
        state1(grid1.n_cells_total()),
        stage1_1(grid1.n_cells_total()),
        scratch1(grid1.n_cells_total()),
        state2(grid2.n_cells_total()),
        stage1_2(grid2.n_cells_total()),
        scratch2(grid2.n_cells_total())
  {
    field.gamma = kGamma;
    if constexpr (Dim == 3) {
      const Scalar dz = span_z / static_cast<Scalar>(nz_cells);
      grid1.dz = dz;
      grid2.dz = dz;
    }

    const std::size_t step_rows = grid1.ny - grid2.ny;

    // Uniform Mach-3 initial condition everywhere in both blocks --
    // "continually fed from the left" (task spec): the inlet BC holds
    // this same state at the low-x ghost layer for the whole run.
    cfe::Vector<Scalar, Dim> inflow_velocity{};
    inflow_velocity[0] = kInflowVelocity;
    const cfe::State<Scalar, N> inflow_state =
        field.conserved_from_primitive(kInflowDensity, inflow_velocity, kInflowPressure);

    fill_uniform(state1, grid1, inflow_state);
    fill_uniform(state2, grid2, inflow_state);

    // j_offset_to_neighbor derivation: block1's padded j is
    // `ngy1 + local_j1` (local_j1 in [0, ny1)); in the shared band
    // (local_j1 >= step_rows), that same physical row is block2's
    // local_j2 = local_j1 - step_rows, i.e. block2's padded j is
    // `ngy2 + local_j2 = ngy2 + (block1_padded_j - ngy1) - step_rows`
    // = `block1_padded_j + (ngy2 - ngy1 - step_rows)`. So
    // `offset_1_to_2 = ngy2 - ngy1 - step_rows` (added to block1's
    // padded j to get block2's); the inverse direction
    // (block2 -> block1) is its negation.
    const long offset_1_to_2 = static_cast<long>(grid2.ngy) - static_cast<long>(grid1.ngy) -
                               static_cast<long>(step_rows);
    const long offset_2_to_1 = -offset_1_to_2;

    // Designated initializers (C++20) rather than six positional,
    // same-typed (`std::size_t`/`long`) arguments -- a field-order
    // transposition here would otherwise compile silently and only show
    // up as a wrong ghost value at runtime (Sally review finding).
    BoundaryX1 boundary_x1{.domain_edge_value = inflow_state,
                           .neighbor_field = state2.view(),
                           .neighbor_grid = grid2,
                           .j_offset_to_neighbor = offset_1_to_2,
                           .j_valid_lo = grid1.ngy + step_rows,
                           .j_valid_hi = grid1.ngy + grid1.ny};
    BoundaryX2 boundary_x2{.domain_edge_value = cfe::State<Scalar, N>{},
                           .neighbor_field = state1.view(),
                           .neighbor_grid = grid1,
                           .j_offset_to_neighbor = offset_2_to_1,
                           .j_valid_lo = grid2.ngy,
                           .j_valid_hi = grid2.ngy + grid2.ny};

    solver1 = Solver1{grid1, field, boundary_x1, BoundaryY{}, BoundaryZ{}};
    solver2 = Solver2{grid2, field, boundary_x2, BoundaryY{}, BoundaryZ{}};
  }

  static void fill_uniform(Storage& state, const cfe::CartesianGrid<Scalar>& grid, const cfe::State<Scalar, N>& q)
  {
    const std::size_t total = grid.n_cells_total();
    for (std::size_t cell = 0; cell < total; ++cell) {
      for (std::size_t c = 0; c < N; ++c) state(cell, c) = q[c];
    }
  }

  // Max |u_axis| + c over every REAL cell of both blocks -- used to
  // size the adaptive CFL `dt`.
  Scalar max_wave_speed() const
  {
    Scalar max_speed = Scalar(0);
    auto scan = [&](const Storage& state, const cfe::CartesianGrid<Scalar>& grid) {
      for (std::size_t k = 0; k < grid.nz; ++k) {
        for (std::size_t j = 0; j < grid.ny; ++j) {
          for (std::size_t i = 0; i < grid.nx; ++i) {
            const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
            cfe::State<Scalar, N> q;
            for (std::size_t c = 0; c < N; ++c) q[c] = state(cell, c);
            const Scalar speed = std::abs(field.velocity_component(q, cfe::Axis::X)) + field.sound_speed(q);
            max_speed = speed > max_speed ? speed : max_speed;
          }
        }
      }
    };
    scan(state1, grid1);
    scan(state2, grid2);
    return max_speed;
  }

  // One SSP-RK2 step, both blocks interleaved stage-by-stage -- see
  // this file's header comment for why.
  void step(Scalar dt)
  {
    // Stage 1: both blocks read each other's CURRENT (stage-0) state
    // through the interface boundary (already the case: boundary_x's
    // neighbor_field already points at state2/state1 after construction
    // and after the end of the previous step's stage 2, below).
    solver1.residual(state1.view(), scratch1.view());
    solver2.residual(state2.view(), scratch2.view());
    cfe::ssp_rk2_combine_stage1<Scalar>(state1.view(), stage1_1.view(), scratch1.view(), dt,
                                        solver1.active_cell_count(), solver1.active_cell_index_map());
    cfe::ssp_rk2_combine_stage1<Scalar>(state2.view(), stage1_2.view(), scratch2.view(), dt,
                                        solver2.active_cell_count(), solver2.active_cell_index_map());

    // Stage 2: each block's interface boundary must now read the
    // OTHER block's STAGE-1 buffer, not its state buffer -- re-point
    // before computing either residual.
    solver1.boundary_x.neighbor_field = stage1_2.view();
    solver2.boundary_x.neighbor_field = stage1_1.view();
    solver1.residual(stage1_1.view(), scratch1.view());
    solver2.residual(stage1_2.view(), scratch2.view());
    cfe::ssp_rk2_combine_stage2<Scalar>(state1.view(), stage1_1.view(), scratch1.view(), dt,
                                        solver1.active_cell_count(), solver1.active_cell_index_map());
    cfe::ssp_rk2_combine_stage2<Scalar>(state2.view(), stage1_2.view(), scratch2.view(), dt,
                                        solver2.active_cell_count(), solver2.active_cell_index_map());

    // Restore both interface boundaries to point at the (now updated)
    // state buffers, ready for the next step's stage 1.
    solver1.boundary_x.neighbor_field = state2.view();
    solver2.boundary_x.neighbor_field = state1.view();
  }

  // Runs to `t_final` with adaptive-CFL `dt`, recomputed every step.
  // Returns the number of steps taken.
  std::size_t run_to(Scalar t_final, Scalar cfl)
  {
    std::size_t steps = 0;
    Scalar t = Scalar(0);
    while (t < t_final) {
      const Scalar speed = max_wave_speed();
      Scalar dt = cfl * grid1.dx / speed;
      if (t + dt > t_final) dt = t_final - t;
      step(dt);
      t += dt;
      ++steps;
    }
    return steps;
  }

  double total_mass() const
  {
    double mass = 0.0;
    auto accumulate = [&](const Storage& state, const cfe::CartesianGrid<Scalar>& grid) {
      const double cell_volume = static_cast<double>(grid.dx) * static_cast<double>(grid.dy) *
                                  static_cast<double>(grid.dz);
      for (std::size_t k = 0; k < grid.nz; ++k) {
        for (std::size_t j = 0; j < grid.ny; ++j) {
          for (std::size_t i = 0; i < grid.nx; ++i) {
            const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
            mass += static_cast<double>(state(cell, 0)) * cell_volume;
          }
        }
      }
    };
    accumulate(state1, grid1);
    accumulate(state2, grid2);
    return mass;
  }

  // Writes one scalar-field VTK snapshot per block (density, pressure,
  // or any other per-cell scalar the caller computes) -- two files per
  // frame (`path_prefix + "_block1.vtk"` / `"_block2.vtk"`), each tiling
  // exactly against the other since both grids' real-cell physical
  // extents are exact (no overlap, no gap): block1 covers
  // x in [0, 0.6), block2 covers x in [0.6, 3.0).
  template <class Compute>
  void write_vtk_frame(const std::string& path_prefix, const char* scalar_name, Compute compute) const
  {
    write_vtk_block(path_prefix + "_block1.vtk", grid1, state1, scalar_name, compute);
    write_vtk_block(path_prefix + "_block2.vtk", grid2, state2, scalar_name, compute);
  }

 private:
  template <class Compute>
  static void write_vtk_block(const std::string& path, const cfe::CartesianGrid<Scalar>& grid,
                               const Storage& state, const char* scalar_name, Compute compute)
  {
    cfe::Field<Scalar, 1> scalar_field(grid.n_cells_total());
    for (std::size_t k = 0; k < grid.nz; ++k) {
      for (std::size_t j = 0; j < grid.ny; ++j) {
        for (std::size_t i = 0; i < grid.nx; ++i) {
          const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
          cfe::State<Scalar, N> q;
          for (std::size_t c = 0; c < N; ++c) q[c] = state(cell, c);
          scalar_field(cell, 0) = compute(q);
        }
      }
    }
    cfe::io::write_vtk_structured_points_cell_scalar<Scalar, cfe::AoSLayout>(path, grid, scalar_field.view(),
                                                                             scalar_name);
  }
};

// Shared step/snapshot/progress-report loop, used by BOTH
// `tutorials/euler_2d_forward_step/forward_step_2d.cpp` (once per
// resolution) and `tutorials/euler_3d_forward_step_extrusion/
// forward_step_3d.cpp` (once, at its one resolution) -- factored out
// after Snorlax's review flagged the two tutorials' `main()` loops as
// near-duplicates of each other. Each tutorial still supplies its own
// `write_frame(frame_index)` callback (which quantities to dump to VTK
// differs between them: density+pressure per block for the 2D
// tutorial, density only -- plus its own separate Z-momentum diagnostic
// -- for the 3D one), and still does its own resolution looping/CSV
// writing with the returned `RunReport`; only the "advance one step,
// write a snapshot on schedule, print progress, time the whole thing"
// mechanics are shared here.
template <class Scalar>
struct RunReport
{
  std::size_t steps = 0;
  double wall_clock_s = 0.0;
  std::size_t frames = 0;
  double final_mass = 0.0;
};

template <class Blocks, class Scalar, class WriteFrame>
RunReport<Scalar> run_and_report(Blocks& blocks, Scalar t_final, Scalar cfl, Scalar snapshot_dt,
                                  WriteFrame write_frame, const char* progress_tag = "",
                                  std::size_t progress_every_n_steps = 200)
{
  std::size_t frame = 0;
  Scalar next_snapshot = Scalar(0);
  Scalar t = Scalar(0);
  std::size_t total_steps = 0;

  const auto wall_start = std::chrono::steady_clock::now();
  write_frame(frame);
  ++frame;
  next_snapshot = snapshot_dt;

  while (t < t_final) {
    const Scalar speed = blocks.max_wave_speed();
    Scalar dt = cfl * blocks.grid1.dx / speed;
    if (t + dt > t_final) dt = t_final - t;
    blocks.step(dt);
    t += dt;
    ++total_steps;

    if (t + Scalar(1e-9) >= next_snapshot || t >= t_final) {
      write_frame(frame);
      ++frame;
      next_snapshot += snapshot_dt;
    }

    if (progress_every_n_steps > 0 && total_steps % progress_every_n_steps == 0) {
      std::printf("  %s: t=%.4f (%.1f%%), steps=%zu\n", progress_tag, static_cast<double>(t),
                  100.0 * static_cast<double>(t) / static_cast<double>(t_final), total_steps);
      std::fflush(stdout);
    }
  }
  const auto wall_end = std::chrono::steady_clock::now();

  RunReport<Scalar> report;
  report.steps = total_steps;
  report.wall_clock_s = std::chrono::duration<double>(wall_end - wall_start).count();
  report.frames = frame;
  report.final_mass = blocks.total_mass();
  return report;
}

}  // namespace cfe_verification
