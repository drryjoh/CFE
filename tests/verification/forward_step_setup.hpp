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
#include "cfe/field/block_set.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/euler/field.hpp"
#include "cfe/grid/boundary/block_interface_boundary.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/boundary/slip_wall_boundary.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/io/restart_file.hpp"
#include "cfe/io/vtk_writer.hpp"
#include "cfe/math/fixed_array.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace cfe_verification {

// Templated on `NumericalFlux` (default `cfe::RusanovFlux`, every
// existing call site unaffected) so task 0011's forward-step Kelvin-
// Helmholtz re-check with HLLC (tutorials/euler_2d_forward_step/) reuses
// this SAME two-block driver with `cfe::HllcFlux` instead of a second,
// duplicated assembly -- the same reasoning sod_shock_tube_runner.hpp's
// own `NumericalFlux` template parameter follows.
template <class Scalar, std::size_t Dim, class NumericalFlux = cfe::RusanovFlux>
struct ForwardStepBlocks
{
  using Field = cfe::EulerField<Scalar, Dim>;
  static constexpr std::size_t N = Field::n_components;
  // One `BlockSet` per buffer KIND (state/stage1/scratch), each one
  // contiguous allocation covering BOTH blocks -- see
  // src/cfe/field/block_set.hpp and ARCHITECTURE.md #21. Block index
  // `kUpstream` (0) is grid1, `kDownstream` (1) is grid2 -- use these
  // constants, not bare `0`/`1`, at every `.view(...)` call site below
  // (the grid1/grid2 naming is 1-indexed, which invites an off-by-one
  // slip against the 0-indexed `view()` -- these constants remove that
  // trap for a future third block).
  static constexpr std::size_t kUpstream = 0;
  static constexpr std::size_t kDownstream = 1;
  using Blocks = cfe::BlockSet<Scalar, N, cfe::AoSLayout>;
  using View = cfe::FieldView<Scalar, N, cfe::AoSLayout>;
  // What `Blocks::view(block)` returns from a `const ForwardStepBlocks`
  // method (`max_wave_speed`/`total_mass`/`write_vtk_frame` below) --
  // read-only, same reasoning as `Field::ConstView` (field.hpp).
  using ConstView = cfe::FieldView<const Scalar, N, cfe::AoSLayout>;

  using BoundaryY = cfe::SlipWallBoundary<Field>;
  using BoundaryZ = cfe::PeriodicBoundary;  // inert (never invoked) when Dim == 2
  using BoundaryX1 = cfe::BlockInterfaceBoundary<Scalar, Field, cfe::Side::High>;
  using BoundaryX2 = cfe::BlockInterfaceBoundary<Scalar, Field, cfe::Side::Low>;

  using Solver1 = cfe::FvmSolver<Scalar, cfe::AoSLayout, Field, BoundaryX1, BoundaryY, BoundaryZ,
                                  cfe::fvm::MusclMinmodReconstruction, NumericalFlux>;
  using Solver2 = cfe::FvmSolver<Scalar, cfe::AoSLayout, Field, BoundaryX2, BoundaryY, BoundaryZ,
                                  cfe::fvm::MusclMinmodReconstruction, NumericalFlux>;

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
  // `BlockSet` (this struct's `Blocks` alias) has no default constructor
  // (its one constructor takes the per-block cell counts) -- these must
  // be built from `grid1`/`grid2` in the member-initializer list below,
  // in DECLARATION order, not reassigned in the constructor body. Each
  // one is a SINGLE allocation covering both blocks' cells for that
  // buffer kind -- `state_blocks.view(kUpstream)`/`view(kDownstream)` replace the former
  // independent `state1`/`state2` Fields (same for stage1/scratch), with
  // zero change to how any per-cell numerics code below consumes them
  // (they still only ever see a plain `FieldView`).
  Blocks state_blocks, stage1_blocks, scratch_blocks;
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
    // Nominal values -- the constructor overwrites both `origin_x` and
    // `origin_y` right after this returns, to the real-space extent
    // `grid1` actually rounded to, which can differ from these nominal
    // constants by a sub-cell amount at a resolution that doesn't divide
    // the geometry evenly (see the constructor's own comment at that
    // assignment for why). `dz` likewise left at the CartesianGrid
    // default (Scalar(1)) for 2D, overwritten by the constructor for
    // Dim==3, same as make_grid1's comment above.
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
        state_blocks{grid1.n_cells_total(), grid2.n_cells_total()},
        stage1_blocks{grid1.n_cells_total(), grid2.n_cells_total()},
        scratch_blocks{grid1.n_cells_total(), grid2.n_cells_total()}
  {
    field.gamma = kGamma;
    if constexpr (Dim == 3) {
      const Scalar dz = span_z / static_cast<Scalar>(nz_cells);
      grid1.dz = dz;
      grid2.dz = dz;
    }

    const std::size_t step_rows = grid1.ny - grid2.ny;

    // `make_grid1`/`make_grid2` each independently round their own cell
    // count from the nominal geometry constants (`kStepPosition`,
    // `kStepHeight`) via `std::lround`, since `cells_per_unit_height`
    // doesn't evenly divide them at every resolution (e.g. at 96,
    // `0.6 * 96 = 57.6` rounds to 58 cells, not 57.6). `make_grid2` then
    // set its own `origin_x`/`origin_y` directly to those SAME nominal
    // constants rather than to `grid1`'s actual (rounded) real extent --
    // harmless for the solve itself (`BlockInterfaceBoundary` matches
    // the two blocks by cell INDEX, never by real-space coordinate, so
    // this never affected any computed result), but it leaves a
    // sub-cell-width gap/overlap between where block1's real cells
    // actually end and where block2's real cells claim to begin,
    // reproducible in `plot_results.py`'s combined view as a thin stray
    // line exactly at the interface for any resolution that doesn't
    // divide the geometry evenly (first seen at cells-per-unit-height=96
    // when the resolution sweep was extended past 20/40/80). Correcting
    // it here, once, right after `step_rows` is derived, keeps the two
    // blocks' real-space extents exactly consistent regardless of
    // rounding at any resolution.
    grid2.origin_x = grid1.origin_x + static_cast<Scalar>(grid1.nx) * grid1.dx;
    grid2.origin_y = grid1.origin_y + static_cast<Scalar>(step_rows) * grid1.dy;

    // Uniform Mach-3 initial condition everywhere in both blocks --
    // "continually fed from the left" (task spec): the inlet BC holds
    // this same state at the low-x ghost layer for the whole run.
    cfe::Vector<Scalar, Dim> inflow_velocity{};
    inflow_velocity[0] = kInflowVelocity;
    const cfe::State<Scalar, N> inflow_state =
        field.conserved_from_primitive(kInflowDensity, inflow_velocity, kInflowPressure);

    fill_uniform(state_blocks.view(kUpstream), grid1, inflow_state);
    fill_uniform(state_blocks.view(kDownstream), grid2, inflow_state);

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
                           .neighbor_field = state_blocks.view(kDownstream),
                           .neighbor_grid = grid2,
                           .j_offset_to_neighbor = offset_1_to_2,
                           .j_valid_lo = grid1.ngy + step_rows,
                           .j_valid_hi = grid1.ngy + grid1.ny};
    BoundaryX2 boundary_x2{.domain_edge_value = cfe::State<Scalar, N>{},
                           .neighbor_field = state_blocks.view(kUpstream),
                           .neighbor_grid = grid1,
                           .j_offset_to_neighbor = offset_2_to_1,
                           .j_valid_lo = grid2.ngy,
                           .j_valid_hi = grid2.ngy + grid2.ny};

    solver1 = Solver1{grid1, field, boundary_x1, BoundaryY{}, BoundaryZ{}};
    solver2 = Solver2{grid2, field, boundary_x2, BoundaryY{}, BoundaryZ{}};
  }

  // Takes a plain `View` (not a `BlockSet` reference): this function, and
  // every helper below it, never needs to know whether the view it was
  // handed is backed by its own allocation or is one block of a shared
  // `BlockSet` buffer -- that is the whole point of `FieldView` being a
  // non-owning pointer+size (see field.hpp).
  static void fill_uniform(View state, const cfe::CartesianGrid<Scalar>& grid, const cfe::State<Scalar, N>& q)
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
    auto scan = [&](ConstView state, const cfe::CartesianGrid<Scalar>& grid) {
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
    scan(state_blocks.view(kUpstream), grid1);
    scan(state_blocks.view(kDownstream), grid2);
    return max_speed;
  }

  // One SSP-RK2 step, both blocks interleaved stage-by-stage -- see
  // this file's header comment for why.
  void step(Scalar dt)
  {
    // Stage 1: both blocks read each other's CURRENT (stage-0) state
    // through the interface boundary (already the case: boundary_x's
    // neighbor_field already points at state_blocks.view(kDownstream) /
    // view(kUpstream) (solver1's / solver2's neighbor, respectively)
    // after construction and after the end of the previous step's
    // stage 2, below).
    solver1.residual(state_blocks.view(kUpstream), scratch_blocks.view(kUpstream));
    solver2.residual(state_blocks.view(kDownstream), scratch_blocks.view(kDownstream));
    cfe::ssp_rk2_combine_stage1<Scalar>(state_blocks.view(kUpstream), stage1_blocks.view(kUpstream),
                                        scratch_blocks.view(kUpstream), dt, solver1.active_cell_count(),
                                        solver1.active_cell_index_map());
    cfe::ssp_rk2_combine_stage1<Scalar>(state_blocks.view(kDownstream), stage1_blocks.view(kDownstream),
                                        scratch_blocks.view(kDownstream), dt, solver2.active_cell_count(),
                                        solver2.active_cell_index_map());

    // Stage 2: each block's interface boundary must now read the
    // OTHER block's STAGE-1 buffer, not its state buffer -- re-point
    // before computing either residual.
    solver1.boundary_x.neighbor_field = stage1_blocks.view(kDownstream);
    solver2.boundary_x.neighbor_field = stage1_blocks.view(kUpstream);
    solver1.residual(stage1_blocks.view(kUpstream), scratch_blocks.view(kUpstream));
    solver2.residual(stage1_blocks.view(kDownstream), scratch_blocks.view(kDownstream));
    cfe::ssp_rk2_combine_stage2<Scalar>(state_blocks.view(kUpstream), stage1_blocks.view(kUpstream),
                                        scratch_blocks.view(kUpstream), dt, solver1.active_cell_count(),
                                        solver1.active_cell_index_map());
    cfe::ssp_rk2_combine_stage2<Scalar>(state_blocks.view(kDownstream), stage1_blocks.view(kDownstream),
                                        scratch_blocks.view(kDownstream), dt, solver2.active_cell_count(),
                                        solver2.active_cell_index_map());

    // Restore both interface boundaries to point at the (now updated)
    // state buffers, ready for the next step's stage 1.
    solver1.boundary_x.neighbor_field = state_blocks.view(kDownstream);
    solver2.boundary_x.neighbor_field = state_blocks.view(kUpstream);
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

  // Runs exactly `n_steps` adaptive-CFL steps from the CURRENT state, with
  // no `t_final` target/clamping -- unlike `run_to` above, which clamps
  // its last `dt` to land exactly on `t_final`. This is the
  // restart-friendly entry point (task tasks/0016-checkpoint-restart.md):
  // the clamping `run_to` does (`if (t + dt > t_final) dt = t_final - t`)
  // depends on the ACCUMULATED `t` value, so reproducing a `run_to`-driven
  // run's exact tail bit-for-bit after a restart would require restoring
  // that same accumulated `t` and re-deriving the same clamping decision
  // -- solvable, but unneeded complexity this task does not require (the
  // user's own description of the workflow is already step-count-based:
  // "checkpoint ~1000 steps before the target, then only simulate those
  // last ~1000 steps"). `run_n_steps` sidesteps the question entirely:
  // every step's `dt` depends only on the CURRENT real-cell state (via
  // `max_wave_speed()`), never on an absolute elapsed-time value, so a
  // sequence of `run_n_steps` calls is trivially restart-resumable --
  // "run N steps" then "run (N-1000) steps, checkpoint, run 1000 more"
  // take the IDENTICAL sequence of steps/dt's either way, which is
  // exactly the property `tests/unit/test_forward_step_restart.cpp`
  // checks bit-for-bit. Returns the elapsed simulated time over those
  // steps (for the caller's own step/time bookkeeping -- see
  // `write_restart`'s own comment for why that bookkeeping never feeds
  // back into the solve itself).
  Scalar run_n_steps(std::size_t n_steps, Scalar cfl)
  {
    Scalar t = Scalar(0);
    for (std::size_t s = 0; s < n_steps; ++s) {
      const Scalar speed = max_wave_speed();
      const Scalar dt = cfl * grid1.dx / speed;
      step(dt);
      t += dt;
    }
    return t;
  }

  // Writes both blocks' CURRENT real-cell state, plus the caller-supplied
  // `step`/`time` bookkeeping pair, to one checkpoint file (see
  // src/cfe/io/restart_file.hpp for the on-disk format and the ghost-
  // cell-exclusion assumption this relies on). `step`/`time` are
  // whatever the caller has been accumulating itself (e.g. the running
  // total from successive `run_n_steps` calls) -- this struct keeps no
  // step/time state of its own, since nothing in the solve ever reads it
  // back; it exists purely so a resumed run can report "resumed at step
  // 11000, t=3.67" instead of restarting that bookkeeping at zero.
  void write_restart(const std::string& path, std::size_t step, Scalar time) const
  {
    cfe::io::write_restart_checkpoint<Scalar, N, cfe::AoSLayout>(
        path, {grid1, grid2}, {state_blocks.view(kUpstream), state_blocks.view(kDownstream)}, step, time);
  }

  // What `read_restart` hands back: the step/time bookkeeping the
  // checkpoint was written with, for the resuming driver to continue
  // accumulating from.
  struct RestartInfo
  {
    std::size_t step = 0;
    Scalar time = Scalar(0);
  };

  // Loads a checkpoint written by `write_restart` into both blocks'
  // CURRENT real-cell state (ghost cells are left as-is -- see
  // src/cfe/io/restart_file.hpp's header comment: the next `step()` call
  // refills them before reading them regardless). Throws if the file's
  // format version, scalar size, component count, or either block's grid
  // shape does not match THIS object's own grids -- i.e. this must be
  // called on a `ForwardStepBlocks` already constructed at the same
  // resolution the checkpoint was written for.
  RestartInfo read_restart(const std::string& path)
  {
    const cfe::io::RestartCheckpointInfo<Scalar> file_info = cfe::io::read_restart_checkpoint<Scalar, N, cfe::AoSLayout>(
        path, {grid1, grid2}, {state_blocks.view(kUpstream), state_blocks.view(kDownstream)});
    return RestartInfo{file_info.step, file_info.time};
  }

  double total_mass() const
  {
    double mass = 0.0;
    auto accumulate = [&](ConstView state, const cfe::CartesianGrid<Scalar>& grid) {
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
    accumulate(state_blocks.view(kUpstream), grid1);
    accumulate(state_blocks.view(kDownstream), grid2);
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
    write_vtk_block(path_prefix + "_block1.vtk", grid1, state_blocks.view(kUpstream), scalar_name, compute);
    write_vtk_block(path_prefix + "_block2.vtk", grid2, state_blocks.view(kDownstream), scalar_name, compute);
  }

 private:
  template <class Compute>
  static void write_vtk_block(const std::string& path, const cfe::CartesianGrid<Scalar>& grid, ConstView state,
                               const char* scalar_name, Compute compute)
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
