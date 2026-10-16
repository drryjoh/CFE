// The core correctness property checkpoint/restart exists to provide
// (task tasks/0016-checkpoint-restart.md, non-negotiable per that task's
// review requirements): a continuous run of N adaptive-CFL steps on the
// real two-block forward-facing-step case must produce the IDENTICAL
// final state (bit-for-bit -- this is a deterministic explicit scheme,
// no randomness anywhere in the step) to: run (N-1000) steps [here, a
// toy-scale (N-20) for a fast test], write a restart checkpoint, read it
// back into a FRESH `ForwardStepBlocks`, then run the remaining steps.
//
// This is deliberately run against the REAL `ForwardStepBlocks` two-
// block/`BlockInterfaceBoundary`/`BlockSet` assembly
// (tests/verification/forward_step_setup.hpp), not a simplified stand-in
// -- the whole point is to verify the restart file's ghost-cell-exclusion
// assumption holds for the actual case this feature was built for,
// including the cross-block interface ghost read (block 1's ghost read
// from block 2's real cells and vice versa), not just a single-block toy
// problem (that is what test_restart_file.cpp already covers, at the
// raw file-format level).
#include <array>
#include <cstdio>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "cfe/core/types.hpp"
#include "cfe/numerics/numerical_flux/hllc.hpp"
#include "forward_step_setup.hpp"
#include "test_framework.hpp"

namespace {

namespace fs = std::filesystem;
using Scalar = double;
constexpr Scalar kCfl = Scalar(0.4);
// Small (cells-per-unit-height=10) and few steps -- a fast regression
// guard, not a reproduction of the production res288_hllc sweep (that
// real-scale timing/speedup demonstration lives in
// tutorials/euler_2d_forward_step/README.md's restart section, run and
// measured separately since it takes tens of minutes, not test-suite
// seconds). The PROPERTY being checked (bit-identical restart-resume) is
// identical in kind at any resolution -- see this file's header comment.
constexpr std::size_t kContinuousSteps = 60;
constexpr std::size_t kCheckpointAtStep = 40;

struct ScratchFile
{
  fs::path path;
  ScratchFile() : path(fs::temp_directory_path() / "cfe_forward_step_restart_test.cferst") {}
  ~ScratchFile() { std::remove(path.string().c_str()); }
};

template <class Blocks>
void check_blocks_bit_identical(const Blocks& a, const Blocks& b)
{
  auto check_block = [](auto view_a, auto view_b, const auto& grid) {
    for (std::size_t k = 0; k < grid.nz; ++k) {
      for (std::size_t j = 0; j < grid.ny; ++j) {
        for (std::size_t i = 0; i < grid.nx; ++i) {
          const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
          for (std::size_t c = 0; c < Blocks::N; ++c) {
            CFE_CHECK(view_a(cell, c) == view_b(cell, c));
          }
        }
      }
    }
  };
  check_block(a.state_blocks.view(a.kUpstream), b.state_blocks.view(b.kUpstream), a.grid1);
  check_block(a.state_blocks.view(a.kDownstream), b.state_blocks.view(b.kDownstream), a.grid2);
}

}  // namespace

CFE_TEST(test_forward_step_restart_resume_matches_continuous_run_bit_for_bit)
{
  using Blocks = cfe_verification::ForwardStepBlocks<Scalar, 2>;
  const ScratchFile scratch;

  // --- Continuous run: kContinuousSteps straight through. ---
  Blocks continuous(10);
  continuous.run_n_steps(kContinuousSteps, kCfl);

  // Sanity: the flow must have actually evolved away from the uniform
  // inflow IC somewhere (otherwise a trivial/degenerate case -- e.g. a
  // bug that left every cell at the IC -- would pass this test for the
  // wrong reason). Downstream block density should differ from the
  // inflow density once the step/shock has had any time to act.
  {
    const auto view = continuous.state_blocks.view(continuous.kDownstream);
    const auto& grid = continuous.grid2;
    bool any_different = false;
    for (std::size_t j = 0; j < grid.ny && !any_different; ++j) {
      for (std::size_t i = 0; i < grid.nx && !any_different; ++i) {
        const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz);
        if (view(cell, 0) != Blocks::kInflowDensity) any_different = true;
      }
    }
    CFE_CHECK(any_different);
  }

  // --- Restart run: kCheckpointAtStep steps, write checkpoint, reload
  // into a FRESH object, run the remaining steps. ---
  Blocks part1(10);
  const Scalar t_at_checkpoint = part1.run_n_steps(kCheckpointAtStep, kCfl);
  part1.write_restart(scratch.path.string(), kCheckpointAtStep, t_at_checkpoint);

  Blocks resumed(10);

  // Direct check that ghost cells are genuinely EXCLUDED from the file --
  // snapshot a handful of `resumed`'s own ghost-cell values from right
  // after construction (the uniform IC, `fill_uniform` fills the WHOLE
  // padded array including ghosts -- see this file's own constructor
  // comment in forward_step_setup.hpp; no boundary condition has run
  // yet), then confirm `read_restart` leaves them EXACTLY as they were --
  // not merely "equal to some plausible value," but bit-identical to
  // whatever was there immediately beforehand, which is the direct,
  // traced proof (not an inference from the end-to-end bit-identical
  // check below) that the restart file carries no ghost-cell data at
  // all. Four distinct locations: the fixed-inflow low-x ghost, the
  // slip-wall low-y ghost, the cross-block-interface high-x ghost, and a
  // riser-face-fallback ghost -- every ghost-fill CASE this case's own
  // boundary conditions have, not just one.
  const auto& grid1 = resumed.grid1;
  const std::vector<std::size_t> ghost_cells_to_check = {
      grid1.flat_index(grid1.ngx - 1, grid1.ngy + grid1.ny / 2, grid1.ngz),               // low-x (inflow)
      grid1.flat_index(grid1.ngx + grid1.nx / 2, grid1.ngy - 1, grid1.ngz),               // low-y (slip wall)
      grid1.flat_index(grid1.ngx + grid1.nx, grid1.ngy + grid1.ny - 1, grid1.ngz),        // high-x (interface)
      grid1.flat_index(grid1.ngx + grid1.nx, grid1.ngy, grid1.ngz)};                      // high-x (riser fallback)
  std::vector<std::array<Scalar, Blocks::N>> ghost_values_before(ghost_cells_to_check.size());
  {
    const auto view = resumed.state_blocks.view(resumed.kUpstream);
    for (std::size_t g = 0; g < ghost_cells_to_check.size(); ++g) {
      for (std::size_t c = 0; c < Blocks::N; ++c) ghost_values_before[g][c] = view(ghost_cells_to_check[g], c);
    }
  }

  const auto restart_info = resumed.read_restart(scratch.path.string());
  CFE_CHECK(restart_info.step == kCheckpointAtStep);
  CFE_CHECK(restart_info.time == t_at_checkpoint);

  {
    const auto view = resumed.state_blocks.view(resumed.kUpstream);
    for (std::size_t g = 0; g < ghost_cells_to_check.size(); ++g) {
      for (std::size_t c = 0; c < Blocks::N; ++c) {
        CFE_CHECK(view(ghost_cells_to_check[g], c) == ghost_values_before[g][c]);
      }
    }
  }

  const std::size_t remaining_steps = kContinuousSteps - kCheckpointAtStep;
  resumed.run_n_steps(remaining_steps, kCfl);

  // --- The actual property: bit-identical final state. ---
  check_blocks_bit_identical(continuous, resumed);
  CFE_CHECK(continuous.total_mass() == resumed.total_mass());
}

// Same property, `HllcFlux` instead of the default `RusanovFlux` --
// `res288_hllc` (the production case this feature was actually built
// for, see `tutorials/euler_2d_forward_step/`) uses HLLC specifically,
// and HLLC's own wave-speed estimate (a short Newton-Raphson iteration,
// `numerics/numerical_flux/hllc.hpp`) is different internal machinery
// from Rusanov's -- this closes the gap of only ever restart-testing
// the flux the production case does NOT use (Violet's review finding
// for this task).
CFE_TEST(test_forward_step_restart_resume_matches_continuous_run_bit_for_bit_hllc)
{
  using Blocks = cfe_verification::ForwardStepBlocks<Scalar, 2, cfe::HllcFlux>;
  const fs::path path = fs::temp_directory_path() / "cfe_forward_step_restart_test_hllc.cferst";
  struct Cleanup
  {
    fs::path p;
    ~Cleanup() { std::remove(p.string().c_str()); }
  } cleanup{path};

  Blocks continuous(10);
  continuous.run_n_steps(kContinuousSteps, kCfl);

  Blocks part1(10);
  const Scalar t_at_checkpoint = part1.run_n_steps(kCheckpointAtStep, kCfl);
  part1.write_restart(path.string(), kCheckpointAtStep, t_at_checkpoint);

  Blocks resumed(10);
  const auto restart_info = resumed.read_restart(path.string());
  CFE_CHECK(restart_info.step == kCheckpointAtStep);
  resumed.run_n_steps(kContinuousSteps - kCheckpointAtStep, kCfl);

  check_blocks_bit_identical(continuous, resumed);
  CFE_CHECK(continuous.total_mass() == resumed.total_mass());
}

CFE_TEST(test_forward_step_restart_resume_matches_continuous_run_bit_for_bit_3d)
{
  // Same property, Dim=3 (a thin periodic-Z extrusion, same shape
  // tests/unit/test_forward_step_sanity.cpp's own 3D case uses) --
  // confirms restart also works through the Z-periodic boundary and a
  // genuinely 3D `BlockSet` layout, not just the 2D case above.
  using Blocks = cfe_verification::ForwardStepBlocks<Scalar, 3>;
  const fs::path path = fs::temp_directory_path() / "cfe_forward_step_restart_test_3d.cferst";
  struct Cleanup
  {
    fs::path p;
    ~Cleanup() { std::remove(p.string().c_str()); }
  } cleanup{path};

  constexpr std::size_t kNz = 4;
  constexpr Scalar kSpanZ = Scalar(0.2);

  Blocks continuous(10, kNz, kSpanZ);
  continuous.run_n_steps(kContinuousSteps, kCfl);

  Blocks part1(10, kNz, kSpanZ);
  const Scalar t_at_checkpoint = part1.run_n_steps(kCheckpointAtStep, kCfl);
  part1.write_restart(path.string(), kCheckpointAtStep, t_at_checkpoint);

  Blocks resumed(10, kNz, kSpanZ);
  const auto restart_info = resumed.read_restart(path.string());
  CFE_CHECK(restart_info.step == kCheckpointAtStep);
  resumed.run_n_steps(kContinuousSteps - kCheckpointAtStep, kCfl);

  check_blocks_bit_identical(continuous, resumed);
  CFE_CHECK(continuous.total_mass() == resumed.total_mass());
}
