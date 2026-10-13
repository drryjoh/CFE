// Woodward & Colella (1984), JCP 54, 115-173, Sec. IV.b: the Mach 3
// forward-facing-step wind tunnel -- CFE's primary deliverable for
// task 0010 (tasks/0010-phase4-forward-step-multiblock.md). Same
// EulerField<Scalar,2> + MusclMinmodReconstruction + SSP-RK2 stack
// every other Euler case in this repo uses, assembled from TWO
// `CartesianGrid` blocks joined by the new
// `BlockInterfaceBoundary` (grid/boundary/block_interface_boundary.hpp)
// -- see tests/verification/forward_step_setup.hpp for the shared
// two-block driver this tutorial just runs and writes output for
// (including the step/snapshot/progress-report loop itself,
// `cfe_verification::run_and_report` -- shared with the 3D sibling
// tutorial, see that function's own doc comment), and
// docs/adr/0011-forward-step-block-interface.md for why two blocks
// rather than one masked grid.
//
// Task 0011 (docs/adr/0012-hllc-numerical-flux.md) added a SECOND run
// of the whole sweep with `HllcFlux` instead of `RusanovFlux` (the
// `ForwardStepBlocks<Scalar,Dim,NumericalFlux>` generalization that task
// added), specifically to check for the Kelvin-Helmholtz "mild waviness"
// Woodward & Colella's own paper describes on the trailing contact
// discontinuity -- MUSCL+Rusanov showed none of it at any resolution
// tested (see this tutorial's README.md's "Looking for Kelvin-Helmholtz
// rollup" section for the honest, already-reported negative result this
// re-run is checking against).
//
// Runs the paper's three reference resolutions (cells-per-unit-height
// 20/40/80, i.e. "20x60"/"40x120"/"80x240" for the full 1x3 domain) to
// t=4 (unsteady until roughly this time, per the paper and task spec),
// writing periodic VTK density/pressure snapshots for qualitative
// inspection: bow shock off the step corner, a Mach-stem reflection off
// the top wall that moves upstream over time, a trailing contact
// discontinuity, and the step corner's own known singular-point
// artifact (documented, not fixed -- see this tutorial's README.md).
//
// Output: data/res<N>_<flux>/summary.csv (wall-clock/step-count per
// resolution/flux) plus data/res<N>_<flux>/frame_XXXX_block{1,2}.vtk
// (ASCII legacy VTK, cfe/io/vtk_writer.hpp) at a fixed time cadence --
// plot_results.py turns these into the actual figures under figures/.
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "cfe/core/types.hpp"
#include "cfe/numerics/numerical_flux/hllc.hpp"
#include "forward_step_setup.hpp"

namespace fs = std::filesystem;

namespace {

using Scalar = cfe::scalar;
constexpr Scalar kTFinal = Scalar(4.0);
constexpr Scalar kCfl = Scalar(0.4);
constexpr Scalar kSnapshotDt = Scalar(0.1);

template <class NumericalFlux>
void run_resolution(std::size_t cells_per_unit_height, const std::string& flux_name)
{
  const std::string tag = "res" + std::to_string(cells_per_unit_height) + "_" + flux_name;
  const fs::path dir = fs::path("data") / tag;
  fs::create_directories(dir);

  std::printf("=== forward step 2D: %s (cells-per-unit-height, %s) ===\n",
              std::to_string(cells_per_unit_height).c_str(), flux_name.c_str());

  cfe_verification::ForwardStepBlocks<Scalar, 2, NumericalFlux> blocks(cells_per_unit_height);

  auto density_of = [&](const cfe::State<Scalar, 4>& q) { return static_cast<Scalar>(blocks.field.density(q)); };
  auto pressure_of = [&](const cfe::State<Scalar, 4>& q) { return static_cast<Scalar>(blocks.field.pressure(q)); };

  auto write_frame = [&](std::size_t frame) {
    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "frame_%04zu", frame);
    blocks.write_vtk_frame((dir / (std::string(prefix) + "_density")).string(), "density", density_of);
    blocks.write_vtk_frame((dir / (std::string(prefix) + "_pressure")).string(), "pressure", pressure_of);
  };

  const auto report =
      cfe_verification::run_and_report(blocks, kTFinal, kCfl, kSnapshotDt, write_frame, tag.c_str());

  std::printf("  %s: done. steps=%zu wall=%.2fs frames=%zu final_mass=%.6f\n", tag.c_str(), report.steps,
              report.wall_clock_s, report.frames, report.final_mass);

  std::ofstream summary(dir / "summary.csv");
  summary << "cells_per_unit_height,steps,wall_clock_s,frames,final_mass\n";
  summary << cells_per_unit_height << "," << report.steps << "," << report.wall_clock_s << "," << report.frames
          << "," << report.final_mass << "\n";
}

}  // namespace

int main()
{
  // 20/40/80 are the paper's own reference grids (see this file's header
  // comment); 96/144 extend the sweep further (144 = 96*1.5) to give a
  // finer convergence illustration than the paper's own three points
  // alone provide; 216/288 (continuing the 1.5x pattern) push further
  // still, specifically to look for Kelvin-Helmholtz rollup on the
  // trailing contact discontinuity that the README's "mild waviness,
  // not dramatic rollup" note says isn't yet visible at lower
  // resolutions with Rusanov.
  const std::vector<std::size_t> resolutions = {std::size_t(20),  std::size_t(40),  std::size_t(80),
                                                 std::size_t(96),  std::size_t(144), std::size_t(216),
                                                 std::size_t(288)};
  for (std::size_t res : resolutions) run_resolution<cfe::RusanovFlux>(res, "rusanov");
  for (std::size_t res : resolutions) run_resolution<cfe::HllcFlux>(res, "hllc");
  return 0;
}
