// 3D CPU extrusion of the forward-facing-step benchmark (task 0010's
// "see it work" deliverable -- tasks/0010-phase4-forward-step-
// multiblock.md): a simple spanwise extrusion of the same 2D geometry
// (tutorials/euler_2d_forward_step/), periodic on the new Z faces, run
// on this machine's CPU backend (Apple Silicon, no CUDA available).
// Shares its step/snapshot/progress loop
// (`cfe_verification::run_and_report`, tests/verification/
// forward_step_setup.hpp) with the 2D sibling tutorial.
//
// This is explicitly NOT a substitute for this project's standing
// separate requirement that every PR adding a Field/scheme include a
// 3D CUDA correctness test + at-scale benchmark on real CUDA hardware
// (see e.g. docs/performance/0007-phase3-euler-cuda-results.md for how
// task 0009 closed that gap on PSC Bridges-2) -- that CUDA 3D work
// remains out of scope here and is called out again in this
// tutorial's README.md and in agent_history.md's completion entry for
// this task.
//
// Because the IC and every boundary condition are uniform in Z and
// `PeriodicBoundary` never introduces Z variation from a uniform
// state, Z-momentum should stay exactly zero throughout and every Z
// slice should evolve identically to a 2D solve -- the same argument
// tests/unit/test_euler_3d_sanity.cpp already makes and verifies for
// the Sod tube; this tutorial is the forward-step's own "does the 3D
// extrusion actually stay 2D" visual/quantitative confirmation (see
// its printed Z-momentum diagnostic below), at one representative
// resolution (cells-per-unit-height=40, comparable to the paper's own
// "40x120" convergence-illustration grid) rather than all three (a
// full 3x3D parameter sweep adds compute cost for no additional
// evidence beyond what the 2D sweep already shows).
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "cfe/core/types.hpp"
#include "forward_step_setup.hpp"

namespace fs = std::filesystem;

namespace {

using Scalar = cfe::scalar;
constexpr std::size_t kCellsPerUnitHeight = 40;
constexpr std::size_t kNzCells = 8;
constexpr Scalar kSpanZ = Scalar(0.2);
constexpr Scalar kTFinal = Scalar(4.0);
constexpr Scalar kCfl = Scalar(0.4);
constexpr Scalar kSnapshotDt = Scalar(0.25);

}  // namespace

int main()
{
  const fs::path dir = fs::path("data");
  fs::create_directories(dir);

  std::printf("=== forward step 3D CPU extrusion: cells-per-unit-height=%zu, nz=%zu, span_z=%.2f ===\n",
              kCellsPerUnitHeight, kNzCells, static_cast<double>(kSpanZ));

  cfe_verification::ForwardStepBlocks<Scalar, 3> blocks(kCellsPerUnitHeight, kNzCells, kSpanZ);

  auto density_of = [&](const cfe::State<Scalar, 5>& q) { return static_cast<Scalar>(blocks.field.density(q)); };

  auto write_frame = [&](std::size_t frame) {
    char prefix[64];
    std::snprintf(prefix, sizeof(prefix), "frame_%04zu", frame);
    blocks.write_vtk_frame((dir / (std::string(prefix) + "_density")).string(), "density", density_of);
  };

  const auto report = cfe_verification::run_and_report(blocks, kTFinal, kCfl, kSnapshotDt, write_frame, "", 100);

  // Z-momentum (component 3: density, mom_x, mom_y, mom_z, energy)
  // magnitude, maxed over every real cell -- the "stays exactly 2D"
  // diagnostic this file's header comment argues for. Checked at the
  // end of the run, not just at t=0.
  double max_abs_z_momentum = 0.0;
  auto scan = [&](const auto& state, const auto& grid) {
    for (std::size_t k = 0; k < grid.nz; ++k) {
      for (std::size_t j = 0; j < grid.ny; ++j) {
        for (std::size_t i = 0; i < grid.nx; ++i) {
          const std::size_t cell = grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k);
          max_abs_z_momentum = std::max(max_abs_z_momentum, std::fabs(static_cast<double>(state(cell, 3))));
        }
      }
    }
  };
  scan(blocks.state_blocks.view(blocks.kUpstream), blocks.grid1);
  scan(blocks.state_blocks.view(blocks.kDownstream), blocks.grid2);

  std::printf("done. steps=%zu wall=%.2fs frames=%zu final_mass=%.6f max|mom_z|=%.3e\n", report.steps,
              report.wall_clock_s, report.frames, report.final_mass, max_abs_z_momentum);

  std::ofstream summary(dir / "summary.csv");
  summary << "cells_per_unit_height,nz_cells,steps,wall_clock_s,frames,final_mass,max_abs_mom_z\n";
  summary << kCellsPerUnitHeight << "," << kNzCells << "," << report.steps << "," << report.wall_clock_s << ","
          << report.frames << "," << report.final_mass << "," << max_abs_z_momentum << "\n";
  return 0;
}
