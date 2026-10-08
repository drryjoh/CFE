// Produces a ParaView-ready time series of a 3D Gaussian bump deforming
// under the inviscid Burgers equation, using the exact same FvmSolver +
// BurgersField + MusclMinmodReconstruction + RusanovFlux + SSP-RK2
// machinery verified by tests/unit/test_burgers_3d_cuda.cu and
// benchmarked by benchmarks/burgers/bench_burgers_3d_cuda.cu -- this is
// that solver's actual output, not a separate toy.
//
// Unlike tutorials/scalar_advection_3d_visualization (linear advection:
// the bump translates unchanged), Burgers' characteristic speed IS the
// local state value, so a localized bump riding on a positive
// background does not just translate -- each face steepens into a
// shock on its leading (downhill-in-state) side, since higher-state
// cells there catch up to the slower background ahead of them, while
// the trailing side spreads into a smooth rarefaction fan, since
// state there decreases away from the peak and those characteristics
// diverge. This happens independently (but visibly) along all three
// axes at once, since Burgers' flux here is the same scalar formula on
// every axis -- the bump should end up looking asymmetric: steep faces
// on the side it is "falling toward," smooth/spread faces on the other.
//
// CPU-only and deliberately small (64^3): this is for visualization, not
// a performance measurement -- see the benchmark above for scale.
//
// Precision is `cfe::scalar` (core/types.hpp, project-wide via the
// `CFE_SCALAR_TYPE` CMake cache variable, `double` by default) -- NOT a
// hardcoded `double` -- matching AGENTS.md #11.
//
// Writes vtk_output/frame_XXXX.vtk (legacy VTK STRUCTURED_POINTS, one
// per snapshot) plus a vtk_output/series.pvd manifest. Open series.pvd
// directly in ParaView to get a time slider over the whole run; the
// individual frame_*.vtk files also open standalone.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/burgers/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/io/vtk_writer.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace {

using Scalar = cfe::scalar;

constexpr std::size_t kN = 64;
constexpr Scalar kDx = Scalar(1.0) / static_cast<Scalar>(kN);
constexpr Scalar kCfl = Scalar(0.3);
constexpr int kTotalSteps = 500;
constexpr int kOutputEvery = 10;  // -> 51 frames, including the initial condition.
constexpr Scalar kBumpCenter = Scalar(0.5);
constexpr Scalar kBumpSigma = Scalar(0.12);
// A positive background plus a smaller bump on top: keeps the state
// strictly positive everywhere (no sonic-point sign change to
// complicate the story), matching the same A+B pattern
// tests/unit/test_burgers_convergence.cpp uses for its smooth IC.
constexpr Scalar kBackground = Scalar(1.0);
constexpr Scalar kBumpAmplitude = Scalar(0.5);

std::string frame_path(const std::filesystem::path& out_dir, int frame)
{
  char name[64];
  std::snprintf(name, sizeof(name), "frame_%04d.vtk", frame);
  return (out_dir / name).string();
}

}  // namespace

int main()
{
  const std::filesystem::path out_dir = "vtk_output";
  std::filesystem::create_directories(out_dir);

  cfe::CartesianGrid<Scalar> grid;
  grid.nx = kN;
  grid.ny = kN;
  grid.nz = kN;
  grid.ngx = 2;
  grid.ngy = 2;
  grid.ngz = 2;
  grid.dx = kDx;
  grid.dy = kDx;
  grid.dz = kDx;

  cfe::Field<Scalar, 1> state(grid.n_cells_total());
  cfe::Field<Scalar, 1> stage1(grid.n_cells_total());
  cfe::Field<Scalar, 1> scratch(grid.n_cells_total());

  Scalar max_abs_u0 = Scalar(0.0);
  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const Scalar x = grid.x_center(grid.ngx + i);
        const Scalar y = grid.y_center(grid.ngy + j);
        const Scalar z = grid.z_center(grid.ngz + k);
        const Scalar r2 = (x - kBumpCenter) * (x - kBumpCenter) + (y - kBumpCenter) * (y - kBumpCenter) +
                           (z - kBumpCenter) * (z - kBumpCenter);
        const Scalar value =
            kBackground + kBumpAmplitude * std::exp(-r2 / (Scalar(2.0) * kBumpSigma * kBumpSigma));
        state(grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k), 0) = value;
        max_abs_u0 = std::max(max_abs_u0, std::abs(value));
      }
    }
  }

  cfe::BurgersField<Scalar, 3> field{};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::BurgersField<Scalar, 3>, cfe::PeriodicBoundary,
                 cfe::PeriodicBoundary, cfe::PeriodicBoundary, cfe::fvm::MusclMinmodReconstruction,
                 cfe::RusanovFlux>
      solver{grid, field, cfe::PeriodicBoundary{}};
  auto residual = [&](cfe::FieldView<Scalar, 1> in, cfe::FieldView<Scalar, 1> out) {
    solver.residual(in, out);
  };

  // Multi-axis CFL (same reasoning as bench_burgers_3d_cuda.cu's dt):
  // Burgers' characteristic speed is the state itself, the same on
  // every axis, so the per-cell residual sees that speed three times
  // (once per active axis) every stage.
  const Scalar dt = kCfl * kDx / (Scalar(3.0) * max_abs_u0);

  std::ofstream pvd(out_dir / "series.pvd");
  pvd << "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\">\n<Collection>\n";

  int frame = 0;
  Scalar t = Scalar(0.0);
  auto write_frame = [&]() {
    const std::string path = frame_path(out_dir, frame);
    cfe::io::write_vtk_structured_points_cell_scalar(path, grid, state.view(), "state");
    pvd << "  <DataSet timestep=\"" << static_cast<double>(t) << "\" file=\""
        << std::filesystem::path(path).filename().string() << "\"/>\n";

    // Seeded from the first REAL cell, not storage index 0 -- that
    // would be a ghost cell (never written before the first ghost-fill,
    // left at whatever the Field's own zero-initialization gives it),
    // which would wrongly pull min_value down to 0 on the very first
    // frame.
    Scalar min_value = state(grid.flat_index(grid.ngx, grid.ngy, grid.ngz), 0);
    Scalar max_value = min_value;
    for (std::size_t k = 0; k < grid.nz; ++k) {
      for (std::size_t j = 0; j < grid.ny; ++j) {
        for (std::size_t i = 0; i < grid.nx; ++i) {
          const Scalar value = state(grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k), 0);
          min_value = std::min(min_value, value);
          max_value = std::max(max_value, value);
        }
      }
    }
    std::printf("wrote %s (t=%.4f, state range [%.4f, %.4f])\n", path.c_str(), static_cast<double>(t),
                static_cast<double>(min_value), static_cast<double>(max_value));
    ++frame;
  };

  write_frame();
  for (int step = 1; step <= kTotalSteps; ++step) {
    cfe::ssp_rk2_step<Scalar>(state.view(), stage1.view(), scratch.view(), dt, residual,
                               solver.active_cell_count(), solver.active_cell_index_map());
    t += dt;
    if (step % kOutputEvery == 0) write_frame();
  }

  pvd << "</Collection>\n</VTKFile>\n";
  std::printf("\n%d frames written to %s/. Open series.pvd in ParaView for a time slider.\n", frame,
              out_dir.string().c_str());
  return 0;
}
