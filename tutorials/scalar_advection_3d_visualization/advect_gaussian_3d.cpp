// Produces a ParaView-ready time series of a 3D Gaussian bump advecting
// (and periodically wrapping) through a unit cube, using the exact same
// FvmSolver + ScalarAdvectionField + SSP-RK2 machinery verified by
// tests/unit/test_scalar_advection_3d_cuda.cu and benchmarked by
// benchmarks/scalar_advection/bench_scalar_advection_3d_cuda.cu -- this
// is that solver's actual output, not a separate toy.
//
// CPU-only and deliberately small (64^3): this is for visualization, not
// a performance measurement -- see the benchmarks above for scale.
//
// Writes vtk_output/frame_XXXX.vtk (legacy VTK STRUCTURED_POINTS, one
// per snapshot) plus a vtk_output/series.pvd manifest. Open series.pvd
// directly in ParaView to get a time slider over the whole run; the
// individual frame_*.vtk files also open standalone.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "cfe/field/field.hpp"
#include "cfe/fields/scalar_advection/field.hpp"
#include "cfe/grid/boundary/boundary_condition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/io/vtk_writer.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace {

constexpr std::size_t kN = 64;
constexpr double kDx = 1.0 / static_cast<double>(kN);
constexpr double kUx = 0.5;
constexpr double kUy = 0.3;
constexpr double kUz = 0.2;
constexpr double kDt = 0.3 * kDx / (kUx + kUy + kUz);
constexpr int kTotalSteps = 640;
constexpr int kOutputEvery = 16;  // -> 41 frames, including the initial condition.
constexpr double kBumpCenter = 0.2;
constexpr double kBumpSigma = 0.08;

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

  cfe::CartesianGrid<double> grid;
  grid.nx = kN;
  grid.ny = kN;
  grid.nz = kN;
  grid.ngx = 2;
  grid.ngy = 2;
  grid.ngz = 2;
  grid.dx = kDx;
  grid.dy = kDx;
  grid.dz = kDx;

  cfe::Field<double, 1> q(grid.n_cells_total());
  cfe::Field<double, 1> stage1(grid.n_cells_total());
  cfe::Field<double, 1> scratch(grid.n_cells_total());

  for (std::size_t k = 0; k < grid.nz; ++k) {
    for (std::size_t j = 0; j < grid.ny; ++j) {
      for (std::size_t i = 0; i < grid.nx; ++i) {
        const double x = grid.x_center(grid.ngx + i);
        const double y = grid.y_center(grid.ngy + j);
        const double z = grid.z_center(grid.ngz + k);
        const double r2 = (x - kBumpCenter) * (x - kBumpCenter) + (y - kBumpCenter) * (y - kBumpCenter) +
                           (z - kBumpCenter) * (z - kBumpCenter);
        q(grid.flat_index(grid.ngx + i, grid.ngy + j, grid.ngz + k), 0) =
            std::exp(-r2 / (2.0 * kBumpSigma * kBumpSigma));
      }
    }
  }

  cfe::Vector<double, 3> velocity;
  velocity[0] = kUx;
  velocity[1] = kUy;
  velocity[2] = kUz;
  cfe::ScalarAdvectionField<double, 3> field{velocity};
  cfe::FvmSolver<double, cfe::AoSLayout, cfe::ScalarAdvectionField<double, 3>, cfe::PeriodicBoundary>
      solver{grid, field, cfe::PeriodicBoundary{}};
  auto residual = [&](cfe::FieldView<double, 1> in, cfe::FieldView<double, 1> out) {
    solver.residual(in, out);
  };

  std::ofstream pvd(out_dir / "series.pvd");
  pvd << "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\">\n<Collection>\n";

  int frame = 0;
  double t = 0.0;
  auto write_frame = [&]() {
    const std::string path = frame_path(out_dir, frame);
    cfe::io::write_vtk_structured_points_cell_scalar(path, grid, q.view(), "q");
    pvd << "  <DataSet timestep=\"" << t << "\" file=\"" << std::filesystem::path(path).filename().string()
        << "\"/>\n";
    std::printf("wrote %s (t=%.4f)\n", path.c_str(), t);
    ++frame;
  };

  write_frame();
  for (int step = 1; step <= kTotalSteps; ++step) {
    cfe::ssp_rk2_step<double>(q.view(), stage1.view(), scratch.view(), kDt, residual);
    t += kDt;
    if (step % kOutputEvery == 0) write_frame();
  }

  pvd << "</Collection>\n</VTKFile>\n";
  std::printf("\n%d frames written to %s/. Open series.pvd in ParaView for a time slider.\n", frame,
              out_dir.string().c_str());
  return 0;
}
