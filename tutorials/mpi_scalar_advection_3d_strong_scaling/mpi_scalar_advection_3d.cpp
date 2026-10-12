// 3D MPI strong-scaling tutorial (task spec 0005): runs the exact same
// FvmSolver + ScalarAdvectionField<Scalar,3> + CentralDifferenceReconstruction
// + UpwindFlux + SSP-RK2 machinery every other scalar-advection
// tutorial/test in this repo uses, decomposed across however many MPI
// ranks this binary is launched with, via CartesianPartition +
// MpiHaloBoundary (task 0005's full 3D block decomposition).
//
// Strong scaling means: the GLOBAL problem (grid resolution, initial
// condition, final time) is held IDENTICAL across every rank count --
// only how it is split up changes. Run this binary once per rank count
// (1, 2, 4, 8 -- see README.md) and compare the reported wall-clock
// time: more ranks splitting the same fixed amount of work should
// finish faster. This is the full-solver complement to task 0004's
// `bench_mpi_halo_exchange`, which only timed the halo exchange itself
// in isolation.
//
// (px,py,pz) -- how many ranks sit along each axis -- is chosen
// automatically from the rank count via a small local search for the
// most "cube-like" factorization (minimizing the spread between the
// largest and smallest of the three factors). This is tutorial-local,
// not a new core type: a real production run would let the launcher/
// scheduler decide this, not the library (see `grid/partition/
// cartesian_partition.hpp`'s own header comment on why `px*py*pz` is a
// caller-supplied parameter, not auto-factored inside that type).
//
// A Gaussian bump translating diagonally (velocity=(1,1,1)), same
// convention `tutorials/scalar_advection_3d_visualization/` already
// uses for its own single-rank version -- unlike that tutorial, each
// rank here writes only ITS OWN local block as a separate VTK file
// (`frame_NNNN_rankNN.vtk`), with that rank's true physical origin (the
// one thing this tutorial does NOT need to avoid, unlike
// `tests/mpi/test_mpi_halo_exchange*.cpp`'s bit-identity requirement --
// nothing here is compared bit-for-bit against anything) -- so every
// rank's piece tiles together correctly in physical space when all
// loaded together in ParaView, with no change needed to the existing,
// unmodified `cfe::io::write_vtk_structured_points_cell_scalar`.
//
// CPU-only; requires CFE_ENABLE_MPI.
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#include "cfe/backend/mpi/mpi_environment.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/scalar_advection/field.hpp"
#include "cfe/grid/boundary/mpi_halo_boundary.hpp"
#include "cfe/grid/partition/cartesian_partition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/io/vtk_writer.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace {

using Scalar = double;

constexpr std::size_t kGlobalN = 128;  // 128^3 global cells, fixed for every rank count
constexpr Scalar kVelocity = 1.0;      // same speed on all three axes
constexpr Scalar kCfl = 0.4;
constexpr Scalar kFinalTime = 0.3;
constexpr int kOutputFrames = 20;  // approximate -- see output-stride comment below
constexpr Scalar kBumpCenter = 0.5;
constexpr Scalar kBumpSigma = 0.08;
constexpr Scalar kBackground = 1.0;
constexpr Scalar kBumpAmplitude = 0.5;

// Most "cube-like" factorization of `size` into `px*py*pz == size`:
// searches every divisor triple and keeps the one minimizing the spread
// between the largest and smallest factor (a tie-break toward a more
// balanced, lower-surface-area-per-rank process grid -- the same shape
// choice a real launcher would make, done here by brute force since
// `size` is always small for a tutorial).
void factor_dims(int size, int& px, int& py, int& pz)
{
  px = 1;
  py = 1;
  pz = size;
  int best_spread = size - 1;
  for (int a = 1; a <= size; ++a) {
    if (size % a != 0) continue;
    const int rem = size / a;
    for (int b = 1; b <= rem; ++b) {
      if (rem % b != 0) continue;
      const int c = rem / b;
      const int spread = std::max({a, b, c}) - std::min({a, b, c});
      if (spread < best_spread) {
        best_spread = spread;
        px = a;
        py = b;
        pz = c;
      }
    }
  }
}

void compute_steps(Scalar dx, int& n_steps, Scalar& dt)
{
  const Scalar dt_target = kCfl / (Scalar(3.0) * kVelocity / dx);
  n_steps = static_cast<int>(std::ceil(kFinalTime / dt_target));
  dt = kFinalTime / static_cast<Scalar>(n_steps);
}

std::string frame_path(const std::filesystem::path& out_dir, int frame, int rank)
{
  char name[96];
  std::snprintf(name, sizeof(name), "frame_%04d_rank%02d.vtk", frame, rank);
  return (out_dir / name).string();
}

}  // namespace

int main(int argc, char** argv)
{
  cfe::backend::mpi::Environment env(&argc, &argv);
  const int rank = cfe::backend::mpi::rank();
  const int size = cfe::backend::mpi::size();

  int px, py, pz;
  factor_dims(size, px, py, pz);

  const auto partition =
      cfe::make_cartesian_partition(kGlobalN, kGlobalN, kGlobalN, px, py, pz, rank, size, true, true, true);

  cfe::CartesianGrid<Scalar> grid;
  grid.nx = partition.local_nx;
  grid.ny = partition.local_ny;
  grid.nz = partition.local_nz;
  grid.ngx = grid.ngy = grid.ngz = 2;
  grid.dx = grid.dy = grid.dz = Scalar(1.0) / static_cast<Scalar>(kGlobalN);
  // Unlike the correctness tests, this tutorial WANTS each rank's true
  // physical origin -- it is never compared bit-for-bit against
  // anything, and the VTK writer needs it to place this rank's piece
  // correctly in world space.
  grid.origin_x = static_cast<Scalar>(partition.global_offset_x) * grid.dx;
  grid.origin_y = static_cast<Scalar>(partition.global_offset_y) * grid.dy;
  grid.origin_z = static_cast<Scalar>(partition.global_offset_z) * grid.dz;

  cfe::Field<Scalar, 1> state(grid.n_cells_total());
  cfe::Field<Scalar, 1> stage1(grid.n_cells_total());
  cfe::Field<Scalar, 1> scratch(grid.n_cells_total());

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
      }
    }
  }

  cfe::ScalarAdvectionField<Scalar, 3> field{cfe::Vector<Scalar, 3>(kVelocity)};
  cfe::MpiHaloBoundary<Scalar, 1> boundary_x{MPI_COMM_WORLD, partition.x_left_rank, partition.x_right_rank};
  cfe::MpiHaloBoundary<Scalar, 1> boundary_y{MPI_COMM_WORLD, partition.y_left_rank, partition.y_right_rank};
  cfe::MpiHaloBoundary<Scalar, 1> boundary_z{MPI_COMM_WORLD, partition.z_left_rank, partition.z_right_rank};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::ScalarAdvectionField<Scalar, 3>, cfe::MpiHaloBoundary<Scalar, 1>>
      solver{grid, field, boundary_x, boundary_y, boundary_z};
  auto residual = [&](cfe::FieldView<Scalar, 1> in, cfe::FieldView<Scalar, 1> out) {
    solver.residual(in, out);
  };

  int n_steps;
  Scalar dt;
  compute_steps(grid.dx, n_steps, dt);
  const int output_every = std::max(1, n_steps / kOutputFrames);

  const std::filesystem::path out_dir = "vtk_output";
  std::filesystem::create_directories(out_dir);

  int frame = 0;
  auto write_frame = [&]() {
    cfe::io::write_vtk_structured_points_cell_scalar(frame_path(out_dir, frame, rank), grid, state.view(),
                                                       "state");
    ++frame;
  };

  write_frame();  // initial condition, every rank's own piece

  // Strong-scaling measurement: the whole run (compute + periodic I/O)
  // is timed as one block, synchronized with a barrier so every rank
  // starts together -- the same `MPI_Barrier` + `std::chrono` pattern
  // task 0004's `bench_mpi_halo_exchange` uses. Each rank's VTK writes
  // are its own local piece only, so total I/O volume across all ranks
  // stays fixed regardless of rank count (just divided among more,
  // smaller files) -- the same "fixed total work" property strong
  // scaling requires of the compute itself.
  MPI_Barrier(MPI_COMM_WORLD);
  const auto t0 = std::chrono::steady_clock::now();

  for (int step = 1; step <= n_steps; ++step) {
    cfe::ssp_rk2_step<Scalar>(state.view(), stage1.view(), scratch.view(), dt, residual,
                               solver.active_cell_count(), solver.active_cell_index_map());
    if (step % output_every == 0 || step == n_steps) write_frame();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const auto t1 = std::chrono::steady_clock::now();
  const double wall_clock_s = std::chrono::duration<double>(t1 - t0).count();

  if (rank == 0) {
    std::printf("ranks,px,py,pz,global_n,n_steps,wall_clock_s\n");
    std::printf("%d,%d,%d,%d,%zu,%d,%.6f\n", size, px, py, pz, kGlobalN, n_steps, wall_clock_s);
    std::printf("\n%d frames per rank written to %s/. Open all ranks' series in ParaView (File > Open,\n",
                frame, out_dir.string().c_str());
    std::printf("select every frame_*_rank*.vtk for one timestep together) to see the whole domain.\n");
  }

  return 0;
}
