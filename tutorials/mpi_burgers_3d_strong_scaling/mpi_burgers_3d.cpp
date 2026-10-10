// 3D MPI Burgers strong-scaling tutorial (task spec 0007, follow-up to
// task 0006's MPI_Allreduce(MAX) CFL fix): runs the exact same
// FvmSolver + BurgersField<Scalar,3> + MusclMinmodReconstruction +
// RusanovFlux + SSP-RK2 machinery `tutorials/burgers_3d_visualization/`
// already uses, decomposed across however many MPI ranks this binary
// is launched with, via CartesianPartition + MpiHaloBoundary (task
// 0005's full 3D block decomposition) and task 0006's collective CFL
// fix (`backend/mpi/mpi_reduce.hpp`'s `allreduce_max` -- without it,
// this tutorial's own spatially-varying Gaussian-bump IC would hit
// exactly the hazard that fix closes: see this file's own `dt`
// computation below).
//
// This is the direct sibling of `tutorials/
// mpi_scalar_advection_3d_strong_scaling/` -- same strong-scaling
// methodology (fixed global problem, 1/2/4/8 ranks, wall-clock timing,
// per-rank VTK output), same `(px,py,pz)` auto-factoring helper -- but
// for the equation that specifically needed task 0006's fix, and whose
// shock-formation behavior is the more scientifically interesting
// thing to actually watch form and propagate across a rank boundary in
// ParaView (`tests/mpi/test_mpi_burgers_steepening.cpp` already proved
// this numerically; this tutorial is where you can SEE it).
//
// A Gaussian bump (same `background + amplitude*exp(-r^2/(2*sigma^2))`
// IC and constants `tutorials/burgers_3d_visualization/` uses) steepens
// into a shock on its leading faces and spreads into a rarefaction fan
// on its trailing faces -- unlike the sibling tutorial's linear
// advection, which only translates the bump unchanged.
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
#include "cfe/backend/mpi/mpi_reduce.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/burgers/field.hpp"
#include "cfe/grid/boundary/mpi_halo_boundary.hpp"
#include "cfe/grid/partition/cartesian_partition.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/io/vtk_writer.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"

namespace {

using Scalar = double;

constexpr std::size_t kGlobalN = 128;  // 128^3 global cells, fixed for every rank count
constexpr Scalar kCfl = 0.3;           // matches tutorials/burgers_3d_visualization/'s own choice
// Matches tutorials/burgers_3d_visualization/'s final physical time
// (500 steps at its own N=64/CFL=0.3 reach t~=0.521) -- same amount of
// physical steepening/rarefaction to look at, just at higher
// resolution and split across ranks here.
constexpr Scalar kFinalTime = 0.521;
constexpr int kOutputFrames = 20;
constexpr Scalar kBumpCenter = 0.5;
constexpr Scalar kBumpSigma = 0.12;
constexpr Scalar kBackground = 1.0;
constexpr Scalar kBumpAmplitude = 0.5;

// Most "cube-like" factorization of `size` into `px*py*pz == size` --
// identical helper to the sibling scalar-advection tutorial's; kept
// duplicated rather than shared, consistent with every other pair of
// this repo's tutorials being independently self-contained.
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
  // True physical origin, same as the sibling tutorial -- this is for
  // visualization, never compared bit-for-bit against anything, unlike
  // tests/mpi/test_mpi_burgers_steepening.cpp's correctness oracle.
  grid.origin_x = static_cast<Scalar>(partition.global_offset_x) * grid.dx;
  grid.origin_y = static_cast<Scalar>(partition.global_offset_y) * grid.dy;
  grid.origin_z = static_cast<Scalar>(partition.global_offset_z) * grid.dz;

  cfe::Field<Scalar, 1> state(grid.n_cells_total());
  cfe::Field<Scalar, 1> stage1(grid.n_cells_total());
  cfe::Field<Scalar, 1> scratch(grid.n_cells_total());

  // Fill this rank's own block with its slice of the Gaussian bump,
  // tracking the largest |value| THIS rank sees -- not yet the global
  // maximum the CFL fix below needs.
  Scalar local_max_abs_u = 0.0;
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
        local_max_abs_u = std::max(local_max_abs_u, std::abs(value));
      }
    }
  }
  // Task 0006's fix: the bump's peak sits in exactly ONE rank's local
  // block (or at most a few, at larger rank counts), so most ranks'
  // own `local_max_abs_u` is noticeably smaller than the true global
  // one -- sizing `dt` from the local value alone would desynchronize
  // ranks' step counts and hang the run (see
  // docs/adr/0009-mpi-domain-decomposition.md's amendment for the full
  // story). One collective call closes that gap.
  const Scalar max_abs_u0 = cfe::backend::mpi::allreduce_max(local_max_abs_u);

  cfe::BurgersField<Scalar, 3> field{};
  cfe::MpiHaloBoundary<Scalar, 1> boundary_x{MPI_COMM_WORLD, partition.x_left_rank, partition.x_right_rank};
  cfe::MpiHaloBoundary<Scalar, 1> boundary_y{MPI_COMM_WORLD, partition.y_left_rank, partition.y_right_rank};
  cfe::MpiHaloBoundary<Scalar, 1> boundary_z{MPI_COMM_WORLD, partition.z_left_rank, partition.z_right_rank};
  cfe::FvmSolver<Scalar, cfe::AoSLayout, cfe::BurgersField<Scalar, 3>, cfe::MpiHaloBoundary<Scalar, 1>,
                 cfe::MpiHaloBoundary<Scalar, 1>, cfe::MpiHaloBoundary<Scalar, 1>,
                 cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux>
      solver{grid, field, boundary_x, boundary_y, boundary_z};
  auto residual = [&](cfe::FieldView<Scalar, 1> in, cfe::FieldView<Scalar, 1> out) {
    solver.residual(in, out);
  };

  // Multi-axis CFL (same reasoning as the single-rank 3D Burgers
  // tutorial/benchmark's own `dt`), now sized from the Allreduce'd
  // GLOBAL maximum instead of a per-rank local one.
  const Scalar dt = kCfl * grid.dx / (Scalar(3.0) * max_abs_u0);
  const int n_steps = static_cast<int>(std::ceil(kFinalTime / dt));

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
  // starts together -- same pattern the sibling scalar-advection
  // tutorial and task 0004's bench_mpi_halo_exchange both use.
  MPI_Barrier(MPI_COMM_WORLD);
  const auto t0 = std::chrono::steady_clock::now();

  const int output_every = std::max(1, n_steps / kOutputFrames);
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
