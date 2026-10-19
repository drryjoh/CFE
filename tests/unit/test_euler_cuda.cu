// CUDA correctness test for the compressible Euler solver (closes the
// policy gap flagged when task 0009 was otherwise ready to commit:
// every PR adding a Field/scheme needs a GPU correctness test, not just
// CPU -- same requirement test_scalar_advection_cuda.cu/
// test_burgers_cuda.cu already satisfy for the N=1 fields). This is
// also the first CUDA coverage for `cfe::SlipWallBoundary`
// (grid/boundary/slip_wall_boundary.hpp) -- CFE's first Field-aware
// BoundaryCondition, previously only unit-tested on CPU
// (test_slip_wall_boundary.cpp). A reflecting (slip-wall) tube at BOTH
// ends, rather than the production verification test's open
// (`StaticBoundary`) ends, is the natural way to exercise it here: the
// same classic Sod (1978) initial condition, now bounded by walls
// instead of far-field states. Runs the identical setup on CPU and GPU
// (same IC, same dt, same step count) and compares cell-by-cell,
// matching VERIFICATION.md #3's cross-backend comparison guidance.
//
// IMPORTANT, found during review: an earlier version of this test ran
// only to t=0.06 (300 steps at the dt below), which a standalone
// CPU-only probe (adaptive-CFL, same IC/BCs, not committed) showed
// never reaches either wall -- the right-moving shock does not arrive
// at x=1 until t~0.28, so that version's CPU-vs-GPU agreement, while
// still a VALID cross-backend check, never actually exercised a wave
// reflecting off `SlipWallBoundary` (both walls stayed bit-identical
// to the initial condition the entire run) despite this file's own
// framing implying otherwise. `kSteps` below was corrected so the run
// genuinely reaches past the first reflection (confirmed: the same
// probe shows velocity at the wall spike to ~0.89 at t~0.28, then
// decay as the reflected wave moves back into the tube). The probe
// also measured the actual post-reflection max |u|+c over t in [0,0.8]
// at 2.21 -- consistent with (not larger than) the pre-reflection
// classic-Sod bound tests/verification/sod_shock_tube_exact.hpp's
// header comment already documents, so `kMaxSpeedEstimate` below did
// not need to change, only `kSteps`.
#include <cmath>
#include <vector>

#include "cfe/backend/cuda/cuda_backend.cuh"
#include "cfe/backend/cuda/device_field.cuh"
#include "cfe/core/types.hpp"
#include "cfe/field/field.hpp"
#include "cfe/fields/euler/field.hpp"
#include "cfe/grid/boundary/slip_wall_boundary.hpp"
#include "cfe/grid/structured/cartesian_grid.hpp"
#include "cfe/math/fixed_array.hpp"
#include "cfe/numerics/fvm/muscl_minmod.hpp"
#include "cfe/numerics/numerical_flux/rusanov.hpp"
#include "cfe/solver/explicit/fvm_solver.hpp"
#include "cfe/solver/time_integration/ssp_rk2.hpp"
#include "test_framework.hpp"

CFE_TEST(test_euler_cuda_matches_cpu_reference)
{
  using Field = cfe::EulerField<double, 1>;
  constexpr std::size_t kN = Field::n_components;  // 3: density, momentum_x, energy

  constexpr cfe::Index kNx = 400;
  constexpr double kGamma = 1.4;
  constexpr double kX0 = 0.5;
  constexpr double kDx = 1.0 / static_cast<double>(kNx);
  // Fixed, conservatively-chosen dt -- this is a CPU/GPU cross-check,
  // not a convergence/exact-solution check, so (unlike
  // sod_shock_tube_runner.hpp's adaptive-CFL scheme) a simple fixed
  // step matches every other CUDA correctness test's convention in this
  // codebase (test_scalar_advection_cuda.cu, test_burgers_cuda.cu).
  // Classic Sod max |u|+c never exceeds ~2.2 even post-shock (see
  // tests/verification/sod_shock_tube_exact.hpp's header comment for
  // the exact wave speeds); a standalone CPU-only probe (adaptive-CFL,
  // same IC/BCs, not committed) measured the actual post-reflection max
  // |u|+c over t in [0,0.8] at 2.21 -- the reflecting tube's wave
  // structure does not exceed the open-tube bound, so 5.0 keeps a
  // comfortable (>2x) margin throughout, not just before the first
  // reflection.
  constexpr double kMaxSpeedEstimate = 5.0;
  constexpr double kCfl = 0.4;
  constexpr double kDt = kCfl * kDx / kMaxSpeedEstimate;
  // 2500 steps * kDt (0.0002) = t=0.5 -- the same probe showed the
  // shock first reaches the right wall at t~0.28 (wall-region |u|
  // spikes to ~0.89, see sod_shock_tube_exact.hpp's values for the
  // pre-reflection shock speed) and has partly dissipated again by
  // t~0.42, so this run genuinely exercises a full reflection cycle at
  // the right wall, not just the inert pre-arrival state the original
  // 300-step version (t=0.06) was limited to.
  constexpr int kSteps = 2500;

  cfe::CartesianGrid<double> grid;
  grid.nx = kNx;
  grid.ngx = 2;
  grid.dx = kDx;

  Field field{};
  field.gamma = kGamma;
  const cfe::Vector<double, 1> zero_velocity{};
  const cfe::State<double, kN> left_q = field.conserved_from_primitive(1.0, zero_velocity, 1.0);
  const cfe::State<double, kN> right_q = field.conserved_from_primitive(0.125, zero_velocity, 0.1);

  std::vector<double> host_ic(grid.n_cells_total() * kN, 0.0);
  for (cfe::Index i = 0; i < grid.nx; ++i) {
    const double x = grid.x_center(grid.ngx + i);
    const cfe::State<double, kN>& q = (x < kX0) ? left_q : right_q;
    const cfe::Index cell = grid.flat_index(grid.ngx + i, 0, 0);
    for (cfe::Index c = 0; c < kN; ++c) host_ic[cell * kN + c] = q[c];
  }

  cfe::SlipWallBoundary<Field> boundary{};

  // --- CPU reference ---
  cfe::Field<double, kN> state_cpu(grid.n_cells_total());
  cfe::Field<double, kN> stage1_cpu(grid.n_cells_total());
  cfe::Field<double, kN> scratch_cpu(grid.n_cells_total());
  for (cfe::Index idx = 0; idx < host_ic.size(); ++idx) state_cpu.data()[idx] = host_ic[idx];

  cfe::FvmSolver<double, cfe::AoSLayout, Field, cfe::SlipWallBoundary<Field>, cfe::SlipWallBoundary<Field>,
                 cfe::SlipWallBoundary<Field>, cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux>
      solver_cpu{grid, field, boundary};
  // A named functor, not a local lambda: nvcc forbids a locally-defined
  // lambda as a template argument to ssp_rk2_step, since its body always
  // contains an extended __device__ lambda in a .cu file regardless of
  // which Backend a call actually selects (see SolverResidual's doc
  // comment in fvm_solver.hpp).
  cfe::SolverResidual<decltype(solver_cpu)> residual_cpu{&solver_cpu};

  // Mass/total-energy BEFORE stepping (the reflecting tube is closed at
  // both ends -- no flux can leave through a `SlipWallBoundary`, so both
  // must be conserved to near machine precision, same acceptance
  // criterion test_euler_sod_shock_tube.cpp's own conservation check
  // uses). No closed-form solution exists for this doubly-reflected
  // problem, so conservation (not an exact-value comparison) is the
  // right physical sanity check that the longer, reflection-including
  // run below is still producing sane physics, independent of the
  // CPU-vs-GPU agreement check that is this file's main point.
  auto total_mass = [&](const cfe::Field<double, kN>& s) {
    double m = 0.0;
    for (cfe::Index i = 0; i < grid.nx; ++i) m += s(grid.flat_index(grid.ngx + i, 0, 0), 0) * grid.dx;
    return m;
  };
  auto total_energy = [&](const cfe::Field<double, kN>& s) {
    double e = 0.0;
    for (cfe::Index i = 0; i < grid.nx; ++i) {
      e += s(grid.flat_index(grid.ngx + i, 0, 0), Field::energy_index()) * grid.dx;
    }
    return e;
  };
  const double mass_before = total_mass(state_cpu);
  const double energy_before = total_energy(state_cpu);

  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double>(state_cpu.view(), stage1_cpu.view(), scratch_cpu.view(), kDt, residual_cpu,
                               solver_cpu.active_cell_count(), solver_cpu.active_cell_index_map());
  }

  CFE_CHECK_NEAR(total_mass(state_cpu), mass_before, 1e-10);
  CFE_CHECK_NEAR(total_energy(state_cpu), energy_before, 1e-10);

  // --- CUDA ---
  cfe::backend::cuda::DeviceField<double, kN> state_gpu(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, kN> stage1_gpu(grid.n_cells_total());
  cfe::backend::cuda::DeviceField<double, kN> scratch_gpu(grid.n_cells_total());
  state_gpu.copy_from_host(host_ic.data());

  cfe::FvmSolver<double, cfe::AoSLayout, Field, cfe::SlipWallBoundary<Field>, cfe::SlipWallBoundary<Field>,
                 cfe::SlipWallBoundary<Field>, cfe::fvm::MusclMinmodReconstruction, cfe::RusanovFlux,
                 cfe::CudaParallelFor>
      solver_gpu{grid, field, boundary};
  cfe::SolverResidual<decltype(solver_gpu)> residual_gpu{&solver_gpu};
  for (int step = 0; step < kSteps; ++step) {
    cfe::ssp_rk2_step<double, cfe::FieldView<double, kN>, decltype(residual_gpu), cfe::CudaParallelFor>(
        state_gpu.view(), stage1_gpu.view(), scratch_gpu.view(), kDt, residual_gpu,
        solver_gpu.active_cell_count(), solver_gpu.active_cell_index_map());
  }
  cfe::backend::cuda::synchronize();

  std::vector<double> host_result(grid.n_cells_total() * kN);
  state_gpu.copy_to_host(host_result.data());

  for (cfe::Index i = 0; i < grid.nx; ++i) {
    const cfe::Index cell = grid.flat_index(grid.ngx + i, 0, 0);
    for (cfe::Index c = 0; c < kN; ++c) {
      CFE_CHECK_NEAR(host_result[cell * kN + c], state_cpu.data()[cell * kN + c], 1e-9);
    }
  }
}
