# Every tutorial subdirectory's CMakeLists.txt is buildable BOTH ways:
# as part of this whole-project build (the normal case, reached via this
# file), and standalone (`cmake -S . -B build` run directly inside that
# tutorial's own directory, with no reference to the repo root) -- for a
# student who just wants to poke at one example without the rest of the
# project. The `if(NOT TARGET cfe_core) ... return()` block at the top of
# each tutorial's CMakeLists.txt is what makes the standalone case work
# (it pulls in this repo's root CMakeLists.txt as a nested subdirectory
# purely to obtain cfe_core); new tutorials should copy that same block.
#
# Pulling in the root this way also makes the *whole* project's targets
# available in the standalone case, not just this one tutorial -- tests,
# benchmarks, and every other tutorial, since their CFE_BUILD_TESTS/
# CFE_BUILD_BENCHMARKS/CFE_BUILD_TUTORIALS options default ON regardless
# of which directory was configured from. A standalone `cmake --build`
# must therefore pass `--target <this-tutorial's-executable>` to actually
# build only this one (each tutorial's own README documents this).
#
# A tiny, dependency-free hands-on example (task spec item 14: "one
# tutorial demonstrating the benchmark"). See benchmarks/memory/ for the
# real 48-combination timing sweep this is deliberately not a replacement
# for.
add_subdirectory(hello_parallel_for)

# 3D scalar-advection VTK time-series demo (2026-09-29 follow-up to
# Phase 1's CUDA correctness/benchmark work, which produces no output to
# actually look at). CPU-only.
add_subdirectory(scalar_advection_3d_visualization)

# 3D Burgers VTK time-series demo (2026-10-02 follow-up to task 0003's
# CUDA correctness/benchmark work, same reasoning as the scalar-advection
# tutorial above, but showing deformation/shock formation instead of
# unchanged translation). CPU-only.
add_subdirectory(burgers_3d_visualization)

# Quantitative 1D/2D Burgers verification tutorials (grid-convergence
# sweeps against exact cut-cell references, first-order vs limited
# second-order reconstruction comparison, plotted via a committed Python
# script) -- a different kind of tutorial than the two above (those are
# single-run visual demos; these are reproducible numerical-evidence
# harnesses). CPU-only.
add_subdirectory(burgers_1d_shock_and_steepening)
add_subdirectory(burgers_2d_diagonal_shock)

# 1D compressible Euler verification tutorial: the Sod (1978) shock
# tube, Phase 3's companion 1D benchmark -- same quantitative-evidence
# style as the Burgers tutorials above (grid-convergence sweep against
# an exact reference, here the exact Riemann solution rather than a
# cut-cell/characteristics formula). CPU-only.
add_subdirectory(euler_1d_sod_shock_tube)

# ROADMAP.md Phase 3's SECOND named Euler canonical case alongside Sod:
# Shu-Osher (Shu & Osher 1989) shock/density-wave interaction, task
# 0012. Same reused EulerField/MUSCL/Rusanov+HLLC/FvmSolver stack;
# verifies (and here, visualizes) the fine post-shock structure question
# -- see tests/verification/test_euler_shu_osher.cpp and
# tasks/0012-phase3-shu-osher.md. CPU-only.
add_subdirectory(euler_1d_shu_osher)

# ROADMAP.md Phase 4's "1D propagating shock demonstration and periodic
# propagation-speed estimate", task 0013: a standalone single
# right-moving shock, tracked across snapshots and compared against the
# analytic Rankine-Hugoniot speed. CPU-only.
add_subdirectory(euler_1d_propagating_shock)

# Phase 4 (pulled forward), task 0010: Woodward & Colella (1984) Mach 3
# forward-facing-step wind tunnel -- CFE's first multi-block case, two
# CartesianGrid blocks joined by the new BlockInterfaceBoundary
# (grid/boundary/block_interface_boundary.hpp). 2D is the primary
# deliverable (three reference resolutions, t~4, qualitative bow-shock/
# Mach-stem/contact checks); the 3D sibling is a CPU-only spanwise
# extrusion "see it work" demonstration, explicitly not a substitute for
# a future CUDA 3D-correctness/at-scale requirement (see that
# tutorial's own file header and docs/adr/0011-forward-step-block-
# interface.md). CPU-only, both.
add_subdirectory(euler_2d_forward_step)
add_subdirectory(euler_3d_forward_step_extrusion)

# 3D MPI strong-scaling demonstration (task spec 0005, follow-up to task
# 0004's MPI domain-decomposition prototype): the full solver, not just
# the halo exchange, run at a fixed global resolution across an
# increasing rank count. MPI-only -- a no-op subdirectory when
# CFE_ENABLE_MPI is off (see this tutorial's own CMakeLists.txt).
add_subdirectory(mpi_scalar_advection_3d_strong_scaling)

# Same strong-scaling story as the tutorial above, but for Burgers'
# equation (task spec 0007, follow-up to task 0006's MPI_Allreduce(MAX)
# CFL fix) -- demonstrates both that the fix works and that a
# decomposed Burgers run scales. MPI-only, same gating.
add_subdirectory(mpi_burgers_3d_strong_scaling)
