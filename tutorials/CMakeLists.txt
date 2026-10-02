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
