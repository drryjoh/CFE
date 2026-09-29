# A tiny, dependency-free hands-on example (task spec item 14: "one
# tutorial demonstrating the benchmark"). See benchmarks/memory/ for the
# real 48-combination timing sweep this is deliberately not a replacement
# for.
add_subdirectory(hello_parallel_for)

# 3D scalar-advection VTK time-series demo (2026-09-29 follow-up to
# Phase 1's CUDA correctness/benchmark work, which produces no output to
# actually look at). CPU-only.
add_subdirectory(scalar_advection_3d_visualization)
