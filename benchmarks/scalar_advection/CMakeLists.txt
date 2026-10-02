# Task spec item 11's CPU benchmark sweep for the scalar-advection kernel
# itself (grid resolution, not component count, is the scaling axis here).
add_executable(cfe_bench_scalar_advection bench_scalar_advection.cpp)
target_link_libraries(cfe_bench_scalar_advection PRIVATE cfe_core)

# Large-scale scalar-advection benchmark: only meaningful on CUDA (the
# CPU sweep above is small-scale; this one specifically demonstrates the
# solver running at scale, per the 2026-09-29 request).
if(CFE_ENABLE_CUDA)
  add_executable(cfe_bench_scalar_advection_cuda bench_scalar_advection_cuda.cu)
  target_link_libraries(cfe_bench_scalar_advection_cuda PRIVATE cfe_core)
  set_target_properties(cfe_bench_scalar_advection_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # 3D counterpart (2026-09-29): exercises all three axis_flux_difference
  # branches together, not just X, at up to ~134M cells (512^3).
  add_executable(cfe_bench_scalar_advection_3d_cuda bench_scalar_advection_3d_cuda.cu)
  target_link_libraries(cfe_bench_scalar_advection_3d_cuda PRIVATE cfe_core)
  set_target_properties(cfe_bench_scalar_advection_3d_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
