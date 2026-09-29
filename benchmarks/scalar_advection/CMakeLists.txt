# Large-scale scalar-advection benchmark: only meaningful on CUDA (task
# spec item 11's benchmark sweep is small-scale/CPU on the memory
# benchmarks already; this one specifically demonstrates the solver
# running at scale, per the 2026-09-29 request).
if(CFE_ENABLE_CUDA)
  add_executable(cfe_bench_scalar_advection_cuda bench_scalar_advection_cuda.cu)
  target_link_libraries(cfe_bench_scalar_advection_cuda PRIVATE cfe_core)
  set_target_properties(cfe_bench_scalar_advection_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
