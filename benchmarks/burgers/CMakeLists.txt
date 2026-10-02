# Burgers benchmark sweep (task 0003/PR #3 follow-up: every PR adding a
# Field/scheme needs an at-scale GPU benchmark, not just correctness
# tests), mirroring benchmarks/scalar_advection/CMakeLists.txt exactly.
add_executable(cfe_bench_burgers bench_burgers.cpp)
target_link_libraries(cfe_bench_burgers PRIVATE cfe_core)

if(CFE_ENABLE_CUDA)
  add_executable(cfe_bench_burgers_cuda bench_burgers_cuda.cu)
  target_link_libraries(cfe_bench_burgers_cuda PRIVATE cfe_core)
  set_target_properties(cfe_bench_burgers_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_executable(cfe_bench_burgers_3d_cuda bench_burgers_3d_cuda.cu)
  target_link_libraries(cfe_bench_burgers_3d_cuda PRIVATE cfe_core)
  set_target_properties(cfe_bench_burgers_3d_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
