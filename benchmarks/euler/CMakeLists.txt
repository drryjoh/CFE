# Euler benchmark sweep (task 0009 follow-up: every PR adding a
# Field/scheme needs an at-scale GPU benchmark, not just correctness
# tests -- same requirement benchmarks/burgers/CMakeLists.txt already
# satisfies for Burgers). GPU-only, mirroring
# benchmarks/burgers/CMakeLists.txt's CUDA targets exactly; no CPU
# benchmark executable is added here since none was requested for this
# follow-up (the task's own "Benchmarks" section for task 0009 already
# covers why no CPU-side benchmark change was needed).
if(CFE_ENABLE_CUDA)
  add_executable(cfe_bench_euler_cuda bench_euler_cuda.cu)
  target_link_libraries(cfe_bench_euler_cuda PRIVATE cfe_core)
  set_target_properties(cfe_bench_euler_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  add_executable(cfe_bench_euler_3d_cuda bench_euler_3d_cuda.cu)
  target_link_libraries(cfe_bench_euler_3d_cuda PRIVATE cfe_core)
  set_target_properties(cfe_bench_euler_3d_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # task 0011: HLLC's own at-scale benchmark, mirroring the Rusanov one
  # above exactly except for the flux -- see bench_euler_hllc_3d_cuda.cu's
  # own header comment.
  add_executable(cfe_bench_euler_hllc_3d_cuda bench_euler_hllc_3d_cuda.cu)
  target_link_libraries(cfe_bench_euler_hllc_3d_cuda PRIVATE cfe_core)
  set_target_properties(cfe_bench_euler_hllc_3d_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
