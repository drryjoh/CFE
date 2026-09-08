add_executable(cfe_bench_field_update bench_field_update.cpp)
target_link_libraries(cfe_bench_field_update PRIVATE cfe_core)

# CUDA sweep: only buildable where nvcc/hardware are present. See
# bench_field_update_cuda.cu for the "unverified" note explaining why this
# could not be compiled during Phase 0 development.
if(CFE_ENABLE_CUDA)
  add_executable(cfe_bench_field_update_cuda bench_field_update_cuda.cu)
  target_link_libraries(cfe_bench_field_update_cuda PRIVATE cfe_core)
  set_target_properties(cfe_bench_field_update_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
