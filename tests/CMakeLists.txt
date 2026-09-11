# Phase 0 unit tests (task spec item 13). A single executable links every
# test translation unit against the tiny header-only framework in
# tests/unit/test_framework.hpp; CTest just runs the resulting binary.

add_executable(cfe_unit_tests
  unit/test_main.cpp
  unit/test_fixed_array.cpp
  unit/test_math_operations.cpp
  unit/test_field_storage.cpp
  unit/test_backend_execution.cpp
)

target_link_libraries(cfe_unit_tests PRIVATE cfe_core)
target_include_directories(cfe_unit_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)

# CUDA correctness test: only added when a CUDA compiler/backend is
# available. See test_backend_execution_cuda.cu for why it is unverified.
if(CFE_ENABLE_CUDA)
  target_sources(cfe_unit_tests PRIVATE unit/test_backend_execution_cuda.cu)
  set_target_properties(cfe_unit_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_test(NAME cfe_unit_tests COMMAND cfe_unit_tests)
