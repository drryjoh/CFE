if(NOT TARGET cfe_core)
  cmake_minimum_required(VERSION 3.18)
  project(cfe_euler_1d_sod_shock_tube_standalone LANGUAGES CXX)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. cfe-root-build)
  return()
endif()

add_executable(cfe_euler_1d_sod_shock_tube sod_shock_tube.cpp)
target_link_libraries(cfe_euler_1d_sod_shock_tube PRIVATE cfe_core)
# Reuses the exact Riemann solver tests/verification/test_euler_sod_shock_tube.cpp
# already verifies against, rather than reimplementing it here.
target_include_directories(cfe_euler_1d_sod_shock_tube PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/verification
)
