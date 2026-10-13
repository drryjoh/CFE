if(NOT TARGET cfe_core)
  cmake_minimum_required(VERSION 3.18)
  project(cfe_euler_2d_forward_step_standalone LANGUAGES CXX)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. cfe-root-build)
  return()
endif()

add_executable(cfe_euler_2d_forward_step forward_step_2d.cpp)
target_link_libraries(cfe_euler_2d_forward_step PRIVATE cfe_core)
# Reuses the shared two-block assembly/stepping driver
# (tests/verification/forward_step_setup.hpp) that
# tests/unit/test_forward_step_sanity.cpp also uses, rather than
# duplicating it here. NOT shared by
# tests/verification/test_forward_step_interface_conservation.cpp --
# that test needs to reach inside the reconstruction/numerical-flux
# pipeline directly (to recompute the interface flux from each block's
# own point of view) in a way FvmSolver::residual() does not expose, so
# it builds its own minimal two-grid scaffold instead.
target_include_directories(cfe_euler_2d_forward_step PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/verification
)
