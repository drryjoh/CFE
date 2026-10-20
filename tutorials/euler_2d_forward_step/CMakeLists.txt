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

# Checkpoint/restart demonstration (task tasks/0016-checkpoint-restart.md)
# -- a separate small executable, not a mode bolted onto
# cfe_euler_2d_forward_step's own sweep `main()`, per that task's own
# scope note ("a flag or separate entry point... rather than a wholesale
# rewrite of the existing sweep driver"). See
# forward_step_2d_restart_demo.cpp's own header comment for its
# `--mode=checkpoint`/`--mode=resume` usage and README.md's "Checkpoint/
# restart" section for the measured wall-clock evidence it produced.
add_executable(cfe_euler_2d_forward_step_restart_demo forward_step_2d_restart_demo.cpp)
target_link_libraries(cfe_euler_2d_forward_step_restart_demo PRIVATE cfe_core)
target_include_directories(cfe_euler_2d_forward_step_restart_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/verification
)
