# Lets this tutorial be configured/built standalone from its own
# directory (`cmake -S . -B build` run here), not only as part of the
# whole CMU-CFE build: if cfe_core doesn't already exist as a target,
# this is the first (outermost) CMake invocation, so pull in the real
# top-level project to define it, then stop -- the nested inclusion
# below re-processes this same file with cfe_core already available.
if(NOT TARGET cfe_core)
  cmake_minimum_required(VERSION 3.18)
  project(cfe_advect_burgers_gaussian_3d_standalone LANGUAGES CXX)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. cfe-root-build)
  return()
endif()

add_executable(cfe_advect_burgers_gaussian_3d advect_burgers_gaussian_3d.cpp)
target_link_libraries(cfe_advect_burgers_gaussian_3d PRIVATE cfe_core)
