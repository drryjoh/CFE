# Lets this tutorial be configured/built standalone from its own
# directory, not only as part of the whole CMU-CFE build -- see
# tutorials/burgers_3d_visualization/CMakeLists.txt's own comment for
# why this block exists.
if(NOT TARGET cfe_core)
  cmake_minimum_required(VERSION 3.18)
  project(cfe_mpi_scalar_advection_3d_standalone LANGUAGES CXX)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. cfe-root-build)
  return()
endif()

# MPI-only tutorial (task spec 0005) -- a no-op subdirectory in a
# non-MPI build of the whole project, same gating convention
# tests/mpi/ and benchmarks/mpi/ use at their own parent CMakeLists.txt.
if(CFE_ENABLE_MPI)
  add_executable(cfe_mpi_scalar_advection_3d mpi_scalar_advection_3d.cpp)
  target_link_libraries(cfe_mpi_scalar_advection_3d PRIVATE cfe_core)
endif()
