if(NOT TARGET cfe_core)
  cmake_minimum_required(VERSION 3.18)
  project(cfe_burgers_1d_standalone LANGUAGES CXX)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. cfe-root-build)
  return()
endif()

add_executable(cfe_burgers_1d burgers_1d.cpp)
target_link_libraries(cfe_burgers_1d PRIVATE cfe_core)
