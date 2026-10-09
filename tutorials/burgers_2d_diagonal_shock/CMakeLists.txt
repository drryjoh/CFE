if(NOT TARGET cfe_core)
  cmake_minimum_required(VERSION 3.18)
  project(cfe_burgers_2d_standalone LANGUAGES CXX)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. cfe-root-build)
  return()
endif()

add_executable(cfe_burgers_2d burgers_2d.cpp)
target_link_libraries(cfe_burgers_2d PRIVATE cfe_core)
