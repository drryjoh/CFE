if(NOT TARGET cfe_core)
  cmake_minimum_required(VERSION 3.18)
  project(cfe_euler_3d_forward_step_extrusion_standalone LANGUAGES CXX)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. cfe-root-build)
  return()
endif()

add_executable(cfe_euler_3d_forward_step_extrusion forward_step_3d.cpp)
target_link_libraries(cfe_euler_3d_forward_step_extrusion PRIVATE cfe_core)
target_include_directories(cfe_euler_3d_forward_step_extrusion PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/verification
)
