if(NOT TARGET cfe_core)
  cmake_minimum_required(VERSION 3.18)
  project(cfe_euler_1d_shu_osher_standalone LANGUAGES CXX)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. cfe-root-build)
  return()
endif()

add_executable(cfe_euler_1d_shu_osher shu_osher.cpp)
target_link_libraries(cfe_euler_1d_shu_osher PRIVATE cfe_core)
target_include_directories(cfe_euler_1d_shu_osher PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/verification
)
