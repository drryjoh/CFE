cmake_minimum_required(VERSION 3.18)

project(CMU-CFE LANGUAGES CXX)

# See docs/adr/0006-language-standard.md: C++20 is an explicit architectural
# decision, not an arbitrary default, and should stay in sync with
# CMAKE_CUDA_STANDARD below.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Phase 0 configuration options ------------------------------------------
# See AGENTS.md #11 and docs/adr/0001-execution-backend.md.

set(CFE_SCALAR_TYPE "double" CACHE STRING "Project-wide default scalar precision (float|double)")
set_property(CACHE CFE_SCALAR_TYPE PROPERTY STRINGS float double)

set(CFE_DEFAULT_BACKEND "serial" CACHE STRING "Default cfe::parallel_for backend (serial|threaded)")
set_property(CACHE CFE_DEFAULT_BACKEND PROPERTY STRINGS serial threaded)

option(CFE_BUILD_TESTS "Build the CFE unit test suite" ON)
option(CFE_BUILD_BENCHMARKS "Build the CFE benchmark executables" ON)
option(CFE_BUILD_TUTORIALS "Build the CFE tutorial/example executables" ON)

# CUDA is opt-in but auto-detected: Phase 0 must not fail a CPU-only build
# just because nvcc/hardware happen to be unavailable (see ADR 0001).
option(CFE_ENABLE_CUDA "Build the CUDA execution backend" OFF)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CFE: CUDA compiler found (${CMAKE_CUDA_COMPILER}); CFE_ENABLE_CUDA may be turned ON")
else()
  if(CFE_ENABLE_CUDA)
    # CFE_ENABLE_CUDA was explicitly requested, not left at its default OFF
    # -- silently downgrading that to a CPU-only build would hide a real
    # configuration problem (missing/broken CUDA toolkit) behind a build
    # that looks successful but is missing the backend the user asked for.
    message(FATAL_ERROR
      "CFE_ENABLE_CUDA=ON was explicitly requested but no CUDA compiler was "
      "found (check_language(CUDA) failed). Install/configure a CUDA "
      "toolkit so nvcc is on PATH, or omit -DCFE_ENABLE_CUDA=ON to build "
      "the CPU-only configuration.")
  endif()
  set(CFE_ENABLE_CUDA OFF CACHE BOOL "Build the CUDA execution backend" FORCE)
endif()

if(CFE_ENABLE_CUDA)
  enable_language(CUDA)
  if(NOT CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
endif()

find_package(Threads REQUIRED)

# --- Core library (header-only) ---------------------------------------------

add_library(cfe_core INTERFACE)
target_include_directories(cfe_core INTERFACE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(cfe_core INTERFACE Threads::Threads)
target_compile_features(cfe_core INTERFACE cxx_std_20)

if(CFE_SCALAR_TYPE STREQUAL "float")
  target_compile_definitions(cfe_core INTERFACE CFE_SCALAR_IS_FLOAT)
endif()

if(CFE_DEFAULT_BACKEND STREQUAL "threaded")
  target_compile_definitions(cfe_core INTERFACE CFE_DEFAULT_BACKEND_THREADED)
endif()

if(CFE_ENABLE_CUDA)
  target_compile_definitions(cfe_core INTERFACE CFE_ENABLE_CUDA)
endif()

# Reasonable warnings-as-signal (not warnings-as-errors: Phase 0 favors fast
# iteration; tighten this later if warnings accumulate unexamined).
target_compile_options(cfe_core INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
)

if(CFE_ENABLE_CUDA)
  # Required for the __device__/__host__ __device__-annotated lambdas used
  # by every parallel_for call site (see backend/cuda/cuda_backend.cuh);
  # without it nvcc rejects them.
  target_compile_options(cfe_core INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif()

if(CFE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(CFE_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(CFE_BUILD_TUTORIALS)
  add_subdirectory(tutorials)
endif()
