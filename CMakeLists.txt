# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Juan Garcia

cmake_minimum_required(VERSION 3.19)

# Single source of truth: the version declared in pyproject.toml.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _statcraft_version_line
  REGEX "^version = \"[0-9]+\\.[0-9]+\\.[0-9]+\""
)
if(NOT _statcraft_version_line)
  message(FATAL_ERROR "Could not read project version from pyproject.toml")
endif()
string(REGEX REPLACE "^version = \"([^\"]+)\".*" "\\1" STATCRAFT_VERSION "${_statcraft_version_line}")

project(
  statcraft
  VERSION ${STATCRAFT_VERSION}
  DESCRIPTION "Engines for statistical model fitting (C++ core) with a lazy, namespaced Python API"
  LANGUAGES CXX
)

# ============================================================================
# Policies
# ============================================================================
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

# ============================================================================
# Toolchain
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Do not scan for C++20 modules
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Position-independent code for extension modules (.pyd / shared objects)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to a Release build on single-config generators
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ============================================================================
# Options
# ============================================================================
option(STATCRAFT_BUILD_PYTHON
  "Build the Python extension modules (nanobind: _glmfit, _nlmefit, _mcglmfit, _saemfit)"
  ON)
option(STATCRAFT_USE_OPENMP
  "Enable OpenMP multi-threading (falls back to single-threaded builds when not available)"
  ON)
option(STATCRAFT_NATIVE
  "Build with native CPU flags. NOT portable across CPUs -- keep OFF for distributable wheels"
  OFF)

# OpenMP is optional: platforms such as stock macOS Clang do not ship it.
set(STATCRAFT_HAVE_OPENMP OFF)
if(STATCRAFT_USE_OPENMP)
  find_package(OpenMP QUIET COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    set(STATCRAFT_HAVE_OPENMP ON)
  else()
    message(STATUS "statcraft: OpenMP not available -- building single-threaded")
  endif()
endif()

# Four heavy translation units can exhaust RAM on small machines when built in
# parallel. Default to sequential module builds only when physical memory is
# low; otherwise let the build tool parallelize normally.
cmake_host_system_information(RESULT _statcraft_ram_mb QUERY TOTAL_PHYSICAL_MEMORY)
# Universal2 builds compile two architectures, so cap parallelism on macOS.
if(APPLE OR _statcraft_ram_mb LESS 6144)
  set(_statcraft_sequential_default ON)
else()
  set(_statcraft_sequential_default OFF)
endif()
option(STATCRAFT_SEQUENTIAL_MODULES
  "Build the four extension modules one at a time to reduce peak RAM usage"
  ${_statcraft_sequential_default})

# ============================================================================
# Dependencies: system packages first, FetchContent as a portable fallback
# ============================================================================
include(FetchContent)

# --- Eigen 3 ---
find_package(Eigen3 3.4 CONFIG QUIET)
if(NOT Eigen3_FOUND)
  message(STATUS "statcraft: Eigen3 not found on the system -- fetching a source copy")
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(Eigen3)
endif()


# --- Boost (headers; the Math module is used by the glm engine) ---
find_package(Boost 1.75 QUIET)
if(NOT Boost_FOUND)
  message(STATUS "statcraft: Boost not found on the system -- fetching the Boost.Math headers")

  FetchContent_Declare(
    BoostMath
    GIT_REPOSITORY https://github.com/boostorg/math.git
    GIT_TAG boost-1.84.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(BoostMath)

  if(NOT TARGET Boost::boost)
    add_library(Boost::boost INTERFACE IMPORTED GLOBAL)
    target_include_directories(Boost::boost INTERFACE ${boostmath_SOURCE_DIR}/include)
    target_compile_definitions(Boost::boost INTERFACE BOOST_MATH_STANDALONE=1)
  endif()
endif()


# ============================================================================
# Shared Core
# ============================================================================
add_library(statcraft_core INTERFACE)
add_library(statcraft::core ALIAS statcraft_core)

# Only this include directory is added; recursive parent-directory includes confuse the compiler.
target_include_directories(statcraft_core SYSTEM INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(statcraft_core INTERFACE
  Eigen3::Eigen
  $<$<BOOL:${STATCRAFT_HAVE_OPENMP}>:OpenMP::OpenMP_CXX>
  $<$<TARGET_EXISTS:Boost::headers>:Boost::headers>
  $<$<NOT:$<TARGET_EXISTS:Boost::headers>>:Boost::boost>
)

set(STATCRAFT_CORE_COMPILE_DEFINITIONS "")
if(STATCRAFT_HAVE_OPENMP)
  list(APPEND STATCRAFT_CORE_COMPILE_DEFINITIONS STATCRAFT_HAVE_OPENMP=1)
endif()
if(WIN32)
  list(APPEND STATCRAFT_CORE_COMPILE_DEFINITIONS
    NOMINMAX
    _USE_MATH_DEFINES
  )
endif()
target_compile_definitions(statcraft_core INTERFACE ${STATCRAFT_CORE_COMPILE_DEFINITIONS})

# Compiler flags. -w / /w only affect this interface target, not downstream code.
target_compile_options(statcraft_core INTERFACE
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-O3>
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-pipe>
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-w>
  $<$<CXX_COMPILER_ID:MSVC>:/O2>
  $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
  $<$<CXX_COMPILER_ID:MSVC>:/EHsc>
  $<$<CXX_COMPILER_ID:MSVC>:/w>
  $<$<CXX_COMPILER_ID:MSVC>:/MP>
)
if(STATCRAFT_NATIVE)
  target_compile_options(statcraft_core INTERFACE
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-march=native>
    $<$<CXX_COMPILER_ID:AppleClang>:-mcpu=native>
  )
endif()

# ============================================================================
# Python Extension Modules (nanobind)
# ============================================================================
if(STATCRAFT_BUILD_PYTHON)
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

  # Detect nanobind through the Python interpreter when no path was supplied.
  if(NOT nanobind_DIR)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE nanobind_DIR
      ERROR_QUIET
    )
  endif()

  # Load nanobind from the system or the detected Python environment.
  find_package(nanobind CONFIG QUIET)

  # Fallback: download nanobind with FetchContent when it is not available locally.
  if(NOT nanobind_FOUND)
    message(STATUS "statcraft: nanobind not found on the system or in the Python environment -- fetching a local copy")

    FetchContent_Declare(
      nanobind
      GIT_REPOSITORY https://github.com/wjakob/nanobind.git
      GIT_TAG v2.4.0  # Adjust this pin when upgrading nanobind
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nanobind)
  endif()

  set(STATCRAFT_PYMODULE_INSTALL "statcraft"
    CACHE STRING "Install destination of the Python modules relative to the install prefix")

  macro(statcraft_python_module TARGET_NAME SOURCE_FILE)
    nanobind_add_module(${TARGET_NAME} NOMINSIZE ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} PRIVATE statcraft_core)
  endmacro()

  statcraft_python_module(_glmfit bindings/_glmfit.cpp)
  statcraft_python_module(_nlmefit bindings/_nlmefit.cpp)
  statcraft_python_module(_mcglmfit bindings/_mcglmfit.cpp)
  statcraft_python_module(_saemfit bindings/_saemfit.cpp)

  if(STATCRAFT_SEQUENTIAL_MODULES)
    message(STATUS "statcraft: building extension modules sequentially (STATCRAFT_SEQUENTIAL_MODULES=ON)")
    add_dependencies(_nlmefit _glmfit)
    add_dependencies(_mcglmfit _nlmefit)
    add_dependencies(_saemfit _mcglmfit)
  else()
    message(STATUS "statcraft: extension modules may build in parallel; set CMAKE_BUILD_PARALLEL_LEVEL to control -j")
  endif()

  install(TARGETS _glmfit _nlmefit _mcglmfit _saemfit
    LIBRARY DESTINATION ${STATCRAFT_PYMODULE_INSTALL}
    RUNTIME DESTINATION ${STATCRAFT_PYMODULE_INSTALL}
  )
endif()
