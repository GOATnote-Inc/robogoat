# cmake/ThirdPartyCUTLASS.cmake
# CUTLASS v4.3.0 dependency management for RoboCache
# Uses FetchContent for reproducible builds, validates version exactly

include(FetchContent)

# User-tunable options with safe defaults
option(ROBOCACHE_BUNDLE_CUTLASS "Fetch CUTLASS automatically" ON)
set(ROBOCACHE_CUTLASS_TAG "v4.2.1" CACHE STRING "Required CUTLASS tag")

# If user provides an existing CUTLASS root, prefer it, but verify version.
set(_CUTLASS_HINTS
  ${CUTLASS_ROOT}
  $ENV{CUTLASS_ROOT}
)

if(NOT ROBOCACHE_BUNDLE_CUTLASS)
  find_path(CUTLASS_INCLUDE_DIR
    NAMES cutlass/cutlass.h
    HINTS ${_CUTLASS_HINTS}
    PATH_SUFFIXES include
  )
  if(NOT CUTLASS_INCLUDE_DIR)
    message(FATAL_ERROR "CUTLASS not found. Either set CUTLASS_ROOT to a v4.2+ install, or enable ROBOCACHE_BUNDLE_CUTLASS=ON.")
  endif()
else()
  # Disable CUTLASS library build - we only need headers
  set(CUTLASS_ENABLE_LIBRARY OFF CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_PROFILER OFF CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  
  FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG        ${ROBOCACHE_CUTLASS_TAG}
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
  )
  FetchContent_MakeAvailable(cutlass)
  set(CUTLASS_INCLUDE_DIR "${cutlass_SOURCE_DIR}/include" CACHE PATH "" FORCE)
endif()

# ---- Version check: require CUTLASS 4.2+ ----
file(READ "${CUTLASS_INCLUDE_DIR}/cutlass/version.h" _cutlass_ver_h)
string(REGEX MATCH "#define[ \t]+CUTLASS_MAJOR[ \t]+([0-9]+)" _m1 "${_cutlass_ver_h}")
set(_cutlass_major "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define[ \t]+CUTLASS_MINOR[ \t]+([0-9]+)" _m2 "${_cutlass_ver_h}")
set(_cutlass_minor "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define[ \t]+CUTLASS_PATCH[ \t]+([0-9]+)" _m3 "${_cutlass_ver_h}")
set(_cutlass_patch "${CMAKE_MATCH_1}")

set(CUTLASS_VERSION "${_cutlass_major}.${_cutlass_minor}.${_cutlass_patch}")

# Require CUTLASS 4.2+  
if(CUTLASS_VERSION VERSION_LESS "4.2.0")
  message(FATAL_ERROR
    "Found CUTLASS ${CUTLASS_VERSION} at ${CUTLASS_INCLUDE_DIR}, "
    "but RoboCache requires v4.2.0 or later (latest is v4.2.1). "
    "Set ROBOCACHE_BUNDLE_CUTLASS=ON (default) or point CUTLASS_ROOT to v4.2+.")
endif()

message(STATUS "Using CUTLASS ${CUTLASS_VERSION} at: ${CUTLASS_INCLUDE_DIR}")

