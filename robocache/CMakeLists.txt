cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(robocache LANGUAGES CXX CUDA)

# ==============================================================================
# Project metadata
# ==============================================================================

set(ROBOCACHE_VERSION_MAJOR 0)
set(ROBOCACHE_VERSION_MINOR 1)
set(ROBOCACHE_VERSION_PATCH 0)
set(ROBOCACHE_VERSION
    "${ROBOCACHE_VERSION_MAJOR}.${ROBOCACHE_VERSION_MINOR}.${ROBOCACHE_VERSION_PATCH}")

message(STATUS "RoboCache version ${ROBOCACHE_VERSION}")
message(STATUS "GPU-accelerated data engine for embodied AI")

# ==============================================================================
# Compiler requirements
# ==============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Require CUDA 13.x for H100 support
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
    message(FATAL_ERROR "CUDA 13.x or later required (found ${CMAKE_CUDA_COMPILER_VERSION})")
endif()

message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")

# ==============================================================================
# Build configuration
# ==============================================================================

# Default to Release build for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ==============================================================================
# CUDA architecture configuration
# ==============================================================================

# H100 = compute capability 9.0 (sm_90)
# Also support A100 (8.0) and RTX 4090 (8.9) for testing
set(CMAKE_CUDA_ARCHITECTURES "90;89;80" CACHE STRING "CUDA architectures")

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ==============================================================================
# CUDA compiler flags
# ==============================================================================

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# Debugging and profiling
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")  # For profilers
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v") # Show register usage

# Additional optimizations for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNDEBUG")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")
endif()

# Debug flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

# ==============================================================================
# Find dependencies
# ==============================================================================

# CUTLASS
set(CUTLASS_DIR "/usr/local/cutlass" CACHE PATH "Path to CUTLASS installation")
find_path(CUTLASS_INCLUDE_DIR
    NAMES cutlass/cutlass.h
    PATHS ${CUTLASS_DIR}/include
          /usr/local/include
          /usr/include
)

if(NOT CUTLASS_INCLUDE_DIR)
    message(FATAL_ERROR
        "CUTLASS not found. Please install CUTLASS 4.3.0 or set CUTLASS_DIR.\n"
        "Download from: https://github.com/NVIDIA/cutlass")
endif()

message(STATUS "CUTLASS include dir: ${CUTLASS_INCLUDE_DIR}")

# PyTorch (for Python bindings)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3 version: ${Python3_VERSION}")
message(STATUS "Python3 executable: ${Python3_EXECUTABLE}")

# Try to find PyTorch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__version__)"
    OUTPUT_VARIABLE TORCH_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_FOUND
)

if(TORCH_FOUND EQUAL 0)
    message(STATUS "PyTorch version: ${TORCH_VERSION}")

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH};${CMAKE_PREFIX_PATH}")
    find_package(Torch REQUIRED)

    set(BUILD_TORCH_EXTENSION ON)
    message(STATUS "PyTorch extension will be built")
else()
    message(WARNING "PyTorch not found. Python bindings will not be built.")
    set(BUILD_TORCH_EXTENSION OFF)
endif()

# ==============================================================================
# Include directories
# ==============================================================================

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cutlass
    ${CUTLASS_INCLUDE_DIR}
)

# ==============================================================================
# Standalone benchmark executable
# ==============================================================================

add_executable(benchmark_trajectory_resample
    benchmarks/benchmark_trajectory_resample.cu
    kernels/cutlass/trajectory_resample.cu
)

target_include_directories(benchmark_trajectory_resample PRIVATE
    ${CUTLASS_INCLUDE_DIR}
)

set_target_properties(benchmark_trajectory_resample PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ==============================================================================
# PyTorch extension (Python module)
# ==============================================================================

if(BUILD_TORCH_EXTENSION)
    add_library(robocache_cuda MODULE
        kernels/cutlass/trajectory_resample.cu
        kernels/cutlass/trajectory_resample_torch.cu
    )

    target_include_directories(robocache_cuda PRIVATE
        ${CUTLASS_INCLUDE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )

    target_link_libraries(robocache_cuda
        ${TORCH_LIBRARIES}
        ${Python3_LIBRARIES}
    )

    # Configure output properties for Python module
    set_target_properties(robocache_cuda PROPERTIES
        PREFIX ""  # Remove 'lib' prefix
        OUTPUT_NAME "robocache_cuda"
        SUFFIX ".so"  # Explicit .so extension
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Determine output directory (inside python package)
    set(PYTHON_MODULE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python/robocache")
    set_target_properties(robocache_cuda PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${PYTHON_MODULE_OUTPUT_DIR}
    )

    message(STATUS "PyTorch extension output: ${PYTHON_MODULE_OUTPUT_DIR}/robocache_cuda.so")
endif()

# ==============================================================================
# Installation rules
# ==============================================================================

install(TARGETS benchmark_trajectory_resample
    RUNTIME DESTINATION bin
)

if(BUILD_TORCH_EXTENSION)
    install(TARGETS robocache_cuda
        LIBRARY DESTINATION ${Python3_SITEARCH}
    )
endif()

# Install headers
install(FILES kernels/cutlass/trajectory_resample.h
    DESTINATION include/robocache
)

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "")
message(STATUS "====================================")
message(STATUS "RoboCache Build Configuration")
message(STATUS "====================================")
message(STATUS "Version: ${ROBOCACHE_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUTLASS: ${CUTLASS_INCLUDE_DIR}")
message(STATUS "PyTorch extension: ${BUILD_TORCH_EXTENSION}")
if(BUILD_TORCH_EXTENSION)
    message(STATUS "PyTorch version: ${TORCH_VERSION}")
endif()
message(STATUS "====================================")
message(STATUS "")
