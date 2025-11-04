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
# Set default to H100 if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures" FORCE)
else()
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures")
endif()

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
# CMake modules
# ==============================================================================

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ==============================================================================
# Find dependencies
# ==============================================================================

# CUTLASS 4.3.0 (fetched via FetchContent or user-supplied)
include(ThirdPartyCUTLASS)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cutlass
    ${CUTLASS_INCLUDE_DIR}
)

set_target_properties(benchmark_trajectory_resample PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Optimization comparison benchmark
add_executable(benchmark_optimization
    benchmarks/benchmark_optimization.cu
    kernels/cutlass/trajectory_resample.cu
    kernels/cutlass/trajectory_resample_optimized.cu
)

target_include_directories(benchmark_optimization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cutlass
    ${CUTLASS_INCLUDE_DIR}
)

set_target_properties(benchmark_optimization PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Multimodal fusion benchmark
add_executable(benchmark_multimodal_fusion
    benchmarks/benchmark_multimodal_fusion.cu
    kernels/cutlass/trajectory_resample.cu
    kernels/cutlass/multimodal_fusion.cu
)

target_include_directories(benchmark_multimodal_fusion PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cutlass
    ${CUTLASS_INCLUDE_DIR}
)

set_target_properties(benchmark_multimodal_fusion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Point cloud voxelization benchmark
add_executable(benchmark_voxelization
    benchmarks/benchmark_voxelization.cu
    kernels/cutlass/point_cloud_voxelization.cu
)

target_include_directories(benchmark_voxelization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cutlass
    ${CUTLASS_INCLUDE_DIR}
)

set_target_properties(benchmark_voxelization PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Comprehensive voxelization benchmark (all modes)
add_executable(benchmark_voxelization_full
    benchmarks/benchmark_voxelization_full.cu
    kernels/cutlass/point_cloud_voxelization.cu
)

target_include_directories(benchmark_voxelization_full PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cutlass
    ${CUTLASS_INCLUDE_DIR}
)

set_target_properties(benchmark_voxelization_full PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ==============================================================================
# PyTorch extension (Python module)
# ==============================================================================

if(BUILD_TORCH_EXTENSION)
    add_library(robocache_cuda MODULE
        kernels/cutlass/trajectory_resample.cu
        kernels/cutlass/trajectory_resample_optimized.cu
        kernels/cutlass/trajectory_resample_production.cu
        kernels/cutlass/trajectory_resample_torch.cu
        kernels/cutlass/multimodal_fusion.cu
        kernels/cutlass/multimodal_fusion_torch.cu
        kernels/cutlass/point_cloud_voxelization.cu
        kernels/cutlass/point_cloud_voxelization_torch.cu
    )

    target_include_directories(robocache_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cutlass
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
install(FILES 
    kernels/cutlass/trajectory_resample.h
    kernels/cutlass/multimodal_fusion.h
    kernels/cutlass/point_cloud_voxelization.h
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
