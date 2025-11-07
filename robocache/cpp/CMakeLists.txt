cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(robocache LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find PyTorch
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)

# CUDA architectures - EXPLICIT targeting
set(CMAKE_CUDA_ARCHITECTURES "80;90")  # A100 (sm_80) + H100 (sm_90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -Xptxas=-v")

# Source files
set(CUDA_SOURCES
    ../csrc/cuda/resample_kernel.cu
    ../csrc/cuda/multimodal_kernel.cu
    ../csrc/cuda/voxelize_kernel.cu
)

set(CPP_SOURCES
    ../csrc/cpp/resample_ops.cpp
    ../csrc/cpp/multimodal_ops.cpp
    ../csrc/cpp/voxelize_ops.cpp
)

# Build extensions
add_library(robocache_cuda MODULE ${CUDA_SOURCES} ${CPP_SOURCES})
target_include_directories(robocache_cuda PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ../csrc/cuda
    ../csrc/cpp
)
target_link_libraries(robocache_cuda ${TORCH_LIBRARIES})
set_target_properties(robocache_cuda PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_cuda_ops"
    SUFFIX ".so"
)

# Install
install(TARGETS robocache_cuda DESTINATION .)

