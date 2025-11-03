// trajectory_resample.h
// Public API for GPU-accelerated trajectory resampling
// Part of RoboCache - Data engine for embodied AI foundation models

#pragma once

#include <cuda_runtime.h>

namespace robocache {
namespace kernels {

/**
 * Resample robot trajectories using BF16 precision (recommended for H100)
 *
 * Converts variable-frequency robot trajectories to a uniform sampling rate
 * using GPU-accelerated linear interpolation. Optimized for NVIDIA H100 with
 * BF16 Tensor Cores and HBM3 memory.
 *
 * Input/Output memory layout:
 *   - source_data: [batch_size, source_length, action_dim] (BF16)
 *   - source_times: [batch_size, source_length] (FP32)
 *   - target_times: [batch_size, target_length] (FP32)
 *   - output_data: [batch_size, target_length, action_dim] (BF16)
 *
 * Requirements:
 *   - source_times must be monotonically increasing for each batch
 *   - All tensors must be GPU-resident (device pointers)
 *   - Memory must be properly aligned (16-byte alignment recommended)
 *
 * Performance characteristics:
 *   - Expected throughput: ~30K trajectories/sec on H100
 *   - Memory bandwidth: ~60% of theoretical HBM3 peak (1.8 TB/s)
 *   - Optimal batch_size: 128-512 for high GPU utilization
 *
 * @param source_data Device pointer to input trajectories (BF16 format)
 * @param source_times Device pointer to source timestamps (FP32)
 * @param target_times Device pointer to target timestamps (FP32)
 * @param output_data Device pointer to output trajectories (BF16 format)
 * @param batch_size Number of trajectories to resample
 * @param source_length Number of frames in each source trajectory
 * @param target_length Number of frames in each output trajectory
 * @param action_dim Dimensionality of robot actions (e.g., 7 for arm, 12 for humanoid)
 * @param use_async Enable asynchronous execution (default: true)
 * @param stream CUDA stream for asynchronous execution (default: 0)
 * @return cudaSuccess on success, error code otherwise
 */
cudaError_t resample_trajectories_bf16(
    const void* source_data,
    const float* source_times,
    const float* target_times,
    void* output_data,
    int batch_size,
    int source_length,
    int target_length,
    int action_dim,
    bool use_async = true,
    cudaStream_t stream = 0
);

/**
 * Resample robot trajectories using FP32 precision
 *
 * Same functionality as BF16 version but with single-precision floating point.
 * Use this for:
 *   - Debugging and validation
 *   - Systems without BF16 support
 *   - When numerical precision is critical
 *
 * Note: FP32 version is ~2-4x slower than BF16 on H100 due to reduced
 * Tensor Core throughput and higher memory bandwidth requirements.
 *
 * @param source_data Device pointer to input trajectories (FP32)
 * @param source_times Device pointer to source timestamps (FP32)
 * @param target_times Device pointer to target timestamps (FP32)
 * @param output_data Device pointer to output trajectories (FP32)
 * @param batch_size Number of trajectories to resample
 * @param source_length Number of frames in each source trajectory
 * @param target_length Number of frames in each output trajectory
 * @param action_dim Dimensionality of robot actions
 * @param stream CUDA stream for asynchronous execution (default: 0)
 * @return cudaSuccess on success, error code otherwise
 */
cudaError_t resample_trajectories_fp32(
    const float* source_data,
    const float* source_times,
    const float* target_times,
    float* output_data,
    int batch_size,
    int source_length,
    int target_length,
    int action_dim,
    cudaStream_t stream = 0
);

/**
 * Resample robot trajectories using FP16 precision
 *
 * Half-precision alternative to BF16. Generally not recommended for robot
 * learning due to narrower dynamic range, but can be useful for:
 *   - Inference on GPUs with strong FP16 support
 *   - Extremely memory-constrained scenarios
 *
 * @param source_data Device pointer to input trajectories (FP16)
 * @param source_times Device pointer to source timestamps (FP32)
 * @param target_times Device pointer to target timestamps (FP32)
 * @param output_data Device pointer to output trajectories (FP16)
 * @param batch_size Number of trajectories to resample
 * @param source_length Number of frames in each source trajectory
 * @param target_length Number of frames in each output trajectory
 * @param action_dim Dimensionality of robot actions
 * @param stream CUDA stream for asynchronous execution (default: 0)
 * @return cudaSuccess on success, error code otherwise
 */
cudaError_t resample_trajectories_fp16(
    const void* source_data,
    const float* source_times,
    const float* target_times,
    void* output_data,
    int batch_size,
    int source_length,
    int target_length,
    int action_dim,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace robocache
