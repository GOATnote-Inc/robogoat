// multimodal_fusion.h
// Production-grade multimodal data fusion for robot foundation models
// Optimized for NVIDIA H100 with CUDA 13.0 + CUTLASS 4.3.0
//
// Copyright (c) 2025 GOATnote Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>
#include <sstream>

namespace robocache {
namespace kernels {
namespace multimodal {

//==============================================================================
// Error Checking Macros
//==============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
               << cudaGetErrorString(error); \
            throw std::runtime_error(ss.str()); \
        } \
    } while(0)

#define CUDA_CHECK_LAST_ERROR() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
               << cudaGetErrorString(error); \
            throw std::runtime_error(ss.str()); \
        } \
    } while(0)

//==============================================================================
// Configuration Structures
//==============================================================================

/// Configuration for multimodal fusion
struct FusionConfig {
    // Batch parameters
    int batch_size;
    int target_seq_length;  // Output sequence length

    // Vision modality
    int vision_src_length;  // Source sequence length (e.g., 30 for 30Hz)
    int vision_dim;         // Feature dimension (e.g., 256)

    // Proprioception modality
    int proprio_src_length; // Source sequence length (e.g., 100 for 100Hz)
    int proprio_dim;        // Feature dimension (e.g., 64)

    // Language modality
    int lang_length;        // Number of tokens (e.g., 77)
    int lang_dim;           // Embedding dimension (e.g., 512)

    // Derived
    int total_dim;          // vision_dim + proprio_dim + lang_dim

    // Options
    bool use_optimized;     // Use warp-optimized kernel
    bool enable_profiling;  // Add profiling markers

    FusionConfig()
        : batch_size(0), target_seq_length(0),
          vision_src_length(0), vision_dim(0),
          proprio_src_length(0), proprio_dim(0),
          lang_length(0), lang_dim(0), total_dim(0),
          use_optimized(true), enable_profiling(false) {}

    void validate() const {
        if (batch_size <= 0) throw std::invalid_argument("batch_size must be > 0");
        if (target_seq_length <= 0) throw std::invalid_argument("target_seq_length must be > 0");
        if (vision_src_length <= 0) throw std::invalid_argument("vision_src_length must be > 0");
        if (vision_dim <= 0) throw std::invalid_argument("vision_dim must be > 0");
        if (proprio_src_length <= 0) throw std::invalid_argument("proprio_src_length must be > 0");
        if (proprio_dim <= 0) throw std::invalid_argument("proprio_dim must be > 0");
        if (lang_length <= 0) throw std::invalid_argument("lang_length must be > 0");
        if (lang_dim <= 0) throw std::invalid_argument("lang_dim must be > 0");

        // Reasonable limits
        if (batch_size > 4096) throw std::invalid_argument("batch_size too large");
        if (target_seq_length > 2048) throw std::invalid_argument("target_seq_length too large");
        if (vision_dim > 2048) throw std::invalid_argument("vision_dim too large");
        if (proprio_dim > 1024) throw std::invalid_argument("proprio_dim too large");
        if (lang_dim > 4096) throw std::invalid_argument("lang_dim too large");
    }

    size_t get_output_size_bytes() const {
        return (size_t)batch_size * target_seq_length * total_dim * sizeof(__nv_bfloat16);
    }
};

//==============================================================================
// Performance Metrics
//==============================================================================

struct FusionMetrics {
    double kernel_time_ms;
    double memory_bandwidth_gbs;
    double achieved_occupancy;
    size_t bytes_transferred;

    FusionMetrics()
        : kernel_time_ms(0.0), memory_bandwidth_gbs(0.0),
          achieved_occupancy(0.0), bytes_transferred(0) {}
};

//==============================================================================
// Main API
//==============================================================================

/// Fuse multimodal data (vision, proprioception, language) with temporal alignment
///
/// All input tensors must be in BF16 format and reside on GPU.
/// Timestamps are in float32 and represent time in seconds.
///
/// @param vision_features     [batch, vision_src_length, vision_dim] BF16
/// @param vision_timestamps   [batch, vision_src_length] FP32
/// @param proprio_features    [batch, proprio_src_length, proprio_dim] BF16
/// @param proprio_timestamps  [batch, proprio_src_length] FP32
/// @param lang_embeddings     [batch, lang_length, lang_dim] BF16
/// @param target_timestamps   [batch, target_seq_length] FP32
/// @param output              [batch, target_seq_length, total_dim] BF16 (output)
/// @param config              Fusion configuration
/// @param stream              CUDA stream (0 = default stream)
/// @param metrics             Optional performance metrics output
///
/// @return cudaSuccess on success, error code otherwise
cudaError_t fuse_multimodal_data(
    const void* vision_features,
    const float* vision_timestamps,
    const void* proprio_features,
    const float* proprio_timestamps,
    const void* lang_embeddings,
    const float* target_timestamps,
    void* output,
    const FusionConfig& config,
    cudaStream_t stream = 0,
    FusionMetrics* metrics = nullptr
);

//==============================================================================
// Helper Functions
//==============================================================================

/// Get device properties for current device
cudaDeviceProp get_device_properties();

/// Check if current GPU supports required features
bool check_gpu_compatibility(std::string* error_msg = nullptr);

/// Estimate shared memory requirements
size_t estimate_shared_memory_bytes(const FusionConfig& config);

/// Estimate optimal block size
dim3 estimate_optimal_block_size(const FusionConfig& config);

/// Estimate optimal grid size
dim3 estimate_optimal_grid_size(const FusionConfig& config);

//==============================================================================
// Validation Functions
//==============================================================================

/// Validate input tensor dimensions and alignment
cudaError_t validate_inputs(
    const void* vision_features,
    const float* vision_timestamps,
    const void* proprio_features,
    const float* proprio_timestamps,
    const void* lang_embeddings,
    const float* target_timestamps,
    void* output,
    const FusionConfig& config
);

/// Check for NaN/Inf in output (debugging)
cudaError_t check_numerical_stability(
    const void* output,
    size_t num_elements,
    cudaStream_t stream = 0
);

} // namespace multimodal
} // namespace kernels
} // namespace robocache
