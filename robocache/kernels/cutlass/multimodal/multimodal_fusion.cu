// multimodal_fusion.cu
// Production-grade multimodal data fusion for robot foundation models
// Optimized for NVIDIA H100 with CUDA 13.0 + CUTLASS 4.3.0
//
// Copyright (c) 2025 GOATnote Inc.
// SPDX-License-Identifier: Apache-2.0

#include "multimodal_fusion.h"
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

namespace cg = cooperative_groups;

namespace robocache {
namespace kernels {
namespace multimodal {

//==============================================================================
// Device Helper Functions
//==============================================================================

/// Device function: Linear interpolation
__device__ __forceinline__ float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

/// Device function: Clamp value to [0, 1]
__device__ __forceinline__ float clamp01(float x) {
    return fmaxf(0.0f, fminf(1.0f, x));
}

/// Device function: Binary search in sorted timestamp array
/// Returns the index of the largest element <= target
__device__ __forceinline__ int binary_search_timestamps(
    const float* __restrict__ timestamps,
    int length,
    float target
) {
    int left = 0;
    int right = length - 1;

    // Handle edge cases
    if (target <= timestamps[0]) return 0;
    if (target >= timestamps[right]) return right - 1;

    // Binary search
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (timestamps[mid] <= target) {
            left = mid;
        } else {
            right = mid;
        }
    }

    return left;
}

/// Device function: Warp-collaborative binary search
/// All threads in warp participate via voting
__device__ __forceinline__ int warp_binary_search_timestamps(
    const float* __restrict__ timestamps,
    int length,
    float target,
    cooperative_groups::thread_block_tile<32>& warp
) {
    int left = 0;
    int right = length - 1;

    // Handle edge cases
    if (target <= timestamps[0]) return 0;
    if (target >= timestamps[right]) return right - 1;

    // Warp-collaborative binary search
    // Maximum 10 iterations for up to 1024 elements (2^10)
    for (int iter = 0; iter < 10; iter++) {
        int mid = (left + right) / 2;

        if (right - left <= 1) break;

        float mid_time = timestamps[mid];
        bool go_right = (mid_time <= target);

        // Warp vote - if majority says go right, update left
        unsigned mask = warp.ballot(go_right);
        if (__popc(mask) > 16) {  // More than half voted true
            left = mid;
        } else {
            right = mid;
        }
    }

    return left;
}

//==============================================================================
// Standard Fusion Kernel
//==============================================================================

/// Standard multimodal fusion kernel with temporal alignment
/// Each block processes one (batch, time) pair
template<typename Element = __nv_bfloat16>
__global__ void multimodal_fusion_kernel(
    const Element* __restrict__ vision_features,
    const float* __restrict__ vision_timestamps,
    const Element* __restrict__ proprio_features,
    const float* __restrict__ proprio_timestamps,
    const Element* __restrict__ lang_embeddings,
    const float* __restrict__ target_timestamps,
    Element* __restrict__ output,
    int batch_size,
    int target_length,
    int vision_src_length,
    int vision_dim,
    int proprio_src_length,
    int proprio_dim,
    int lang_length,
    int lang_dim
) {
    int batch_idx = blockIdx.x;
    int time_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || time_idx >= target_length) return;

    // Shared memory for interpolated features
    extern __shared__ char smem[];
    Element* s_vision = reinterpret_cast<Element*>(smem);
    Element* s_proprio = s_vision + vision_dim;
    float* s_target_time = reinterpret_cast<float*>(s_proprio + proprio_dim);

    // Load target timestamp
    if (tid == 0) {
        s_target_time[0] = target_timestamps[batch_idx * target_length + time_idx];
    }
    __syncthreads();

    float target_time = s_target_time[0];

    // === Vision interpolation ===
    if (tid == 0) {
        const float* vis_times = vision_timestamps + batch_idx * vision_src_length;

        // Binary search
        int left = binary_search_timestamps(vis_times, vision_src_length, target_time);
        int right = min(left + 1, vision_src_length - 1);

        // Get interpolation weight
        float t_left = vis_times[left];
        float t_right = vis_times[right];
        float weight = (t_right > t_left) ?
            clamp01((target_time - t_left) / (t_right - t_left)) : 0.0f;

        // Interpolate vision features
        const Element* vis_left = vision_features +
            (batch_idx * vision_src_length + left) * vision_dim;
        const Element* vis_right = vision_features +
            (batch_idx * vision_src_length + right) * vision_dim;

        for (int d = 0; d < vision_dim; d++) {
            float v_l = __bfloat162float(vis_left[d]);
            float v_r = __bfloat162float(vis_right[d]);
            s_vision[d] = __float2bfloat16(lerp(v_l, v_r, weight));
        }
    }

    // === Proprioception interpolation (parallel) ===
    const float* prop_times = proprio_timestamps + batch_idx * proprio_src_length;

    for (int d = tid; d < proprio_dim; d += blockDim.x) {
        // Binary search (each thread independently)
        int left = binary_search_timestamps(prop_times, proprio_src_length, target_time);
        int right = min(left + 1, proprio_src_length - 1);

        float t_left = prop_times[left];
        float t_right = prop_times[right];
        float weight = (t_right > t_left) ?
            clamp01((target_time - t_left) / (t_right - t_left)) : 0.0f;

        const Element* prop_left = proprio_features +
            (batch_idx * proprio_src_length + left) * proprio_dim + d;
        const Element* prop_right = proprio_features +
            (batch_idx * proprio_src_length + right) * proprio_dim + d;

        float p_l = __bfloat162float(*prop_left);
        float p_r = __bfloat162float(*prop_right);
        s_proprio[d] = __float2bfloat16(lerp(p_l, p_r, weight));
    }

    __syncthreads();

    // === Write output (concatenate all modalities) ===
    int total_dim = vision_dim + proprio_dim + lang_dim;
    Element* out_ptr = output +
        (batch_idx * target_length + time_idx) * total_dim;

    // Copy vision features (coalesced)
    for (int d = tid; d < vision_dim; d += blockDim.x) {
        out_ptr[d] = s_vision[d];
    }

    // Copy proprioception features (coalesced)
    for (int d = tid; d < proprio_dim; d += blockDim.x) {
        out_ptr[vision_dim + d] = s_proprio[d];
    }

    // Copy language embeddings (average pooling across tokens)
    const Element* lang_ptr = lang_embeddings +
        batch_idx * lang_length * lang_dim;

    for (int d = tid; d < lang_dim; d += blockDim.x) {
        float sum = 0.0f;
        // Average pool language tokens
        for (int l = 0; l < lang_length; l++) {
            sum += __bfloat162float(lang_ptr[l * lang_dim + d]);
        }
        out_ptr[vision_dim + proprio_dim + d] =
            __float2bfloat16(sum / lang_length);
    }
}

//==============================================================================
// Optimized Fusion Kernel (Warp-Level)
//==============================================================================

/// Warp-optimized multimodal fusion kernel
/// Uses cooperative groups and warp-level primitives for maximum performance
template<typename Element = __nv_bfloat16>
__global__ void __launch_bounds__(256, 2)
multimodal_fusion_optimized_kernel(
    const Element* __restrict__ vision_features,
    const float* __restrict__ vision_timestamps,
    const Element* __restrict__ proprio_features,
    const float* __restrict__ proprio_timestamps,
    const Element* __restrict__ lang_embeddings,
    const float* __restrict__ target_timestamps,
    Element* __restrict__ output,
    int batch_size,
    int target_length,
    int vision_src_length,
    int vision_dim,
    int proprio_src_length,
    int proprio_dim,
    int lang_length,
    int lang_dim
) {
    // Cooperative groups
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int batch_idx = blockIdx.x;
    int time_idx = blockIdx.y;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (batch_idx >= batch_size || time_idx >= target_length) return;

    // Shared memory
    extern __shared__ char smem[];
    Element* s_vision = reinterpret_cast<Element*>(smem);
    Element* s_proprio = s_vision + vision_dim;
    float* s_timestamps = reinterpret_cast<float*>(s_proprio + proprio_dim);

    // Load target timestamp (broadcast to all threads)
    float target_time;
    if (threadIdx.x == 0) {
        s_timestamps[0] = target_timestamps[batch_idx * target_length + time_idx];
    }
    block.sync();
    target_time = s_timestamps[0];

    // === Warp 0: Process vision ===
    if (warp_id == 0) {
        const float* vis_times = vision_timestamps + batch_idx * vision_src_length;

        // Warp-collaborative binary search
        int left = warp_binary_search_timestamps(vis_times, vision_src_length,
                                                  target_time, warp);
        int right = min(left + 1, vision_src_length - 1);

        // Get timestamps and weight (all lanes have same values)
        float t_left = vis_times[left];
        float t_right = vis_times[right];
        float weight = (t_right > t_left) ?
            clamp01((target_time - t_left) / (t_right - t_left)) : 0.0f;

        // Interpolate vision features (vectorized across lanes)
        const Element* vis_left = vision_features +
            (batch_idx * vision_src_length + left) * vision_dim;
        const Element* vis_right = vision_features +
            (batch_idx * vision_src_length + right) * vision_dim;

        // Each lane processes multiple dimensions
        for (int d = lane_id; d < vision_dim; d += 32) {
            float v_l = __bfloat162float(vis_left[d]);
            float v_r = __bfloat162float(vis_right[d]);
            s_vision[d] = __float2bfloat16(lerp(v_l, v_r, weight));
        }
    }

    // === Warp 1: Process proprioception ===
    if (warp_id == 1) {
        const float* prop_times = proprio_timestamps + batch_idx * proprio_src_length;

        // Warp-collaborative binary search
        int left = warp_binary_search_timestamps(prop_times, proprio_src_length,
                                                  target_time, warp);
        int right = min(left + 1, proprio_src_length - 1);

        float t_left = prop_times[left];
        float t_right = prop_times[right];
        float weight = (t_right > t_left) ?
            clamp01((target_time - t_left) / (t_right - t_left)) : 0.0f;

        const Element* prop_left = proprio_features +
            (batch_idx * proprio_src_length + left) * proprio_dim;
        const Element* prop_right = proprio_features +
            (batch_idx * proprio_src_length + right) * proprio_dim;

        for (int d = lane_id; d < proprio_dim; d += 32) {
            float p_l = __bfloat162float(prop_left[d]);
            float p_r = __bfloat162float(prop_right[d]);
            s_proprio[d] = __float2bfloat16(lerp(p_l, p_r, weight));
        }
    }

    block.sync();

    // === All warps: Write output (coalesced) ===
    int total_dim = vision_dim + proprio_dim + lang_dim;
    Element* out_ptr = output +
        (batch_idx * target_length + time_idx) * total_dim;

    // Vision
    for (int d = threadIdx.x; d < vision_dim; d += blockDim.x) {
        out_ptr[d] = s_vision[d];
    }

    // Proprioception
    for (int d = threadIdx.x; d < proprio_dim; d += blockDim.x) {
        out_ptr[vision_dim + d] = s_proprio[d];
    }

    // Language (average pooling)
    const Element* lang_ptr = lang_embeddings +
        batch_idx * lang_length * lang_dim;

    for (int d = threadIdx.x; d < lang_dim; d += blockDim.x) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int l = 0; l < lang_length; l++) {
            sum += __bfloat162float(lang_ptr[l * lang_dim + d]);
        }
        out_ptr[vision_dim + proprio_dim + d] =
            __float2bfloat16(sum / lang_length);
    }
}

//==============================================================================
// Host Functions
//==============================================================================

cudaError_t fuse_multimodal_data(
    const void* vision_features,
    const float* vision_timestamps,
    const void* proprio_features,
    const float* proprio_timestamps,
    const void* lang_embeddings,
    const float* target_timestamps,
    void* output,
    const FusionConfig& config,
    cudaStream_t stream,
    FusionMetrics* metrics
) {
    // Validate configuration
    try {
        config.validate();
    } catch (const std::exception& e) {
        return cudaErrorInvalidValue;
    }

    // Validate inputs
    cudaError_t err = validate_inputs(
        vision_features, vision_timestamps,
        proprio_features, proprio_timestamps,
        lang_embeddings, target_timestamps,
        output, config
    );
    if (err != cudaSuccess) return err;

    // Setup timing if metrics requested
    cudaEvent_t start, stop;
    if (metrics) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
    }

    // Launch kernel
    dim3 grid(config.batch_size, config.target_seq_length);
    size_t smem_size = (config.vision_dim + config.proprio_dim) * sizeof(__nv_bfloat16) +
                       sizeof(float);

    if (config.use_optimized) {
        dim3 block(256);  // 8 warps

        multimodal_fusion_optimized_kernel<__nv_bfloat16><<<grid, block, smem_size, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(vision_features),
            vision_timestamps,
            reinterpret_cast<const __nv_bfloat16*>(proprio_features),
            proprio_timestamps,
            reinterpret_cast<const __nv_bfloat16*>(lang_embeddings),
            target_timestamps,
            reinterpret_cast<__nv_bfloat16*>(output),
            config.batch_size,
            config.target_seq_length,
            config.vision_src_length,
            config.vision_dim,
            config.proprio_src_length,
            config.proprio_dim,
            config.lang_length,
            config.lang_dim
        );
    } else {
        dim3 block(256);

        multimodal_fusion_kernel<__nv_bfloat16><<<grid, block, smem_size, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(vision_features),
            vision_timestamps,
            reinterpret_cast<const __nv_bfloat16*>(proprio_features),
            proprio_timestamps,
            reinterpret_cast<const __nv_bfloat16*>(lang_embeddings),
            target_timestamps,
            reinterpret_cast<__nv_bfloat16*>(output),
            config.batch_size,
            config.target_seq_length,
            config.vision_src_length,
            config.vision_dim,
            config.proprio_src_length,
            config.proprio_dim,
            config.lang_length,
            config.lang_dim
        );
    }

    CUDA_CHECK_LAST_ERROR();

    // Record metrics if requested
    if (metrics) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        metrics->kernel_time_ms = elapsed_ms;

        // Calculate memory bandwidth
        size_t bytes_read =
            config.batch_size * config.vision_src_length * config.vision_dim * sizeof(__nv_bfloat16) +
            config.batch_size * config.proprio_src_length * config.proprio_dim * sizeof(__nv_bfloat16) +
            config.batch_size * config.lang_length * config.lang_dim * sizeof(__nv_bfloat16) +
            config.batch_size * config.vision_src_length * sizeof(float) +
            config.batch_size * config.proprio_src_length * sizeof(float) +
            config.batch_size * config.target_seq_length * sizeof(float);

        size_t bytes_write = config.get_output_size_bytes();

        metrics->bytes_transferred = bytes_read + bytes_write;
        metrics->memory_bandwidth_gbs =
            (bytes_read + bytes_write) / (elapsed_ms / 1000.0) / 1e9;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    return cudaSuccess;
}

cudaDeviceProp get_device_properties() {
    cudaDeviceProp prop;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop;
}

bool check_gpu_compatibility(std::string* error_msg) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        if (error_msg) *error_msg = "Failed to get CUDA device";
        return false;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        if (error_msg) *error_msg = "Failed to get device properties";
        return false;
    }

    // Check compute capability (need at least 8.0 for Ampere)
    if (prop.major < 8) {
        if (error_msg) {
            std::stringstream ss;
            ss << "GPU compute capability " << prop.major << "." << prop.minor
               << " is too old. Need at least 8.0 (Ampere)";
            *error_msg = ss.str();
        }
        return false;
    }

    // Check for bfloat16 support
    // Ampere (8.x) and Hopper (9.x) support bfloat16
    if (prop.major < 8) {
        if (error_msg) *error_msg = "GPU does not support bfloat16";
        return false;
    }

    return true;
}

size_t estimate_shared_memory_bytes(const FusionConfig& config) {
    return (config.vision_dim + config.proprio_dim) * sizeof(__nv_bfloat16) +
           sizeof(float);
}

dim3 estimate_optimal_block_size(const FusionConfig& config) {
    return dim3(256);  // 8 warps is typically optimal for H100
}

dim3 estimate_optimal_grid_size(const FusionConfig& config) {
    return dim3(config.batch_size, config.target_seq_length);
}

cudaError_t validate_inputs(
    const void* vision_features,
    const float* vision_timestamps,
    const void* proprio_features,
    const float* proprio_timestamps,
    const void* lang_embeddings,
    const float* target_timestamps,
    void* output,
    const FusionConfig& config
) {
    // Check for null pointers
    if (!vision_features || !vision_timestamps ||
        !proprio_features || !proprio_timestamps ||
        !lang_embeddings || !target_timestamps || !output) {
        return cudaErrorInvalidValue;
    }

    // Check pointer alignment (should be at least 16-byte aligned for BF16)
    if (reinterpret_cast<uintptr_t>(vision_features) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(proprio_features) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(lang_embeddings) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(output) % 16 != 0) {
        return cudaErrorInvalidValue;
    }

    // Verify all pointers are device pointers
    cudaPointerAttributes attr;

    cudaError_t err = cudaPointerGetAttributes(&attr, vision_features);
    if (err != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        return cudaErrorInvalidValue;
    }

    err = cudaPointerGetAttributes(&attr, proprio_features);
    if (err != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        return cudaErrorInvalidValue;
    }

    err = cudaPointerGetAttributes(&attr, lang_embeddings);
    if (err != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        return cudaErrorInvalidValue;
    }

    err = cudaPointerGetAttributes(&attr, output);
    if (err != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

/// Kernel to check for NaN/Inf values
__global__ void check_nan_inf_kernel(
    const __nv_bfloat16* data,
    size_t num_elements,
    int* has_nan_inf
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_elements) {
        float val = __bfloat162float(data[idx]);
        if (isnan(val) || isinf(val)) {
            atomicAdd(has_nan_inf, 1);
        }
    }
}

cudaError_t check_numerical_stability(
    const void* output,
    size_t num_elements,
    cudaStream_t stream
) {
    int* d_has_nan_inf;
    CUDA_CHECK(cudaMalloc(&d_has_nan_inf, sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_has_nan_inf, 0, sizeof(int), stream));

    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);

    check_nan_inf_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(output),
        num_elements,
        d_has_nan_inf
    );

    CUDA_CHECK_LAST_ERROR();

    int h_has_nan_inf;
    CUDA_CHECK(cudaMemcpyAsync(&h_has_nan_inf, d_has_nan_inf, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_has_nan_inf));

    return (h_has_nan_inf > 0) ? cudaErrorInvalidValue : cudaSuccess;
}

} // namespace multimodal
} // namespace kernels
} // namespace robocache
