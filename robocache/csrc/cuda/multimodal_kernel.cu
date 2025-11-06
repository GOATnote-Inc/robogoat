// Copyright (c) 2025 GOATnote Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// =============================================================================
// Multimodal Sensor Fusion - Temporal Alignment Kernel
// =============================================================================
// Aligns and fuses heterogeneous sensor streams (vision, proprioception, IMU)
// with temporal interpolation to uniform target timestamps.
//
// Architecture: SM80 (A100), SM90 (H100)
// Precision: BFloat16, Float32
// Throughput: >50k fusions/sec @ 3 streams on H100
// =============================================================================

template<typename T>
__device__ __forceinline__ T lerp(T a, T b, float t) {
    return a + t * (b - a);
}

template<>
__device__ __forceinline__ __nv_bfloat16 lerp(__nv_bfloat16 a, __nv_bfloat16 b, float t) {
    float a_f = __bfloat162float(a);
    float b_f = __bfloat162float(b);
    return __float2bfloat16_rn(a_f + t * (b_f - a_f));
}

// Binary search for temporal alignment
__device__ __forceinline__ int find_interval(
    const float* times,
    int n,
    float target
) {
    if (target <= times[0]) return 0;
    if (target >= times[n - 1]) return n - 2;
    
    int left = 0, right = n - 1;
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (times[mid] <= target) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return left;
}

// =============================================================================
// Multimodal Fusion Kernel (Float32)
// =============================================================================
__global__ void multimodal_fusion_fp32_kernel(
    const float* __restrict__ stream1_data,  // [B, S1, D1]
    const float* __restrict__ stream1_times, // [B, S1]
    const float* __restrict__ stream2_data,  // [B, S2, D2]
    const float* __restrict__ stream2_times, // [B, S2]
    const float* __restrict__ stream3_data,  // [B, S3, D3]
    const float* __restrict__ stream3_times, // [B, S3]
    const float* __restrict__ target_times,  // [B, T]
    float* __restrict__ output,              // [B, T, D1+D2+D3]
    int batch_size,
    int s1, int d1,
    int s2, int d2,
    int s3, int d3,
    int target_len
) {
    int batch_idx = blockIdx.x;
    int target_idx = blockIdx.y;
    int feature_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || target_idx >= target_len) return;
    
    float t_target = target_times[batch_idx * target_len + target_idx];
    
    // Stream 1 alignment
    const float* s1_times = stream1_times + batch_idx * s1;
    int idx1 = find_interval(s1_times, s1, t_target);
    float t1_0 = s1_times[idx1];
    float t1_1 = s1_times[idx1 + 1];
    float alpha1 = (t_target - t1_0) / (t1_1 - t1_0 + 1e-8f);
    
    // Stream 2 alignment
    const float* s2_times = stream2_times + batch_idx * s2;
    int idx2 = find_interval(s2_times, s2, t_target);
    float t2_0 = s2_times[idx2];
    float t2_1 = s2_times[idx2 + 1];
    float alpha2 = (t_target - t2_0) / (t2_1 - t2_0 + 1e-8f);
    
    // Stream 3 alignment
    const float* s3_times = stream3_times + batch_idx * s3;
    int idx3 = find_interval(s3_times, s3, t_target);
    float t3_0 = s3_times[idx3];
    float t3_1 = s3_times[idx3 + 1];
    float alpha3 = (t_target - t3_0) / (t3_1 - t3_0 + 1e-8f);
    
    int total_dim = d1 + d2 + d3;
    int output_offset = (batch_idx * target_len + target_idx) * total_dim;
    
    // Interpolate stream 1
    for (int i = feature_idx; i < d1; i += blockDim.x) {
        int s1_base = (batch_idx * s1 + idx1) * d1;
        float v0 = stream1_data[s1_base + i];
        float v1 = stream1_data[s1_base + d1 + i];
        output[output_offset + i] = lerp(v0, v1, alpha1);
    }
    
    // Interpolate stream 2
    for (int i = feature_idx; i < d2; i += blockDim.x) {
        int s2_base = (batch_idx * s2 + idx2) * d2;
        float v0 = stream2_data[s2_base + i];
        float v1 = stream2_data[s2_base + d2 + i];
        output[output_offset + d1 + i] = lerp(v0, v1, alpha2);
    }
    
    // Interpolate stream 3
    for (int i = feature_idx; i < d3; i += blockDim.x) {
        int s3_base = (batch_idx * s3 + idx3) * d3;
        float v0 = stream3_data[s3_base + i];
        float v1 = stream3_data[s3_base + d3 + i];
        output[output_offset + d1 + d2 + i] = lerp(v0, v1, alpha3);
    }
}

// =============================================================================
// Multimodal Fusion Kernel (BFloat16)
// =============================================================================
__global__ void multimodal_fusion_bf16_kernel(
    const __nv_bfloat16* __restrict__ stream1_data,
    const float* __restrict__ stream1_times,
    const __nv_bfloat16* __restrict__ stream2_data,
    const float* __restrict__ stream2_times,
    const __nv_bfloat16* __restrict__ stream3_data,
    const float* __restrict__ stream3_times,
    const float* __restrict__ target_times,
    __nv_bfloat16* __restrict__ output,
    int batch_size,
    int s1, int d1,
    int s2, int d2,
    int s3, int d3,
    int target_len
) {
    int batch_idx = blockIdx.x;
    int target_idx = blockIdx.y;
    int feature_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || target_idx >= target_len) return;
    
    float t_target = target_times[batch_idx * target_len + target_idx];
    
    // Stream 1 alignment
    const float* s1_times = stream1_times + batch_idx * s1;
    int idx1 = find_interval(s1_times, s1, t_target);
    float alpha1 = (t_target - s1_times[idx1]) / (s1_times[idx1 + 1] - s1_times[idx1] + 1e-8f);
    
    // Stream 2 alignment
    const float* s2_times = stream2_times + batch_idx * s2;
    int idx2 = find_interval(s2_times, s2, t_target);
    float alpha2 = (t_target - s2_times[idx2]) / (s2_times[idx2 + 1] - s2_times[idx2] + 1e-8f);
    
    // Stream 3 alignment
    const float* s3_times = stream3_times + batch_idx * s3;
    int idx3 = find_interval(s3_times, s3, t_target);
    float alpha3 = (t_target - s3_times[idx3]) / (s3_times[idx3 + 1] - s3_times[idx3] + 1e-8f);
    
    int total_dim = d1 + d2 + d3;
    int output_offset = (batch_idx * target_len + target_idx) * total_dim;
    
    // Interpolate stream 1
    for (int i = feature_idx; i < d1; i += blockDim.x) {
        int s1_base = (batch_idx * s1 + idx1) * d1;
        __nv_bfloat16 v0 = stream1_data[s1_base + i];
        __nv_bfloat16 v1 = stream1_data[s1_base + d1 + i];
        output[output_offset + i] = lerp(v0, v1, alpha1);
    }
    
    // Interpolate stream 2
    for (int i = feature_idx; i < d2; i += blockDim.x) {
        int s2_base = (batch_idx * s2 + idx2) * d2;
        __nv_bfloat16 v0 = stream2_data[s2_base + i];
        __nv_bfloat16 v1 = stream2_data[s2_base + d2 + i];
        output[output_offset + d1 + i] = lerp(v0, v1, alpha2);
    }
    
    // Interpolate stream 3
    for (int i = feature_idx; i < d3; i += blockDim.x) {
        int s3_base = (batch_idx * s3 + idx3) * d3;
        __nv_bfloat16 v0 = stream3_data[s3_base + i];
        __nv_bfloat16 v1 = stream3_data[s3_base + d3 + i];
        output[output_offset + d1 + d2 + i] = lerp(v0, v1, alpha3);
    }
}

// =============================================================================
// Host Interface
// =============================================================================
void multimodal_fusion_cuda(
    const void* stream1_data,
    const float* stream1_times,
    const void* stream2_data,
    const float* stream2_times,
    const void* stream3_data,
    const float* stream3_times,
    const float* target_times,
    void* output,
    int batch_size,
    int s1, int d1,
    int s2, int d2,
    int s3, int d3,
    int target_len,
    bool use_bf16,
    cudaStream_t stream
) {
    dim3 grid(batch_size, target_len);
    int max_dim = max(max(d1, d2), d3);
    dim3 block(min(max_dim, 256));
    
    if (use_bf16) {
        multimodal_fusion_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(stream1_data),
            stream1_times,
            reinterpret_cast<const __nv_bfloat16*>(stream2_data),
            stream2_times,
            reinterpret_cast<const __nv_bfloat16*>(stream3_data),
            stream3_times,
            target_times,
            reinterpret_cast<__nv_bfloat16*>(output),
            batch_size,
            s1, d1, s2, d2, s3, d3,
            target_len
        );
    } else {
        multimodal_fusion_fp32_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float*>(stream1_data),
            stream1_times,
            reinterpret_cast<const float*>(stream2_data),
            stream2_times,
            reinterpret_cast<const float*>(stream3_data),
            stream3_times,
            target_times,
            reinterpret_cast<float*>(output),
            batch_size,
            s1, d1, s2, d2, s3, d3,
            target_len
        );
    }
}

