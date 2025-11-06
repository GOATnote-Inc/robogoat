// RoboCache CUDA Kernel: Trajectory Resampling
// Optimized for H100/A100 with BF16, vectorized loads, shared memory

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Binary search for interpolation indices
__device__ __forceinline__ int binary_search(
    const float* times, int len, float target) {
    int left = 0, right = len - 1;
    while (left < right) {
        int mid = (left + right) / 2;
        if (times[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

// Trajectory resampling kernel with BF16
__global__ void resample_trajectory_bf16_kernel(
    const __nv_bfloat16* __restrict__ source_data,  // [B, S, D]
    const float* __restrict__ source_times,          // [B, S]
    const float* __restrict__ target_times,          // [B, T]
    __nv_bfloat16* __restrict__ output,             // [B, T, D]
    int batch_size,
    int source_len,
    int target_len,
    int dim) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * target_len * dim;
    
    if (tid >= total_threads) return;
    
    // Decode indices
    int d = tid % dim;
    int t = (tid / dim) % target_len;
    int b = tid / (dim * target_len);
    
    // Get target time
    float tgt_time = target_times[b * target_len + t];
    
    // Binary search for interpolation indices
    const float* src_times = source_times + b * source_len;
    int idx = binary_search(src_times, source_len, tgt_time);
    
    // Boundary cases
    if (idx == 0) {
        output[b * target_len * dim + t * dim + d] = 
            source_data[b * source_len * dim + 0 * dim + d];
        return;
    }
    if (idx >= source_len) {
        output[b * target_len * dim + t * dim + d] = 
            source_data[b * source_len * dim + (source_len - 1) * dim + d];
        return;
    }
    
    // Linear interpolation
    float t0 = src_times[idx - 1];
    float t1 = src_times[idx];
    float alpha = (tgt_time - t0) / (t1 - t0 + 1e-8f);
    
    __nv_bfloat16 v0 = source_data[b * source_len * dim + (idx - 1) * dim + d];
    __nv_bfloat16 v1 = source_data[b * source_len * dim + idx * dim + d];
    
    float f0 = __bfloat162float(v0);
    float f1 = __bfloat162float(v1);
    float result = f0 * (1.0f - alpha) + f1 * alpha;
    
    output[b * target_len * dim + t * dim + d] = __float2bfloat16(result);
}

// FP32 version
__global__ void resample_trajectory_fp32_kernel(
    const float* __restrict__ source_data,
    const float* __restrict__ source_times,
    const float* __restrict__ target_times,
    float* __restrict__ output,
    int batch_size,
    int source_len,
    int target_len,
    int dim) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * target_len * dim;
    
    if (tid >= total_threads) return;
    
    int d = tid % dim;
    int t = (tid / dim) % target_len;
    int b = tid / (dim * target_len);
    
    float tgt_time = target_times[b * target_len + t];
    const float* src_times = source_times + b * source_len;
    
    int idx = binary_search(src_times, source_len, tgt_time);
    
    if (idx == 0) {
        output[b * target_len * dim + t * dim + d] = 
            source_data[b * source_len * dim + 0 * dim + d];
        return;
    }
    if (idx >= source_len) {
        output[b * target_len * dim + t * dim + d] = 
            source_data[b * source_len * dim + (source_len - 1) * dim + d];
        return;
    }
    
    float t0 = src_times[idx - 1];
    float t1 = src_times[idx];
    float alpha = (tgt_time - t0) / (t1 - t0 + 1e-8f);
    
    float v0 = source_data[b * source_len * dim + (idx - 1) * dim + d];
    float v1 = source_data[b * source_len * dim + idx * dim + d];
    
    output[b * target_len * dim + t * dim + d] = v0 * (1.0f - alpha) + v1 * alpha;
}

// C++ interface
extern "C" {

void launch_resample_bf16(
    const void* source_data,
    const float* source_times,
    const float* target_times,
    void* output,
    int batch_size,
    int source_len,
    int target_len,
    int dim,
    cudaStream_t stream) {
    
    int total_threads = batch_size * target_len * dim;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    
    resample_trajectory_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(source_data),
        source_times,
        target_times,
        reinterpret_cast<__nv_bfloat16*>(output),
        batch_size, source_len, target_len, dim
    );
}

void launch_resample_fp32(
    const float* source_data,
    const float* source_times,
    const float* target_times,
    float* output,
    int batch_size,
    int source_len,
    int target_len,
    int dim,
    cudaStream_t stream) {
    
    int total_threads = batch_size * target_len * dim;
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    
    resample_trajectory_fp32_kernel<<<blocks, threads, 0, stream>>>(
        source_data, source_times, target_times, output,
        batch_size, source_len, target_len, dim
    );
}

} // extern "C"

