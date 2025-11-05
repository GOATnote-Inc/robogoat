// Optimized trajectory resampling kernel - Phase 1+2: Shared Memory + Vectorization
#include <cuda_runtime.h>
#include <cuda_bf16.h>

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_SHARED_TIMESTAMPS = 4096;  // 16KB for float timestamps

__device__ __forceinline__ void compute_interpolation_weights_shared(
    float target_time,
    const float* __restrict__ s_source_times,
    int source_length,
    int& left_idx,
    int& right_idx,
    float& weight
) {
    // Binary search on shared memory (much faster than global)
    int left = 0;
    int right = source_length - 1;
    
    // Handle edge cases
    if (target_time <= s_source_times[0]) {
        left_idx = 0;
        right_idx = 0;
        weight = 0.0f;
        return;
    }
    if (target_time >= s_source_times[source_length - 1]) {
        left_idx = source_length - 1;
        right_idx = source_length - 1;
        weight = 0.0f;
        return;
    }
    
    // Binary search
    while (left < right - 1) {
        int mid = (left + right) >> 1;
        if (s_source_times[mid] <= target_time) {
            left = mid;
        } else {
            right = mid;
        }
    }
    
    left_idx = left;
    right_idx = right;
    
    float t_left = s_source_times[left];
    float t_right = s_source_times[right];
    float delta = t_right - t_left;
    
    if (delta > 1e-6f) {
        weight = (target_time - t_left) / delta;
        weight = fminf(fmaxf(weight, 0.0f), 1.0f);
    } else {
        weight = 0.0f;
    }
}

// Optimized kernel with shared memory caching
template<typename Element = __nv_bfloat16>
__global__ void trajectory_resample_optimized_kernel(
    const Element* __restrict__ source_data,
    const float* __restrict__ source_times,
    const float* __restrict__ target_times,
    Element* __restrict__ output_data,
    int batch_size,
    int source_length,
    int target_length,
    int action_dim
) {
    // Shared memory for timestamps (cooperative loading)
    __shared__ float s_source_times[MAX_SHARED_TIMESTAMPS];
    __shared__ float s_target_time;
    __shared__ int s_left_idx;
    __shared__ int s_right_idx;
    __shared__ float s_weight;
    
    int batch_idx = blockIdx.x;
    int target_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || target_idx >= target_length) return;
    
    // Cooperative loading of source_times to shared memory
    const float* batch_source_times = source_times + batch_idx * source_length;
    for (int i = tid; i < source_length; i += BLOCK_SIZE) {
        s_source_times[i] = batch_source_times[i];
    }
    __syncthreads();
    
    // Single thread computes interpolation weights using shared memory
    if (tid == 0) {
        s_target_time = target_times[batch_idx * target_length + target_idx];
        compute_interpolation_weights_shared(
            s_target_time,
            s_source_times,
            source_length,
            s_left_idx,
            s_right_idx,
            s_weight
        );
    }
    __syncthreads();
    
    // Load parameters
    int left_idx = s_left_idx;
    int right_idx = s_right_idx;
    float weight = s_weight;
    float inv_weight = 1.0f - weight;
    
    // Compute base pointers
    const Element* batch_source = source_data + batch_idx * source_length * action_dim;
    Element* batch_output = output_data + batch_idx * target_length * action_dim;
    
    const Element* left_frame = batch_source + left_idx * action_dim;
    const Element* right_frame = batch_source + right_idx * action_dim;
    Element* output_frame = batch_output + target_idx * action_dim;
    
    // Vectorized processing with float4 (16 bytes = 8 BF16 values)
    constexpr int VEC_SIZE = 8;  // BF16 elements per float4
    int num_vec = action_dim / VEC_SIZE;
    int remainder = action_dim % VEC_SIZE;
    
    // Process vectorized portion
    using Vec = float4;
    const Vec* left_vec = reinterpret_cast<const Vec*>(left_frame);
    const Vec* right_vec = reinterpret_cast<const Vec*>(right_frame);
    Vec* output_vec = reinterpret_cast<Vec*>(output_frame);
    
    for (int v = tid; v < num_vec; v += BLOCK_SIZE) {
        Vec left_val = left_vec[v];
        Vec right_val = right_vec[v];
        
        // Cast to __nv_bfloat162 for efficient processing
        __nv_bfloat162* left_bf16 = reinterpret_cast<__nv_bfloat162*>(&left_val);
        __nv_bfloat162* right_bf16 = reinterpret_cast<__nv_bfloat162*>(&right_val);
        __nv_bfloat162 result_bf16[4];
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 left_f = __bfloat1622float2(left_bf16[i]);
            float2 right_f = __bfloat1622float2(right_bf16[i]);
            
            float2 result_f;
            result_f.x = fmaf(weight, right_f.x - left_f.x, left_f.x);
            result_f.y = fmaf(weight, right_f.y - left_f.y, left_f.y);
            
            result_bf16[i] = __float22bfloat162_rn(result_f);
        }
        
        Vec* result_vec = reinterpret_cast<Vec*>(result_bf16);
        output_vec[v] = *result_vec;
    }
    
    // Handle remainder elements
    int remainder_start = num_vec * VEC_SIZE;
    for (int dim = remainder_start + tid; dim < action_dim; dim += BLOCK_SIZE) {
        float left_f = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(left_frame)[dim]);
        float right_f = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(right_frame)[dim]);
        float result_f = fmaf(weight, right_f - left_f, left_f);
        output_frame[dim] = __float2bfloat16(result_f);
    }
}

// Host interface
extern "C" cudaError_t launch_trajectory_resample_optimized(
    const void* source_data,
    const float* source_times,
    const float* target_times,
    void* output_data,
    int batch_size,
    int source_length,
    int target_length,
    int action_dim,
    cudaStream_t stream
) {
    if (source_length > MAX_SHARED_TIMESTAMPS) {
        return cudaErrorInvalidValue;
    }
    
    dim3 grid(batch_size, target_length);
    dim3 block(BLOCK_SIZE);
    
    trajectory_resample_optimized_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(source_data),
        source_times,
        target_times,
        reinterpret_cast<__nv_bfloat16*>(output_data),
        batch_size,
        source_length,
        target_length,
        action_dim
    );
    
    return cudaGetLastError();
}

