// Hopper-optimized trajectory resampling with warp-level primitives
// Target: 22x-500x speedup over PyTorch baseline
// Optimizations: Warp-cooperative search, persistent threads, vectorized loads

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
constexpr int MAX_SHARED_TIMESTAMPS = 4096;

//==============================================================================
// Warp-Cooperative Binary Search
// All threads in warp cooperate to find search result, then broadcast
//==============================================================================
__device__ __forceinline__ void warp_cooperative_binary_search(
    float target_time,
    const float* __restrict__ s_source_times,
    int source_length,
    int& left_idx,
    int& right_idx,
    float& weight,
    const cg::thread_block_tile<32>& warp
) {
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
    
    // Warp-cooperative binary search using ballot + shfl
    int left = 0;
    int right = source_length - 1;
    
    while (right - left > 1) {
        int mid = (left + right) >> 1;
        
        // All threads in warp check the same midpoint
        float mid_time = s_source_times[mid];
        bool predicate = (mid_time <= target_time);
        
        // Use ballot to get consensus (all threads should have same result)
        unsigned mask = warp.ballot(predicate);
        
        // Update search bounds based on consensus
        if (mask) {
            left = mid;
        } else {
            right = mid;
        }
    }
    
    left_idx = left;
    right_idx = right;
    
    // Compute interpolation weight
    float t_left = s_source_times[left];
    float t_right = s_source_times[right];
    float delta = t_right - t_left;
    
    if (delta > 1e-6f) {
        weight = __fdividef(target_time - t_left, delta);  // Fast division
        weight = fminf(fmaxf(weight, 0.0f), 1.0f);
    } else {
        weight = 0.0f;
    }
}

//==============================================================================
// Hopper-Optimized Persistent Kernel
// Uses persistent thread blocks to amortize launch overhead
//==============================================================================
template<typename Element = __nv_bfloat16>
__global__ void __launch_bounds__(BLOCK_SIZE, 2)  // Max 2 blocks per SM for high occupancy
trajectory_resample_hopper_persistent(
    const Element* __restrict__ source_data,
    const float* __restrict__ source_times,
    const float* __restrict__ target_times,
    Element* __restrict__ output_data,
    int batch_size,
    int source_length,
    int target_length,
    int action_dim,
    int total_work
) {
    // Shared memory for timestamps - cooperatively loaded once
    __shared__ float s_source_times[MAX_SHARED_TIMESTAMPS];
    
    // Cooperative groups
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Persistent thread block: Process multiple work items
    for (int work_id = blockIdx.x; work_id < total_work; work_id += gridDim.x) {
        // Decode work_id into (batch_idx, target_idx)
        int batch_idx = work_id / target_length;
        int target_idx = work_id % target_length;
        
        if (batch_idx >= batch_size) break;
        
        // Cooperatively load source timestamps to shared memory (once per batch)
        if (target_idx == 0 || blockIdx.x == work_id) {
            const float* batch_source_times = source_times + batch_idx * source_length;
            for (int i = tid; i < source_length; i += BLOCK_SIZE) {
                s_source_times[i] = batch_source_times[i];
            }
        }
        block.sync();
        
        // Load target time for this work item
        float target_time = target_times[batch_idx * target_length + target_idx];
        
        // Warp-level binary search (all warps do the same search, but we'll optimize this)
        int left_idx, right_idx;
        float weight;
        
        if (warp_id == 0 && lane_id == 0) {
            // Only one thread does the search, then broadcasts
            int left = 0, right = source_length - 1;
            
            if (target_time <= s_source_times[0]) {
                left_idx = 0; right_idx = 0; weight = 0.0f;
            } else if (target_time >= s_source_times[source_length - 1]) {
                left_idx = source_length - 1; right_idx = source_length - 1; weight = 0.0f;
            } else {
                // Binary search
                while (right - left > 1) {
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
                weight = (delta > 1e-6f) ? __fdividef(target_time - t_left, delta) : 0.0f;
                weight = fminf(fmaxf(weight, 0.0f), 1.0f);
            }
        }
        
        // Broadcast search results to all threads in block using shared memory
        __shared__ int s_left_idx, s_right_idx;
        __shared__ float s_weight;
        
        if (tid == 0) {
            s_left_idx = left_idx;
            s_right_idx = right_idx;
            s_weight = weight;
        }
        block.sync();
        
        left_idx = s_left_idx;
        right_idx = s_right_idx;
        weight = s_weight;
        
        // Compute base pointers
        const Element* batch_source = source_data + batch_idx * source_length * action_dim;
        Element* batch_output = output_data + batch_idx * target_length * action_dim;
        
        const Element* left_frame = batch_source + left_idx * action_dim;
        const Element* right_frame = batch_source + right_idx * action_dim;
        Element* output_frame = batch_output + target_idx * action_dim;
        
        // Vectorized interpolation - all threads cooperate
        constexpr int VEC_SIZE = 8;  // BF16 elements per float4
        int num_vec = action_dim / VEC_SIZE;
        
        using Vec = float4;
        const Vec* left_vec = reinterpret_cast<const Vec*>(left_frame);
        const Vec* right_vec = reinterpret_cast<const Vec*>(right_frame);
        Vec* output_vec = reinterpret_cast<Vec*>(output_frame);
        
        // Each thread processes multiple vectors with stride
        for (int v = tid; v < num_vec; v += BLOCK_SIZE) {
            Vec left_val = left_vec[v];
            Vec right_val = right_vec[v];
            
            __nv_bfloat162* left_bf16 = reinterpret_cast<__nv_bfloat162*>(&left_val);
            __nv_bfloat162* right_bf16 = reinterpret_cast<__nv_bfloat162*>(&right_val);
            __nv_bfloat162 result_bf16[4];
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 left_f = __bfloat1622float2(left_bf16[i]);
                float2 right_f = __bfloat1622float2(right_bf16[i]);
                
                // Fused multiply-add for interpolation
                float2 result_f;
                result_f.x = __fmaf_rn(weight, right_f.x - left_f.x, left_f.x);
                result_f.y = __fmaf_rn(weight, right_f.y - left_f.y, left_f.y);
                
                result_bf16[i] = __float22bfloat162_rn(result_f);
            }
            
            Vec* result_vec = reinterpret_cast<Vec*>(result_bf16);
            output_vec[v] = *result_vec;
        }
        
        // Handle remainder
        int remainder_start = num_vec * VEC_SIZE;
        for (int dim = remainder_start + tid; dim < action_dim; dim += BLOCK_SIZE) {
            float left_f = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(left_frame)[dim]);
            float right_f = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(right_frame)[dim]);
            float result_f = __fmaf_rn(weight, right_f - left_f, left_f);
            output_frame[dim] = __float2bfloat16(result_f);
        }
        
        block.sync();
    }
}

//==============================================================================
// Host Interface
//==============================================================================
extern "C" cudaError_t launch_trajectory_resample_hopper(
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
    
    // Use persistent thread blocks
    int total_work = batch_size * target_length;
    int num_blocks = min(total_work, 1024);  // Limit to 1024 blocks for persistence
    
    trajectory_resample_hopper_persistent<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(source_data),
        source_times,
        target_times,
        reinterpret_cast<__nv_bfloat16*>(output_data),
        batch_size,
        source_length,
        target_length,
        action_dim,
        total_work
    );
    
    return cudaGetLastError();
}

