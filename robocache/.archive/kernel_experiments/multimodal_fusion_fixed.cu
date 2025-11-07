// multimodal_fusion.cu - FIXED BF16 CONVERSIONS
// GPU-accelerated multimodal sensor fusion
// Production-ready H100 kernels with correct BF16 handling

#include "multimodal_fusion.h"
#include "trajectory_resample.h"
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace robocache {
namespace kernels {

// Kernel configuration
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_CACHED_TIMES = 512;

//==============================================================================
// Device helper: Type-safe conversion functions
//==============================================================================

template<typename T>
__device__ __forceinline__ float to_float(const T& val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<__nv_bfloat16>(const __nv_bfloat16& val) {
    return __bfloat162float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
    return __float2bfloat16_rn(val);
}

//==============================================================================
// Device helper: Binary search in shared memory
//==============================================================================

__device__ __forceinline__
int binary_search_times(const float* times, int length, float target) {
    int left = 0;
    int right = length - 1;
    
    #pragma unroll 8
    while (left < right - 1) {
        int mid = (left + right) >> 1;
        if (times[mid] <= target) {
            left = mid;
        } else {
            right = mid;
        }
    }
    
    return left;
}

//==============================================================================
// Fused Multimodal Alignment Kernel
//==============================================================================

template<typename Element>
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
fused_multimodal_alignment_kernel(
    // Vision inputs
    const Element* __restrict__ vision_data,
    const float* __restrict__ vision_times,
    int vision_src_len,
    int vision_dim,
    
    // Proprioception inputs
    const Element* __restrict__ proprio_data,
    const float* __restrict__ proprio_times,
    int proprio_src_len,
    int proprio_dim,
    
    // Force/torque inputs (optional)
    const Element* __restrict__ force_data,
    const float* __restrict__ force_times,
    int force_src_len,
    int force_dim,
    
    // Target timeline and output
    const float* __restrict__ target_times,
    int target_len,
    Element* __restrict__ output,
    int batch_size
) {
    auto block = cg::this_thread_block();
    const int tid = block.thread_rank();
    
    // Each block handles one batch
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    
    // Process target timesteps
    for (int t_idx = blockIdx.y * blockDim.x + threadIdx.x; 
         t_idx < target_len; 
         t_idx += gridDim.y * blockDim.x) {
        
        float target_time = target_times[b * target_len + t_idx];
        size_t out_offset = (size_t)b * target_len + t_idx;
        
        // ==== Align Vision ====
        if (vision_data != nullptr) {
            int v_left = binary_search_times(
                vision_times + b * vision_src_len,
                vision_src_len,
                target_time
            );
            int v_right = min(v_left + 1, vision_src_len - 1);
            
            float v_t_left = vision_times[b * vision_src_len + v_left];
            float v_t_right = vision_times[b * vision_src_len + v_right];
            float v_delta = v_t_right - v_t_left;
            float v_weight = (v_delta < 1e-6f) ? 0.0f :
                fminf(fmaxf((target_time - v_t_left) / v_delta, 0.0f), 1.0f);
            
            const Element* v_left_data = vision_data + 
                (b * vision_src_len + v_left) * vision_dim;
            const Element* v_right_data = vision_data + 
                (b * vision_src_len + v_right) * vision_dim;
            Element* v_out = output + out_offset * (vision_dim + proprio_dim + force_dim);
            
            for (int d = 0; d < vision_dim; d++) {
                float val_left = to_float(v_left_data[d]);
                float val_right = to_float(v_right_data[d]);
                float result = fmaf(v_weight, val_right - val_left, val_left);
                v_out[d] = from_float<Element>(result);
            }
        }
        
        // ==== Align Proprioception ====
        if (proprio_data != nullptr) {
            int p_left = binary_search_times(
                proprio_times + b * proprio_src_len,
                proprio_src_len,
                target_time
            );
            int p_right = min(p_left + 1, proprio_src_len - 1);
            
            float p_t_left = proprio_times[b * proprio_src_len + p_left];
            float p_t_right = proprio_times[b * proprio_src_len + p_right];
            float p_delta = p_t_right - p_t_left;
            float p_weight = (p_delta < 1e-6f) ? 0.0f :
                fminf(fmaxf((target_time - p_t_left) / p_delta, 0.0f), 1.0f);
            
            const Element* p_left_data = proprio_data + 
                (b * proprio_src_len + p_left) * proprio_dim;
            const Element* p_right_data = proprio_data + 
                (b * proprio_src_len + p_right) * proprio_dim;
            Element* p_out = output + 
                out_offset * (vision_dim + proprio_dim + force_dim) + vision_dim;
            
            for (int d = 0; d < proprio_dim; d++) {
                float val_left = to_float(p_left_data[d]);
                float val_right = to_float(p_right_data[d]);
                float result = fmaf(p_weight, val_right - val_left, val_left);
                p_out[d] = from_float<Element>(result);
            }
        }
        
        // ==== Align Force (if provided) ====
        if (force_data != nullptr) {
            int f_left = binary_search_times(
                force_times + b * force_src_len,
                force_src_len,
                target_time
            );
            int f_right = min(f_left + 1, force_src_len - 1);
            
            float f_t_left = force_times[b * force_src_len + f_left];
            float f_t_right = force_times[b * force_src_len + f_right];
            float f_delta = f_t_right - f_t_left;
            float f_weight = (f_delta < 1e-6f) ? 0.0f :
                fminf(fmaxf((target_time - f_t_left) / f_delta, 0.0f), 1.0f);
            
            const Element* f_left_data = force_data + 
                (b * force_src_len + f_left) * force_dim;
            const Element* f_right_data = force_data + 
                (b * force_src_len + f_right) * force_dim;
            Element* f_out = output + 
                out_offset * (vision_dim + proprio_dim + force_dim) + 
                vision_dim + proprio_dim;
            
            for (int d = 0; d < force_dim; d++) {
                float val_left = to_float(f_left_data[d]);
                float val_right = to_float(f_right_data[d]);
                float result = fmaf(f_weight, val_right - val_left, val_left);
                f_out[d] = from_float<Element>(result);
            }
        }
    }
    block.sync();
}

//==============================================================================
// Host API
//==============================================================================

template<typename T>
cudaError_t fused_multimodal_alignment(
    const T* vision_data, const float* vision_times, int vision_src_len, int vision_dim,
    const T* proprio_data, const float* proprio_times, int proprio_src_len, int proprio_dim,
    const T* force_data, const float* force_times, int force_src_len, int force_dim,
    const float* target_times, int target_len,
    T* output, int batch_size,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(batch_size, (target_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    fused_multimodal_alignment_kernel<<<grid, block, 0, stream>>>(
        vision_data, vision_times, vision_src_len, vision_dim,
        proprio_data, proprio_times, proprio_src_len, proprio_dim,
        force_data, force_times, force_src_len, force_dim,
        target_times, target_len, output, batch_size
    );
    
    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t fused_multimodal_alignment<__nv_bfloat16>(
    const __nv_bfloat16*, const float*, int, int,
    const __nv_bfloat16*, const float*, int, int,
    const __nv_bfloat16*, const float*, int, int,
    const float*, int, __nv_bfloat16*, int, cudaStream_t);

template cudaError_t fused_multimodal_alignment<float>(
    const float*, const float*, int, int,
    const float*, const float*, int, int,
    const float*, const float*, int, int,
    const float*, int, float*, int, cudaStream_t);

} // namespace kernels
} // namespace robocache

