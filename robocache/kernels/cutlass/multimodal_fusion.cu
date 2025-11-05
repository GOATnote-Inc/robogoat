// multimodal_fusion.cu
// GPU-accelerated multimodal sensor fusion
// Phase 2 implementation - Production-ready H100 kernels

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
    // Force inputs
    const Element* __restrict__ force_data,
    const float* __restrict__ force_times,
    int force_src_len,
    int force_dim,
    // Target timestamps
    const float* __restrict__ target_times,
    int target_len,
    // Output
    Element* __restrict__ output,
    int batch_size
) {
    auto block = cg::this_thread_block();
    int tid = threadIdx.x;
    
    // Shared memory for target times (reused across all modalities)
    __shared__ float s_target_times[MAX_CACHED_TIMES];
    
    // Cache target times (shared across vision/proprio/force)
    int times_to_cache = min(target_len, MAX_CACHED_TIMES);
    
    // Process batches persistently
    for (int b = blockIdx.x; b < batch_size; b += gridDim.x) {
        // Load target times for this batch
        for (int i = tid; i < times_to_cache; i += BLOCK_SIZE) {
            s_target_times[i] = target_times[b * target_len + i];
        }
        block.sync();
        
        // Process each target time
        for (int t = tid; t < target_len; t += BLOCK_SIZE) {
            float target_time = s_target_times[t];
            int out_offset = b * target_len + t;
            
            // ==== Align Vision ====
            if (vision_data != nullptr) {
                // Binary search for vision timestamps
                int v_left = binary_search_times(
                    vision_times + b * vision_src_len,
                    vision_src_len,
                    target_time
                );
                int v_right = min(v_left + 1, vision_src_len - 1);
                
                // Compute interpolation weight
                float v_t_left = vision_times[b * vision_src_len + v_left];
                float v_t_right = vision_times[b * vision_src_len + v_right];
                float v_delta = v_t_right - v_t_left;
                float v_weight = (v_delta < 1e-6f) ? 0.0f :
                    fminf(fmaxf((target_time - v_t_left) / v_delta, 0.0f), 1.0f);
                
                // Interpolate vision features
                const Element* v_left_data = vision_data + 
                    (b * vision_src_len + v_left) * vision_dim;
                const Element* v_right_data = vision_data + 
                    (b * vision_src_len + v_right) * vision_dim;
                Element* v_out = output + out_offset * (vision_dim + proprio_dim + force_dim);
                
                for (int d = 0; d < vision_dim; d++) {
                    float val_left = (sizeof(Element) == 2) ? __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(v_left_data[d])) : static_cast<float>(v_left_data[d]);
                    float val_right = (sizeof(Element) == 2) ? __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(v_right_data[d])) : static_cast<float>(v_right_data[d]);
                    float result = fmaf(v_weight, val_right - val_left, val_left);
                    v_out[d] = (sizeof(Element) == 2) ? reinterpret_cast<Element&>(__float2bfloat16(result)) : static_cast<Element>(result);
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
                    float val_left = (sizeof(Element) == 2) ? __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(p_left_data[d])) : static_cast<float>(p_left_data[d]);
                    float val_right = (sizeof(Element) == 2) ? __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(p_right_data[d])) : static_cast<float>(p_right_data[d]);
                    float result = fmaf(p_weight, val_right - val_left, val_left);
                    p_out[d] = (sizeof(Element) == 2) ? reinterpret_cast<Element&>(__float2bfloat16(result)) : static_cast<Element>(result);
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
                    float val_left = (sizeof(Element) == 2) ? __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(f_left_data[d])) : static_cast<float>(f_left_data[d]);
                    float val_right = (sizeof(Element) == 2) ? __bfloat162float(reinterpret_cast<const __nv_bfloat16&>(f_right_data[d])) : static_cast<float>(f_right_data[d]);
                    float result = fmaf(f_weight, val_right - val_left, val_left);
                    f_out[d] = (sizeof(Element) == 2) ? reinterpret_cast<Element&>(__float2bfloat16(result)) : static_cast<Element>(result);
                }
            }
        }
        block.sync();
    }
}

//==============================================================================
// Host API
//==============================================================================

template<typename T>
cudaError_t fused_multimodal_alignment(
    const T* vision_data,
    const float* vision_times,
    int vision_src_len,
    int vision_dim,
    const T* proprio_data,
    const float* proprio_times,
    int proprio_src_len,
    int proprio_dim,
    const T* force_data,
    const float* force_times,
    int force_src_len,
    int force_dim,
    const float* target_times,
    int target_len,
    T* output,
    int batch_size,
    cudaStream_t stream
) {
    // Persistent kernel: use SM count * 2-4 blocks
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int num_blocks = prop.multiProcessorCount * 2;
    
    dim3 grid(num_blocks);
    dim3 block(BLOCK_SIZE);
    
    fused_multimodal_alignment_kernel<T><<<grid, block, 0, stream>>>(
        vision_data, vision_times, vision_src_len, vision_dim,
        proprio_data, proprio_times, proprio_src_len, proprio_dim,
        force_data, force_times, force_src_len, force_dim,
        target_times, target_len,
        output, batch_size
    );
    
    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t fused_multimodal_alignment<float>(
    const float*, const float*, int, int,
    const float*, const float*, int, int,
    const float*, const float*, int, int,
    const float*, int, float*, int, cudaStream_t);

template cudaError_t fused_multimodal_alignment<__nv_bfloat16>(
    const __nv_bfloat16*, const float*, int, int,
    const __nv_bfloat16*, const float*, int, int,
    const __nv_bfloat16*, const float*, int, int,
    const float*, int, __nv_bfloat16*, int, cudaStream_t);

//==============================================================================
// Individual stream alignment kernels (for flexibility)
//==============================================================================

template<typename T>
cudaError_t align_vision_stream(
    const T* vision_data,
    const float* vision_times,
    const float* target_times,
    T* output,
    int batch_size,
    int vision_src_len,
    int target_len,
    int vision_dim,
    cudaStream_t stream
) {
    // Reuse trajectory resampling kernel (same algorithm)
    return robocache::kernels::resample_trajectories_fp32(
        reinterpret_cast<const float*>(vision_data),
        vision_times,
        target_times,
        reinterpret_cast<float*>(output),
        batch_size, vision_src_len, target_len, vision_dim,
        stream
    );
}

template<typename T>
cudaError_t align_proprio_stream(
    const T* proprio_data,
    const float* proprio_times,
    const float* target_times,
    T* output,
    int batch_size,
    int proprio_src_len,
    int target_len,
    int proprio_dim,
    cudaStream_t stream
) {
    // Reuse trajectory resampling kernel
    return robocache::kernels::resample_trajectories_fp32(
        reinterpret_cast<const float*>(proprio_data),
        proprio_times,
        target_times,
        reinterpret_cast<float*>(output),
        batch_size, proprio_src_len, target_len, proprio_dim,
        stream
    );
}

// Explicit instantiations for individual streams
template cudaError_t align_vision_stream<float>(
    const float*, const float*, const float*, float*,
    int, int, int, int, cudaStream_t);

template cudaError_t align_proprio_stream<float>(
    const float*, const float*, const float*, float*,
    int, int, int, int, cudaStream_t);

} // namespace kernels
} // namespace robocache

