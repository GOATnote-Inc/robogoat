// trajectory_resample_bulk.cu
// Bulk index computation architecture - Target: 25-35% HBM3 efficiency
//
// Key Innovation: Decouple search from interpolation
// Phase 1: Bulk compute ALL indices in parallel (no dependencies)
// Phase 2: Bulk interpolate using precomputed indices (fully parallel)
//
// Benefits:
// - Eliminates per-thread latency serialization
// - Enables aggressive vectorization
// - Better memory coalescing
// - Can overlap phases with streams

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace robocache {
namespace kernels {
namespace bulk {

constexpr int BLOCK_SIZE = 256;
constexpr int ITEMS_PER_THREAD = 4;

//==============================================================================
// Phase 1: Bulk Index Computation (All Independent)
//==============================================================================

/**
 * Compute interpolation indices for ALL targets in parallel
 * 
 * Each thread processes multiple targets independently - no dependencies
 * Result stored in global memory for Phase 2
 */
__global__ void compute_all_indices(
    const float* __restrict__ source_times,
    const float* __restrict__ target_times,
    int* __restrict__ left_indices,
    int* __restrict__ right_indices,
    float* __restrict__ weights,
    int batch_size,
    int source_length,
    int target_length
) {
    __shared__ float s_source_times[512];
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Load source times into shared memory (done once per batch)
    const float* batch_src_times = source_times + batch_idx * source_length;
    int cache_size = min(source_length, 512);
    
    for (int i = tid; i < cache_size; i += BLOCK_SIZE) {
        s_source_times[i] = batch_src_times[i];
    }
    __syncthreads();
    
    // Each thread processes multiple targets independently
    const float* batch_tgt_times = target_times + batch_idx * target_length;
    int* batch_left = left_indices + batch_idx * target_length;
    int* batch_right = right_indices + batch_idx * target_length;
    float* batch_weights = weights + batch_idx * target_length;
    
    for (int t = tid; t < target_length; t += BLOCK_SIZE) {
        float target_time = batch_tgt_times[t];
        
        // Binary search in shared memory (fast, no global memory stalls)
        int left = 0, right = cache_size - 1;
        
        #pragma unroll 8
        while (left < right - 1) {
            int mid = (left + right) >> 1;
            if (s_source_times[mid] <= target_time) left = mid;
            else right = mid;
        }
        
        right = min(left + 1, source_length - 1);
        
        // Compute weight
        float t_left = s_source_times[left];
        float t_right = s_source_times[right];
        float delta = t_right - t_left;
        float weight = (delta < 1e-6f) ? 0.0f : 
                      fminf(fmaxf((target_time - t_left) / delta, 0.0f), 1.0f);
        
        // Store results (coalesced writes)
        batch_left[t] = left;
        batch_right[t] = right;
        batch_weights[t] = weight;
    }
}

//==============================================================================
// Phase 2: Bulk Interpolation (Fully Parallel with Vectorization)
//==============================================================================

/**
 * Perform interpolation using precomputed indices
 * 
 * Fully parallel - no dependencies between threads
 * Enables aggressive vectorization and memory coalescing
 */
template<typename Element = __nv_bfloat16>
__global__ void __launch_bounds__(256, 4)
bulk_interpolate(
    const Element* __restrict__ source_data,
    const int* __restrict__ left_indices,
    const int* __restrict__ right_indices,
    const float* __restrict__ weights,
    Element* __restrict__ output_data,
    int batch_size,
    int source_length,
    int target_length,
    int action_dim
) {
    int batch_idx = blockIdx.x;
    int target_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || target_idx >= target_length) return;
    
    // Load precomputed indices and weight (cached in registers)
    int idx_offset = batch_idx * target_length + target_idx;
    int left_idx = left_indices[idx_offset];
    int right_idx = right_indices[idx_offset];
    float weight = weights[idx_offset];
    
    // Compute pointers
    const Element* batch_source = source_data + batch_idx * source_length * action_dim;
    const Element* src_left = batch_source + left_idx * action_dim;
    const Element* src_right = batch_source + right_idx * action_dim;
    Element* output = output_data + batch_idx * target_length * action_dim + target_idx * action_dim;
    
    // Vectorized interpolation (8 BF16 values = 16 bytes at once)
    if (sizeof(Element) == 2 && action_dim % 8 == 0) {
        // Ultra-fast path: 128-bit vectorized loads
        using Vec8 = uint4;  // 16 bytes = 8×BF16
        const Vec8* v_left = reinterpret_cast<const Vec8*>(src_left);
        const Vec8* v_right = reinterpret_cast<const Vec8*>(src_right);
        Vec8* v_out = reinterpret_cast<Vec8*>(output);
        
        int num_vecs = action_dim / 8;
        for (int v = tid; v < num_vecs; v += BLOCK_SIZE) {
            Vec8 vl = v_left[v];
            Vec8 vr = v_right[v];
            
            // Unpack and interpolate
            Element* pl = reinterpret_cast<Element*>(&vl);
            Element* pr = reinterpret_cast<Element*>(&vr);
            Element result[8];
            
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float fl = static_cast<float>(pl[i]);
                float fr = static_cast<float>(pr[i]);
                result[i] = static_cast<Element>(fmaf(weight, fr - fl, fl));
            }
            
            v_out[v] = *reinterpret_cast<Vec8*>(result);
        }
    } else {
        // Scalar fallback
        for (int dim = tid; dim < action_dim; dim += BLOCK_SIZE) {
            float val_left = static_cast<float>(src_left[dim]);
            float val_right = static_cast<float>(src_right[dim]);
            output[dim] = static_cast<Element>(fmaf(weight, val_right - val_left, val_left));
        }
    }
}

//==============================================================================
// Host API with Two-Phase Launch
//==============================================================================

template<typename Element>
cudaError_t resample_trajectories_bulk(
    const Element* source_data,
    const float* source_times,
    const float* target_times,
    Element* output_data,
    int batch_size,
    int source_length,
    int target_length,
    int action_dim,
    cudaStream_t stream = 0
) {
    // Allocate temporary storage for indices and weights
    int* d_left_indices;
    int* d_right_indices;
    float* d_weights;
    
    size_t indices_size = batch_size * target_length * sizeof(int);
    size_t weights_size = batch_size * target_length * sizeof(float);
    
    cudaMallocAsync(&d_left_indices, indices_size, stream);
    cudaMallocAsync(&d_right_indices, indices_size, stream);
    cudaMallocAsync(&d_weights, weights_size, stream);
    
    // Phase 1: Compute all indices (one block per batch)
    dim3 grid1(batch_size);
    dim3 block1(BLOCK_SIZE);
    
    compute_all_indices<<<grid1, block1, 0, stream>>>(
        source_times, target_times,
        d_left_indices, d_right_indices, d_weights,
        batch_size, source_length, target_length
    );
    
    // Phase 2: Bulk interpolation (grid covers all work)
    dim3 grid2(batch_size, target_length);
    dim3 block2(BLOCK_SIZE);
    
    bulk_interpolate<Element><<<grid2, block2, 0, stream>>>(
        source_data,
        d_left_indices, d_right_indices, d_weights,
        output_data,
        batch_size, source_length, target_length, action_dim
    );
    
    // Free temporary storage
    cudaFreeAsync(d_left_indices, stream);
    cudaFreeAsync(d_right_indices, stream);
    cudaFreeAsync(d_weights, stream);
    
    return cudaGetLastError();
}

// Explicit instantiations
template cudaError_t resample_trajectories_bulk<float>(
    const float*, const float*, const float*, float*,
    int, int, int, int, cudaStream_t);

template cudaError_t resample_trajectories_bulk<__nv_bfloat16>(
    const __nv_bfloat16*, const float*, const float*, __nv_bfloat16*,
    int, int, int, int, cudaStream_t);

template cudaError_t resample_trajectories_bulk<__half>(
    const __half*, const float*, const float*, __half*,
    int, int, int, int, cudaStream_t);

}}} // namespace robocache::kernels::bulk

