// multimodal_fusion.h
// GPU-accelerated multimodal sensor fusion for robot learning
// Phase 2: Temporal alignment and sensor synchronization

#ifndef ROBOCACHE_MULTIMODAL_FUSION_H
#define ROBOCACHE_MULTIMODAL_FUSION_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace robocache {
namespace kernels {

/**
 * Temporal Sensor Alignment
 * 
 * Synchronizes multiple sensor streams sampled at different frequencies
 * to a common target frequency using GPU-accelerated interpolation.
 * 
 * Common use case:
 * - Vision: 30 Hz RGB-D camera
 * - Proprioception: 100 Hz joint encoders
 * - Force: 333 Hz force-torque sensor
 * - Language: Variable (command arrival times)
 * → Align all to 50 Hz for transformer input
 * 
 * Performance target: 10x faster than CPU, 50-60% HBM utilization
 */

/**
 * Align vision sensor data (RGB-D) to target timestamps
 * 
 * Args:
 *   vision_data: [batch, vision_src_len, vision_dim] (BF16/FP32)
 *   vision_times: [batch, vision_src_len] (FP32)
 *   target_times: [batch, target_len] (FP32)
 *   output: [batch, target_len, vision_dim] (same dtype as input)
 *   
 * Returns: cudaSuccess or error code
 */
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
    cudaStream_t stream = 0
);

/**
 * Align proprioceptive sensor data (joint positions, velocities) to target timestamps
 * 
 * Args:
 *   proprio_data: [batch, proprio_src_len, proprio_dim] (BF16/FP32)
 *   proprio_times: [batch, proprio_src_len] (FP32)
 *   target_times: [batch, target_len] (FP32)
 *   output: [batch, target_len, proprio_dim] (same dtype as input)
 */
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
    cudaStream_t stream = 0
);

/**
 * Fused multimodal alignment
 * 
 * Aligns multiple sensor streams in a single kernel launch for efficiency.
 * Outputs concatenated aligned features ready for transformer input.
 * 
 * Args:
 *   vision_data: [batch, vision_src_len, vision_dim]
 *   vision_times: [batch, vision_src_len]
 *   proprio_data: [batch, proprio_src_len, proprio_dim]
 *   proprio_times: [batch, proprio_src_len]
 *   force_data: [batch, force_src_len, force_dim]
 *   force_times: [batch, force_src_len]
 *   target_times: [batch, target_len]
 *   output: [batch, target_len, total_dim] where total_dim = vision_dim + proprio_dim + force_dim
 *   
 * Benefits of fusion:
 * - Single kernel launch (reduced overhead)
 * - Shared target time loading
 * - Better memory coalescing
 * - 20-30% faster than separate alignments
 */
template<typename T>
cudaError_t fused_multimodal_alignment(
    // Vision inputs
    const T* vision_data,
    const float* vision_times,
    int vision_src_len,
    int vision_dim,
    // Proprioception inputs
    const T* proprio_data,
    const float* proprio_times,
    int proprio_src_len,
    int proprio_dim,
    // Force inputs (optional, can be nullptr)
    const T* force_data,
    const float* force_times,
    int force_src_len,
    int force_dim,
    // Target timestamps
    const float* target_times,
    int target_len,
    // Output
    T* output,
    int batch_size,
    cudaStream_t stream = 0
);

/**
 * Missing data handling with forward-fill
 * 
 * When sensor data is missing (e.g., vision dropout), fills with last known value.
 * Common in real robot deployments with unreliable sensors.
 * 
 * Args:
 *   data: [batch, seq_len, feature_dim] - input data with NaN for missing
 *   mask: [batch, seq_len] - 1.0 for valid, 0.0 for missing
 *   output: [batch, seq_len, feature_dim] - filled data
 */
template<typename T>
cudaError_t forward_fill_missing(
    const T* data,
    const float* mask,
    T* output,
    int batch_size,
    int seq_len,
    int feature_dim,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace robocache

#endif // ROBOCACHE_MULTIMODAL_FUSION_H

