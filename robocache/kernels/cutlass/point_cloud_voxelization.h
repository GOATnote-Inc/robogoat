// point_cloud_voxelization.h
// GPU-accelerated point cloud voxelization for robot learning
// Phase 3: Dense 3D data processing

#ifndef ROBOCACHE_POINT_CLOUD_VOXELIZATION_H
#define ROBOCACHE_POINT_CLOUD_VOXELIZATION_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace robocache {
namespace kernels {

/**
 * Point Cloud Voxelization
 * 
 * Convert unstructured 3D point clouds to structured voxel grids for
 * transformer-based 3D reasoning in robot manipulation.
 * 
 * Common robot vision pipeline:
 * - RGB-D camera → Point cloud (100K+ points)
 * - Voxelization → [D, H, W] grid
 * - 3D CNN/Transformer → Scene understanding → Grasp planning
 * 
 * Performance target: 50-100x faster than CPU, ~40% HBM utilization
 */

enum VoxelizationMode {
    OCCUPANCY,      // Binary occupancy (1 if point exists)
    DENSITY,        // Count points per voxel
    FEATURE_MAX,    // Max-pooling of features
    FEATURE_MEAN,   // Average features per voxel
    TSDF            // Truncated Signed Distance Function
};

/**
 * Occupancy Voxelization
 * 
 * Create binary occupancy grid from point cloud.
 * 
 * Args:
 *   points: [batch, num_points, 3] (XYZ coordinates)
 *   voxel_grid: [batch, depth, height, width] (output, binary)
 *   voxel_size: Size of each voxel (meters)
 *   origin: [3] Origin of voxel grid (XYZ)
 *   
 * Returns: cudaSuccess or error code
 */
cudaError_t voxelize_occupancy(
    const float* points,
    float* voxel_grid,
    int batch_size,
    int num_points,
    int depth,
    int height,
    int width,
    float voxel_size,
    const float* origin,
    cudaStream_t stream = 0
);

/**
 * Density Voxelization
 * 
 * Count number of points per voxel (useful for attention weights).
 * 
 * Args:
 *   points: [batch, num_points, 3]
 *   voxel_grid: [batch, depth, height, width] (output, float counts)
 */
cudaError_t voxelize_density(
    const float* points,
    float* voxel_grid,
    int batch_size,
    int num_points,
    int depth,
    int height,
    int width,
    float voxel_size,
    const float* origin,
    cudaStream_t stream = 0
);

/**
 * Feature Voxelization with Max Pooling
 * 
 * Aggregate point features into voxels using max pooling.
 * Common for RGB features, semantic labels.
 * 
 * Args:
 *   points: [batch, num_points, 3] (XYZ)
 *   features: [batch, num_points, feature_dim] (e.g., RGB, semantics)
 *   voxel_grid: [batch, depth, height, width, feature_dim] (output)
 */
cudaError_t voxelize_feature_max(
    const float* points,
    const float* features,
    float* voxel_grid,
    int batch_size,
    int num_points,
    int depth,
    int height,
    int width,
    int feature_dim,
    float voxel_size,
    const float* origin,
    cudaStream_t stream = 0
);

/**
 * Feature Voxelization with Mean Pooling
 * 
 * Aggregate point features into voxels using mean pooling.
 * Better for continuous features (colors, normals).
 * 
 * Args:
 *   points: [batch, num_points, 3]
 *   features: [batch, num_points, feature_dim]
 *   voxel_grid: [batch, depth, height, width, feature_dim]
 */
cudaError_t voxelize_feature_mean(
    const float* points,
    const float* features,
    float* voxel_grid,
    int batch_size,
    int num_points,
    int depth,
    int height,
    int width,
    int feature_dim,
    float voxel_size,
    const float* origin,
    cudaStream_t stream = 0
);

/**
 * TSDF Voxelization
 * 
 * Create Truncated Signed Distance Function for 3D reconstruction.
 * Used in object reconstruction, scene completion.
 * 
 * Args:
 *   points: [batch, num_points, 3]
 *   normals: [batch, num_points, 3] (surface normals)
 *   tsdf_grid: [batch, depth, height, width] (signed distance)
 *   weight_grid: [batch, depth, height, width] (confidence weights)
 *   truncation_distance: Max distance to consider (meters)
 */
cudaError_t voxelize_tsdf(
    const float* points,
    const float* normals,
    float* tsdf_grid,
    float* weight_grid,
    int batch_size,
    int num_points,
    int depth,
    int height,
    int width,
    float voxel_size,
    const float* origin,
    float truncation_distance,
    cudaStream_t stream = 0
);

/**
 * Fused Multi-Feature Voxelization
 * 
 * Voxelize multiple feature types in single pass (more efficient).
 * 
 * Args:
 *   points: [batch, num_points, 3]
 *   rgb: [batch, num_points, 3] (color)
 *   normals: [batch, num_points, 3] (surface normals)
 *   semantics: [batch, num_points, num_classes] (semantic probs)
 *   voxel_rgb: [batch, D, H, W, 3] (output)
 *   voxel_normals: [batch, D, H, W, 3] (output)
 *   voxel_semantics: [batch, D, H, W, num_classes] (output)
 *   voxel_counts: [batch, D, H, W] (for averaging)
 */
cudaError_t voxelize_multi_feature(
    const float* points,
    const float* rgb,
    const float* normals,
    const float* semantics,
    float* voxel_rgb,
    float* voxel_normals,
    float* voxel_semantics,
    float* voxel_counts,
    int batch_size,
    int num_points,
    int depth,
    int height,
    int width,
    int num_classes,
    float voxel_size,
    const float* origin,
    cudaStream_t stream = 0
);

/**
 * Helper: Point Cloud Bounds
 * 
 * Compute bounding box of point cloud for automatic grid sizing.
 * 
 * Args:
 *   points: [batch, num_points, 3]
 *   min_bounds: [batch, 3] (output)
 *   max_bounds: [batch, 3] (output)
 */
cudaError_t compute_point_cloud_bounds(
    const float* points,
    float* min_bounds,
    float* max_bounds,
    int batch_size,
    int num_points,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace robocache

#endif // ROBOCACHE_POINT_CLOUD_VOXELIZATION_H

