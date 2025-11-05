// point_cloud_voxelization.cu
// GPU-accelerated point cloud voxelization
// Phase 3 implementation - Production-ready H100 kernels

#include "point_cloud_voxelization.h"
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace robocache {
namespace kernels {

// Kernel configuration
constexpr int BLOCK_SIZE = 256;
constexpr int POINTS_PER_THREAD = 4;

//==============================================================================
// Helper: Point to Voxel Index
//==============================================================================

__device__ __forceinline__
bool point_to_voxel_idx(
    float px, float py, float pz,
    const float* origin,
    float voxel_size,
    int depth, int height, int width,
    int& vx, int& vy, int& vz
) {
    // Convert world coordinates to voxel indices
    vx = __float2int_rd((px - origin[0]) / voxel_size);
    vy = __float2int_rd((py - origin[1]) / voxel_size);
    vz = __float2int_rd((pz - origin[2]) / voxel_size);
    
    // Check bounds
    return (vx >= 0 && vx < width &&
            vy >= 0 && vy < height &&
            vz >= 0 && vz < depth);
}

__device__ __forceinline__
int voxel_idx_to_linear(int vx, int vy, int vz, int depth, int height, int width) {
    return vz * (height * width) + vy * width + vx;
}

//==============================================================================
// Occupancy Voxelization Kernel
//==============================================================================

// PRODUCTION FIX: Use atomic counts (deterministic) instead of atomicExch (non-deterministic)
// This matches voxelization-kit-secure best practices.
// Strategy: Accumulate counts with atomicAdd, then convert to binary occupancy.

__global__ void voxelize_occupancy_kernel(
    const float* __restrict__ points,
    float* __restrict__ voxel_grid,
    int batch_size,
    int num_points,
    int depth,
    int height,
    int width,
    float voxel_size,
    const float* __restrict__ origin
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = batch_size * num_points;
    
    // Grid-stride loop for large point clouds
    for (int point_idx = tid; point_idx < total_points; point_idx += gridDim.x * blockDim.x) {
        int batch_idx = point_idx / num_points;
        int local_point_idx = point_idx % num_points;
        
        // Load point coordinates
        int point_offset = batch_idx * num_points * 3 + local_point_idx * 3;
        float px = points[point_offset + 0];
        float py = points[point_offset + 1];
        float pz = points[point_offset + 2];
        
        // Convert to voxel index (using floor rule for CPU/GPU parity)
        int vx, vy, vz;
        if (point_to_voxel_idx(px, py, pz, origin, voxel_size, depth, height, width, vx, vy, vz)) {
            // CRITICAL FIX: Use atomicAdd for deterministic accumulation (not atomicExch)
            // This ensures CPU/GPU parity and eliminates race conditions
            int voxel_linear = voxel_idx_to_linear(vx, vy, vz, depth, height, width);
            int voxel_offset = batch_idx * (depth * height * width) + voxel_linear;
            
            // Accumulate counts (any value > 0 means occupied)
            atomicAdd(&voxel_grid[voxel_offset], 1.0f);
        }
    }
}

// Convert counts to binary occupancy (optional second pass)
__global__ void counts_to_occupancy_kernel(
    float* __restrict__ voxel_grid,
    int total_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_voxels) {
        // Convert count to binary: count > 0 → 1.0, else 0.0
        voxel_grid[idx] = (voxel_grid[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

//==============================================================================
// Density Voxelization Kernel
//==============================================================================

__global__ void voxelize_density_kernel(
    const float* __restrict__ points,
    float* __restrict__ voxel_grid,
    int batch_size,
    int num_points,
    int depth,
    int height,
    int width,
    float voxel_size,
    const float* __restrict__ origin
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = batch_size * num_points;
    
    for (int point_idx = tid; point_idx < total_points; point_idx += gridDim.x * blockDim.x) {
        int batch_idx = point_idx / num_points;
        int local_point_idx = point_idx % num_points;
        
        int point_offset = batch_idx * num_points * 3 + local_point_idx * 3;
        float px = points[point_offset + 0];
        float py = points[point_offset + 1];
        float pz = points[point_offset + 2];
        
        int vx, vy, vz;
        if (point_to_voxel_idx(px, py, pz, origin, voxel_size, depth, height, width, vx, vy, vz)) {
            int voxel_linear = voxel_idx_to_linear(vx, vy, vz, depth, height, width);
            int voxel_offset = batch_idx * (depth * height * width) + voxel_linear;
            
            // Increment count atomically
            atomicAdd(&voxel_grid[voxel_offset], 1.0f);
        }
    }
}

//==============================================================================
// Feature Max Pooling Voxelization Kernel
//==============================================================================

__global__ void voxelize_feature_max_kernel(
    const float* __restrict__ points,
    const float* __restrict__ features,
    float* __restrict__ voxel_grid,
    int batch_size,
    int num_points,
    int depth,
    int height,
    int width,
    int feature_dim,
    float voxel_size,
    const float* __restrict__ origin
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = batch_size * num_points;
    
    for (int point_idx = tid; point_idx < total_points; point_idx += gridDim.x * blockDim.x) {
        int batch_idx = point_idx / num_points;
        int local_point_idx = point_idx % num_points;
        
        // Load point
        int point_offset = batch_idx * num_points * 3 + local_point_idx * 3;
        float px = points[point_offset + 0];
        float py = points[point_offset + 1];
        float pz = points[point_offset + 2];
        
        int vx, vy, vz;
        if (point_to_voxel_idx(px, py, pz, origin, voxel_size, depth, height, width, vx, vy, vz)) {
            int voxel_linear = voxel_idx_to_linear(vx, vy, vz, depth, height, width);
            
            // Load and update features with max pooling
            int feature_offset = batch_idx * num_points * feature_dim + local_point_idx * feature_dim;
            int voxel_offset = batch_idx * (depth * height * width * feature_dim) + voxel_linear * feature_dim;
            
            for (int d = 0; d < feature_dim; d++) {
                float feat = features[feature_offset + d];
                // Atomic max using atomicMax for integers, or CAS loop for floats
                unsigned int* addr = (unsigned int*)(&voxel_grid[voxel_offset + d]);
                unsigned int old = *addr;
                unsigned int assumed;
                
                do {
                    assumed = old;
                    float old_val = __uint_as_float(assumed);
                    float new_val = fmaxf(old_val, feat);
                    old = atomicCAS(addr, assumed, __float_as_uint(new_val));
                } while (assumed != old);
            }
        }
    }
}

//==============================================================================
// Feature Mean Pooling Voxelization Kernel (Two-pass)
//==============================================================================

__global__ void voxelize_feature_mean_kernel_pass1(
    const float* __restrict__ points,
    const float* __restrict__ features,
    float* __restrict__ voxel_grid,
    float* __restrict__ voxel_counts,
    int batch_size,
    int num_points,
    int depth,
    int height,
    int width,
    int feature_dim,
    float voxel_size,
    const float* __restrict__ origin
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = batch_size * num_points;
    
    for (int point_idx = tid; point_idx < total_points; point_idx += gridDim.x * blockDim.x) {
        int batch_idx = point_idx / num_points;
        int local_point_idx = point_idx % num_points;
        
        int point_offset = batch_idx * num_points * 3 + local_point_idx * 3;
        float px = points[point_offset + 0];
        float py = points[point_offset + 1];
        float pz = points[point_offset + 2];
        
        int vx, vy, vz;
        if (point_to_voxel_idx(px, py, pz, origin, voxel_size, depth, height, width, vx, vy, vz)) {
            int voxel_linear = voxel_idx_to_linear(vx, vy, vz, depth, height, width);
            int voxel_offset = batch_idx * (depth * height * width * feature_dim) + voxel_linear * feature_dim;
            int count_offset = batch_idx * (depth * height * width) + voxel_linear;
            
            // Accumulate features
            int feature_offset = batch_idx * num_points * feature_dim + local_point_idx * feature_dim;
            for (int d = 0; d < feature_dim; d++) {
                atomicAdd(&voxel_grid[voxel_offset + d], features[feature_offset + d]);
            }
            
            // Increment count
            atomicAdd(&voxel_counts[count_offset], 1.0f);
        }
    }
}

__global__ void voxelize_feature_mean_kernel_pass2(
    float* __restrict__ voxel_grid,
    const float* __restrict__ voxel_counts,
    int batch_size,
    int depth,
    int height,
    int width,
    int feature_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = batch_size * depth * height * width;
    
    for (int voxel_idx = tid; voxel_idx < total_voxels; voxel_idx += gridDim.x * blockDim.x) {
        float count = voxel_counts[voxel_idx];
        
        if (count > 0.0f) {
            int voxel_offset = voxel_idx * feature_dim;
            float inv_count = 1.0f / count;
            
            // Normalize by count
            for (int d = 0; d < feature_dim; d++) {
                voxel_grid[voxel_offset + d] *= inv_count;
            }
        }
    }
}

//==============================================================================
// TSDF Voxelization Kernel
//==============================================================================

__global__ void voxelize_tsdf_kernel(
    const float* __restrict__ points,
    const float* __restrict__ normals,
    float* __restrict__ tsdf_grid,
    float* __restrict__ weight_grid,
    int batch_size,
    int num_points,
    int depth,
    int height,
    int width,
    float voxel_size,
    const float* __restrict__ origin,
    float truncation_distance
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = batch_size * num_points;
    
    float inv_voxel_size = 1.0f / voxel_size;
    float inv_trunc = 1.0f / truncation_distance;
    int trunc_voxels = __float2int_ru(truncation_distance * inv_voxel_size) + 1;
    
    // Each thread processes one point and updates nearby voxels
    for (int point_idx = tid; point_idx < total_points; point_idx += gridDim.x * blockDim.x) {
        int batch_idx = point_idx / num_points;
        int local_point_idx = point_idx % num_points;
        
        // Load point and normal
        int point_offset = batch_idx * num_points * 3 + local_point_idx * 3;
        float px = points[point_offset + 0];
        float py = points[point_offset + 1];
        float pz = points[point_offset + 2];
        
        float nx = 0.0f, ny = 0.0f, nz = 0.0f;
        if (normals != nullptr) {
            int normal_offset = batch_idx * num_points * 3 + local_point_idx * 3;
            nx = normals[normal_offset + 0];
            ny = normals[normal_offset + 1];
            nz = normals[normal_offset + 2];
        }
        
        // Convert point to voxel coordinates
        int center_vx = __float2int_rd((px - origin[0]) * inv_voxel_size);
        int center_vy = __float2int_rd((py - origin[1]) * inv_voxel_size);
        int center_vz = __float2int_rd((pz - origin[2]) * inv_voxel_size);
        
        // Update all voxels within truncation distance
        for (int dz = -trunc_voxels; dz <= trunc_voxels; dz++) {
            int vz = center_vz + dz;
            if (vz < 0 || vz >= depth) continue;
            
            for (int dy = -trunc_voxels; dy <= trunc_voxels; dy++) {
                int vy = center_vy + dy;
                if (vy < 0 || vy >= height) continue;
                
                for (int dx = -trunc_voxels; dx <= trunc_voxels; dx++) {
                    int vx = center_vx + dx;
                    if (vx < 0 || vx >= width) continue;
                    
                    // Compute voxel center
                    float voxel_cx = origin[0] + (vx + 0.5f) * voxel_size;
                    float voxel_cy = origin[1] + (vy + 0.5f) * voxel_size;
                    float voxel_cz = origin[2] + (vz + 0.5f) * voxel_size;
                    
                    // Distance to point
                    float dx_world = voxel_cx - px;
                    float dy_world = voxel_cy - py;
                    float dz_world = voxel_cz - pz;
                    float dist = sqrtf(dx_world*dx_world + dy_world*dy_world + dz_world*dz_world);
                    
                    if (dist < truncation_distance) {
                        // Compute signed distance using normal
                        float signed_dist = dist;
                        if (normals != nullptr) {
                            float dot = nx * dx_world + ny * dy_world + nz * dz_world;
                            if (dot > 0) signed_dist = -dist; // Inside surface
                        }
                        
                        // Normalize and clamp
                        float tsdf_val = fmaxf(-1.0f, fminf(1.0f, signed_dist * inv_trunc));
                        
                        // Update voxel atomically (use min for now, proper fusion needs averaging)
                        int voxel_idx = batch_idx * (depth * height * width) + 
                                       vz * (height * width) + vy * width + vx;
                        
                        // Atomic min for TSDF (closer points win)
                        unsigned int* tsdf_addr = (unsigned int*)(&tsdf_grid[voxel_idx]);
                        unsigned int old = *tsdf_addr;
                        unsigned int assumed;
                        do {
                            assumed = old;
                            float old_val = __uint_as_float(assumed);
                            float new_val = (old_val == 0.0f) ? tsdf_val : fminf(fabsf(old_val), fabsf(tsdf_val)) * ((old_val < 0) ? -1.0f : 1.0f);
                            old = atomicCAS(tsdf_addr, assumed, __float_as_uint(new_val));
                        } while (assumed != old);
                        
                        // Increment weight
                        atomicAdd(&weight_grid[voxel_idx], 1.0f);
                    }
                }
            }
        }
    }
}

//==============================================================================
// Bounds Computation Kernel
//==============================================================================

__global__ void compute_bounds_kernel(
    const float* __restrict__ points,
    float* __restrict__ min_bounds,
    float* __restrict__ max_bounds,
    int batch_size,
    int num_points
) {
    __shared__ float s_min[3][32];
    __shared__ float s_max[3][32];
    
    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    
    // Initialize local min/max
    float local_min[3] = {1e10f, 1e10f, 1e10f};
    float local_max[3] = {-1e10f, -1e10f, -1e10f};
    
    // Process points
    for (int p = tid; p < num_points; p += blockDim.x) {
        int offset = batch_idx * num_points * 3 + p * 3;
        float px = points[offset + 0];
        float py = points[offset + 1];
        float pz = points[offset + 2];
        
        local_min[0] = fminf(local_min[0], px);
        local_min[1] = fminf(local_min[1], py);
        local_min[2] = fminf(local_min[2], pz);
        
        local_max[0] = fmaxf(local_max[0], px);
        local_max[1] = fmaxf(local_max[1], py);
        local_max[2] = fmaxf(local_max[2], pz);
    }
    
    // Warp-level reduction
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    for (int dim = 0; dim < 3; dim++) {
        for (int offset = 16; offset > 0; offset /= 2) {
            local_min[dim] = fminf(local_min[dim], __shfl_down_sync(0xffffffff, local_min[dim], offset));
            local_max[dim] = fmaxf(local_max[dim], __shfl_down_sync(0xffffffff, local_max[dim], offset));
        }
        
        if (lane_id == 0) {
            s_min[dim][warp_id] = local_min[dim];
            s_max[dim][warp_id] = local_max[dim];
        }
    }
    __syncthreads();
    
    // Final reduction
    if (tid == 0) {
        for (int dim = 0; dim < 3; dim++) {
            float final_min = s_min[dim][0];
            float final_max = s_max[dim][0];
            for (int w = 1; w < (blockDim.x + 31) / 32; w++) {
                final_min = fminf(final_min, s_min[dim][w]);
                final_max = fmaxf(final_max, s_max[dim][w]);
            }
            min_bounds[batch_idx * 3 + dim] = final_min;
            max_bounds[batch_idx * 3 + dim] = final_max;
        }
    }
}

//==============================================================================
// Host API
//==============================================================================

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
    cudaStream_t stream
) {
    // Clear voxel grid
    size_t grid_size = batch_size * depth * height * width * sizeof(float);
    cudaMemsetAsync(voxel_grid, 0, grid_size, stream);
    
    // Pass 1: Accumulate point counts per voxel (deterministic with atomicAdd)
    int total_points = batch_size * num_points;
    int num_blocks_pass1 = (total_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
    voxelize_occupancy_kernel<<<num_blocks_pass1, BLOCK_SIZE, 0, stream>>>(
        points, voxel_grid, batch_size, num_points,
        depth, height, width, voxel_size, origin
    );
    
    // Pass 2: Convert counts to binary occupancy (0 or 1)
    // This ensures backward compatibility with the original API expectation
    int total_voxels = batch_size * depth * height * width;
    int num_blocks_pass2 = (total_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    counts_to_occupancy_kernel<<<num_blocks_pass2, BLOCK_SIZE, 0, stream>>>(
        voxel_grid, total_voxels
    );
    
    return cudaGetLastError();
}

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
    cudaStream_t stream
) {
    size_t grid_size = batch_size * depth * height * width * sizeof(float);
    cudaMemsetAsync(voxel_grid, 0, grid_size, stream);
    
    int num_blocks = (batch_size * num_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
    voxelize_density_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        points, voxel_grid, batch_size, num_points,
        depth, height, width, voxel_size, origin
    );
    
    return cudaGetLastError();
}

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
    cudaStream_t stream
) {
    size_t grid_size = batch_size * depth * height * width * feature_dim * sizeof(float);
    cudaMemsetAsync(voxel_grid, 0, grid_size, stream);
    
    int num_blocks = (batch_size * num_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
    voxelize_feature_max_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        points, features, voxel_grid, batch_size, num_points,
        depth, height, width, feature_dim, voxel_size, origin
    );
    
    return cudaGetLastError();
}

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
    cudaStream_t stream
) {
    // Allocate temporary count buffer
    float* voxel_counts;
    size_t count_size = batch_size * depth * height * width * sizeof(float);
    cudaMalloc(&voxel_counts, count_size);
    cudaMemsetAsync(voxel_counts, 0, count_size, stream);
    
    size_t grid_size = batch_size * depth * height * width * feature_dim * sizeof(float);
    cudaMemsetAsync(voxel_grid, 0, grid_size, stream);
    
    // Pass 1: Accumulate features and counts
    int num_blocks = (batch_size * num_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
    voxelize_feature_mean_kernel_pass1<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        points, features, voxel_grid, voxel_counts, batch_size, num_points,
        depth, height, width, feature_dim, voxel_size, origin
    );
    
    // Pass 2: Normalize by counts
    int total_voxels = batch_size * depth * height * width;
    num_blocks = (total_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    voxelize_feature_mean_kernel_pass2<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        voxel_grid, voxel_counts, batch_size, depth, height, width, feature_dim
    );
    
    cudaFree(voxel_counts);
    return cudaGetLastError();
}

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
    cudaStream_t stream
) {
    size_t grid_size = batch_size * depth * height * width * sizeof(float);
    cudaMemsetAsync(tsdf_grid, 0, grid_size, stream);
    cudaMemsetAsync(weight_grid, 0, grid_size, stream);
    
    int total_voxels = batch_size * depth * height * width;
    int num_blocks = (total_voxels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    voxelize_tsdf_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        points, normals, tsdf_grid, weight_grid, batch_size, num_points,
        depth, height, width, voxel_size, origin, truncation_distance
    );
    
    return cudaGetLastError();
}

cudaError_t compute_point_cloud_bounds(
    const float* points,
    float* min_bounds,
    float* max_bounds,
    int batch_size,
    int num_points,
    cudaStream_t stream
) {
    compute_bounds_kernel<<<batch_size, 256, 0, stream>>>(
        points, min_bounds, max_bounds, batch_size, num_points
    );
    
    return cudaGetLastError();
}

} // namespace kernels
} // namespace robocache

