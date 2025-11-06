// Copyright (c) 2025 GOATnote Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0

#include <torch/extension.h>
#include <vector>

enum VoxelMode {
    COUNT = 0,
    OCCUPANCY = 1,
    MEAN = 2,
    MAX = 3
};

// CUDA forward declaration
void voxelize_cuda(
    const float* points,
    const float* features,
    void* voxel_grid,
    void* voxel_counts,
    int num_points,
    int num_features,
    float3 grid_min,
    float voxel_size,
    int grid_x, int grid_y, int grid_z,
    VoxelMode mode,
    cudaStream_t stream
);

std::vector<torch::Tensor> voxelize_pointcloud_cuda(
    torch::Tensor points,            // [N, 3] point coordinates
    torch::Tensor features,          // [N, F] point features (optional)
    std::vector<float> grid_min,     // [x, y, z] minimum corner
    float voxel_size,                // voxel edge length
    std::vector<int> grid_size,      // [X, Y, Z] grid dimensions
    std::string mode                 // "count", "occupancy", "mean", "max"
) {
    // Input validation
    TORCH_CHECK(points.is_cuda(), "points must be on CUDA device");
    TORCH_CHECK(points.dim() == 2, "points must be 2D [N, 3]");
    TORCH_CHECK(points.size(1) == 3, "points must have 3 coordinates (x, y, z)");
    
    TORCH_CHECK(grid_min.size() == 3, "grid_min must have 3 elements [x, y, z]");
    TORCH_CHECK(grid_size.size() == 3, "grid_size must have 3 elements [X, Y, Z]");
    TORCH_CHECK(voxel_size > 0, "voxel_size must be positive");
    
    int num_points = points.size(0);
    int grid_x = grid_size[0];
    int grid_y = grid_size[1];
    int grid_z = grid_size[2];
    
    // Parse mode
    VoxelMode voxel_mode;
    if (mode == "count") {
        voxel_mode = COUNT;
    } else if (mode == "occupancy") {
        voxel_mode = OCCUPANCY;
    } else if (mode == "mean") {
        voxel_mode = MEAN;
    } else if (mode == "max") {
        voxel_mode = MAX;
    } else {
        TORCH_CHECK(false, "Invalid mode: " + mode + ". Must be 'count', 'occupancy', 'mean', or 'max'");
    }
    
    // Validate features for mean/max modes
    int num_features = 0;
    if (voxel_mode == MEAN || voxel_mode == MAX) {
        TORCH_CHECK(features.defined(), "features required for mean/max modes");
        TORCH_CHECK(features.is_cuda(), "features must be on CUDA device");
        TORCH_CHECK(features.dim() == 2, "features must be 2D [N, F]");
        TORCH_CHECK(features.size(0) == num_points, "features must match points length");
        num_features = features.size(1);
        features = features.to(torch::kFloat32).contiguous();
    }
    
    // Ensure points are contiguous float32
    points = points.to(torch::kFloat32).contiguous();
    
    // Allocate output
    torch::Tensor voxel_grid;
    torch::Tensor voxel_counts;
    
    if (voxel_mode == COUNT || voxel_mode == OCCUPANCY) {
        voxel_grid = torch::zeros({grid_x, grid_y, grid_z}, 
                                   torch::TensorOptions()
                                       .dtype(torch::kInt32)
                                       .device(points.device()));
    } else {
        voxel_grid = torch::zeros({grid_x, grid_y, grid_z, num_features}, 
                                   torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(points.device()));
        if (voxel_mode == MEAN) {
            voxel_counts = torch::zeros({grid_x, grid_y, grid_z}, 
                                         torch::TensorOptions()
                                             .dtype(torch::kInt32)
                                             .device(points.device()));
        }
    }
    
    // Prepare grid_min
    float3 grid_min_f3;
    grid_min_f3.x = grid_min[0];
    grid_min_f3.y = grid_min[1];
    grid_min_f3.z = grid_min[2];
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    voxelize_cuda(
        points.data_ptr<float>(),
        (voxel_mode == MEAN || voxel_mode == MAX) ? features.data_ptr<float>() : nullptr,
        voxel_grid.data_ptr(),
        (voxel_mode == MEAN) ? voxel_counts.data_ptr() : nullptr,
        num_points,
        num_features,
        grid_min_f3,
        voxel_size,
        grid_x, grid_y, grid_z,
        voxel_mode,
        stream
    );
    
    // Return results
    if (voxel_mode == MEAN) {
        return {voxel_grid, voxel_counts};
    } else {
        return {voxel_grid};
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_pointcloud", &voxelize_pointcloud_cuda, "Point Cloud Voxelization (CUDA)");
}

