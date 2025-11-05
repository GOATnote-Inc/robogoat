// point_cloud_voxelization_torch.cu
// PyTorch bindings for point cloud voxelization kernels (Production-Grade Error Handling)

#include <torch/extension.h>
#include "point_cloud_voxelization.h"
#include "error_handling.cuh"

namespace robocache {
namespace kernels {

using namespace robocache::error;

//==============================================================================
// PyTorch API Wrappers (Production-Grade with Error Handling)
//==============================================================================

torch::Tensor voxelize_occupancy_torch(
    torch::Tensor points,              // [batch, num_points, 3]
    torch::Tensor grid_size,           // [3] (D, H, W)
    float voxel_size,
    torch::Tensor origin               // [3]
) {
    // ===== INPUT VALIDATION =====
    validate_tensor(points, "points", 3, true, torch::kFloat32);
    validate_tensor(grid_size, "grid_size", 1, true, torch::kInt32);
    validate_tensor(origin, "origin", 1, true, torch::kFloat32);
    
    TORCH_CHECK(
        points.size(2) == 3,
        "points must have shape [batch, num_points, 3], got shape [",
        points.size(0), ", ", points.size(1), ", ", points.size(2), "]"
    );
    
    TORCH_CHECK(
        grid_size.size(0) == 3,
        "grid_size must have 3 elements [depth, height, width], got ", grid_size.size(0)
    );
    
    TORCH_CHECK(
        origin.size(0) == 3,
        "origin must have 3 elements [x, y, z], got ", origin.size(0)
    );
    
    TORCH_CHECK(
        voxel_size > 0.0f,
        "voxel_size must be positive, got ", voxel_size
    );
    
    int batch_size = points.size(0);
    int num_points = points.size(1);
    int depth = grid_size[0].item<int>();
    int height = grid_size[1].item<int>();
    int width = grid_size[2].item<int>();
    
    TORCH_CHECK(
        batch_size > 0 && num_points > 0,
        "Empty point cloud. batch_size=", batch_size, ", num_points=", num_points
    );
    
    TORCH_CHECK(
        depth > 0 && height > 0 && width > 0,
        "Invalid grid dimensions. depth=", depth, ", height=", height, ", width=", width
    );
    
    TORCH_CHECK(
        depth <= 512 && height <= 512 && width <= 512,
        "Grid dimensions too large (max 512). Got: ", depth, "x", height, "x", width
    );
    
    // ===== MEMORY CHECK =====
    size_t output_size = batch_size * depth * height * width * sizeof(float);
    size_t input_size = batch_size * num_points * 3 * sizeof(float);
    size_t required = output_size + input_size;
    
    if (!check_memory_available(required)) {
        size_t free_bytes, total_bytes;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        
        TORCH_WARN(
            "Voxelization may fail due to insufficient GPU memory.\n",
            "  Required: ", format_bytes(required), "\n",
            "  Available: ", format_bytes(free_bytes), "\n",
            "  Device: ", get_device_info(), "\n",
            "Hint: Reduce batch size, grid resolution, or use CPU processing."
        );
    }
    
    // ===== ALLOCATE OUTPUT =====
    torch::Tensor voxel_grid;
    try {
        voxel_grid = torch::zeros(
            {batch_size, depth, height, width},
            torch::TensorOptions().dtype(torch::kFloat32).device(points.device())
        );
    } catch (const c10::Error& e) {
        TORCH_CHECK(
            false,
            "Failed to allocate voxel grid [", batch_size, ", ", depth, ", ",
            height, ", ", width, "] = ", format_bytes(output_size), "\n",
            "Error: ", e.what(), "\n",
            get_device_info(), "\n",
            "Hint: Reduce batch_size or grid resolution, or use CPU processing."
        );
    }
    
    // ===== CALL CUDA KERNEL =====
    cudaError_t err = voxelize_occupancy(
        points.data_ptr<float>(),
        voxel_grid.data_ptr<float>(),
        batch_size, num_points,
        depth, height, width,
        voxel_size,
        origin.data_ptr<float>(),
        c10::cuda::getCurrentCUDAStream()
    );
    
    if (err != cudaSuccess) {
        TORCH_CHECK(
            false,
            "Voxelization kernel failed: ", cudaGetErrorString(err), "\n",
            get_device_info(), "\n",
            "Input shape: [", batch_size, ", ", num_points, ", 3]\n",
            "Grid shape: [", batch_size, ", ", depth, ", ", height, ", ", width, "]"
        );
    }
    
    return voxel_grid;
}

torch::Tensor voxelize_density_torch(
    torch::Tensor points,
    torch::Tensor grid_size,
    float voxel_size,
    torch::Tensor origin
) {
    TORCH_CHECK(points.is_cuda(), "points must be CUDA tensor");
    TORCH_CHECK(points.dim() == 3 && points.size(2) == 3, "points must be [batch, N, 3]");
    
    int batch_size = points.size(0);
    int num_points = points.size(1);
    int depth = grid_size[0].item<int>();
    int height = grid_size[1].item<int>();
    int width = grid_size[2].item<int>();
    
    auto voxel_grid = torch::zeros(
        {batch_size, depth, height, width},
        torch::TensorOptions().dtype(torch::kFloat32).device(points.device())
    );
    
    voxelize_density(
        points.data_ptr<float>(),
        voxel_grid.data_ptr<float>(),
        batch_size, num_points,
        depth, height, width,
        voxel_size,
        origin.data_ptr<float>(),
        c10::cuda::getCurrentCUDAStream()
    );
    
    return voxel_grid;
}

torch::Tensor voxelize_feature_max_torch(
    torch::Tensor points,              // [batch, N, 3]
    torch::Tensor features,            // [batch, N, F]
    torch::Tensor grid_size,           // [3]
    float voxel_size,
    torch::Tensor origin               // [3]
) {
    TORCH_CHECK(points.is_cuda() && features.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(points.size(0) == features.size(0) && points.size(1) == features.size(1),
                "points and features must have same batch and num_points");
    
    int batch_size = points.size(0);
    int num_points = points.size(1);
    int feature_dim = features.size(2);
    int depth = grid_size[0].item<int>();
    int height = grid_size[1].item<int>();
    int width = grid_size[2].item<int>();
    
    auto voxel_grid = torch::zeros(
        {batch_size, depth, height, width, feature_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(points.device())
    );
    
    voxelize_feature_max(
        points.data_ptr<float>(),
        features.data_ptr<float>(),
        voxel_grid.data_ptr<float>(),
        batch_size, num_points,
        depth, height, width, feature_dim,
        voxel_size,
        origin.data_ptr<float>(),
        c10::cuda::getCurrentCUDAStream()
    );
    
    return voxel_grid;
}

torch::Tensor voxelize_feature_mean_torch(
    torch::Tensor points,
    torch::Tensor features,
    torch::Tensor grid_size,
    float voxel_size,
    torch::Tensor origin
) {
    TORCH_CHECK(points.is_cuda() && features.is_cuda(), "tensors must be CUDA");
    
    int batch_size = points.size(0);
    int num_points = points.size(1);
    int feature_dim = features.size(2);
    int depth = grid_size[0].item<int>();
    int height = grid_size[1].item<int>();
    int width = grid_size[2].item<int>();
    
    auto voxel_grid = torch::zeros(
        {batch_size, depth, height, width, feature_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(points.device())
    );
    
    voxelize_feature_mean(
        points.data_ptr<float>(),
        features.data_ptr<float>(),
        voxel_grid.data_ptr<float>(),
        batch_size, num_points,
        depth, height, width, feature_dim,
        voxel_size,
        origin.data_ptr<float>(),
        c10::cuda::getCurrentCUDAStream()
    );
    
    return voxel_grid;
}

std::tuple<torch::Tensor, torch::Tensor> voxelize_tsdf_torch(
    torch::Tensor points,              // [batch, N, 3]
    torch::Tensor normals,             // [batch, N, 3]
    torch::Tensor grid_size,           // [3]
    float voxel_size,
    torch::Tensor origin,              // [3]
    float truncation_distance
) {
    TORCH_CHECK(points.is_cuda() && normals.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(points.sizes() == normals.sizes(), "points and normals must have same shape");
    
    int batch_size = points.size(0);
    int num_points = points.size(1);
    int depth = grid_size[0].item<int>();
    int height = grid_size[1].item<int>();
    int width = grid_size[2].item<int>();
    
    auto tsdf_grid = torch::zeros(
        {batch_size, depth, height, width},
        torch::TensorOptions().dtype(torch::kFloat32).device(points.device())
    );
    
    auto weight_grid = torch::zeros(
        {batch_size, depth, height, width},
        torch::TensorOptions().dtype(torch::kFloat32).device(points.device())
    );
    
    voxelize_tsdf(
        points.data_ptr<float>(),
        normals.data_ptr<float>(),
        tsdf_grid.data_ptr<float>(),
        weight_grid.data_ptr<float>(),
        batch_size, num_points,
        depth, height, width,
        voxel_size,
        origin.data_ptr<float>(),
        truncation_distance,
        c10::cuda::getCurrentCUDAStream()
    );
    
    return std::make_tuple(tsdf_grid, weight_grid);
}

std::tuple<torch::Tensor, torch::Tensor> compute_point_cloud_bounds_torch(
    torch::Tensor points              // [batch, N, 3]
) {
    TORCH_CHECK(points.is_cuda(), "points must be CUDA tensor");
    TORCH_CHECK(points.dim() == 3 && points.size(2) == 3, "points must be [batch, N, 3]");
    
    int batch_size = points.size(0);
    int num_points = points.size(1);
    
    auto min_bounds = torch::empty(
        {batch_size, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(points.device())
    );
    
    auto max_bounds = torch::empty(
        {batch_size, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(points.device())
    );
    
    compute_point_cloud_bounds(
        points.data_ptr<float>(),
        min_bounds.data_ptr<float>(),
        max_bounds.data_ptr<float>(),
        batch_size, num_points,
        c10::cuda::getCurrentCUDAStream()
    );
    
    return std::make_tuple(min_bounds, max_bounds);
}

} // namespace kernels
} // namespace robocache

//==============================================================================
// Python Module Definition
//==============================================================================


// Note: PYBIND11_MODULE removed - now defined in robocache_bindings_all.cu
