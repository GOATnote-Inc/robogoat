// point_cloud_voxelization_torch.cu
// PyTorch bindings for point cloud voxelization kernels

#include <torch/extension.h>
#include "point_cloud_voxelization.h"

namespace robocache {
namespace kernels {

//==============================================================================
// PyTorch API Wrappers
//==============================================================================

torch::Tensor voxelize_occupancy_torch(
    torch::Tensor points,              // [batch, num_points, 3]
    torch::Tensor grid_size,           // [3] (D, H, W)
    float voxel_size,
    torch::Tensor origin               // [3]
) {
    TORCH_CHECK(points.is_cuda(), "points must be CUDA tensor");
    TORCH_CHECK(points.dim() == 3 && points.size(2) == 3, "points must be [batch, N, 3]");
    TORCH_CHECK(grid_size.size(0) == 3, "grid_size must be [3]");
    TORCH_CHECK(origin.size(0) == 3, "origin must be [3]");
    
    int batch_size = points.size(0);
    int num_points = points.size(1);
    int depth = grid_size[0].item<int>();
    int height = grid_size[1].item<int>();
    int width = grid_size[2].item<int>();
    
    // Allocate output grid
    auto voxel_grid = torch::zeros(
        {batch_size, depth, height, width},
        torch::TensorOptions().dtype(torch::kFloat32).device(points.device())
    );
    
    // Call CUDA kernel
    voxelize_occupancy(
        points.data_ptr<float>(),
        voxel_grid.data_ptr<float>(),
        batch_size, num_points,
        depth, height, width,
        voxel_size,
        origin.data_ptr<float>(),
        (cudaStream_t)c10::cuda::getCurrentCUDAStream().stream()
    );
    
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
        (cudaStream_t)c10::cuda::getCurrentCUDAStream().stream()
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
        (cudaStream_t)c10::cuda::getCurrentCUDAStream().stream()
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
        (cudaStream_t)c10::cuda::getCurrentCUDAStream().stream()
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
        (cudaStream_t)c10::cuda::getCurrentCUDAStream().stream()
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
        (cudaStream_t)c10::cuda::getCurrentCUDAStream().stream()
    );
    
    return std::make_tuple(min_bounds, max_bounds);
}

} // namespace kernels
} // namespace robocache

//==============================================================================
// Python Module Definition
//==============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_occupancy", &robocache::kernels::voxelize_occupancy_torch,
          "Occupancy voxelization (CUDA)",
          py::arg("points"), py::arg("grid_size"), py::arg("voxel_size"), py::arg("origin"));
    
    m.def("voxelize_density", &robocache::kernels::voxelize_density_torch,
          "Density voxelization (CUDA)",
          py::arg("points"), py::arg("grid_size"), py::arg("voxel_size"), py::arg("origin"));
    
    m.def("voxelize_feature_max", &robocache::kernels::voxelize_feature_max_torch,
          "Feature max pooling voxelization (CUDA)",
          py::arg("points"), py::arg("features"), py::arg("grid_size"),
          py::arg("voxel_size"), py::arg("origin"));
    
    m.def("voxelize_feature_mean", &robocache::kernels::voxelize_feature_mean_torch,
          "Feature mean pooling voxelization (CUDA)",
          py::arg("points"), py::arg("features"), py::arg("grid_size"),
          py::arg("voxel_size"), py::arg("origin"));
    
    m.def("voxelize_tsdf", &robocache::kernels::voxelize_tsdf_torch,
          "TSDF voxelization (CUDA)",
          py::arg("points"), py::arg("normals"), py::arg("grid_size"),
          py::arg("voxel_size"), py::arg("origin"), py::arg("truncation_distance"));
    
    m.def("compute_point_cloud_bounds", &robocache::kernels::compute_point_cloud_bounds_torch,
          "Compute point cloud bounding box (CUDA)",
          py::arg("points"));
}

