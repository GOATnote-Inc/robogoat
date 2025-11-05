// Unified PyBind11 bindings for all RoboCache CUDA kernels
// This file consolidates trajectory resampling, multimodal fusion, and voxelization

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Forward declarations for trajectory resampling
extern "C" cudaError_t launch_trajectory_resample_optimized(
    const void* source_data, const float* source_times, const float* target_times, void* output_data,
    int batch_size, int source_length, int target_length, int action_dim, cudaStream_t stream);

torch::Tensor resample_trajectories_optimized(
    torch::Tensor source_data, torch::Tensor source_times, torch::Tensor target_times) {
    TORCH_CHECK(source_data.is_cuda() && source_times.is_cuda() && target_times.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(source_data.dtype() == torch::kBFloat16, "source_data must be BFloat16");
    TORCH_CHECK(source_times.dtype() == torch::kFloat32 && target_times.dtype() == torch::kFloat32, "Times must be Float32");
    
    int batch_size = source_data.size(0), source_length = source_data.size(1), action_dim = source_data.size(2), target_length = target_times.size(1);
    auto output = torch::empty({batch_size, target_length, action_dim},
                              torch::TensorOptions().dtype(torch::kBFloat16).device(source_data.device()));
    cudaStream_t stream = 0;  // Use default stream
    cudaError_t err = launch_trajectory_resample_optimized(source_data.data_ptr(), source_times.data_ptr<float>(),
        target_times.data_ptr<float>(), output.data_ptr(), batch_size, source_length, target_length, action_dim, stream);
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

// Forward declarations for multimodal fusion (from multimodal_fusion_torch.cu implementation)
namespace robocache { namespace bindings {
torch::Tensor fused_multimodal_alignment_torch(
    torch::Tensor vision_data, torch::Tensor vision_times,
    torch::Tensor proprio_data, torch::Tensor proprio_times,
    torch::optional<torch::Tensor> force_data, torch::optional<torch::Tensor> force_times,
    torch::Tensor target_times);
}}

// Forward declarations for voxelization (from point_cloud_voxelization_torch.cu implementation)
namespace robocache { namespace bindings {
torch::Tensor voxelize_point_cloud_torch(
    torch::Tensor points, torch::Tensor features,
    float voxel_size, int grid_size,
    std::string mode);
}}

// Unified PyBind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RoboCache CUDA kernels for robot learning data preprocessing";
    
    // Trajectory resampling
    m.def("resample_trajectories", &resample_trajectories_optimized,
          "Optimized trajectory resampling with shared memory and vectorization",
          py::arg("source_data"), py::arg("source_times"), py::arg("target_times"));
    
    // Multimodal fusion
    m.def("fused_multimodal_alignment", &robocache::bindings::fused_multimodal_alignment_torch,
          "Fused multimodal sensor alignment (CUDA)",
          py::arg("vision_data"), py::arg("vision_times"),
          py::arg("proprio_data"), py::arg("proprio_times"),
          py::arg("force_data") = py::none(), py::arg("force_times") = py::none(),
          py::arg("target_times"));
    
    // Point cloud voxelization
    m.def("voxelize_point_cloud", &robocache::bindings::voxelize_point_cloud_torch,
          "Point cloud voxelization with TSDF/occupancy/density modes",
          py::arg("points"), py::arg("features"),
          py::arg("voxel_size"), py::arg("grid_size"),
          py::arg("mode") = "occupancy");
}

