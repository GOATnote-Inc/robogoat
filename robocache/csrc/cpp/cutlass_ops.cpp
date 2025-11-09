// RoboCache PyTorch C++ Extension: CUTLASS-optimized Operations
// Unified bindings for production-validated H100/A100 kernels
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ==============================================================================
// Trajectory Resampling (Production kernel from kernels/cutlass/)
// ==============================================================================

extern "C" {
cudaError_t launch_trajectory_resample_optimized(
    const void* source_data, const float* source_times, const float* target_times, void* output_data,
    int batch_size, int source_length, int target_length, int action_dim, cudaStream_t stream);
}

torch::Tensor resample_trajectories_cutlass(
    torch::Tensor source_data,
    torch::Tensor source_times,
    torch::Tensor target_times) {
    
    TORCH_CHECK(source_data.is_cuda(), "source_data must be CUDA tensor");
    TORCH_CHECK(source_times.is_cuda(), "source_times must be CUDA tensor");
    TORCH_CHECK(target_times.is_cuda(), "target_times must be CUDA tensor");
    TORCH_CHECK(source_data.dtype() == torch::kBFloat16, "source_data must be BFloat16");
    TORCH_CHECK(source_times.dtype() == torch::kFloat32, "source_times must be Float32");
    TORCH_CHECK(target_times.dtype() == torch::kFloat32, "target_times must be Float32");
    
    source_data = source_data.contiguous();
    source_times = source_times.contiguous();
    target_times = target_times.contiguous();
    
    int batch_size = source_data.size(0);
    int source_length = source_data.size(1);
    int action_dim = source_data.size(2);
    int target_length = target_times.size(1);
    
    auto output = torch::empty({batch_size, target_length, action_dim},
                              torch::TensorOptions()
                                  .dtype(torch::kBFloat16)
                                  .device(source_data.device()));
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    cudaError_t err = launch_trajectory_resample_optimized(
        source_data.data_ptr(),
        source_times.data_ptr<float>(),
        target_times.data_ptr<float>(),
        output.data_ptr(),
        batch_size, source_length, target_length, action_dim,
        stream
    );
    
    TORCH_CHECK(err == cudaSuccess, 
                "CUTLASS trajectory resample kernel failed: ", 
                cudaGetErrorString(err));
    
    return output;
}

// ==============================================================================
// PyBind11 Module
// ==============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "RoboCache CUTLASS-optimized CUDA kernels (H100/A100 validated)";
    
    m.def("resample_trajectories_cutlass", &resample_trajectories_cutlass,
          "CUTLASS-optimized trajectory resampling (BF16, production kernel)",
          py::arg("source_data"), py::arg("source_times"), py::arg("target_times"));
}

