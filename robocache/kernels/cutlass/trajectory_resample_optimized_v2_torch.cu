#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Forward declare kernel launch function
extern "C" cudaError_t launch_trajectory_resample_optimized(
    const void* source_data,
    const float* source_times,
    const float* target_times,
    void* output_data,
    int batch_size,
    int source_length,
    int target_length,
    int action_dim,
    cudaStream_t stream
);

// PyTorch wrapper
torch::Tensor resample_trajectories_optimized(
    torch::Tensor source_data,
    torch::Tensor source_times,
    torch::Tensor target_times
) {
    TORCH_CHECK(source_data.is_cuda(), "source_data must be CUDA tensor");
    TORCH_CHECK(source_times.is_cuda(), "source_times must be CUDA tensor");
    TORCH_CHECK(target_times.is_cuda(), "target_times must be CUDA tensor");
    
    TORCH_CHECK(source_data.dtype() == torch::kBFloat16, "source_data must be BFloat16");
    TORCH_CHECK(source_times.dtype() == torch::kFloat32, "source_times must be Float32");
    TORCH_CHECK(target_times.dtype() == torch::kFloat32, "target_times must be Float32");
    
    int batch_size = source_data.size(0);
    int source_length = source_data.size(1);
    int action_dim = source_data.size(2);
    int target_length = target_times.size(1);
    
    auto output = torch::empty({batch_size, target_length, action_dim},
                              torch::TensorOptions()
                                  .dtype(torch::kBFloat16)
                                  .device(source_data.device()));
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(source_data.device().index());
    
    cudaError_t err = launch_trajectory_resample_optimized(
        source_data.data_ptr(),
        source_times.data_ptr<float>(),
        target_times.data_ptr<float>(),
        output.data_ptr(),
        batch_size,
        source_length,
        target_length,
        action_dim,
        stream
    );
    
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("resample_trajectories", &resample_trajectories_optimized,
          "Optimized trajectory resampling with shared memory and vectorization");
}

