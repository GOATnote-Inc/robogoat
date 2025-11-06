// RoboCache PyTorch C++ Extension: Trajectory Resampling
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// CUDA kernel launches (declared in .cu file)
extern "C" {
void launch_resample_bf16(
    const void* source_data,
    const float* source_times,
    const float* target_times,
    void* output,
    int batch_size,
    int source_len,
    int target_len,
    int dim,
    cudaStream_t stream);

void launch_resample_fp32(
    const float* source_data,
    const float* source_times,
    const float* target_times,
    float* output,
    int batch_size,
    int source_len,
    int target_len,
    int dim,
    cudaStream_t stream);
}

// PyTorch interface
torch::Tensor resample_trajectories_cuda(
    torch::Tensor source_data,     // [B, S, D]
    torch::Tensor source_times,    // [B, S]
    torch::Tensor target_times) {  // [B, T]
    
    // Validate inputs
    TORCH_CHECK(source_data.is_cuda(), "source_data must be CUDA tensor");
    TORCH_CHECK(source_times.is_cuda(), "source_times must be CUDA tensor");
    TORCH_CHECK(target_times.is_cuda(), "target_times must be CUDA tensor");
    TORCH_CHECK(source_data.dim() == 3, "source_data must be 3D [B, S, D]");
    TORCH_CHECK(source_times.dim() == 2, "source_times must be 2D [B, S]");
    TORCH_CHECK(target_times.dim() == 2, "target_times must be 2D [B, T]");
    
    int batch_size = source_data.size(0);
    int source_len = source_data.size(1);
    int dim = source_data.size(2);
    int target_len = target_times.size(1);
    
    TORCH_CHECK(source_times.size(0) == batch_size, "Batch size mismatch");
    TORCH_CHECK(source_times.size(1) == source_len, "Source length mismatch");
    TORCH_CHECK(target_times.size(0) == batch_size, "Batch size mismatch");
    
    // Ensure times are float32
    source_times = source_times.to(torch::kFloat32);
    target_times = target_times.to(torch::kFloat32);
    
    // Allocate output
    auto output = torch::empty({batch_size, target_len, dim}, 
                               torch::TensorOptions()
                                   .dtype(source_data.dtype())
                                   .device(source_data.device()));
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel based on dtype
    if (source_data.dtype() == torch::kBFloat16) {
        launch_resample_bf16(
            source_data.data_ptr(),
            source_times.data_ptr<float>(),
            target_times.data_ptr<float>(),
            output.data_ptr(),
            batch_size, source_len, target_len, dim,
            stream
        );
    } else if (source_data.dtype() == torch::kFloat32) {
        launch_resample_fp32(
            source_data.data_ptr<float>(),
            source_times.data_ptr<float>(),
            target_times.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, source_len, target_len, dim,
            stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype. Use float32 or bfloat16.");
    }
    
    return output;
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("resample_trajectories_cuda", &resample_trajectories_cuda,
          "CUDA-accelerated trajectory resampling (BF16/FP32)");
}

