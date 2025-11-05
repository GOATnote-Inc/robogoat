// PyTorch bindings for streaming resampler
// Includes contiguity checks and proper stream usage

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Forward declaration of CUDA kernel launcher
extern "C" cudaError_t launch_trajectory_resample_streaming(
    const void* src, const float* st, const float* tt, void* out,
    int B, int S, int T, int D, cudaStream_t stream);

torch::Tensor resample_trajectories_streaming(
    torch::Tensor source_data,
    torch::Tensor source_times,
    torch::Tensor target_times)
{
    // Input validation
    TORCH_CHECK(source_data.is_cuda() && source_times.is_cuda() && target_times.is_cuda(),
                "All tensors must be CUDA");
    TORCH_CHECK(source_data.dtype() == torch::kBFloat16,
                "source_data must be BFloat16");
    TORCH_CHECK(source_times.dtype() == torch::kFloat32 && target_times.dtype() == torch::kFloat32,
                "Times must be Float32");

    // Ensure contiguity for coalesced access (CRITICAL for performance)
    source_data = source_data.contiguous();
    source_times = source_times.contiguous();
    target_times = target_times.contiguous();

    int B = source_data.size(0);  // batch
    int S = source_data.size(1);  // source length
    int D = source_data.size(2);  // action dim
    int T = target_times.size(1); // target length

    // Validate D % 8 == 0 for vectorized loads (TODO: add tail handling)
    TORCH_CHECK(D % 8 == 0,
                "Action dimension must be multiple of 8 for vectorized loads. Got D=", D);

    // Allocate output
    auto output = torch::empty({B, T, D},
                              torch::TensorOptions()
                                  .dtype(torch::kBFloat16)
                                  .device(source_data.device()));

    // USE PYTORCH'S CURRENT STREAM (not stream 0)
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(source_data.device().index());

    // Launch streaming kernel
    cudaError_t err = launch_trajectory_resample_streaming(
        source_data.data_ptr(),
        source_times.data_ptr<float>(),
        target_times.data_ptr<float>(),
        output.data_ptr(),
        B, S, T, D,
        stream
    );

    TORCH_CHECK(err == cudaSuccess,
                "Streaming resampler kernel failed: ", cudaGetErrorString(err));

    return output;
}

// Note: PYBIND11_MODULE defined in robocache_bindings_all.cu

