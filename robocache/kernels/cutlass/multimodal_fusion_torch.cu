// multimodal_fusion_torch.cu
// PyTorch bindings for multimodal sensor fusion

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "multimodal_fusion.h"

namespace robocache {
namespace bindings {

/**
 * Fused multimodal alignment - PyTorch interface
 * 
 * Aligns vision, proprioception, and force sensors to common timestamps.
 * 
 * Args:
 *   vision_data: [batch, vision_src_len, vision_dim] tensor (BF16/FP32)
 *   vision_times: [batch, vision_src_len] tensor (FP32)
 *   proprio_data: [batch, proprio_src_len, proprio_dim] tensor (BF16/FP32)
 *   proprio_times: [batch, proprio_src_len] tensor (FP32)
 *   force_data: [batch, force_src_len, force_dim] tensor (BF16/FP32), optional
 *   force_times: [batch, force_src_len] tensor (FP32), optional
 *   target_times: [batch, target_len] tensor (FP32)
 * 
 * Returns:
 *   output: [batch, target_len, total_dim] where total_dim = vision_dim + proprio_dim + force_dim
 */
torch::Tensor fused_multimodal_alignment_torch(
    torch::Tensor vision_data,
    torch::Tensor vision_times,
    torch::Tensor proprio_data,
    torch::Tensor proprio_times,
    torch::optional<torch::Tensor> force_data,
    torch::optional<torch::Tensor> force_times,
    torch::Tensor target_times
) {
    // Validate inputs
    TORCH_CHECK(vision_data.is_cuda(), "vision_data must be CUDA tensor");
    TORCH_CHECK(vision_times.is_cuda(), "vision_times must be CUDA tensor");
    TORCH_CHECK(proprio_data.is_cuda(), "proprio_data must be CUDA tensor");
    TORCH_CHECK(proprio_times.is_cuda(), "proprio_times must be CUDA tensor");
    TORCH_CHECK(target_times.is_cuda(), "target_times must be CUDA tensor");
    
    TORCH_CHECK(vision_data.dim() == 3, "vision_data must be 3D [batch, src_len, dim]");
    TORCH_CHECK(proprio_data.dim() == 3, "proprio_data must be 3D [batch, src_len, dim]");
    TORCH_CHECK(vision_times.dim() == 2, "vision_times must be 2D [batch, src_len]");
    TORCH_CHECK(proprio_times.dim() == 2, "proprio_times must be 2D [batch, src_len]");
    TORCH_CHECK(target_times.dim() == 2, "target_times must be 2D [batch, target_len]");
    
    TORCH_CHECK(vision_times.scalar_type() == torch::kFloat32, "vision_times must be FP32");
    TORCH_CHECK(proprio_times.scalar_type() == torch::kFloat32, "proprio_times must be FP32");
    TORCH_CHECK(target_times.scalar_type() == torch::kFloat32, "target_times must be FP32");
    
    int batch_size = vision_data.size(0);
    int vision_src_len = vision_data.size(1);
    int vision_dim = vision_data.size(2);
    int proprio_src_len = proprio_data.size(1);
    int proprio_dim = proprio_data.size(2);
    int target_len = target_times.size(1);
    
    TORCH_CHECK(proprio_data.size(0) == batch_size, "Batch size mismatch");
    TORCH_CHECK(vision_times.size(0) == batch_size, "Batch size mismatch");
    TORCH_CHECK(proprio_times.size(0) == batch_size, "Batch size mismatch");
    TORCH_CHECK(target_times.size(0) == batch_size, "Batch size mismatch");
    
    // Handle optional force data
    int force_src_len = 0;
    int force_dim = 0;
    const void* force_data_ptr = nullptr;
    const float* force_times_ptr = nullptr;
    
    if (force_data.has_value() && force_times.has_value()) {
        auto f_data = force_data.value();
        auto f_times = force_times.value();
        
        TORCH_CHECK(f_data.is_cuda(), "force_data must be CUDA tensor");
        TORCH_CHECK(f_times.is_cuda(), "force_times must be CUDA tensor");
        TORCH_CHECK(f_data.dim() == 3, "force_data must be 3D");
        TORCH_CHECK(f_times.dim() == 2, "force_times must be 2D");
        TORCH_CHECK(f_data.size(0) == batch_size, "force batch size mismatch");
        TORCH_CHECK(f_times.size(0) == batch_size, "force times batch size mismatch");
        
        force_src_len = f_data.size(1);
        force_dim = f_data.size(2);
        force_data_ptr = f_data.data_ptr();
        force_times_ptr = f_times.data_ptr<float>();
    }
    
    int total_dim = vision_dim + proprio_dim + force_dim;
    
    // Allocate output
    auto output = torch::empty(
        {batch_size, target_len, total_dim},
        torch::TensorOptions()
            .dtype(vision_data.scalar_type())
            .device(vision_data.device())
    );
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Dispatch based on dtype
    cudaError_t err;
    if (vision_data.scalar_type() == torch::kBFloat16) {
        err = kernels::fused_multimodal_alignment<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(vision_data.data_ptr()),
            vision_times.data_ptr<float>(),
            vision_src_len, vision_dim,
            reinterpret_cast<const __nv_bfloat16*>(proprio_data.data_ptr()),
            proprio_times.data_ptr<float>(),
            proprio_src_len, proprio_dim,
            reinterpret_cast<const __nv_bfloat16*>(force_data_ptr),
            force_times_ptr,
            force_src_len, force_dim,
            target_times.data_ptr<float>(),
            target_len,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            batch_size,
            stream
        );
    } else if (vision_data.scalar_type() == torch::kFloat32) {
        err = kernels::fused_multimodal_alignment<float>(
            reinterpret_cast<const float*>(vision_data.data_ptr()),
            vision_times.data_ptr<float>(),
            vision_src_len, vision_dim,
            reinterpret_cast<const float*>(proprio_data.data_ptr()),
            proprio_times.data_ptr<float>(),
            proprio_src_len, proprio_dim,
            reinterpret_cast<const float*>(force_data_ptr),
            force_times_ptr,
            force_src_len, force_dim,
            target_times.data_ptr<float>(),
            target_len,
            reinterpret_cast<float*>(output.data_ptr()),
            batch_size,
            stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype, use BF16 or FP32");
    }
    
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

} // namespace bindings
} // namespace robocache
// Note: PYBIND11_MODULE removed - now defined in robocache_bindings_all.cu
