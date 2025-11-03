// multimodal_fusion_torch.cu
// PyTorch C++ extension for multimodal fusion
//
// Copyright (c) 2025 GOATnote Inc.
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "multimodal_fusion.h"

using namespace robocache::kernels::multimodal;

/// PyTorch binding for fuse_multimodal_data
torch::Tensor fuse_multimodal_torch(
    torch::Tensor vision_features,     // [B, T_vis, D_vis]
    torch::Tensor vision_timestamps,   // [B, T_vis]
    torch::Tensor proprio_features,    // [B, T_prop, D_prop]
    torch::Tensor proprio_timestamps,  // [B, T_prop]
    torch::Tensor lang_embeddings,     // [B, L, D_lang]
    torch::Tensor target_timestamps,   // [B, T]
    bool use_optimized = true
) {
    // Validation
    TORCH_CHECK(vision_features.is_cuda(), "vision_features must be on CUDA");
    TORCH_CHECK(proprio_features.is_cuda(), "proprio_features must be on CUDA");
    TORCH_CHECK(lang_embeddings.is_cuda(), "lang_embeddings must be on CUDA");
    TORCH_CHECK(vision_timestamps.is_cuda(), "vision_timestamps must be on CUDA");
    TORCH_CHECK(proprio_timestamps.is_cuda(), "proprio_timestamps must be on CUDA");
    TORCH_CHECK(target_timestamps.is_cuda(), "target_timestamps must be on CUDA");

    TORCH_CHECK(vision_features.dtype() == torch::kBFloat16 ||
                vision_features.dtype() == torch::kFloat16 ||
                vision_features.dtype() == torch::kFloat32,
                "vision_features must be BF16, FP16, or FP32");

    TORCH_CHECK(vision_features.dim() == 3, "vision_features must be 3D [B, T, D]");
    TORCH_CHECK(proprio_features.dim() == 3, "proprio_features must be 3D [B, T, D]");
    TORCH_CHECK(lang_embeddings.dim() == 3, "lang_embeddings must be 3D [B, L, D]");
    TORCH_CHECK(vision_timestamps.dim() == 2, "vision_timestamps must be 2D [B, T]");
    TORCH_CHECK(proprio_timestamps.dim() == 2, "proprio_timestamps must be 2D [B, T]");
    TORCH_CHECK(target_timestamps.dim() == 2, "target_timestamps must be 2D [B, T]");

    TORCH_CHECK(vision_timestamps.dtype() == torch::kFloat32,
                "Timestamps must be FP32");
    TORCH_CHECK(proprio_timestamps.dtype() == torch::kFloat32,
                "Timestamps must be FP32");
    TORCH_CHECK(target_timestamps.dtype() == torch::kFloat32,
                "Timestamps must be FP32");

    // Ensure contiguous
    vision_features = vision_features.contiguous();
    proprio_features = proprio_features.contiguous();
    lang_embeddings = lang_embeddings.contiguous();
    vision_timestamps = vision_timestamps.contiguous();
    proprio_timestamps = proprio_timestamps.contiguous();
    target_timestamps = target_timestamps.contiguous();

    // Extract dimensions
    int batch_size = vision_features.size(0);
    int vision_src_len = vision_features.size(1);
    int vision_dim = vision_features.size(2);

    int proprio_src_len = proprio_features.size(1);
    int proprio_dim = proprio_features.size(2);

    int lang_len = lang_embeddings.size(1);
    int lang_dim = lang_embeddings.size(2);

    int target_len = target_timestamps.size(1);
    int total_dim = vision_dim + proprio_dim + lang_dim;

    // Validate batch sizes match
    TORCH_CHECK(proprio_features.size(0) == batch_size,
                "proprio_features batch size mismatch");
    TORCH_CHECK(lang_embeddings.size(0) == batch_size,
                "lang_embeddings batch size mismatch");
    TORCH_CHECK(vision_timestamps.size(0) == batch_size,
                "vision_timestamps batch size mismatch");
    TORCH_CHECK(proprio_timestamps.size(0) == batch_size,
                "proprio_timestamps batch size mismatch");
    TORCH_CHECK(target_timestamps.size(0) == batch_size,
                "target_timestamps batch size mismatch");

    // Validate timestamp lengths
    TORCH_CHECK(vision_timestamps.size(1) == vision_src_len,
                "vision_timestamps length mismatch");
    TORCH_CHECK(proprio_timestamps.size(1) == proprio_src_len,
                "proprio_timestamps length mismatch");

    // Create output tensor
    auto output = torch::empty({batch_size, target_len, total_dim},
                               vision_features.options());

    // Setup config
    FusionConfig config;
    config.batch_size = batch_size;
    config.target_seq_length = target_len;
    config.vision_src_length = vision_src_len;
    config.vision_dim = vision_dim;
    config.proprio_src_length = proprio_src_len;
    config.proprio_dim = proprio_dim;
    config.lang_length = lang_len;
    config.lang_dim = lang_dim;
    config.total_dim = total_dim;
    config.use_optimized = use_optimized;

    try {
        config.validate();
    } catch (const std::exception& e) {
        TORCH_CHECK(false, "Configuration validation failed: ", e.what());
    }

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Call kernel
    cudaError_t status = fuse_multimodal_data(
        vision_features.data_ptr(),
        vision_timestamps.data_ptr<float>(),
        proprio_features.data_ptr(),
        proprio_timestamps.data_ptr<float>(),
        lang_embeddings.data_ptr(),
        target_timestamps.data_ptr<float>(),
        output.data_ptr(),
        config,
        stream,
        nullptr  // metrics
    );

    TORCH_CHECK(status == cudaSuccess,
                "Multimodal fusion failed: ", cudaGetErrorString(status));

    return output;
}

/// Python module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Multimodal fusion for robot foundation models";

    m.def("fuse_multimodal",
          &fuse_multimodal_torch,
          "Fuse vision, proprioception, and language modalities with temporal alignment",
          py::arg("vision_features"),
          py::arg("vision_timestamps"),
          py::arg("proprio_features"),
          py::arg("proprio_timestamps"),
          py::arg("lang_embeddings"),
          py::arg("target_timestamps"),
          py::arg("use_optimized") = true);

    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__cuda_version__") = CUDART_VERSION;

    // Check GPU compatibility
    m.def("check_gpu_compatibility", []() {
        std::string error_msg;
        bool compatible = check_gpu_compatibility(&error_msg);
        if (!compatible) {
            throw std::runtime_error("GPU not compatible: " + error_msg);
        }
        return compatible;
    }, "Check if current GPU supports multimodal fusion");

    // Get device info
    m.def("get_device_info", []() {
        cudaDeviceProp prop = get_device_properties();
        py::dict info;
        info["name"] = prop.name;
        info["compute_capability"] = std::to_string(prop.major) + "." + std::to_string(prop.minor);
        info["total_memory_gb"] = prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0;
        info["clock_rate_mhz"] = prop.clockRate / 1000.0;
        info["memory_clock_rate_mhz"] = prop.memoryClockRate / 1000.0;
        info["memory_bus_width"] = prop.memoryBusWidth;
        info["multiprocessor_count"] = prop.multiProcessorCount;
        return info;
    }, "Get current GPU device information");
}
