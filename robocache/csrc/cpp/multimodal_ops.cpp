// Copyright (c) 2025 GOATnote Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0

#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
void multimodal_fusion_cuda(
    const void* stream1_data,
    const float* stream1_times,
    const void* stream2_data,
    const float* stream2_times,
    const void* stream3_data,
    const float* stream3_times,
    const float* target_times,
    void* output,
    int batch_size,
    int s1, int d1,
    int s2, int d2,
    int s3, int d3,
    int target_len,
    bool use_bf16,
    cudaStream_t stream
);

torch::Tensor fuse_multimodal_cuda(
    torch::Tensor stream1_data,    // [B, S1, D1]
    torch::Tensor stream1_times,   // [B, S1]
    torch::Tensor stream2_data,    // [B, S2, D2]
    torch::Tensor stream2_times,   // [B, S2]
    torch::Tensor stream3_data,    // [B, S3, D3]
    torch::Tensor stream3_times,   // [B, S3]
    torch::Tensor target_times     // [B, T]
) {
    // Input validation
    TORCH_CHECK(stream1_data.is_cuda(), "stream1_data must be on CUDA device");
    TORCH_CHECK(stream2_data.is_cuda(), "stream2_data must be on CUDA device");
    TORCH_CHECK(stream3_data.is_cuda(), "stream3_data must be on CUDA device");
    
    TORCH_CHECK(stream1_data.dim() == 3, "stream1_data must be 3D [B, S1, D1]");
    TORCH_CHECK(stream2_data.dim() == 3, "stream2_data must be 3D [B, S2, D2]");
    TORCH_CHECK(stream3_data.dim() == 3, "stream3_data must be 3D [B, S3, D3]");
    
    TORCH_CHECK(stream1_times.dim() == 2, "stream1_times must be 2D [B, S1]");
    TORCH_CHECK(stream2_times.dim() == 2, "stream2_times must be 2D [B, S2]");
    TORCH_CHECK(stream3_times.dim() == 2, "stream3_times must be 2D [B, S3]");
    TORCH_CHECK(target_times.dim() == 2, "target_times must be 2D [B, T]");
    
    int batch_size = stream1_data.size(0);
    TORCH_CHECK(stream2_data.size(0) == batch_size, "Batch size mismatch");
    TORCH_CHECK(stream3_data.size(0) == batch_size, "Batch size mismatch");
    
    int s1 = stream1_data.size(1);
    int d1 = stream1_data.size(2);
    int s2 = stream2_data.size(1);
    int d2 = stream2_data.size(2);
    int s3 = stream3_data.size(1);
    int d3 = stream3_data.size(2);
    int target_len = target_times.size(1);
    
    TORCH_CHECK(stream1_times.size(1) == s1, "stream1_times length mismatch");
    TORCH_CHECK(stream2_times.size(1) == s2, "stream2_times length mismatch");
    TORCH_CHECK(stream3_times.size(1) == s3, "stream3_times length mismatch");
    
    // Ensure all data tensors have same dtype
    TORCH_CHECK(stream1_data.dtype() == stream2_data.dtype(), "Stream dtypes must match");
    TORCH_CHECK(stream2_data.dtype() == stream3_data.dtype(), "Stream dtypes must match");
    
    // Ensure time tensors are float32
    stream1_times = stream1_times.to(torch::kFloat32);
    stream2_times = stream2_times.to(torch::kFloat32);
    stream3_times = stream3_times.to(torch::kFloat32);
    target_times = target_times.to(torch::kFloat32);
    
    // Ensure contiguous layout
    stream1_data = stream1_data.contiguous();
    stream2_data = stream2_data.contiguous();
    stream3_data = stream3_data.contiguous();
    stream1_times = stream1_times.contiguous();
    stream2_times = stream2_times.contiguous();
    stream3_times = stream3_times.contiguous();
    target_times = target_times.contiguous();
    
    // Allocate output
    int total_dim = d1 + d2 + d3;
    auto output = torch::empty({batch_size, target_len, total_dim}, 
                               torch::TensorOptions()
                                   .dtype(stream1_data.dtype())
                                   .device(stream1_data.device()));
    
    // Determine precision
    bool use_bf16 = (stream1_data.dtype() == torch::kBFloat16);
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    multimodal_fusion_cuda(
        stream1_data.data_ptr(),
        stream1_times.data_ptr<float>(),
        stream2_data.data_ptr(),
        stream2_times.data_ptr<float>(),
        stream3_data.data_ptr(),
        stream3_times.data_ptr<float>(),
        target_times.data_ptr<float>(),
        output.data_ptr(),
        batch_size,
        s1, d1,
        s2, d2,
        s3, d3,
        target_len,
        use_bf16,
        stream
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fuse_multimodal", &fuse_multimodal_cuda, "Multimodal Sensor Fusion (CUDA)");
}

