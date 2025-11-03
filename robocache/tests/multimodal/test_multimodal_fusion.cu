// test_multimodal_fusion.cu
// Comprehensive unit tests for multimodal fusion kernel
//
// Copyright (c) 2025 GOATnote Inc.
// SPDX-License-Identifier: Apache-2.0

#include "../../kernels/cutlass/multimodal/multimodal_fusion.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <cassert>

using namespace robocache::kernels::multimodal;

//==============================================================================
// Test Utilities
//==============================================================================

class TestLogger {
public:
    void log_test_start(const std::string& name) {
        std::cout << "\n[TEST] " << name << " ... ";
        current_test_ = name;
    }

    void log_test_pass() {
        std::cout << "✓ PASS" << std::endl;
        passed_++;
    }

    void log_test_fail(const std::string& reason) {
        std::cout << "✗ FAIL: " << reason << std::endl;
        failed_++;
    }

    void print_summary() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Test Summary:" << std::endl;
        std::cout << "  Passed: " << passed_ << std::endl;
        std::cout << "  Failed: " << failed_ << std::endl;
        std::cout << "  Total:  " << (passed_ + failed_) << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        if (failed_ > 0) {
            std::cout << "\n❌ TESTS FAILED" << std::endl;
        } else {
            std::cout << "\n✅ ALL TESTS PASSED" << std::endl;
        }
    }

    int get_failures() const { return failed_; }

private:
    std::string current_test_;
    int passed_ = 0;
    int failed_ = 0;
};

/// Helper: Generate random bfloat16 data
std::vector<__nv_bfloat16> generate_random_bf16(size_t n, float min_val = -1.0f, float max_val = 1.0f) {
    std::vector<__nv_bfloat16> data(n);
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(min_val, max_val);

    for (size_t i = 0; i < n; i++) {
        data[i] = __float2bfloat16(dist(gen));
    }

    return data;
}

/// Helper: Generate sorted timestamps
std::vector<float> generate_timestamps(size_t n, float start = 0.0f, float end = 1.0f) {
    std::vector<float> timestamps(n);
    for (size_t i = 0; i < n; i++) {
        timestamps[i] = start + (end - start) * i / (n - 1);
    }
    return timestamps;
}

/// Helper: Compare two bfloat16 arrays
bool compare_bf16_arrays(const std::vector<__nv_bfloat16>& a,
                         const std::vector<__nv_bfloat16>& b,
                         float rtol = 1e-2f, float atol = 1e-3f) {
    if (a.size() != b.size()) return false;

    for (size_t i = 0; i < a.size(); i++) {
        float va = __bfloat162float(a[i]);
        float vb = __bfloat162float(b[i]);

        if (std::isnan(va) || std::isnan(vb)) return false;
        if (std::isinf(va) || std::isinf(vb)) return false;

        float diff = std::abs(va - vb);
        float threshold = atol + rtol * std::abs(vb);

        if (diff > threshold) {
            std::cout << "\n    Mismatch at index " << i << ": "
                      << va << " vs " << vb << " (diff: " << diff << ")" << std::endl;
            return false;
        }
    }

    return true;
}

//==============================================================================
// Test Cases
//==============================================================================

/// Test 1: GPU compatibility check
bool test_gpu_compatibility(TestLogger& logger) {
    logger.log_test_start("GPU Compatibility Check");

    std::string error_msg;
    bool compatible = check_gpu_compatibility(&error_msg);

    if (!compatible) {
        logger.log_test_fail("GPU not compatible: " + error_msg);
        return false;
    }

    cudaDeviceProp prop = get_device_properties();
    std::cout << "\n    GPU: " << prop.name
              << " (Compute " << prop.major << "." << prop.minor << ")";

    logger.log_test_pass();
    return true;
}

/// Test 2: Configuration validation
bool test_config_validation(TestLogger& logger) {
    logger.log_test_start("Configuration Validation");

    // Valid config
    FusionConfig config;
    config.batch_size = 32;
    config.target_seq_length = 50;
    config.vision_src_length = 30;
    config.vision_dim = 256;
    config.proprio_src_length = 100;
    config.proprio_dim = 64;
    config.lang_length = 77;
    config.lang_dim = 512;
    config.total_dim = config.vision_dim + config.proprio_dim + config.lang_dim;

    try {
        config.validate();
    } catch (const std::exception& e) {
        logger.log_test_fail(std::string("Valid config rejected: ") + e.what());
        return false;
    }

    // Invalid config (negative batch size)
    FusionConfig bad_config = config;
    bad_config.batch_size = -1;

    try {
        bad_config.validate();
        logger.log_test_fail("Invalid config accepted (negative batch size)");
        return false;
    } catch (const std::exception&) {
        // Expected
    }

    logger.log_test_pass();
    return true;
}

/// Test 3: Simple fusion with constant data
bool test_simple_fusion(TestLogger& logger) {
    logger.log_test_start("Simple Fusion (Constant Data)");

    // Small test case
    FusionConfig config;
    config.batch_size = 2;
    config.target_seq_length = 5;
    config.vision_src_length = 3;
    config.vision_dim = 8;
    config.proprio_src_length = 5;
    config.proprio_dim = 4;
    config.lang_length = 3;
    config.lang_dim = 8;
    config.total_dim = config.vision_dim + config.proprio_dim + config.lang_dim;
    config.use_optimized = false;

    try {
        config.validate();
    } catch (const std::exception& e) {
        logger.log_test_fail(std::string("Config validation failed: ") + e.what());
        return false;
    }

    // Create constant input data
    size_t vision_size = config.batch_size * config.vision_src_length * config.vision_dim;
    size_t proprio_size = config.batch_size * config.proprio_src_length * config.proprio_dim;
    size_t lang_size = config.batch_size * config.lang_length * config.lang_dim;
    size_t output_size = config.batch_size * config.target_seq_length * config.total_dim;

    std::vector<__nv_bfloat16> h_vision(vision_size, __float2bfloat16(1.0f));
    std::vector<__nv_bfloat16> h_proprio(proprio_size, __float2bfloat16(2.0f));
    std::vector<__nv_bfloat16> h_lang(lang_size, __float2bfloat16(3.0f));
    std::vector<__nv_bfloat16> h_output(output_size);

    std::vector<float> h_vision_times = generate_timestamps(config.vision_src_length * config.batch_size);
    std::vector<float> h_proprio_times = generate_timestamps(config.proprio_src_length * config.batch_size);
    std::vector<float> h_target_times = generate_timestamps(config.target_seq_length * config.batch_size);

    // Allocate device memory
    void *d_vision, *d_proprio, *d_lang, *d_output;
    float *d_vision_times, *d_proprio_times, *d_target_times;

    cudaMalloc(&d_vision, vision_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_proprio, proprio_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_lang, lang_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output, output_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_vision_times, h_vision_times.size() * sizeof(float));
    cudaMalloc(&d_proprio_times, h_proprio_times.size() * sizeof(float));
    cudaMalloc(&d_target_times, h_target_times.size() * sizeof(float));

    // Copy to device
    cudaMemcpy(d_vision, h_vision.data(), vision_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proprio, h_proprio.data(), proprio_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lang, h_lang.data(), lang_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vision_times, h_vision_times.data(), h_vision_times.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proprio_times, h_proprio_times.data(), h_proprio_times.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_times, h_target_times.data(), h_target_times.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    FusionMetrics metrics;
    cudaError_t err = fuse_multimodal_data(
        d_vision, d_vision_times,
        d_proprio, d_proprio_times,
        d_lang, d_target_times,
        d_output,
        config,
        0,
        &metrics
    );

    if (err != cudaSuccess) {
        logger.log_test_fail(std::string("Kernel failed: ") + cudaGetErrorString(err));
        cudaFree(d_vision); cudaFree(d_proprio); cudaFree(d_lang); cudaFree(d_output);
        cudaFree(d_vision_times); cudaFree(d_proprio_times); cudaFree(d_target_times);
        return false;
    }

    // Copy result back
    cudaMemcpy(h_output.data(), d_output, output_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Verify output
    // Vision features should be ~1.0, proprio ~2.0, lang ~3.0
    bool correct = true;
    for (int b = 0; b < config.batch_size && correct; b++) {
        for (int t = 0; t < config.target_seq_length && correct; t++) {
            int offset = (b * config.target_seq_length + t) * config.total_dim;

            // Check vision
            for (int d = 0; d < config.vision_dim && correct; d++) {
                float val = __bfloat162float(h_output[offset + d]);
                if (std::abs(val - 1.0f) > 0.1f) {
                    std::cout << "\n    Vision mismatch: " << val << " (expected ~1.0)";
                    correct = false;
                }
            }

            // Check proprio
            for (int d = 0; d < config.proprio_dim && correct; d++) {
                float val = __bfloat162float(h_output[offset + config.vision_dim + d]);
                if (std::abs(val - 2.0f) > 0.1f) {
                    std::cout << "\n    Proprio mismatch: " << val << " (expected ~2.0)";
                    correct = false;
                }
            }

            // Check lang (average of 3.0 is 3.0)
            for (int d = 0; d < config.lang_dim && correct; d++) {
                float val = __bfloat162float(h_output[offset + config.vision_dim + config.proprio_dim + d]);
                if (std::abs(val - 3.0f) > 0.1f) {
                    std::cout << "\n    Lang mismatch: " << val << " (expected ~3.0)";
                    correct = false;
                }
            }
        }
    }

    // Cleanup
    cudaFree(d_vision); cudaFree(d_proprio); cudaFree(d_lang); cudaFree(d_output);
    cudaFree(d_vision_times); cudaFree(d_proprio_times); cudaFree(d_target_times);

    if (!correct) {
        logger.log_test_fail("Output values incorrect");
        return false;
    }

    std::cout << "\n    Kernel time: " << std::fixed << std::setprecision(3)
              << metrics.kernel_time_ms << " ms";

    logger.log_test_pass();
    return true;
}

/// Test 4: Temporal interpolation correctness
bool test_temporal_interpolation(TestLogger& logger) {
    logger.log_test_start("Temporal Interpolation");

    FusionConfig config;
    config.batch_size = 1;
    config.target_seq_length = 5;
    config.vision_src_length = 3;
    config.vision_dim = 4;
    config.proprio_src_length = 3;
    config.proprio_dim = 2;
    config.lang_length = 2;
    config.lang_dim = 4;
    config.total_dim = config.vision_dim + config.proprio_dim + config.lang_dim;
    config.use_optimized = false;

    // Create vision data: [0, 1, 2] at times [0.0, 0.5, 1.0]
    std::vector<__nv_bfloat16> h_vision(config.vision_src_length * config.vision_dim);
    for (int t = 0; t < config.vision_src_length; t++) {
        for (int d = 0; d < config.vision_dim; d++) {
            h_vision[t * config.vision_dim + d] = __float2bfloat16((float)t);
        }
    }

    std::vector<float> h_vision_times = {0.0f, 0.5f, 1.0f};
    std::vector<float> h_target_times = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};

    // Proprio and lang (not testing interpolation for these)
    std::vector<__nv_bfloat16> h_proprio(config.proprio_src_length * config.proprio_dim, __float2bfloat16(0.0f));
    std::vector<__nv_bfloat16> h_lang(config.lang_length * config.lang_dim, __float2bfloat16(0.0f));
    std::vector<float> h_proprio_times = {0.0f, 0.5f, 1.0f};

    // Allocate device memory
    void *d_vision, *d_proprio, *d_lang, *d_output;
    float *d_vision_times, *d_proprio_times, *d_target_times;

    size_t vision_size = h_vision.size();
    size_t proprio_size = h_proprio.size();
    size_t lang_size = h_lang.size();
    size_t output_size = config.target_seq_length * config.total_dim;

    cudaMalloc(&d_vision, vision_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_proprio, proprio_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_lang, lang_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output, output_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_vision_times, h_vision_times.size() * sizeof(float));
    cudaMalloc(&d_proprio_times, h_proprio_times.size() * sizeof(float));
    cudaMalloc(&d_target_times, h_target_times.size() * sizeof(float));

    cudaMemcpy(d_vision, h_vision.data(), vision_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proprio, h_proprio.data(), proprio_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lang, h_lang.data(), lang_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vision_times, h_vision_times.data(), h_vision_times.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proprio_times, h_proprio_times.data(), h_proprio_times.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_times, h_target_times.data(), h_target_times.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel
    cudaError_t err = fuse_multimodal_data(
        d_vision, d_vision_times,
        d_proprio, d_proprio_times,
        d_lang, d_target_times,
        d_output,
        config,
        0,
        nullptr
    );

    if (err != cudaSuccess) {
        logger.log_test_fail(std::string("Kernel failed: ") + cudaGetErrorString(err));
        cudaFree(d_vision); cudaFree(d_proprio); cudaFree(d_lang); cudaFree(d_output);
        cudaFree(d_vision_times); cudaFree(d_proprio_times); cudaFree(d_target_times);
        return false;
    }

    // Copy result
    std::vector<__nv_bfloat16> h_output(output_size);
    cudaMemcpy(h_output.data(), d_output, output_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Verify interpolation
    // t=0.0 → 0.0, t=0.25 → 0.5, t=0.5 → 1.0, t=0.75 → 1.5, t=1.0 → 2.0
    std::vector<float> expected = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};

    bool correct = true;
    for (int t = 0; t < config.target_seq_length && correct; t++) {
        int offset = t * config.total_dim;
        float val = __bfloat162float(h_output[offset]);  // First vision dim

        if (std::abs(val - expected[t]) > 0.1f) {
            std::cout << "\n    Interpolation error at t=" << h_target_times[t]
                      << ": got " << val << ", expected " << expected[t];
            correct = false;
        }
    }

    cudaFree(d_vision); cudaFree(d_proprio); cudaFree(d_lang); cudaFree(d_output);
    cudaFree(d_vision_times); cudaFree(d_proprio_times); cudaFree(d_target_times);

    if (!correct) {
        logger.log_test_fail("Interpolation incorrect");
        return false;
    }

    logger.log_test_pass();
    return true;
}

/// Test 5: Optimized vs standard kernel equivalence
bool test_kernel_equivalence(TestLogger& logger) {
    logger.log_test_start("Optimized vs Standard Kernel Equivalence");

    FusionConfig config;
    config.batch_size = 16;
    config.target_seq_length = 50;
    config.vision_src_length = 30;
    config.vision_dim = 256;
    config.proprio_src_length = 100;
    config.proprio_dim = 64;
    config.lang_length = 77;
    config.lang_dim = 512;
    config.total_dim = config.vision_dim + config.proprio_dim + config.lang_dim;

    // Generate random input
    size_t vision_size = config.batch_size * config.vision_src_length * config.vision_dim;
    size_t proprio_size = config.batch_size * config.proprio_src_length * config.proprio_dim;
    size_t lang_size = config.batch_size * config.lang_length * config.lang_dim;
    size_t output_size = config.batch_size * config.target_seq_length * config.total_dim;

    auto h_vision = generate_random_bf16(vision_size);
    auto h_proprio = generate_random_bf16(proprio_size);
    auto h_lang = generate_random_bf16(lang_size);
    auto h_vision_times = generate_timestamps(config.batch_size * config.vision_src_length);
    auto h_proprio_times = generate_timestamps(config.batch_size * config.proprio_src_length);
    auto h_target_times = generate_timestamps(config.batch_size * config.target_seq_length);

    // Allocate device memory
    void *d_vision, *d_proprio, *d_lang, *d_output_std, *d_output_opt;
    float *d_vision_times, *d_proprio_times, *d_target_times;

    cudaMalloc(&d_vision, vision_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_proprio, proprio_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_lang, lang_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output_std, output_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output_opt, output_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_vision_times, h_vision_times.size() * sizeof(float));
    cudaMalloc(&d_proprio_times, h_proprio_times.size() * sizeof(float));
    cudaMalloc(&d_target_times, h_target_times.size() * sizeof(float));

    cudaMemcpy(d_vision, h_vision.data(), vision_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proprio, h_proprio.data(), proprio_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lang, h_lang.data(), lang_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vision_times, h_vision_times.data(), h_vision_times.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proprio_times, h_proprio_times.data(), h_proprio_times.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_times, h_target_times.data(), h_target_times.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Run standard kernel
    config.use_optimized = false;
    FusionMetrics metrics_std;
    cudaError_t err = fuse_multimodal_data(
        d_vision, d_vision_times,
        d_proprio, d_proprio_times,
        d_lang, d_target_times,
        d_output_std,
        config,
        0,
        &metrics_std
    );

    if (err != cudaSuccess) {
        logger.log_test_fail(std::string("Standard kernel failed: ") + cudaGetErrorString(err));
        // Cleanup...
        return false;
    }

    // Run optimized kernel
    config.use_optimized = true;
    FusionMetrics metrics_opt;
    err = fuse_multimodal_data(
        d_vision, d_vision_times,
        d_proprio, d_proprio_times,
        d_lang, d_target_times,
        d_output_opt,
        config,
        0,
        &metrics_opt
    );

    if (err != cudaSuccess) {
        logger.log_test_fail(std::string("Optimized kernel failed: ") + cudaGetErrorString(err));
        // Cleanup...
        return false;
    }

    // Compare outputs
    std::vector<__nv_bfloat16> h_output_std(output_size);
    std::vector<__nv_bfloat16> h_output_opt(output_size);
    cudaMemcpy(h_output_std.data(), d_output_std, output_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_opt.data(), d_output_opt, output_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    bool equivalent = compare_bf16_arrays(h_output_std, h_output_opt);

    // Cleanup
    cudaFree(d_vision); cudaFree(d_proprio); cudaFree(d_lang);
    cudaFree(d_output_std); cudaFree(d_output_opt);
    cudaFree(d_vision_times); cudaFree(d_proprio_times); cudaFree(d_target_times);

    if (!equivalent) {
        logger.log_test_fail("Outputs differ");
        return false;
    }

    std::cout << "\n    Standard: " << metrics_std.kernel_time_ms << " ms"
              << "\n    Optimized: " << metrics_opt.kernel_time_ms << " ms"
              << "\n    Speedup: " << (metrics_std.kernel_time_ms / metrics_opt.kernel_time_ms) << "x";

    logger.log_test_pass();
    return true;
}

/// Test 6: Numerical stability check
bool test_numerical_stability(TestLogger& logger) {
    logger.log_test_start("Numerical Stability (NaN/Inf Check)");

    FusionConfig config;
    config.batch_size = 32;
    config.target_seq_length = 50;
    config.vision_src_length = 30;
    config.vision_dim = 256;
    config.proprio_src_length = 100;
    config.proprio_dim = 64;
    config.lang_length = 77;
    config.lang_dim = 512;
    config.total_dim = config.vision_dim + config.proprio_dim + config.lang_dim;
    config.use_optimized = true;

    size_t vision_size = config.batch_size * config.vision_src_length * config.vision_dim;
    size_t proprio_size = config.batch_size * config.proprio_src_length * config.proprio_dim;
    size_t lang_size = config.batch_size * config.lang_length * config.lang_dim;
    size_t output_size = config.batch_size * config.target_seq_length * config.total_dim;

    auto h_vision = generate_random_bf16(vision_size, -10.0f, 10.0f);
    auto h_proprio = generate_random_bf16(proprio_size, -10.0f, 10.0f);
    auto h_lang = generate_random_bf16(lang_size, -10.0f, 10.0f);
    auto h_vision_times = generate_timestamps(config.batch_size * config.vision_src_length);
    auto h_proprio_times = generate_timestamps(config.batch_size * config.proprio_src_length);
    auto h_target_times = generate_timestamps(config.batch_size * config.target_seq_length);

    void *d_vision, *d_proprio, *d_lang, *d_output;
    float *d_vision_times, *d_proprio_times, *d_target_times;

    cudaMalloc(&d_vision, vision_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_proprio, proprio_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_lang, lang_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_output, output_size * sizeof(__nv_bfloat16));
    cudaMalloc(&d_vision_times, h_vision_times.size() * sizeof(float));
    cudaMalloc(&d_proprio_times, h_proprio_times.size() * sizeof(float));
    cudaMalloc(&d_target_times, h_target_times.size() * sizeof(float));

    cudaMemcpy(d_vision, h_vision.data(), vision_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proprio, h_proprio.data(), proprio_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lang, h_lang.data(), lang_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vision_times, h_vision_times.data(), h_vision_times.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_proprio_times, h_proprio_times.data(), h_proprio_times.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_times, h_target_times.data(), h_target_times.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t err = fuse_multimodal_data(
        d_vision, d_vision_times,
        d_proprio, d_proprio_times,
        d_lang, d_target_times,
        d_output,
        config,
        0,
        nullptr
    );

    if (err != cudaSuccess) {
        logger.log_test_fail(std::string("Kernel failed: ") + cudaGetErrorString(err));
        return false;
    }

    // Check for NaN/Inf
    err = check_numerical_stability(d_output, output_size);

    cudaFree(d_vision); cudaFree(d_proprio); cudaFree(d_lang); cudaFree(d_output);
    cudaFree(d_vision_times); cudaFree(d_proprio_times); cudaFree(d_target_times);

    if (err != cudaSuccess) {
        logger.log_test_fail("Output contains NaN or Inf");
        return false;
    }

    logger.log_test_pass();
    return true;
}

//==============================================================================
// Main Test Runner
//==============================================================================

int main() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Multimodal Fusion Kernel - Unit Tests" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    TestLogger logger;

    // Run tests
    test_gpu_compatibility(logger);
    test_config_validation(logger);
    test_simple_fusion(logger);
    test_temporal_interpolation(logger);
    test_kernel_equivalence(logger);
    test_numerical_stability(logger);

    // Print summary
    logger.print_summary();

    return (logger.get_failures() == 0) ? 0 : 1;
}
