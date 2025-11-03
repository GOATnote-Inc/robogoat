// benchmark_multimodal_fusion.cu
// Comprehensive performance benchmarks for multimodal fusion
//
// Copyright (c) 2025 GOATnote Inc.
// SPDX-License-Identifier: Apache-2.0

#include "../../kernels/cutlass/multimodal/multimodal_fusion.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>

using namespace robocache::kernels::multimodal;

//==============================================================================
// Benchmark Utilities
//==============================================================================

struct BenchmarkConfig {
    std::string name;
    int batch_size;
    int target_seq_length;
    int vision_src_length;
    int vision_dim;
    int proprio_src_length;
    int proprio_dim;
    int lang_length;
    int lang_dim;
};

std::vector<BenchmarkConfig> get_benchmark_configs() {
    return {
        // Small (typical real-time inference)
        {"Small-RealTime", 1, 50, 30, 256, 100, 64, 77, 512},

        // Medium (single robot training)
        {"Medium-Training", 32, 50, 30, 256, 100, 64, 77, 512},

        // Large (typical training batch)
        {"Large-Training", 256, 50, 30, 256, 100, 64, 77, 512},

        // Extra Large (distributed training)
        {"XLarge-Distributed", 512, 50, 30, 256, 100, 64, 77, 512},

        // High-dimensional (vision transformer)
        {"HighDim-Vision", 128, 50, 30, 512, 100, 64, 77, 512},

        // Long sequence (extended trajectory)
        {"LongSeq-Trajectory", 64, 200, 120, 256, 400, 64, 77, 512},
    };
}

/// Generate random bfloat16 data
std::vector<__nv_bfloat16> generate_random_bf16(size_t n, int seed = 42) {
    std::vector<__nv_bfloat16> data(n);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < n; i++) {
        data[i] = __float2bfloat16(dist(gen));
    }

    return data;
}

/// Generate timestamps
std::vector<float> generate_timestamps(size_t n) {
    std::vector<float> times(n);
    for (size_t i = 0; i < n; i++) {
        times[i] = (float)i / (n - 1);
    }
    return times;
}

//==============================================================================
// Benchmark Runner
//==============================================================================

class BenchmarkRunner {
public:
    BenchmarkRunner(int warmup_iters = 10, int bench_iters = 100)
        : warmup_iters_(warmup_iters), bench_iters_(bench_iters) {}

    void run_benchmark(const BenchmarkConfig& cfg) {
        std::cout << "\n" << std::string(70, '-') << std::endl;
        std::cout << "Benchmark: " << cfg.name << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        // Setup config
        FusionConfig config;
        config.batch_size = cfg.batch_size;
        config.target_seq_length = cfg.target_seq_length;
        config.vision_src_length = cfg.vision_src_length;
        config.vision_dim = cfg.vision_dim;
        config.proprio_src_length = cfg.proprio_src_length;
        config.proprio_dim = cfg.proprio_dim;
        config.lang_length = cfg.lang_length;
        config.lang_dim = cfg.lang_dim;
        config.total_dim = config.vision_dim + config.proprio_dim + config.lang_dim;

        std::cout << "  Config:" << std::endl;
        std::cout << "    Batch size: " << config.batch_size << std::endl;
        std::cout << "    Target sequence: " << config.target_seq_length << std::endl;
        std::cout << "    Vision: [" << config.vision_src_length << ", " << config.vision_dim << "]" << std::endl;
        std::cout << "    Proprio: [" << config.proprio_src_length << ", " << config.proprio_dim << "]" << std::endl;
        std::cout << "    Language: [" << config.lang_length << ", " << config.lang_dim << "]" << std::endl;
        std::cout << "    Output dim: " << config.total_dim << std::endl;

        // Generate data
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

        // Calculate sizes
        size_t bytes_read = vision_size * sizeof(__nv_bfloat16) +
                            proprio_size * sizeof(__nv_bfloat16) +
                            lang_size * sizeof(__nv_bfloat16) +
                            h_vision_times.size() * sizeof(float) +
                            h_proprio_times.size() * sizeof(float) +
                            h_target_times.size() * sizeof(float);
        size_t bytes_write = output_size * sizeof(__nv_bfloat16);
        size_t total_bytes = bytes_read + bytes_write;

        std::cout << "\n  Data Transfer:" << std::endl;
        std::cout << "    Input: " << (bytes_read / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "    Output: " << (bytes_write / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "    Total: " << (total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;

        // Benchmark both kernels
        std::cout << "\n  Results:" << std::endl;

        for (bool use_opt : {false, true}) {
            config.use_optimized = use_opt;
            std::string kernel_name = use_opt ? "Optimized" : "Standard ";

            // Warmup
            for (int i = 0; i < warmup_iters_; i++) {
                fuse_multimodal_data(
                    d_vision, d_vision_times,
                    d_proprio, d_proprio_times,
                    d_lang, d_target_times,
                    d_output,
                    config,
                    0,
                    nullptr
                );
            }
            cudaDeviceSynchronize();

            // Benchmark
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            for (int i = 0; i < bench_iters_; i++) {
                fuse_multimodal_data(
                    d_vision, d_vision_times,
                    d_proprio, d_proprio_times,
                    d_lang, d_target_times,
                    d_output,
                    config,
                    0,
                    nullptr
                );
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            float avg_time_ms = elapsed_ms / bench_iters_;

            // Calculate metrics
            double bandwidth_gbs = total_bytes / (avg_time_ms / 1000.0) / 1e9;
            double samples_per_sec = config.batch_size / (avg_time_ms / 1000.0);
            double latency_per_sample_us = (avg_time_ms * 1000.0) / config.batch_size;

            std::cout << "    " << kernel_name << ":" << std::endl;
            std::cout << "      Latency: " << std::fixed << std::setprecision(3)
                      << avg_time_ms << " ms" << std::endl;
            std::cout << "      Throughput: " << std::fixed << std::setprecision(1)
                      << samples_per_sec << " samples/sec" << std::endl;
            std::cout << "      Latency/sample: " << std::fixed << std::setprecision(2)
                      << latency_per_sample_us << " µs" << std::endl;
            std::cout << "      Bandwidth: " << std::fixed << std::setprecision(2)
                      << bandwidth_gbs << " GB/s" << std::endl;

            // Save results
            BenchmarkResult result;
            result.config_name = cfg.name;
            result.kernel_name = kernel_name;
            result.batch_size = config.batch_size;
            result.avg_time_ms = avg_time_ms;
            result.bandwidth_gbs = bandwidth_gbs;
            result.samples_per_sec = samples_per_sec;
            result.latency_per_sample_us = latency_per_sample_us;
            results_.push_back(result);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        // Cleanup
        cudaFree(d_vision);
        cudaFree(d_proprio);
        cudaFree(d_lang);
        cudaFree(d_output);
        cudaFree(d_vision_times);
        cudaFree(d_proprio_times);
        cudaFree(d_target_times);
    }

    void save_results_csv(const std::string& filename) {
        std::ofstream file(filename);

        file << "Config,Kernel,BatchSize,AvgTime(ms),Bandwidth(GB/s),Throughput(samples/s),Latency/Sample(us)\n";

        for (const auto& r : results_) {
            file << r.config_name << ","
                 << r.kernel_name << ","
                 << r.batch_size << ","
                 << std::fixed << std::setprecision(3) << r.avg_time_ms << ","
                 << std::fixed << std::setprecision(2) << r.bandwidth_gbs << ","
                 << std::fixed << std::setprecision(1) << r.samples_per_sec << ","
                 << std::fixed << std::setprecision(2) << r.latency_per_sample_us << "\n";
        }

        file.close();
        std::cout << "\nResults saved to: " << filename << std::endl;
    }

    void print_summary() {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "Benchmark Summary" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        // Find best results
        double best_bandwidth = 0.0;
        double best_throughput = 0.0;
        double best_latency = 1e9;

        for (const auto& r : results_) {
            if (r.kernel_name.find("Optimized") != std::string::npos) {
                best_bandwidth = std::max(best_bandwidth, r.bandwidth_gbs);
                best_throughput = std::max(best_throughput, r.samples_per_sec);
                best_latency = std::min(best_latency, r.latency_per_sample_us);
            }
        }

        std::cout << "\nBest Optimized Kernel Performance:" << std::endl;
        std::cout << "  Peak Bandwidth: " << std::fixed << std::setprecision(2)
                  << best_bandwidth << " GB/s" << std::endl;
        std::cout << "  Peak Throughput: " << std::fixed << std::setprecision(1)
                  << best_throughput << " samples/sec" << std::endl;
        std::cout << "  Best Latency: " << std::fixed << std::setprecision(2)
                  << best_latency << " µs/sample" << std::endl;

        // Get GPU peak bandwidth
        cudaDeviceProp prop = get_device_properties();
        double peak_bandwidth_gbs = (prop.memoryClockRate * 1000.0 *
                                     (prop.memoryBusWidth / 8.0) * 2.0) / 1e9;
        double bandwidth_efficiency = (best_bandwidth / peak_bandwidth_gbs) * 100.0;

        std::cout << "\n  GPU: " << prop.name << std::endl;
        std::cout << "  Theoretical Peak Bandwidth: " << std::fixed << std::setprecision(2)
                  << peak_bandwidth_gbs << " GB/s" << std::endl;
        std::cout << "  Achieved Efficiency: " << std::fixed << std::setprecision(1)
                  << bandwidth_efficiency << "%" << std::endl;
    }

private:
    struct BenchmarkResult {
        std::string config_name;
        std::string kernel_name;
        int batch_size;
        double avg_time_ms;
        double bandwidth_gbs;
        double samples_per_sec;
        double latency_per_sample_us;
    };

    int warmup_iters_;
    int bench_iters_;
    std::vector<BenchmarkResult> results_;
};

//==============================================================================
// Reproducibility Test
//==============================================================================

void test_reproducibility(int num_runs = 5) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Reproducibility Test" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    FusionConfig config;
    config.batch_size = 64;
    config.target_seq_length = 50;
    config.vision_src_length = 30;
    config.vision_dim = 256;
    config.proprio_src_length = 100;
    config.proprio_dim = 64;
    config.lang_length = 77;
    config.lang_dim = 512;
    config.total_dim = config.vision_dim + config.proprio_dim + config.lang_dim;
    config.use_optimized = true;

    // Generate fixed input
    size_t vision_size = config.batch_size * config.vision_src_length * config.vision_dim;
    size_t proprio_size = config.batch_size * config.proprio_src_length * config.proprio_dim;
    size_t lang_size = config.batch_size * config.lang_length * config.lang_dim;
    size_t output_size = config.batch_size * config.target_seq_length * config.total_dim;

    auto h_vision = generate_random_bf16(vision_size, 42);
    auto h_proprio = generate_random_bf16(proprio_size, 42);
    auto h_lang = generate_random_bf16(lang_size, 42);
    auto h_vision_times = generate_timestamps(config.batch_size * config.vision_src_length);
    auto h_proprio_times = generate_timestamps(config.batch_size * config.proprio_src_length);
    auto h_target_times = generate_timestamps(config.batch_size * config.target_seq_length);

    // Allocate
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

    // Run multiple times
    std::vector<std::vector<__nv_bfloat16>> outputs;
    std::vector<double> timings;

    for (int run = 0; run < num_runs; run++) {
        FusionMetrics metrics;
        fuse_multimodal_data(
            d_vision, d_vision_times,
            d_proprio, d_proprio_times,
            d_lang, d_target_times,
            d_output,
            config,
            0,
            &metrics
        );

        std::vector<__nv_bfloat16> h_output(output_size);
        cudaMemcpy(h_output.data(), d_output, output_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

        outputs.push_back(h_output);
        timings.push_back(metrics.kernel_time_ms);

        std::cout << "  Run " << (run + 1) << ": " << std::fixed << std::setprecision(3)
                  << metrics.kernel_time_ms << " ms" << std::endl;
    }

    // Check bitwise reproducibility
    bool all_identical = true;
    for (int run = 1; run < num_runs; run++) {
        for (size_t i = 0; i < output_size; i++) {
            if (outputs[run][i] != outputs[0][i]) {
                all_identical = false;
                std::cout << "\n  ✗ Outputs differ at index " << i
                          << " (run 0 vs run " << run << ")" << std::endl;
                break;
            }
        }
        if (!all_identical) break;
    }

    if (all_identical) {
        std::cout << "\n  ✓ Bitwise identical across all " << num_runs << " runs" << std::endl;
    }

    // Timing statistics
    double avg_time = 0.0;
    double min_time = timings[0];
    double max_time = timings[0];

    for (double t : timings) {
        avg_time += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }
    avg_time /= num_runs;

    double variance = 0.0;
    for (double t : timings) {
        variance += (t - avg_time) * (t - avg_time);
    }
    variance /= num_runs;
    double stddev = std::sqrt(variance);

    std::cout << "\n  Timing Statistics:" << std::endl;
    std::cout << "    Mean: " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
    std::cout << "    Std Dev: " << std::fixed << std::setprecision(4) << stddev << " ms" << std::endl;
    std::cout << "    Min: " << std::fixed << std::setprecision(3) << min_time << " ms" << std::endl;
    std::cout << "    Max: " << std::fixed << std::setprecision(3) << max_time << " ms" << std::endl;
    std::cout << "    Variance: " << std::fixed << std::setprecision(4)
              << (stddev / avg_time * 100.0) << "%" << std::endl;

    cudaFree(d_vision);
    cudaFree(d_proprio);
    cudaFree(d_lang);
    cudaFree(d_output);
    cudaFree(d_vision_times);
    cudaFree(d_proprio_times);
    cudaFree(d_target_times);
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char** argv) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Multimodal Fusion - Performance Benchmarks" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Print GPU info
    cudaDeviceProp prop = get_device_properties();
    std::cout << "\nGPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0) << " GB" << std::endl;
    std::cout << "Clock Rate: " << (prop.clockRate / 1000.0) << " MHz" << std::endl;
    std::cout << "Memory Clock: " << (prop.memoryClockRate / 1000.0) << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;

    // Run benchmarks
    BenchmarkRunner runner(10, 100);

    for (const auto& cfg : get_benchmark_configs()) {
        runner.run_benchmark(cfg);
    }

    runner.print_summary();
    runner.save_results_csv("benchmark_results.csv");

    // Reproducibility test
    test_reproducibility(5);

    std::cout << "\n✅ Benchmarking complete!" << std::endl;
    std::cout << "\nFor NCU profiling, run:" << std::endl;
    std::cout << "  ncu --set full -o multimodal_fusion_profile ./benchmark_multimodal_fusion" << std::endl;

    return 0;
}
