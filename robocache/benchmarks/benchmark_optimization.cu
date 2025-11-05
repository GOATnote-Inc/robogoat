// benchmark_optimization.cu
// Comprehensive benchmark comparing baseline vs optimized kernels
// Measures: throughput, bandwidth, latency, and efficiency improvements

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>

// Include both kernels
#include "../kernels/cutlass/trajectory_resample.cu"
#include "../kernels/cutlass/trajectory_resample_optimized.cu"

using namespace std;
using namespace robocache::kernels;

//==============================================================================
// Benchmark Configuration
//==============================================================================

struct BenchmarkConfig {
    int batch_size;
    int source_length;
    int target_length;
    int action_dim;
    int num_warmup;
    int num_iterations;
    
    double bytes_read() const {
        return batch_size * source_length * action_dim * sizeof(float) +
               batch_size * (source_length + target_length) * sizeof(float);
    }
    
    double bytes_write() const {
        return batch_size * target_length * action_dim * sizeof(float);
    }
    
    double total_bytes() const {
        return bytes_read() + bytes_write();
    }
    
    long long flops() const {
        // Per output element: 2 loads, 2 FMAs = 4 FLOPs
        return (long long)batch_size * target_length * action_dim * 4;
    }
};

//==============================================================================
// Benchmark Runner
//==============================================================================

struct BenchmarkResult {
    double time_ms;
    double bandwidth_gb_s;
    double throughput_samples_s;
    double efficiency_percent;  // vs H100 peak 3TB/s
};

template<typename KernelFunc>
BenchmarkResult run_benchmark(
    KernelFunc kernel_func,
    const float* d_source_data,
    const float* d_source_times,
    const float* d_target_times,
    float* d_output_data,
    const BenchmarkConfig& config
) {
    // Warmup
    for (int i = 0; i < config.num_warmup; i++) {
        kernel_func(d_source_data, d_source_times, d_target_times, d_output_data,
                   config.batch_size, config.source_length, config.target_length,
                   config.action_dim, 0);
    }
    cudaDeviceSynchronize();
    
    // Benchmark
    auto start = chrono::high_resolution_clock::now();
    
    for (int i = 0; i < config.num_iterations; i++) {
        kernel_func(d_source_data, d_source_times, d_target_times, d_output_data,
                   config.batch_size, config.source_length, config.target_length,
                   config.action_dim, 0);
    }
    cudaDeviceSynchronize();
    
    auto end = chrono::high_resolution_clock::now();
    double elapsed_ms = chrono::duration<double, milli>(end - start).count() / config.num_iterations;
    
    // Calculate metrics
    BenchmarkResult result;
    result.time_ms = elapsed_ms;
    result.bandwidth_gb_s = (config.total_bytes() / 1e9) / (elapsed_ms / 1000.0);
    result.throughput_samples_s = (config.batch_size * config.target_length * config.action_dim) / 
                                  (elapsed_ms / 1000.0);
    result.efficiency_percent = (result.bandwidth_gb_s / 3000.0) * 100.0;  // H100 PCIe: ~3TB/s
    
    return result;
}

//==============================================================================
// Main Benchmark
//==============================================================================

int main(int argc, char** argv) {
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    cout << "================================================================================\n";
    cout << "          RoboCache Optimization Benchmark: Baseline vs Optimized\n";
    cout << "================================================================================\n\n";
    
    cout << "GPU: " << prop.name << "\n";
    cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    cout << "Peak Memory Bandwidth: ~3000 GB/s (H100 PCIe HBM3)\n";
    cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB\n";
    cout << "\n";
    
    // Test configurations
    vector<BenchmarkConfig> configs = {
        // Small: Fast iteration
        {32, 100, 50, 32, 20, 100},
        
        // Medium: Typical workload
        {256, 100, 50, 32, 20, 100},
        
        // Large batch: Test scaling
        {1024, 100, 50, 32, 20, 50},
        
        // Long trajectories: Test shared memory benefit
        {256, 500, 250, 32, 10, 50},
        
        // High dimensions: Test vectorization
        {256, 100, 50, 128, 20, 100},
        
        // Very long source: Exceeds shared memory cache
        {256, 1000, 500, 64, 10, 50},
    };
    
    cout << "Running benchmarks...\n\n";
    
    // Print header
    cout << setw(6) << "Batch" << " | ";
    cout << setw(4) << "Src" << " | ";
    cout << setw(4) << "Tgt" << " | ";
    cout << setw(4) << "Dim" << " | ";
    cout << setw(12) << "Kernel" << " | ";
    cout << setw(10) << "Time(ms)" << " | ";
    cout << setw(10) << "BW(GB/s)" << " | ";
    cout << setw(10) << "Eff(%)" << " | ";
    cout << setw(10) << "Speedup" << "\n";
    cout << string(100, '-') << "\n";
    
    for (const auto& config : configs) {
        // Allocate memory
        float* d_source_data;
        float* d_source_times;
        float* d_target_times;
        float* d_output_data;
        
        cudaMalloc(&d_source_data, config.batch_size * config.source_length * config.action_dim * sizeof(float));
        cudaMalloc(&d_source_times, config.batch_size * config.source_length * sizeof(float));
        cudaMalloc(&d_target_times, config.batch_size * config.target_length * sizeof(float));
        cudaMalloc(&d_output_data, config.batch_size * config.target_length * config.action_dim * sizeof(float));
        
        // Initialize with random data
        vector<float> h_source_data(config.batch_size * config.source_length * config.action_dim);
        vector<float> h_source_times(config.batch_size * config.source_length);
        vector<float> h_target_times(config.batch_size * config.target_length);
        
        for (int b = 0; b < config.batch_size; b++) {
            for (int t = 0; t < config.source_length; t++) {
                h_source_times[b * config.source_length + t] = float(t) / config.source_length;
            }
            for (int t = 0; t < config.target_length; t++) {
                h_target_times[b * config.target_length + t] = float(t) / config.target_length;
            }
        }
        
        cudaMemcpy(d_source_data, h_source_data.data(), h_source_data.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_source_times, h_source_times.data(), h_source_times.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_target_times, h_target_times.data(), h_target_times.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Benchmark baseline kernel
        auto baseline_result = run_benchmark(
            [](const float* src_data, const float* src_times, const float* tgt_times, float* out_data,
               int batch, int src_len, int tgt_len, int act_dim, cudaStream_t stream) {
                TrajectoryResamplerGEMM<float>::resample_batch(
                    src_data, src_times, tgt_times, out_data,
                    batch, src_len, tgt_len, act_dim, stream
                );
            },
            d_source_data, d_source_times, d_target_times, d_output_data, config
        );
        
        // Benchmark optimized kernel
        auto optimized_result = run_benchmark(
            optimized::resample_trajectories_optimized<float>,
            d_source_data, d_source_times, d_target_times, d_output_data, config
        );
        
        // Calculate speedup
        double speedup = baseline_result.time_ms / optimized_result.time_ms;
        
        // Print baseline results
        cout << setw(6) << config.batch_size << " | ";
        cout << setw(4) << config.source_length << " | ";
        cout << setw(4) << config.target_length << " | ";
        cout << setw(4) << config.action_dim << " | ";
        cout << setw(12) << "Baseline" << " | ";
        cout << setw(10) << fixed << setprecision(3) << baseline_result.time_ms << " | ";
        cout << setw(10) << fixed << setprecision(1) << baseline_result.bandwidth_gb_s << " | ";
        cout << setw(10) << fixed << setprecision(2) << baseline_result.efficiency_percent << " | ";
        cout << setw(10) << "-" << "\n";
        
        // Print optimized results
        cout << setw(6) << config.batch_size << " | ";
        cout << setw(4) << config.source_length << " | ";
        cout << setw(4) << config.target_length << " | ";
        cout << setw(4) << config.action_dim << " | ";
        cout << setw(12) << "Optimized" << " | ";
        cout << setw(10) << fixed << setprecision(3) << optimized_result.time_ms << " | ";
        cout << setw(10) << fixed << setprecision(1) << optimized_result.bandwidth_gb_s << " | ";
        cout << setw(10) << fixed << setprecision(2) << optimized_result.efficiency_percent << " | ";
        cout << setw(10) << fixed << setprecision(2) << speedup << "x\n";
        
        cout << string(100, '-') << "\n";
        
        // Cleanup
        cudaFree(d_source_data);
        cudaFree(d_source_times);
        cudaFree(d_target_times);
        cudaFree(d_output_data);
    }
    
    cout << "\n";
    cout << "================================================================================\n";
    cout << "Optimization Summary\n";
    cout << "================================================================================\n";
    cout << "Key Improvements:\n";
    cout << "  1. Shared memory caching of time arrays (reduces global memory latency)\n";
    cout << "  2. Cooperative warp-level binary search (better ILP)\n";
    cout << "  3. Process multiple targets per block (amortizes overhead)\n";
    cout << "  4. Improved memory coalescing patterns\n";
    cout << "\n";
    cout << "Expected: 30-100% speedup depending on workload characteristics\n";
    cout << "Greatest benefit: Long source sequences that fit in shared memory\n";
    cout << "================================================================================\n";
    
    return 0;
}

