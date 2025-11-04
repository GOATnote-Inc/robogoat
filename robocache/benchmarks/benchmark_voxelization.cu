// benchmark_voxelization.cu
// Benchmark for point cloud voxelization
// Compares CUDA implementation against CPU baseline

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include "point_cloud_voxelization.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

//==============================================================================
// CPU Reference Implementation
//==============================================================================

void voxelize_occupancy_cpu(
    const float* points,
    float* voxel_grid,
    int batch_size,
    int num_points,
    int depth,
    int height,
    int width,
    float voxel_size,
    const float* origin
) {
    // Clear grid
    size_t grid_size = batch_size * depth * height * width;
    std::fill(voxel_grid, voxel_grid + grid_size, 0.0f);
    
    // Voxelize points
    for (int b = 0; b < batch_size; b++) {
        for (int p = 0; p < num_points; p++) {
            int point_offset = b * num_points * 3 + p * 3;
            float px = points[point_offset + 0];
            float py = points[point_offset + 1];
            float pz = points[point_offset + 2];
            
            // Convert to voxel index (floor, matching GPU __float2int_rd)
            int vx = std::floor((px - origin[0]) / voxel_size);
            int vy = std::floor((py - origin[1]) / voxel_size);
            int vz = std::floor((pz - origin[2]) / voxel_size);
            
            // Check bounds
            if (vx >= 0 && vx < width &&
                vy >= 0 && vy < height &&
                vz >= 0 && vz < depth) {
                int voxel_idx = b * (depth * height * width) +
                               vz * (height * width) +
                               vy * width +
                               vx;
                voxel_grid[voxel_idx] = 1.0f;
            }
        }
    }
}

void voxelize_density_cpu(
    const float* points,
    float* voxel_grid,
    int batch_size,
    int num_points,
    int depth,
    int height,
    int width,
    float voxel_size,
    const float* origin
) {
    size_t grid_size = batch_size * depth * height * width;
    std::fill(voxel_grid, voxel_grid + grid_size, 0.0f);
    
    for (int b = 0; b < batch_size; b++) {
        for (int p = 0; p < num_points; p++) {
            int point_offset = b * num_points * 3 + p * 3;
            float px = points[point_offset + 0];
            float py = points[point_offset + 1];
            float pz = points[point_offset + 2];
            
            int vx = static_cast<int>((px - origin[0]) / voxel_size);
            int vy = static_cast<int>((py - origin[1]) / voxel_size);
            int vz = static_cast<int>((pz - origin[2]) / voxel_size);
            
            if (vx >= 0 && vx < width &&
                vy >= 0 && vy < height &&
                vz >= 0 && vz < depth) {
                int voxel_idx = b * (depth * height * width) +
                               vz * (height * width) +
                               vy * width +
                               vx;
                voxel_grid[voxel_idx] += 1.0f;
            }
        }
    }
}

//==============================================================================
// Benchmark Configuration
//==============================================================================

struct BenchmarkConfig {
    std::string name;
    int batch_size;
    int num_points;
    int depth;
    int height;
    int width;
    float voxel_size;
    
    size_t point_data_mb() const {
        return (batch_size * num_points * 3 * sizeof(float)) / (1024 * 1024);
    }
    
    size_t voxel_data_mb() const {
        return (batch_size * depth * height * width * sizeof(float)) / (1024 * 1024);
    }
    
    size_t total_data_mb() const {
        return point_data_mb() + voxel_data_mb();
    }
};

//==============================================================================
// Benchmark Runner
//==============================================================================

void run_benchmark(const BenchmarkConfig& config) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Configuration: " << config.name << "\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Batch size:      " << config.batch_size << "\n";
    std::cout << "Num points:      " << config.num_points << "\n";
    std::cout << "Grid size:       " << config.depth << " x " << config.height << " x " << config.width << "\n";
    std::cout << "Voxel size:      " << config.voxel_size << " m\n";
    std::cout << "Point data:      " << config.point_data_mb() << " MB\n";
    std::cout << "Voxel data:      " << config.voxel_data_mb() << " MB\n";
    std::cout << "Total data:      " << config.total_data_mb() << " MB\n";
    
    // Allocate host memory
    size_t point_size = config.batch_size * config.num_points * 3;
    size_t voxel_size = config.batch_size * config.depth * config.height * config.width;
    
    std::vector<float> h_points(point_size);
    std::vector<float> h_voxel_cpu(voxel_size);
    std::vector<float> h_voxel_gpu(voxel_size);
    float origin[3] = {-1.0f, -1.0f, -1.0f};
    
    // Generate random points in [-1, 1]^3
    for (size_t i = 0; i < point_size; i++) {
        h_points[i] = -1.0f + 2.0f * (rand() / (float)RAND_MAX);
    }
    
    // Allocate device memory
    float *d_points, *d_voxels, *d_origin;
    CUDA_CHECK(cudaMalloc(&d_points, point_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_voxels, voxel_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_origin, 3 * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), point_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_origin, origin, 3 * sizeof(float), cudaMemcpyHostToDevice));
    
    // ========== Benchmark GPU (Occupancy) ==========
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "GPU OCCUPANCY VOXELIZATION\n";
    std::cout << std::string(80, '-') << "\n";
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        robocache::kernels::voxelize_occupancy(
            d_points, d_voxels,
            config.batch_size, config.num_points,
            config.depth, config.height, config.width,
            config.voxel_size, d_origin, 0
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int num_iters = 1000;
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        robocache::kernels::voxelize_occupancy(
            d_points, d_voxels,
            config.batch_size, config.num_points,
            config.depth, config.height, config.width,
            config.voxel_size, d_origin, 0
        );
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    
    float gpu_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, end));
    float gpu_latency = gpu_time_ms / num_iters;
    
    // Copy result back for verification
    CUDA_CHECK(cudaMemcpy(h_voxel_gpu.data(), d_voxels, voxel_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // ========== Benchmark CPU ==========
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "CPU OCCUPANCY VOXELIZATION\n";
    std::cout << std::string(80, '-') << "\n";
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    voxelize_occupancy_cpu(
        h_points.data(), h_voxel_cpu.data(),
        config.batch_size, config.num_points,
        config.depth, config.height, config.width,
        config.voxel_size, origin
    );
    auto cpu_end = std::chrono::high_resolution_clock::now();
    
    float cpu_latency = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
    
    // ========== Results ==========
    std::cout << "\n" << std::string(80, '-') << "\n";
    std::cout << "RESULTS\n";
    std::cout << std::string(80, '-') << "\n";
    
    double gpu_bandwidth = (config.total_data_mb() / 1024.0) / (gpu_latency / 1000.0);
    double cpu_bandwidth = (config.total_data_mb() / 1024.0) / (cpu_latency / 1000.0);
    double speedup = cpu_latency / gpu_latency;
    
    // Get H100 peak bandwidth
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bandwidth_gbs = 3000.0; // H100 PCIe peak
    double gpu_efficiency = (gpu_bandwidth / peak_bandwidth_gbs) * 100.0;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "GPU latency:        " << gpu_latency << " ms\n";
    std::cout << "CPU latency:        " << cpu_latency << " ms\n";
    std::cout << "Speedup:            " << speedup << "x\n";
    std::cout << "GPU bandwidth:      " << gpu_bandwidth << " GB/s\n";
    std::cout << "CPU bandwidth:      " << cpu_bandwidth << " GB/s\n";
    std::cout << "GPU HBM efficiency: " << gpu_efficiency << "%\n";
    std::cout << "GPU throughput:     " << (config.batch_size * 1000.0 / gpu_latency) << " clouds/sec\n";
    
    // Verify correctness
    int differences = 0;
    for (size_t i = 0; i < voxel_size; i++) {
        if (h_voxel_cpu[i] != h_voxel_gpu[i]) {
            differences++;
        }
    }
    
    std::cout << "\nCorrectness: " << (differences == 0 ? "✅ PASS" : "❌ FAIL") << "\n";
    if (differences > 0) {
        std::cout << "  Differences: " << differences << " / " << voxel_size << "\n";
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_voxels));
    CUDA_CHECK(cudaFree(d_origin));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
}

//==============================================================================
// Main
//==============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          RoboCache Phase 3: Point Cloud Voxelization                ║\n";
    std::cout << "║                  H100 Performance Benchmark                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";
    
    // Check GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "\nGPU: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0) << " GB\n";
    
    // Benchmark configurations
    std::vector<BenchmarkConfig> configs = {
        // Small: Single depth frame
        {
            .name = "Small (640x480 depth, batch=8)",
            .batch_size = 8,
            .num_points = 50000,   // Typical for 640x480 depth
            .depth = 64,
            .height = 64,
            .width = 64,
            .voxel_size = 0.02f    // 2cm voxels
        },
        
        // Medium: Tabletop manipulation
        {
            .name = "Medium (tabletop, batch=32)",
            .batch_size = 32,
            .num_points = 100000,  // Denser scene
            .depth = 128,
            .height = 128,
            .width = 128,
            .voxel_size = 0.01f    // 1cm voxels
        },
        
        // Large: Full room scan
        {
            .name = "Large (room scan, batch=64)",
            .batch_size = 64,
            .num_points = 200000,  // Full room
            .depth = 256,
            .height = 256,
            .width = 256,
            .voxel_size = 0.05f    // 5cm voxels
        }
    };
    
    for (const auto& config : configs) {
        run_benchmark(config);
    }
    
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  SUMMARY                                                             ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nPoint cloud voxelization is critical for 3D robot manipulation:\n";
    std::cout << "  • RGB-D camera → Point cloud → Voxel grid → 3D CNN/Transformer\n";
    std::cout << "  • Used in grasp planning, scene understanding, object reconstruction\n";
    std::cout << "  • Expected: 50-100x speedup vs CPU (highly parallel workload)\n";
    std::cout << "  • Target: 30-50% HBM efficiency (atomic operations overhead)\n\n";
    
    return 0;
}

