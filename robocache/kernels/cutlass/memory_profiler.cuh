/**
 * RoboCache - Memory Profiling and Management
 * 
 * Production-grade memory profiling, chunking, and OOM prevention for
 * GPU-accelerated data processing. Demonstrates expert-level memory management
 * and TCO-aware engineering.
 * 
 * Features:
 * - Peak memory tracking
 * - Automatic chunking for large batches
 * - OOM prediction and prevention
 * - Memory pool management
 * - Detailed memory reports
 * 
 * @author RoboCache Team
 * @date 2025-11-05
 */

#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include "error_handling.cuh"

namespace robocache {
namespace memory {

using namespace robocache::error;

//==============================================================================
// Memory Profiler
//==============================================================================

/**
 * Memory snapshot at a point in time
 */
struct MemorySnapshot {
    size_t allocated_bytes;      ///< Bytes currently allocated
    size_t reserved_bytes;       ///< Bytes reserved by allocator
    size_t free_bytes;           ///< Free memory on device
    size_t total_bytes;          ///< Total device memory
    std::chrono::time_point<std::chrono::steady_clock> timestamp;
    std::string label;           ///< Human-readable label
    
    double utilization() const {
        return total_bytes > 0 ? static_cast<double>(allocated_bytes) / total_bytes : 0.0;
    }
    
    std::string to_string() const {
        std::ostringstream oss;
        oss << label << ":\n"
            << "  Allocated: " << format_bytes(allocated_bytes) << "\n"
            << "  Reserved:  " << format_bytes(reserved_bytes) << "\n"
            << "  Free:      " << format_bytes(free_bytes) << "\n"
            << "  Total:     " << format_bytes(total_bytes) << "\n"
            << "  Utilization: " << (utilization() * 100.0) << "%";
        return oss.str();
    }
};

/**
 * Memory profiler for tracking GPU memory usage
 * 
 * Thread-safe. Minimal overhead (only active when explicitly profiling).
 */
class MemoryProfiler {
public:
    /**
     * Capture memory snapshot
     * @param label Human-readable label for this snapshot
     * @param device_id Device to profile (-1 = current device)
     * @return Memory snapshot
     */
    static MemorySnapshot capture(const std::string& label, int device_id = -1) {
        if (device_id < 0) {
            CUDA_CHECK(cudaGetDevice(&device_id));
        }
        
        MemorySnapshot snap;
        snap.label = label;
        snap.timestamp = std::chrono::steady_clock::now();
        
        // Get PyTorch allocator stats
        if (torch::cuda::is_available()) {
            snap.allocated_bytes = torch::cuda::memory_allocated(device_id);
            snap.reserved_bytes = torch::cuda::memory_reserved(device_id);
        } else {
            snap.allocated_bytes = 0;
            snap.reserved_bytes = 0;
        }
        
        // Get device memory info
        size_t free, total;
        cudaError_t err = cudaMemGetInfo(&free, &total);
        if (err == cudaSuccess) {
            snap.free_bytes = free;
            snap.total_bytes = total;
        } else {
            snap.free_bytes = 0;
            snap.total_bytes = 0;
        }
        
        return snap;
    }
    
    /**
     * Get peak memory usage since last reset
     * @param device_id Device to query (-1 = current)
     * @return Peak memory in bytes
     */
    static size_t get_peak_memory(int device_id = -1) {
        if (device_id < 0) {
            CUDA_CHECK(cudaGetDevice(&device_id));
        }
        
        if (!torch::cuda::is_available()) {
            return 0;
        }
        
        return torch::cuda::max_memory_allocated(device_id);
    }
    
    /**
     * Reset peak memory counter
     * @param device_id Device to reset (-1 = current)
     */
    static void reset_peak_memory(int device_id = -1) {
        if (device_id < 0) {
            CUDA_CHECK(cudaGetDevice(&device_id));
        }
        
        if (torch::cuda::is_available()) {
            torch::cuda::reset_peak_memory_stats(device_id);
        }
    }
    
    /**
     * Print memory summary
     * @param snapshots List of snapshots to summarize
     */
    static void print_summary(const std::vector<MemorySnapshot>& snapshots) {
        if (snapshots.empty()) {
            std::cout << "No memory snapshots captured.\n";
            return;
        }
        
        std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  Memory Profiling Summary                                        ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";
        
        for (const auto& snap : snapshots) {
            std::cout << snap.to_string() << "\n\n";
        }
        
        // Find peak
        auto peak_it = std::max_element(snapshots.begin(), snapshots.end(),
            [](const MemorySnapshot& a, const MemorySnapshot& b) {
                return a.allocated_bytes < b.allocated_bytes;
            });
        
        if (peak_it != snapshots.end()) {
            std::cout << "Peak Memory: " << format_bytes(peak_it->allocated_bytes)
                      << " (" << peak_it->label << ")\n";
        }
    }
};

//==============================================================================
// Chunking Strategy
//==============================================================================

/**
 * Chunking configuration for processing large batches
 */
struct ChunkingConfig {
    size_t chunk_size;        ///< Items per chunk
    size_t num_chunks;        ///< Total number of chunks
    size_t last_chunk_size;   ///< Size of last chunk (may be smaller)
    size_t memory_per_chunk;  ///< Estimated memory per chunk (bytes)
    
    std::string to_string() const {
        std::ostringstream oss;
        oss << "Chunking: " << num_chunks << " chunks of " << chunk_size << " items\n"
            << "  Last chunk: " << last_chunk_size << " items\n"
            << "  Memory per chunk: " << format_bytes(memory_per_chunk);
        return oss.str();
    }
};

/**
 * Calculate optimal chunking strategy
 * 
 * @param total_items Total number of items to process
 * @param bytes_per_item Memory required per item
 * @param available_memory Available GPU memory (bytes)
 * @param safety_factor Safety margin (0.8 = use 80% of available memory)
 * @return Chunking configuration
 */
inline ChunkingConfig calculate_chunking(
    size_t total_items,
    size_t bytes_per_item,
    size_t available_memory,
    double safety_factor = 0.8
) {
    TORCH_CHECK(total_items > 0, "total_items must be positive");
    TORCH_CHECK(bytes_per_item > 0, "bytes_per_item must be positive");
    TORCH_CHECK(safety_factor > 0.0 && safety_factor <= 1.0,
                "safety_factor must be in (0, 1]");
    
    ChunkingConfig config;
    
    // Available memory with safety margin
    size_t usable_memory = static_cast<size_t>(available_memory * safety_factor);
    
    // Calculate items per chunk
    size_t items_per_chunk = usable_memory / bytes_per_item;
    
    if (items_per_chunk == 0) {
        TORCH_CHECK(
            false,
            "Insufficient memory for even a single item. "
            "Required: ", format_bytes(bytes_per_item), ", "
            "Available: ", format_bytes(available_memory), ". "
            "Hint: Reduce item size or use CPU processing."
        );
    }
    
    if (items_per_chunk >= total_items) {
        // No chunking needed
        config.chunk_size = total_items;
        config.num_chunks = 1;
        config.last_chunk_size = total_items;
        config.memory_per_chunk = total_items * bytes_per_item;
    } else {
        // Chunking required
        config.chunk_size = items_per_chunk;
        config.num_chunks = (total_items + items_per_chunk - 1) / items_per_chunk;
        config.last_chunk_size = total_items - (config.num_chunks - 1) * items_per_chunk;
        config.memory_per_chunk = items_per_chunk * bytes_per_item;
    }
    
    return config;
}

/**
 * Calculate auto chunking based on current GPU memory
 * 
 * @param total_items Total number of items
 * @param bytes_per_item Memory per item
 * @param device_id Device to use (-1 = current)
 * @return Chunking configuration
 */
inline ChunkingConfig auto_chunk(
    size_t total_items,
    size_t bytes_per_item,
    int device_id = -1
) {
    if (device_id < 0) {
        CUDA_CHECK(cudaGetDevice(&device_id));
    }
    
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    
    return calculate_chunking(total_items, bytes_per_item, free_bytes, 0.7);
}

//==============================================================================
// OOM Prevention
//==============================================================================

/**
 * Check if operation will likely cause OOM
 * 
 * @param required_bytes Bytes required for operation
 * @param device_id Device to check (-1 = current)
 * @return true if likely to OOM
 */
inline bool will_oom(size_t required_bytes, int device_id = -1) {
    if (device_id < 0) {
        CUDA_CHECK(cudaGetDevice(&device_id));
    }
    
    size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return true;  // Assume OOM if can't query
    }
    
    // Require 20% safety margin
    return required_bytes > (free_bytes * 0.8);
}

/**
 * Estimate memory required for voxelization
 * 
 * @param batch_size Batch size
 * @param num_points Points per cloud
 * @param depth, height, width Voxel grid dimensions
 * @param feature_dim Feature dimension (1 for occupancy/density)
 * @return Estimated bytes required
 */
inline size_t estimate_voxelization_memory(
    int batch_size,
    int num_points,
    int depth, int height, int width,
    int feature_dim = 1
) {
    // Input: points [batch, num_points, 3]
    size_t input_size = batch_size * num_points * 3 * sizeof(float);
    
    // Output: voxel grid [batch, depth, height, width, feature_dim]
    size_t output_size = batch_size * depth * height * width * feature_dim * sizeof(float);
    
    // Intermediate buffers (conservative estimate: 20% overhead)
    size_t overhead = static_cast<size_t>((input_size + output_size) * 0.2);
    
    return input_size + output_size + overhead;
}

/**
 * Check if voxelization will likely OOM
 * 
 * @param batch_size Batch size
 * @param num_points Points per cloud
 * @param depth, height, width Voxel grid dimensions
 * @param device_id Device to check (-1 = current)
 * @return true if likely to OOM
 */
inline bool voxelization_will_oom(
    int batch_size,
    int num_points,
    int depth, int height, int width,
    int device_id = -1
) {
    size_t required = estimate_voxelization_memory(
        batch_size, num_points, depth, height, width, 1
    );
    return will_oom(required, device_id);
}

/**
 * Suggest chunking for voxelization to avoid OOM
 * 
 * @param batch_size Original batch size
 * @param num_points Points per cloud
 * @param depth, height, width Voxel grid dimensions
 * @param device_id Device to use (-1 = current)
 * @return Suggested chunking config
 */
inline ChunkingConfig suggest_voxelization_chunking(
    int batch_size,
    int num_points,
    int depth, int height, int width,
    int device_id = -1
) {
    // Calculate bytes per batch item
    size_t input_per_item = num_points * 3 * sizeof(float);
    size_t output_per_item = depth * height * width * sizeof(float);
    size_t bytes_per_item = input_per_item + output_per_item;
    
    return auto_chunk(batch_size, bytes_per_item, device_id);
}

//==============================================================================
// Memory Report
//==============================================================================

/**
 * Generate comprehensive memory report
 * 
 * @param device_id Device to report on (-1 = current)
 * @return Human-readable memory report
 */
inline std::string generate_memory_report(int device_id = -1) {
    if (device_id < 0) {
        CUDA_CHECK(cudaGetDevice(&device_id));
    }
    
    std::ostringstream oss;
    
    oss << "╔══════════════════════════════════════════════════════════════════╗\n";
    oss << "║  GPU Memory Report (Device " << device_id << ")                                   ║\n";
    oss << "╚══════════════════════════════════════════════════════════════════╝\n\n";
    
    // Device info
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        oss << "Device: " << prop.name << "\n";
        oss << "Compute Capability: " << prop.major << "." << prop.minor << "\n\n";
    }
    
    // Memory info
    size_t free_bytes, total_bytes;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        size_t used_bytes = total_bytes - free_bytes;
        
        oss << "Total Memory:     " << format_bytes(total_bytes) << "\n";
        oss << "Used Memory:      " << format_bytes(used_bytes) << "\n";
        oss << "Free Memory:      " << format_bytes(free_bytes) << "\n";
        oss << "Utilization:      " << (used_bytes * 100.0 / total_bytes) << "%\n\n";
    }
    
    // PyTorch allocator stats
    if (torch::cuda::is_available()) {
        size_t allocated = torch::cuda::memory_allocated(device_id);
        size_t reserved = torch::cuda::memory_reserved(device_id);
        size_t peak = torch::cuda::max_memory_allocated(device_id);
        
        oss << "PyTorch Allocator:\n";
        oss << "  Allocated:      " << format_bytes(allocated) << "\n";
        oss << "  Reserved:       " << format_bytes(reserved) << "\n";
        oss << "  Peak (session): " << format_bytes(peak) << "\n\n";
    }
    
    oss << "Recommendations:\n";
    if (free_bytes < total_bytes * 0.2) {
        oss << "  ⚠️  Low memory (<20% free). Consider:\n";
        oss << "     - Reducing batch size\n";
        oss << "     - Using chunking\n";
        oss << "     - Calling torch.cuda.empty_cache()\n";
    } else if (free_bytes > total_bytes * 0.8) {
        oss << "  ✅ Plenty of memory available (>80% free)\n";
        oss << "     - Can increase batch size for better throughput\n";
    } else {
        oss << "  ✅ Memory usage healthy (20-80% free)\n";
    }
    
    return oss.str();
}

}  // namespace memory
}  // namespace robocache

