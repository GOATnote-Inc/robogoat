/**
 * RoboCache - Multi-GPU Safety and Stream Management
 * 
 * Production-grade multi-GPU programming utilities for safe and efficient
 * multi-device execution. Demonstrates expert CUDA stream semantics and
 * device management.
 * 
 * Features:
 * - CUDAGuard for safe device switching
 * - Stream pool management
 * - Cross-device synchronization
 * - Multi-GPU workload distribution
 * - Device affinity management
 * 
 * @author RoboCache Team
 * @date 2025-11-05
 */

#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include "error_handling.cuh"

namespace robocache {
namespace multigpu {

using namespace robocache::error;

//==============================================================================
// CUDAGuard: RAII Device Switcher
//==============================================================================

/**
 * RAII-style CUDA device guard
 * 
 * Automatically restores previous device on scope exit.
 * Thread-safe. No overhead if device doesn't change.
 * 
 * Usage:
 *   {
 *       CUDAGuard guard(target_device);
 *       // Work on target_device
 *   }  // Automatically restores previous device
 * 
 * Similar to PyTorch's c10::cuda::CUDAGuard but with explicit error checking.
 */
class CUDAGuard {
public:
    /**
     * Constructor: Switch to target device
     * @param device Target device ID (-1 = no-op guard)
     */
    explicit CUDAGuard(int device = -1)
        : prev_device_(-1), target_device_(device) {
        if (device >= 0) {
            CUDA_CHECK(cudaGetDevice(&prev_device_));
            if (prev_device_ != target_device_) {
                CUDA_CHECK(cudaSetDevice(target_device_));
            } else {
                // No switch needed
                prev_device_ = -1;
            }
        }
    }
    
    /**
     * Constructor: Switch to tensor's device
     * @param tensor Tensor whose device to switch to
     */
    explicit CUDAGuard(const torch::Tensor& tensor)
        : CUDAGuard(tensor.is_cuda() ? tensor.device().index() : -1) {}
    
    /**
     * Destructor: Restore previous device
     */
    ~CUDAGuard() {
        if (prev_device_ >= 0) {
            // Don't throw in destructor - just log error
            cudaError_t err = cudaSetDevice(prev_device_);
            if (err != cudaSuccess) {
                std::cerr << "Warning: Failed to restore device " << prev_device_
                          << " in CUDAGuard destructor: " << cudaGetErrorString(err) << "\n";
            }
        }
    }
    
    // Non-copyable, non-movable (RAII semantics)
    CUDAGuard(const CUDAGuard&) = delete;
    CUDAGuard& operator=(const CUDAGuard&) = delete;
    CUDAGuard(CUDAGuard&&) = delete;
    CUDAGuard& operator=(CUDAGuard&&) = delete;
    
    /**
     * Get current device (after guard applied)
     */
    int current_device() const { return target_device_; }
    
    /**
     * Get previous device (before guard)
     */
    int previous_device() const { return prev_device_; }

private:
    int prev_device_;    ///< Device to restore on destruction
    int target_device_;  ///< Target device (for this guard)
};

//==============================================================================
// StreamGuard: RAII Stream Switcher
//==============================================================================

/**
 * RAII-style CUDA stream guard
 * 
 * Automatically restores previous stream on scope exit.
 * Useful for temporarily switching to a different stream.
 * 
 * Usage:
 *   {
 *       StreamGuard guard(my_stream);
 *       // Work on my_stream
 *   }  // Automatically restores previous stream
 */
class StreamGuard {
public:
    explicit StreamGuard(cudaStream_t stream)
        : prev_stream_(c10::cuda::getCurrentCUDAStream()), stream_(stream) {
        c10::cuda::setCurrentCUDAStream(c10::cuda::getStreamFromPool(false, stream_));
    }
    
    ~StreamGuard() {
        c10::cuda::setCurrentCUDAStream(prev_stream_);
    }
    
    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
    
    cudaStream_t stream() const { return stream_; }

private:
    c10::cuda::CUDAStream prev_stream_;
    cudaStream_t stream_;
};

//==============================================================================
// Stream Pool Management
//==============================================================================

/**
 * Per-device stream pool for efficient multi-stream execution
 * 
 * Creates and manages multiple CUDA streams per device.
 * Streams are reused to avoid creation overhead.
 * Thread-safe.
 */
class StreamPool {
public:
    /**
     * Get or create stream pool for device
     * @param device_id Device ID
     * @param num_streams Number of streams per device (default: 4)
     * @return Reference to stream pool
     */
    static std::vector<cudaStream_t>& get_streams(int device_id, int num_streams = 4) {
        static std::unordered_map<int, std::vector<cudaStream_t>> pools;
        static std::mutex mutex;
        
        std::lock_guard<std::mutex> lock(mutex);
        
        auto it = pools.find(device_id);
        if (it == pools.end()) {
            // Create streams for this device
            CUDAGuard guard(device_id);
            
            std::vector<cudaStream_t> streams(num_streams);
            for (int i = 0; i < num_streams; i++) {
                CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
            }
            
            pools[device_id] = streams;
            return pools[device_id];
        }
        
        return it->second;
    }
    
    /**
     * Synchronize all streams on device
     * @param device_id Device ID (-1 = current device)
     */
    static void synchronize_all(int device_id = -1) {
        if (device_id < 0) {
            CUDA_CHECK(cudaGetDevice(&device_id));
        }
        
        auto& streams = get_streams(device_id);
        CUDAGuard guard(device_id);
        
        for (auto stream : streams) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    
    /**
     * Destroy all streams (cleanup)
     * Call this before program exit to avoid CUDA warnings
     */
    static void destroy_all() {
        static std::unordered_map<int, std::vector<cudaStream_t>> pools;
        static std::mutex mutex;
        
        std::lock_guard<std::mutex> lock(mutex);
        
        for (auto& [device_id, streams] : pools) {
            CUDAGuard guard(device_id);
            for (auto stream : streams) {
                CUDA_WARN(cudaStreamDestroy(stream));
            }
        }
        
        pools.clear();
    }
};

//==============================================================================
// Multi-Device Utilities
//==============================================================================

/**
 * Get number of available CUDA devices
 * @return Number of devices (0 if CUDA unavailable)
 */
inline int get_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

/**
 * Check if multi-GPU is available
 * @return true if 2+ GPUs available
 */
inline bool is_multigpu_available() {
    return get_device_count() >= 2;
}

/**
 * Get device ID from PyTorch tensor
 * @param tensor Input tensor
 * @return Device ID, or -1 if not CUDA tensor
 */
inline int get_tensor_device(const torch::Tensor& tensor) {
    if (!tensor.is_cuda()) {
        return -1;
    }
    return tensor.device().index();
}

/**
 * Check if two tensors are on the same device
 */
inline bool same_device(const torch::Tensor& a, const torch::Tensor& b) {
    if (!a.is_cuda() || !b.is_cuda()) {
        return !a.is_cuda() && !b.is_cuda();  // Both CPU = same
    }
    return a.device() == b.device();
}

/**
 * Check if tensors can be accessed with peer-to-peer
 * 
 * @param device_a First device
 * @param device_b Second device
 * @return true if P2P access enabled
 */
inline bool can_access_peer(int device_a, int device_b) {
    if (device_a == device_b) {
        return true;  // Same device
    }
    
    int can_access = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_access, device_a, device_b);
    if (err != cudaSuccess) {
        return false;
    }
    
    return can_access != 0;
}

/**
 * Enable peer-to-peer access between two devices
 * 
 * @param device_a First device
 * @param device_b Second device
 * @return true if P2P enabled (or already enabled)
 */
inline bool enable_peer_access(int device_a, int device_b) {
    if (device_a == device_b) {
        return true;
    }
    
    if (!can_access_peer(device_a, device_b)) {
        return false;
    }
    
    CUDAGuard guard(device_a);
    cudaError_t err = cudaDeviceEnablePeerAccess(device_b, 0);
    
    // Already enabled = success
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();  // Clear error
        return true;
    }
    
    return err == cudaSuccess;
}

//==============================================================================
// Multi-GPU Workload Distribution
//==============================================================================

/**
 * Split batch across multiple GPUs
 * 
 * Distributes batch_size items across available GPUs.
 * Returns (device_id, start_idx, count) for each GPU.
 * 
 * @param batch_size Total batch size
 * @param num_devices Number of GPUs to use (-1 = all available)
 * @return Vector of (device_id, start_idx, count)
 * 
 * Example:
 *   auto splits = split_batch(100, 4);
 *   // splits = [(0, 0, 25), (1, 25, 25), (2, 50, 25), (3, 75, 25)]
 */
inline std::vector<std::tuple<int, int, int>> split_batch(int batch_size, int num_devices = -1) {
    if (num_devices < 0) {
        num_devices = get_device_count();
    }
    
    TORCH_CHECK(num_devices > 0, "No CUDA devices available");
    TORCH_CHECK(batch_size > 0, "batch_size must be positive");
    
    std::vector<std::tuple<int, int, int>> splits;
    
    int items_per_device = (batch_size + num_devices - 1) / num_devices;
    
    for (int device_id = 0; device_id < num_devices; device_id++) {
        int start_idx = device_id * items_per_device;
        if (start_idx >= batch_size) {
            break;  // No more items for this device
        }
        
        int count = std::min(items_per_device, batch_size - start_idx);
        splits.emplace_back(device_id, start_idx, count);
    }
    
    return splits;
}

/**
 * Synchronize across all devices
 * 
 * Ensures all kernels on all devices have completed.
 * Useful before returning results to user.
 */
inline void synchronize_all_devices() {
    int num_devices = get_device_count();
    for (int i = 0; i < num_devices; i++) {
        CUDAGuard guard(i);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

//==============================================================================
// Cross-Device Data Transfer
//==============================================================================

/**
 * Copy tensor to another device (async)
 * 
 * @param src Source tensor
 * @param target_device Target device ID
 * @param stream Stream for async copy (default: current stream)
 * @return Tensor on target device
 */
inline torch::Tensor copy_to_device_async(
    const torch::Tensor& src,
    int target_device,
    cudaStream_t stream = 0
) {
    TORCH_CHECK(src.is_cuda(), "Source tensor must be on CUDA");
    
    int src_device = src.device().index();
    if (src_device == target_device) {
        return src;  // Already on target device
    }
    
    // Allocate on target device
    CUDAGuard guard(target_device);
    auto dst = torch::empty_like(src, torch::TensorOptions().device(torch::Device(torch::kCUDA, target_device)));
    
    // Copy (supports P2P if available)
    CUDA_CHECK(cudaMemcpyAsync(
        dst.data_ptr(),
        src.data_ptr(),
        src.nbytes(),
        cudaMemcpyDeviceToDevice,
        stream
    ));
    
    return dst;
}

/**
 * Gather tensors from multiple devices to single device
 * 
 * @param tensors Tensors on different devices
 * @param target_device Target device to gather to
 * @return Concatenated tensor on target device
 */
inline torch::Tensor gather_from_devices(
    const std::vector<torch::Tensor>& tensors,
    int target_device
) {
    TORCH_CHECK(!tensors.empty(), "Cannot gather from empty tensor list");
    
    // Copy all tensors to target device
    std::vector<torch::Tensor> device_tensors;
    for (const auto& tensor : tensors) {
        if (tensor.device().index() == target_device) {
            device_tensors.push_back(tensor);
        } else {
            device_tensors.push_back(copy_to_device_async(tensor, target_device));
        }
    }
    
    // Synchronize to ensure all copies complete
    CUDAGuard guard(target_device);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Concatenate along batch dimension
    return torch::cat(device_tensors, 0);
}

//==============================================================================
// Device Affinity Hints
//==============================================================================

/**
 * Get optimal device for current thread
 * 
 * Returns device ID based on thread affinity or round-robin.
 * Useful for multi-threaded data loading.
 * 
 * @return Device ID (0 to num_devices-1)
 */
inline int get_optimal_device_for_thread() {
    static std::atomic<int> counter{0};
    int num_devices = get_device_count();
    if (num_devices <= 1) {
        return 0;
    }
    
    // Simple round-robin for now
    // TODO: Consider CPU affinity, NUMA nodes, etc.
    return (counter.fetch_add(1, std::memory_order_relaxed)) % num_devices;
}

}  // namespace multigpu
}  // namespace robocache

