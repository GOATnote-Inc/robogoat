/**
 * RoboCache - Production-Grade CUDA Error Handling
 * 
 * Expert-level error handling for GPU-accelerated robotics data processing.
 * Demonstrates defensive programming and fault tolerance required for
 * production CUDA libraries.
 * 
 * Features:
 * - Context-rich error messages with file/line/function info
 * - CUDA error checking with device info
 * - PyTorch tensor validation
 * - Graceful CPU fallback for OOM/device errors
 * - Thread-safe error reporting
 * 
 * @author RoboCache Team
 * @date 2025-11-05
 */

#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <sstream>
#include <string>

namespace robocache {
namespace error {

// ============================================================================
// CUDA Error Checking Macros
// ============================================================================

/**
 * CUDA_CHECK: Check CUDA runtime API calls
 * 
 * Usage:
 *   CUDA_CHECK(cudaMalloc(&ptr, size));
 *   CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
 * 
 * On error, throws std::runtime_error with:
 * - Error code and message
 * - File, line, function
 * - Device ID and name
 */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::ostringstream oss;                                            \
            oss << "CUDA Error at " << __FILE__ << ":" << __LINE__            \
                << " in " << __FUNCTION__ << "\n"                              \
                << "  Error: " << cudaGetErrorName(err) << " ("               \
                << cudaGetErrorString(err) << ")\n"                            \
                << "  Call: " << #call << "\n";                                \
            int device;                                                        \
            if (cudaGetDevice(&device) == cudaSuccess) {                       \
                cudaDeviceProp prop;                                           \
                if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {   \
                    oss << "  Device: " << device << " (" << prop.name << ")\n"; \
                }                                                              \
            }                                                                  \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

/**
 * CUDA_CHECK_LAST: Check for errors in last kernel launch
 * 
 * Usage:
 *   my_kernel<<<grid, block>>>(args);
 *   CUDA_CHECK_LAST();
 * 
 * Checks both cudaGetLastError() and cudaDeviceSynchronize()
 */
#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            std::ostringstream oss;                                            \
            oss << "CUDA Kernel Error at " << __FILE__ << ":" << __LINE__     \
                << " in " << __FUNCTION__ << "\n"                              \
                << "  Error: " << cudaGetErrorName(err) << " ("               \
                << cudaGetErrorString(err) << ")\n";                           \
            int device;                                                        \
            if (cudaGetDevice(&device) == cudaSuccess) {                       \
                cudaDeviceProp prop;                                           \
                if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {   \
                    oss << "  Device: " << device << " (" << prop.name << ")\n"; \
                }                                                              \
            }                                                                  \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
        err = cudaDeviceSynchronize();                                         \
        if (err != cudaSuccess) {                                              \
            std::ostringstream oss;                                            \
            oss << "CUDA Synchronization Error at " << __FILE__ << ":"        \
                << __LINE__ << " in " << __FUNCTION__ << "\n"                  \
                << "  Error: " << cudaGetErrorName(err) << " ("               \
                << cudaGetErrorString(err) << ")\n";                           \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

/**
 * CUDA_WARN: Non-fatal CUDA error (logs but doesn't throw)
 * 
 * Usage:
 *   CUDA_WARN(cudaMemcpy(...));  // Log error but continue
 */
#define CUDA_WARN(call)                                                        \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Warning at " << __FILE__ << ":" << __LINE__    \
                      << " in " << __FUNCTION__ << "\n"                        \
                      << "  Error: " << cudaGetErrorName(err) << " ("         \
                      << cudaGetErrorString(err) << ")\n"                      \
                      << "  Call: " << #call << "\n";                          \
        }                                                                      \
    } while (0)

// ============================================================================
// PyTorch Tensor Validation
// ============================================================================

/**
 * Validate tensor properties for CUDA operations
 * 
 * Checks:
 * - Tensor is defined (not empty)
 * - Tensor is on CUDA device
 * - Tensor is contiguous (if required)
 * - Expected number of dimensions
 * - Expected dtype (if specified)
 * 
 * @param tensor Tensor to validate
 * @param name Human-readable tensor name (for error messages)
 * @param expected_ndim Expected number of dimensions (-1 = any)
 * @param require_contiguous Whether tensor must be contiguous
 * @param expected_dtype Expected data type (c10::nullopt = any)
 */
inline void validate_tensor(
    const torch::Tensor& tensor,
    const std::string& name,
    int expected_ndim = -1,
    bool require_contiguous = true,
    c10::optional<torch::ScalarType> expected_dtype = c10::nullopt
) {
    // Check tensor is defined
    TORCH_CHECK(
        tensor.defined(),
        "Tensor '", name, "' is undefined (empty tensor)"
    );
    
    // Check tensor is on CUDA device
    TORCH_CHECK(
        tensor.is_cuda(),
        "Tensor '", name, "' must be on CUDA device, got device: ",
        tensor.device()
    );
    
    // Check contiguity
    if (require_contiguous) {
        TORCH_CHECK(
            tensor.is_contiguous(),
            "Tensor '", name, "' must be contiguous. "
            "Call .contiguous() on the tensor before passing to RoboCache."
        );
    }
    
    // Check number of dimensions
    if (expected_ndim >= 0) {
        TORCH_CHECK(
            tensor.ndimension() == expected_ndim,
            "Tensor '", name, "' has wrong number of dimensions. "
            "Expected: ", expected_ndim, ", got: ", tensor.ndimension()
        );
    }
    
    // Check dtype
    if (expected_dtype.has_value()) {
        TORCH_CHECK(
            tensor.scalar_type() == expected_dtype.value(),
            "Tensor '", name, "' has wrong dtype. "
            "Expected: ", expected_dtype.value(), ", got: ", tensor.scalar_type()
        );
    }
}

/**
 * Validate tensor shape matches expected dimensions
 * 
 * @param tensor Tensor to validate
 * @param name Human-readable tensor name
 * @param expected_shape Expected shape (use -1 for "any size")
 * 
 * Example:
 *   validate_tensor_shape(points, "points", {-1, 4096, 3});  // [batch, 4096, 3]
 */
inline void validate_tensor_shape(
    const torch::Tensor& tensor,
    const std::string& name,
    const std::vector<int64_t>& expected_shape
) {
    TORCH_CHECK(
        tensor.ndimension() == static_cast<int64_t>(expected_shape.size()),
        "Tensor '", name, "' has wrong number of dimensions. "
        "Expected: ", expected_shape.size(), ", got: ", tensor.ndimension()
    );
    
    for (size_t i = 0; i < expected_shape.size(); i++) {
        if (expected_shape[i] >= 0) {  // -1 means "any size"
            TORCH_CHECK(
                tensor.size(i) == expected_shape[i],
                "Tensor '", name, "' has wrong size at dimension ", i, ". "
                "Expected: ", expected_shape[i], ", got: ", tensor.size(i)
            );
        }
    }
}

/**
 * Validate all tensors are on the same device
 * 
 * @param tensors List of tensors to check
 * @param names List of tensor names (for error messages)
 */
inline void validate_same_device(
    const std::vector<torch::Tensor>& tensors,
    const std::vector<std::string>& names
) {
    TORCH_CHECK(
        tensors.size() == names.size(),
        "Internal error: tensors.size() != names.size()"
    );
    
    if (tensors.empty()) return;
    
    auto device = tensors[0].device();
    for (size_t i = 1; i < tensors.size(); i++) {
        TORCH_CHECK(
            tensors[i].device() == device,
            "Tensor '", names[i], "' is on different device. "
            "Expected: ", device, ", got: ", tensors[i].device(), ". "
            "All tensors must be on the same device."
        );
    }
}

// ============================================================================
// Memory Management
// ============================================================================

/**
 * Check if device has enough free memory for allocation
 * 
 * @param required_bytes Bytes needed
 * @param device_id Device to check (-1 = current device)
 * @return true if enough memory available
 */
inline bool check_memory_available(size_t required_bytes, int device_id = -1) {
    if (device_id < 0) {
        CUDA_CHECK(cudaGetDevice(&device_id));
    }
    
    size_t free_bytes, total_bytes;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        return false;  // Can't determine, assume insufficient
    }
    
    // Reserve 10% safety margin
    return free_bytes > (required_bytes * 1.1);
}

/**
 * Estimate memory required for operation
 * 
 * @param tensor_sizes List of tensor sizes (in elements)
 * @param element_size Size of each element (bytes)
 * @param overhead Additional overhead (e.g., for intermediate buffers)
 * @return Estimated bytes required
 */
inline size_t estimate_memory_required(
    const std::vector<int64_t>& tensor_sizes,
    size_t element_size,
    double overhead = 1.2  // 20% overhead by default
) {
    size_t total = 0;
    for (auto size : tensor_sizes) {
        total += size * element_size;
    }
    return static_cast<size_t>(total * overhead);
}

/**
 * Get human-readable memory size string
 * 
 * @param bytes Size in bytes
 * @return String like "1.5 GB" or "256 MB"
 */
inline std::string format_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }
    
    std::ostringstream oss;
    oss.precision(2);
    oss << std::fixed << size << " " << units[unit_idx];
    return oss.str();
}

// ============================================================================
// Graceful Fallback
// ============================================================================

/**
 * Error types that support CPU fallback
 */
enum class FallbackReason {
    OOM,                // Out of memory
    DEVICE_ERROR,       // Device error (reset, hung, etc.)
    UNSUPPORTED,        // Operation not supported on current device
    PERFORMANCE,        // CPU would be faster (e.g., tiny batch)
    UNKNOWN             // Unknown error
};

/**
 * Check if error should trigger CPU fallback
 * 
 * @param err CUDA error code
 * @return Fallback reason (or UNKNOWN if no fallback)
 */
inline FallbackReason should_fallback(cudaError_t err) {
    switch (err) {
        case cudaErrorMemoryAllocation:
        case cudaErrorOutOfMemory:
            return FallbackReason::OOM;
        
        case cudaErrorDevicesUnavailable:
        case cudaErrorDeviceUninitialized:
        case cudaErrorNoDevice:
        case cudaErrorInsufficientDriver:
            return FallbackReason::DEVICE_ERROR;
        
        case cudaErrorNotSupported:
        case cudaErrorNotPermitted:
            return FallbackReason::UNSUPPORTED;
        
        default:
            return FallbackReason::UNKNOWN;
    }
}

/**
 * Get human-readable fallback reason
 */
inline const char* fallback_reason_str(FallbackReason reason) {
    switch (reason) {
        case FallbackReason::OOM:
            return "Out of GPU memory";
        case FallbackReason::DEVICE_ERROR:
            return "GPU device error";
        case FallbackReason::UNSUPPORTED:
            return "Operation not supported on GPU";
        case FallbackReason::PERFORMANCE:
            return "CPU would be faster for this input size";
        case FallbackReason::UNKNOWN:
            return "Unknown error";
        default:
            return "Unspecified";
    }
}

// ============================================================================
// Device Information
// ============================================================================

/**
 * Get detailed device information for error reporting
 * 
 * @param device_id Device to query (-1 = current device)
 * @return String with device info (name, memory, compute capability)
 */
inline std::string get_device_info(int device_id = -1) {
    if (device_id < 0) {
        if (cudaGetDevice(&device_id) != cudaSuccess) {
            return "Device: unknown (failed to query)";
        }
    }
    
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        return "Device: unknown (failed to get properties)";
    }
    
    size_t free_bytes, total_bytes;
    std::string mem_info = "unknown";
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        mem_info = format_bytes(free_bytes) + " free / " + format_bytes(total_bytes) + " total";
    }
    
    std::ostringstream oss;
    oss << "Device " << device_id << ": " << prop.name << "\n"
        << "  Compute Capability: " << prop.major << "." << prop.minor << "\n"
        << "  Memory: " << mem_info << "\n"
        << "  SMs: " << prop.multiProcessorCount;
    
    return oss.str();
}

/**
 * Check if current device supports required compute capability
 * 
 * @param major Required major compute capability
 * @param minor Required minor compute capability
 * @return true if supported
 */
inline bool check_compute_capability(int major, int minor) {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return false;
    }
    
    return (prop.major > major) || (prop.major == major && prop.minor >= minor);
}

}  // namespace error
}  // namespace robocache

