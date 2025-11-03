# Build Instructions

Complete guide to building RoboCache from source on various platforms.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installing CUTLASS 4.3.0](#installing-cutlass-430)
- [Building RoboCache](#building-robocache)
- [Installation Verification](#installation-verification)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+, CentOS 7+)
- **GPU**: NVIDIA GPU with Compute Capability 8.0+ (A100, H100, RTX 4090)
  - H100 (sm_90) - Recommended
  - A100 (sm_80) - Supported
  - RTX 4090 (sm_89) - Supported for testing

### Software Requirements

1. **CUDA Toolkit 13.x or later**
   ```bash
   # Check CUDA version
   nvcc --version

   # Should show: cuda_13.x or later
   ```

   If not installed, download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

2. **CMake 3.18+**
   ```bash
   cmake --version

   # If too old, install newer version:
   wget https://github.com/Kitware/CMake/releases/download/v3.27.0/cmake-3.27.0-linux-x86_64.sh
   sudo sh cmake-3.27.0-linux-x86_64.sh --prefix=/usr/local --skip-license
   ```

3. **Python 3.8+**
   ```bash
   python3 --version
   ```

4. **PyTorch 2.0+ with CUDA support**
   ```bash
   python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

   # Should show: PyTorch 2.x.x, CUDA 13.x

   # If not installed:
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

5. **Build tools**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install build-essential git python3-dev

   # CentOS/RHEL
   sudo yum groupinstall "Development Tools"
   sudo yum install python3-devel
   ```

## Installing CUTLASS 4.3.0

### Option 1: System-wide Installation (Recommended)

```bash
# Clone CUTLASS repository
cd /tmp
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v4.3.0

# Copy headers to system include directory
sudo mkdir -p /usr/local/include
sudo cp -r include/cutlass /usr/local/include/

# Verify installation
ls /usr/local/include/cutlass/cutlass.h
```

### Option 2: Custom Installation Path

```bash
# Clone to custom location
cd $HOME
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v4.3.0

# Note the path - you'll need it for CMake
export CUTLASS_DIR=$HOME/cutlass
```

When building RoboCache with custom CUTLASS path:
```bash
cmake .. -DCUTLASS_DIR=$HOME/cutlass
```

## Building RoboCache

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/robocache.git
cd robocache
```

### Step 2: Create Build Directory

```bash
mkdir build
cd build
```

### Step 3: Configure with CMake

**For H100 (Recommended):**
```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90
```

**For A100:**
```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80
```

**For RTX 4090:**
```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89
```

**For Multiple Architectures:**
```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80;89;90"
```

**With Custom CUTLASS Path:**
```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUTLASS_DIR=$HOME/cutlass
```

**Debug Build (for development):**
```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CUDA_ARCHITECTURES=90
```

### Step 4: Build

```bash
# Use all CPU cores
make -j$(nproc)

# Or specify number of parallel jobs
make -j8
```

Expected output:
```
Scanning dependencies of target benchmark_trajectory_resample
[ 25%] Building CUDA object CMakeFiles/benchmark_trajectory_resample.dir/benchmarks/benchmark_trajectory_resample.cu.o
[ 50%] Linking CUDA executable benchmark_trajectory_resample
[ 50%] Built target benchmark_trajectory_resample
Scanning dependencies of target robocache_cuda
[ 75%] Building CUDA object CMakeFiles/robocache_cuda.dir/kernels/cutlass/trajectory_resample.cu.o
[100%] Building CUDA object CMakeFiles/robocache_cuda.dir/kernels/cutlass/trajectory_resample_torch.cu.o
[100%] Linking CUDA shared module ../python/robocache/robocache_cuda.so
[100%] Built target robocache_cuda
```

### Step 5: Install Python Package

```bash
# From robocache root directory
cd ..
pip install -e .
```

## Installation Verification

### Test 1: Check Python Import

```python
python3 << EOF
import robocache
robocache.print_installation_info()
EOF
```

Expected output:
```
============================================================
RoboCache Installation Info
============================================================
Version:              0.1.0
CUDA Extension:       ✓
PyTorch:              ✓
PyTorch Version:      2.1.0
CUDA Available:       ✓
CUDA Version:         13.1
GPU Count:            1
GPU Name:             NVIDIA H100 PCIe
GPU Memory:           80.0 GB
============================================================
```

### Test 2: Run Benchmark

```bash
cd build
./benchmark_trajectory_resample
```

Should complete without errors and show performance results.

### Test 3: Run Python Example

```bash
cd examples
python3 basic_usage.py
```

Should show trajectory resampling examples and performance comparison.

### Test 4: Basic Functionality Test

```python
python3 << EOF
import torch
import robocache

# Create test data
data = torch.randn(64, 100, 32, dtype=torch.bfloat16, device='cuda')
src_times = torch.linspace(0, 1, 100, device='cuda').expand(64, -1)
tgt_times = torch.linspace(0, 1, 50, device='cuda').expand(64, -1)

# Resample
result = robocache.resample_trajectories(data, src_times, tgt_times)

print(f"Input shape:  {data.shape}")
print(f"Output shape: {result.shape}")
print("✓ RoboCache is working!")
EOF
```

## Troubleshooting

### Problem: CMake can't find CUTLASS

**Error:**
```
CMake Error: CUTLASS not found. Please install CUTLASS 4.3.0 or set CUTLASS_DIR.
```

**Solution:**
```bash
# Verify CUTLASS installation
ls /usr/local/include/cutlass/cutlass.h

# If not found, reinstall CUTLASS (see above)
# Or specify custom path:
cmake .. -DCUTLASS_DIR=/path/to/cutlass
```

### Problem: CUDA version too old

**Error:**
```
CMake Error: CUDA 13.x or later required (found 12.1)
```

**Solution:**
Install CUDA 13.x from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

### Problem: PyTorch not found

**Error:**
```
WARNING: PyTorch not found. Python bindings will not be built.
```

**Solution:**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Problem: Architecture mismatch

**Error:**
```
no kernel image is available for execution on the device
```

**Solution:**
Your GPU's compute capability doesn't match the compiled architecture. Rebuild with correct architecture:

```bash
# Check your GPU's compute capability
python3 -c "import torch; print(torch.cuda.get_device_capability())"

# Example output: (9, 0) for H100
# Rebuild with matching architecture:
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
make clean && make -j$(nproc)
```

### Problem: Out of memory during compilation

**Error:**
```
c++: fatal error: Killed signal terminated program cc1plus
```

**Solution:**
Reduce parallel compilation:
```bash
make -j2  # Use only 2 parallel jobs instead of all cores
```

Or increase swap space:
```bash
# Create 16GB swap file
sudo dd if=/dev/zero of=/swapfile bs=1G count=16
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Problem: Import error in Python

**Error:**
```python
ImportError: librobocache_cuda.so: cannot open shared object file
```

**Solution:**
The extension wasn't copied to the right location. Reinstall:
```bash
cd robocache
pip install -e . --force-reinstall
```

### Problem: Slow performance

**Issue:** Benchmark shows much slower than expected performance.

**Debugging steps:**

1. Check GPU frequency scaling:
   ```bash
   nvidia-smi -q -d CLOCK | grep -A 2 "Graphics"
   # Should show maximum clock speed

   # If throttled, check power/thermal:
   nvidia-smi -q -d POWER,TEMPERATURE
   ```

2. Verify correct GPU is being used:
   ```python
   import torch
   print(torch.cuda.get_device_name(0))  # Should show H100/A100/etc.
   ```

3. Check CUDA compilation arch:
   ```bash
   # Look for "ptxas info" in build output
   make VERBOSE=1
   # Should show: --gpu-name=sm_90 for H100
   ```

4. Profile with nsys:
   ```bash
   nsys profile --stats=true ./benchmark_trajectory_resample
   ```

## Advanced Build Options

### Enable verbose build output

```bash
make VERBOSE=1
```

Shows all compiler commands for debugging.

### Custom CUDA flags

```bash
cmake .. -DCMAKE_CUDA_FLAGS="-O3 -use_fast_math -Xcompiler -march=native"
```

### Build only C++ benchmark (no Python)

```bash
cmake .. -DBUILD_TORCH_EXTENSION=OFF
make benchmark_trajectory_resample
```

### Install to custom prefix

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/robocache-install
make install
```

## Platform-Specific Notes

### Ubuntu 20.04/22.04

No special configuration needed. Follow standard instructions.

### CentOS 7

Requires newer devtoolset for C++17 support:
```bash
sudo yum install centos-release-scl
sudo yum install devtoolset-11
scl enable devtoolset-11 bash
```

### Docker

See `Dockerfile` for containerized build:
```bash
docker build -t robocache:latest .
docker run --gpus all -it robocache:latest
```

## Next Steps

After successful installation:

1. **Run examples**: `cd examples && python3 basic_usage.py`
2. **Read documentation**: Check `docs/` for more information
3. **Benchmark your system**: `cd build && ./benchmark_trajectory_resample`
4. **Integrate with your code**: See README.md for usage examples

## Getting Help

- **Issues**: https://github.com/yourusername/robocache/issues
- **Discussions**: https://github.com/yourusername/robocache/discussions
- **Email**: [email protected]
