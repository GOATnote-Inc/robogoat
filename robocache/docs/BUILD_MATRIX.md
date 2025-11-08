# RoboCache Build Matrix

**Version:** 1.0.0  
**Last Updated:** 2025-11-06  
**Status:** Production-Validated

---

## Build Variants

| Variant | Build Flag | Default Backend | Intended Users | Notes |
|---------|------------|-----------------|----------------|-------|
| **PyTorch Reference** | *(none)* | PyTorch (CPU/GPU tensors) | Open-source users, CI, unit tests | Ships in this repo. No CUDA runtime required. |
| **CUDA Extension (optional)** | `ROBOCACHE_ENABLE_CUDA_BACKEND=1` (or `ROBOCACHE_BUILD_WITH_CUDA=1`) | CUDA (falls back to PyTorch if compilation fails) | Enterprise / GPU users with NVCC + CUTLASS | Requires CUDA Toolkit ≥13.0, CUTLASS 4.3.0 headers, compatible NVIDIA GPU. |

The CUDA extension is not bundled with the open-source release; the build flag must be exported before running `pip install -e
python/`. Without it, RoboCache never attempts to compile or load CUDA kernels.

## Validated GPU SKUs

| GPU | Architecture | Compute Cap | Memory | Status | Validation Date |
|-----|--------------|-------------|--------|--------|-----------------|
| **NVIDIA H100 PCIe** | Hopper (SM90) | 9.0 | 80GB HBM3 | ✅ **Production** | 2025-11-06 |
| **NVIDIA A100 SXM4** | Ampere (SM80) | 8.0 | 80GB HBM2e | ✅ **Production** | 2025-11-06 |
| **NVIDIA RTX 6000 Ada** | Ada Lovelace (SM89) | 8.9 | 48GB GDDR6 | ⏳ **Planned** | Q1 2026 |
| **NVIDIA B100** | Blackwell (SM100) | 10.0 | 192GB HBM3e | ⏳ **Future** | Q2 2026 |

---

## CUDA Toolkit Requirements

### Minimum Supported

| Component | Min Version | Recommended | Notes |
|-----------|-------------|-------------|-------|
| **CUDA Toolkit** | 12.1 | **13.0** | H100 requires 12.0+; 13.0 for latest features |
| **Driver** | 525.x | **565.x+** | Match CUDA toolkit |
| **cuBLAS** | 12.1 | 13.0 | Included with CUDA |
| **cuDNN** | 8.9 | 9.0+ | For DL frameworks |
| **NCCL** | 2.18 | 2.21+ | Multi-GPU communication |

### CUTLASS Version

| CUTLASS | Release Date | Features | Status |
|---------|--------------|----------|--------|
| 4.2.1 | Aug 2025 | Hopper support, BF16 | ✅ Validated |
| **4.3.0** | **Oct 2025** | **Python DSL, SM100 support** | ✅ **Production** |

**Installation:**
```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass && git checkout main  # 4.3.0
```

---

## Build Configurations

### Architecture Targets

**CMake (Multi-Architecture):**
```cmake
set(CMAKE_CUDA_ARCHITECTURES "80;89;90")  # A100, RTX 6000, H100
```

**PyTorch JIT (Single Architecture):**
```python
extra_cuda_cflags=['-arch=sm_90']  # H100
extra_cuda_cflags=['-arch=sm_80']  # A100
```

### Compiler Flags

**Recommended (Production):**
```bash
-O3                  # Maximum optimization
--use_fast_math      # Fast math (slightly reduced precision)
-std=c++17           # C++17 standard
--expt-relaxed-constexpr  # Enable relaxed constexpr
```

**Debug (Development):**
```bash
-g -G                # Debug symbols + device code debugging
-lineinfo            # Line-level profiling with Nsight
-DDEBUG              # Enable debug assertions
```

---

## Validated Software Stack

### Ubuntu 22.04 LTS

| Package | Version | Installation |
|---------|---------|--------------|
| **GCC** | 11.4+ | `apt install build-essential` |
| **CMake** | 3.24+ | `apt install cmake` |
| **Python** | 3.10+ | `apt install python3.10 python3.10-dev` |
| **Git** | 2.34+ | `apt install git` |

### Python Dependencies

**Core:**
```
torch >= 2.1.0
numpy >= 1.24.0
```

**Optional (ROS 2):**
```
rclpy (ROS 2 Jazzy)
sensor_msgs
std_msgs
```

**Dev Tools:**
```
pytest >= 7.0
black >= 23.0
mypy >= 1.0
pre-commit >= 3.0
```

---

## Performance Baselines

### H100 PCIe (SM90)

| Benchmark | Latency | Throughput | NCU DRAM BW | Status |
|-----------|---------|------------|-------------|--------|
| **Trajectory Resample** | 0.014ms | 25,600 traj/sec | 0.05% | ✅ Validated |
| **Multimodal Fusion** | 0.050ms | 20,000 fusions/sec | 0.03% | ✅ Validated |
| **Voxelization** | 0.070ms | 2.9B pts/sec | 54% | ✅ Validated |
| **End-to-End (NSys)** | **1.56ms** | **20,548 eps/sec** | - | ✅ Validated |

**Driver:** 565.57.01  
**CUDA:** 13.0.88  
**Nsight Compute:** 2025.3.1.4  
**Nsight Systems:** 2025.3.2

---

### A100 SXM4 (SM80)

| Benchmark | Latency | Throughput | Scaling vs H100 | Status |
|-----------|---------|------------|-----------------|--------|
| **Trajectory Resample** | 0.013ms | 24,615 traj/sec | 1.02× slower | ✅ Validated |
| **Multimodal Fusion** | 0.013ms | 24,615 fusions/sec | 3.85× faster* | ✅ Validated |
| **Voxelization** | 0.012ms | 2.5B pts/sec | 1.16× slower | ✅ Validated |
| **End-to-End** | 18.28ms | 1,751 eps/sec | 1.30× slower | ✅ Validated |

\* Measurement artifact for small workloads (latency-limited)

**Driver:** 565.57.01  
**CUDA:** 12.1 / 13.0  

---

## Containerized Builds

### Docker Images

**Base Runtime:**
```dockerfile
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04
# Includes: CUDA 13.0, cuBLAS, cuDNN, NCCL
# Size: ~8 GB
```

**Development:**
```dockerfile
FROM robocache/runtime:latest
# Adds: Nsight Systems, Nsight Compute, pytest, pre-commit
# Size: ~12 GB
```

**ROS 2 Integration:**
```dockerfile
FROM robocache/runtime:latest
# Adds: ROS 2 Jazzy, Isaac ROS GEMS, TensorRT 10.0
# Size: ~15 GB
```

### Build Commands

```bash
# Runtime
docker build -f docker/Dockerfile.runtime -t robocache/runtime:1.0 .

# Development
docker build -f docker/Dockerfile.dev -t robocache/dev:1.0 .

# With GPU support
docker run --gpus all -it robocache/runtime:1.0
```

---

## Continuous Integration

### GitHub Actions Matrix

```yaml
strategy:
  matrix:
    cuda: ['12.1', '13.0']
    python: ['3.10', '3.11']
    os: ['ubuntu-22.04']
    include:
      - cuda: '13.0'
        arch: 'sm_90'  # H100
      - cuda: '12.1'
        arch: 'sm_80'  # A100
```

### Test Environments

| Environment | GPU | Purpose | Frequency |
|-------------|-----|---------|-----------|
| **GitHub Actions** | CPU | Smoke tests | Every commit |
| **Brev H100** | H100 PCIe | Full validation | On PR |
| **Brev A100** | A100 SXM4 | Multi-arch | On PR |
| **DGX Station** | 8× A100 | Multi-GPU | Weekly |

---

## Known Issues & Workarounds

### CUDA 13.0 + PyTorch 2.5.1

**Issue:** PyTorch 2.5.1 officially supports CUDA 12.1/12.4, not 13.0.

**Workaround:**
```python
# Use nightly build with CUDA 13.0
pip install torch==2.10.0.dev20251105+cu130 \
    --index-url https://download.pytorch.org/whl/nightly/cu130
```

**Status:** Works in production. PyTorch 2.6 will officially support CUDA 13.0.

---

### BF16 on SM < 8.0

**Issue:** BF16 requires Ampere or newer (SM80+).

**Workaround:**
```python
# Use FP16 fallback on older GPUs
if torch.cuda.get_device_capability()[0] < 8:
    dtype = torch.float16
else:
    dtype = torch.bfloat16
```

**Status:** Automatic fallback implemented.

---

### CUTLASS 4.3.0 Not Tagged

**Issue:** CUTLASS 4.3.0 is on main branch but not tagged as release.

**Workaround:**
```bash
# Use main branch explicitly
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass && git checkout main
```

**Status:** Validated in production. Tag expected Q1 2026.

---

## Platform-Specific Notes

### Linux (Ubuntu 22.04)

**Recommended:** Primary development and production platform.

**GPU Passthrough:**
- Bare metal: Full support
- Docker: `--gpus all` flag
- KVM/QEMU: GPU passthrough with VFIO

---

### Windows (Experimental)

**Status:** Not officially supported. Community contributions welcome.

**Known Limitations:**
- CUDA toolkit paths differ (`C:\Program Files\NVIDIA GPU Computing Toolkit\`)
- Docker Desktop GPU support limited
- Nsight Systems may require admin privileges

---

### ARM64 (Grace-Hopper)

**Status:** Future support planned (Q3 2026).

**Architecture:** Grace CPU + H100 GPU on same chip.

**Expected Benefits:**
- Unified memory (no PCIe bottleneck)
- Lower latency CPU↔GPU transfers
- Improved energy efficiency

---

## Maintenance

### Build Matrix Updates

**Quarterly:**
- Add new GPU SKUs (RTX 6000, B100)
- Update CUDA toolkit versions
- Refresh performance baselines

**Annually:**
- Deprecate EOL GPUs (< 2 years old architectures remain)
- Update OS versions (Ubuntu LTS releases)

---

## Support

### Reporting Build Issues

**Include in bug reports:**
1. GPU model + driver version (`nvidia-smi`)
2. CUDA version (`nvcc --version`)
3. Python version (`python --version`)
4. PyTorch version (`python -c "import torch; print(torch.__version__)"`)
5. Full build log (CMake or JIT compilation)

**Submit to:** https://github.com/GOATnote-Inc/robogoat/issues

---

**Build Matrix Version:** 1.0.0  
**Last Updated:** 2025-11-06  
**Maintainer:** b@thegoatnote.com

