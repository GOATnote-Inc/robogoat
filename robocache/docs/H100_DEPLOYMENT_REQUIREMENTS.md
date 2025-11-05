# H100 Deployment Requirements

**Objective:** Document what's needed to deploy and validate RoboCache on H100  
**Date:** November 5, 2025  
**Status:** Build Environment Blocked

---

## Current Status

### ✅ Completed
1. **Kernel implementations** - All CUDA kernels written and locally validated
2. **NCU profiling infrastructure** - Automated benchmark scripts ready
3. **Warp optimizations** - Phase 1 complete (persistent threads, warp primitives)
4. **Documentation** - Expert-level technical specifications

### ❌ Blocked
1. **H100 build environment** - PyTorch C++ extension build failing
2. **TMA kernel validation** - Requires working H100 build
3. **End-to-end NCU profiling** - Requires deployed kernels

---

## H100 Environment Details

**Instance:** `awesome-gpu-name` (Shadeform via Brev)
```
GPU: NVIDIA H100 PCIe (80 GB HBM3)
CUDA: 13.0
PyTorch: 2.10.0.dev20251101+cu130
Python: 3.10
OS: Ubuntu 20.04 / 22.04
```

**Access:**
```bash
brev shell awesome-gpu-name --dir /workspace
```

---

## Build Environment Issues

### Issue 1: PyTorch ABI Mismatch

**Symptom:**
```python
RuntimeError: RoboCache CUDA extension is not available.
Import error: undefined symbol: _ZN8pybind116detail11type_casterIN2at6TensorEvE4loadENS_6handleEb
```

**Root Cause:**
- Old `.so` file built with different PyTorch version
- C++ ABI incompatibility between build-time and runtime PyTorch

**Required Fix:**
```bash
# Remove old cached builds
rm -f python/robocache/robocache_cuda.so
rm -rf /root/.cache/torch_extensions/py310_cu130/robocache_cuda

# Force JIT recompilation
# (Current issue: permission denied on cache directory)
sudo rm -rf /root/.cache/torch_extensions/py310_cu130/robocache_cuda

# Then import robocache (should trigger JIT rebuild)
python3 -c "import sys; sys.path.insert(0, '/workspace/robocache/python'); import robocache"
```

**Status:** ⏳ Pending execution on H100

---

### Issue 2: CMake Build Path

**Alternative Approach:** Manual CMake build (bypass JIT)

**Requirements:**
```bash
# Check CUDA compiler
nvcc --version  # Should be 13.x

# Check CMake
cmake --version  # Should be 3.18+

# Build manually
cd /workspace/robocache
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90  # SM90 for H100
make -j$(nproc)

# Install extension
cp robocache_cuda.so ../python/robocache/
```

**Status:** ⏳ Not yet attempted (requires H100 shell access)

---

## Deployment Checklist

### Phase 1: Environment Setup ⏳

- [ ] SSH into H100 instance
- [ ] Remove old cached builds
- [ ] Verify CUDA 13.x installation
- [ ] Verify PyTorch 2.10+ with CUDA support
- [ ] Test basic CUDA kernel compilation

**Commands:**
```bash
brev shell awesome-gpu-name --dir /workspace
sudo rm -rf /root/.cache/torch_extensions
nvcc --version
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

### Phase 2: Build Validation ⏳

- [ ] Attempt JIT compilation (via `import robocache`)
- [ ] If JIT fails, try manual CMake build
- [ ] Verify kernel functions are loadable
- [ ] Run quick smoke test (small batch)

**Commands:**
```bash
cd /workspace/robocache
python3 << EOF
import sys
sys.path.insert(0, '/workspace/robocache/python')
import torch
import robocache

print(f"RoboCache version: {robocache.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Smoke test
B, S, T, D = 2, 10, 8, 16
src = torch.randn(B, S, D, dtype=torch.bfloat16, device='cuda')
src_t = torch.linspace(0, 1, S, device='cuda').unsqueeze(0).expand(B, -1).contiguous()
tgt_t = torch.linspace(0, 1, T, device='cuda').unsqueeze(0).expand(B, -1).contiguous()

result = robocache.resample_trajectories(src, src_t, tgt_t)
print(f"✅ Smoke test passed: {result.shape}")
EOF
```

---

### Phase 3: Baseline Kernel NCU Profiling ⏳

- [ ] Run baseline kernel (`trajectory_resample_optimized_v2`)
- [ ] Profile with NCU (DRAM BW, L1 cache, SM util, latency)
- [ ] Validate 23.76% DRAM BW from previous measurements
- [ ] Document in `docs/profiling/NCU_BASELINE_H100.md`

**Commands:**
```bash
cd /workspace/robocache

# Run benchmark
python3 benchmarks/benchmark_trajectory_baseline.py

# NCU profiling
/usr/local/cuda/bin/ncu \
  --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,smsp__inst_executed.avg.per_cycle_active \
  --target-processes all \
  python3 benchmarks/benchmark_trajectory_baseline.py
```

---

### Phase 4: Warp Kernel Deployment ⏳

- [ ] Rebuild with warp-optimized kernel (`trajectory_resample_tma_v2`)
- [ ] Run correctness tests (compare with baseline)
- [ ] Profile with NCU (expect improved DRAM BW)
- [ ] Document results

**Commands:**
```bash
cd /workspace/robocache
python3 benchmarks/benchmark_tma_comparison.py
```

---

### Phase 5: TMA Integration (Future) ⏳

- [ ] Integrate CuTe TMA wrappers
- [ ] Profile with NCU (target 60-80% DRAM BW)
- [ ] Validate 2-3x speedup over baseline
- [ ] Document TMA benefits

---

### Phase 6: Voxelization NCU Profiling ⏳

- [ ] Deploy voxelization kernel
- [ ] Profile occupancy computation
- [ ] Measure atomic operation throughput
- [ ] Document results

---

### Phase 7: Multimodal Fusion Validation ✅

- **Status:** Already validated in previous context
- **Results:** 81.66 µs, 20.45% L1 cache, 0.52% DRAM (optimal)
- **Action:** Re-run for confirmation

---

## Known Environment Constraints

### 1. Permission Issues
```
rm: cannot remove '/root/.cache/torch_extensions': Permission denied
```

**Workaround:** Use `sudo` or switch to user-writable cache directory
```bash
export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
```

### 2. Brev Shell Limitations
- Commands must be piped via `brev shell awesome-gpu-name --dir /workspace`
- Interactive sessions timeout after inactivity
- File syncing (`brev rsync`) sometimes unreliable

**Workaround:** Use direct `cat` injection for code updates
```bash
cat << 'EOF' | brev shell awesome-gpu-name --dir /workspace
cat > /workspace/robocache/python/robocache/fix.py << 'PYEOF'
# Fixed code here
PYEOF
EOF
```

### 3. Build Cache Contamination
- Old `.so` files persist across sessions
- Need to manually clear before rebuilding

**Workaround:** Always remove cache before builds
```bash
rm -rf /root/.cache/torch_extensions
rm -f python/robocache/robocache_cuda.so
```

---

## Success Criteria

### Minimum Viable Deployment ✅
- [x] At least one kernel (trajectory resample) deployed and functional
- [x] NCU profiling data collected (23.76% DRAM BW confirmed)
- [x] Baseline established for warp optimizations

### Full Deployment ⏳
- [ ] All kernels (trajectory, multimodal, voxelization) deployed
- [ ] Warp-optimized kernels validated on H100
- [ ] NCU profiling for all kernels
- [ ] End-to-end dataloader benchmarks (RT-X, CALVIN, RoboMimic)

### Production Deployment (Future) ⏳
- [ ] TMA integration (60-80% DRAM BW target)
- [ ] Prebuilt wheels for H100 (avoid user JIT compilation)
- [ ] Docker container with all dependencies
- [ ] CI/CD pipeline for automated H100 testing

---

## Alternative Deployment Strategies

### Option A: Local Development → Remote Testing
**Current Approach**

**Pros:**
- Write code locally with full IDE support
- Test basic correctness on local GPU (if available)
- Deploy to H100 only for performance validation

**Cons:**
- Brev shell latency (commands take 10-30 seconds)
- File sync unreliable
- Build environment mismatches

---

### Option B: Remote Development on H100
**Direct Development**

**Pros:**
- No sync issues (write code directly on H100)
- Immediate feedback (no deploy delay)
- Guaranteed environment consistency

**Cons:**
- No local IDE (must use vim/nano or remote VS Code)
- Session timeouts require reconnection
- Less comfortable development experience

---

### Option C: Docker Container (Recommended for Production)

**Approach:** Ship prebuilt container with all dependencies

**Dockerfile:**
```dockerfile
FROM nvcr.io/nvidia/pytorch:24.10-py3  # H100-optimized PyTorch

# Install CUTLASS 4.3.0
RUN git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass && \
    cd /opt/cutlass && git checkout v4.3.0

# Install RoboCache
COPY . /opt/robocache
RUN cd /opt/robocache && pip install -e .

# Prebuild CUDA extensions
RUN python3 -c "import robocache; print('Extensions built successfully')"

# Set environment
ENV TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
ENV CUTLASS_PATH=/opt/cutlass

CMD ["/bin/bash"]
```

**Build & Deploy:**
```bash
docker build -t robocache:h100 .
docker run --gpus all -it robocache:h100

# Inside container
python3 -c "import robocache; print(robocache.__version__)"
```

**Pros:**
- Reproducible environment
- No JIT compilation (extensions prebuilt)
- Easy distribution to users

**Cons:**
- Large container size (~10 GB)
- Requires Docker registry (Docker Hub, NVIDIA NGC)

---

## Next Steps

### Immediate Actions (This Session)
1. ✅ Document deployment requirements (this file)
2. ⏳ Attempt H100 build fix (clear cache, JIT rebuild)
3. ⏳ Validate baseline kernel on H100
4. ⏳ Document results

### Short-term (Next Session with H100 Access)
1. ⏳ Deploy warp-optimized kernel
2. ⏳ NCU profiling comparison (baseline vs warp)
3. ⏳ Voxelization NCU profiling
4. ⏳ RT-X/CALVIN baseline benchmarks

### Long-term (Production Release)
1. ⏳ TMA integration and validation
2. ⏳ Docker container for H100
3. ⏳ Prebuilt manylinux wheels
4. ⏳ CI/CD for automated H100 testing

---

## Contact & Support

**H100 Access:** Shadeform via Brev (`awesome-gpu-name`)  
**Build Issues:** PyTorch ABI mismatch, cache permissions  
**Workarounds:** Manual CMake build, cache clearing  
**Status:** Build environment repair in progress

**Last Updated:** November 5, 2025  
**Next Update:** After successful H100 deployment

