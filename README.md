# RoboCache

> **Archived August 2026.** Last GPU validation 2025-11-08 on H100 PCIe at commit
> `0db3726`; numbers below are from that run and have not been re-validated.

**GPU-Accelerated Data Engine for Robot Foundation Models**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%2B-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

[Quick Start](#quick-start) | [Installation](#installation) | [Performance](#performance) | [Documentation](robocache/README.md)

---

## Overview

RoboCache is a CUDA library for sensor preprocessing in robotics: GPU-accelerated
temporal alignment (trajectory resampling, multimodal fusion) and point cloud
voxelization, with a pure-PyTorch fallback.

**Scope of validation:**
- CUDA kernels were benchmarked on H100 PCIe and A100 SXM4 in November 2025.
- All benchmark inputs were synthetic tensors (`torch.randn` / `torch.rand`),
  including the "dataset" benchmarks, which use tensors shaped like Isaac Gym,
  TartanAir, nuScenes, and KITTI samples - no real dataset files were loaded.
- The CPU fallback has known correctness bugs (see [Known Issues](#known-issues)).

---

## Quick Start

```python
import torch
import robocache

# 3-stream multimodal fusion (vision + proprioception + IMU)
vision = torch.randn(4, 30, 512, dtype=torch.bfloat16, device='cuda')
vision_times = torch.linspace(0, 1, 30, device='cuda').expand(4, -1)

proprio = torch.randn(4, 100, 64, dtype=torch.bfloat16, device='cuda')
proprio_times = torch.linspace(0, 1, 100, device='cuda').expand(4, -1)

imu = torch.randn(4, 200, 12, dtype=torch.bfloat16, device='cuda')
imu_times = torch.linspace(0, 1, 200, device='cuda').expand(4, -1)

target_times = torch.linspace(0, 1, 50, device='cuda').expand(4, -1)

# Fuse all streams to common timeline
fused = robocache.fuse_multimodal(
    vision, vision_times,
    proprio, proprio_times,
    imu, imu_times,
    target_times
)
# Output: (4, 50, 588) - batch x time x (512+64+12)
```

**Point Cloud Voxelization:**
```python
# LiDAR -> 3D voxel grid
points = torch.rand(500000, 3, device='cuda') * 20.0 - 10.0

voxel_grid = robocache.voxelize_pointcloud(
    points,
    grid_min=[-10.0, -10.0, -10.0],
    voxel_size=0.05,  # 5cm voxels
    grid_size=[128, 128, 128],
    mode='occupancy'
)
```

---

## Installation

Not published to PyPI. Install from source:

```bash
git clone https://github.com/GOATnote-Inc/robogoat.git
cd robogoat/robocache

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Build CUDA extensions
python setup.py develop

# Verify
python -c "import robocache; robocache.self_test()"
```

**Requirements:**
- NVIDIA GPU (Compute Capability >= 8.0)
- CUDA 12.1+ or 13.0+
- PyTorch 2.0+

---

## Performance

All numbers are from the November 2025 validation run on a single NVIDIA H100
PCIe 80GB (CUDA 13.0, driver 580.95) and have not been re-validated since.

### Kernel microbenchmarks (H100, CUDA kernel vs. PyTorch on CPU)

Source: [`robocache/bench/results/benchmark_h100_20251106_172811.csv`](robocache/bench/results/benchmark_h100_20251106_172811.csv)
(5 seeds x 50 repeats = 250 measurements per CUDA config; synthetic bf16 tensors).

| Trajectory resample config (B x S -> T, D) | CUDA P50 | PyTorch CPU P50 | Speedup |
|---|---|---|---|
| 8 x 250, 128 (small) | 0.184 ms | 20.14 ms | ~110x |
| 32 x 500, 256 (medium) | 2.605 ms | 38.39 ms | ~15x |
| 64 x 1000, 512 (large) | 20.05 ms | 75.69 ms | ~3.8x |

These are kernel-vs-CPU microbenchmarks, not end-to-end training comparisons.

### End-to-end training (H100, measured)

Source: [`PRODUCTION_STATUS.md`](PRODUCTION_STATUS.md) and
[`robocache/profiling/NCU_H100_TRAJECTORY_RESAMPLE.md`](robocache/profiling/NCU_H100_TRAJECTORY_RESAMPLE.md).

| Pipeline | ms/step | Speedup |
|---|---|---|
| Baseline (PyTorch preprocessing) | 18.28 | 1.00x |
| RoboCache preprocessing | 14.04 | **1.30x** |

The measured end-to-end training speedup is **1.30x**, driven by preprocessing
being a minority of step time once the model forward/backward is included.

### Known regression (measured, documented)

Source: [`robocache/benchmarks/results/h100_validated_20251105.json`](robocache/benchmarks/results/h100_validated_20251105.json).

| Config (B x S -> T, D) | RoboCache | PyTorch GPU | Result |
|---|---|---|---|
| 64 x 4096 -> 1024, 32 | 0.190 ms | 0.140 ms | **0.74x (slower)** |

For long sequences (> ~2000 timesteps) the per-thread binary search falls out
of L1 cache and native PyTorch GPU interpolation is faster. See
[`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md).

### Profiling artifacts

Nsight Compute and Nsight Systems text captures from the H100/A100 runs are
committed under [`artifacts/h100/`](artifacts/h100/) and
[`artifacts/a100/`](artifacts/a100/) (with GPU and driver stamps). Binary
`.ncu-rep` / `.nsys-rep` files are not committed.

---

## Examples

- ROS 2 node: [`examples/ros2_node/`](examples/ros2_node/)
- Isaac Sim demo (synthetic fallback data): [`examples/isaac_sim_demo/`](examples/isaac_sim_demo/)
- Multi-GPU benchmark: [`examples/multi_gpu/`](examples/multi_gpu/)

---

## Testing

```bash
cd robocache
pytest tests/ -v
```

**CI:** Lint + CPU tests on every PR (`.github/workflows/ci.yml`). CUDA kernels
are not exercised in CI; GPU validation was manual (see
[`docs/validation/`](docs/validation/)). The self-hosted GPU runners used for
that validation no longer exist.

---

## Known Issues

- **CPU voxelization fallback is incorrect.** `ops_fallback.voxelize_pointcloud_cpu`
  clamps out-of-bounds points into boundary voxels instead of dropping them
  (diverging from the CUDA kernel), and raises `IndexError` on empty or
  single-point clouds. The corresponding CPU tests are marked `xfail`. Do not
  use the CPU fallback where voxel occupancy correctness matters.
- **Stale multimodal-fusion tests.** `robocache/tests/test_multimodal_fusion.py`
  targets a pre-1.0 two-stream `fuse_multimodal` API and is skipped; the current
  API takes three streams.
- **Compute Sanitizer / 24h burn-in were never run in CI.** The stress-test code
  exists but there is no committed memcheck/racecheck log.
- **Voxelization out-of-bounds behavior (CUDA):** points outside the grid are
  clipped, no error is thrown.
- **Timestamp monotonicity is not enforced;** callers must supply monotonically
  increasing timestamps.

---

## Documentation

- [API Reference](docs/sphinx/index.rst)
- [Performance Tuning](docs/KERNEL_TUNING_GUIDE.md)
- [Known Limitations](KNOWN_LIMITATIONS.md)
- [Validation notes](docs/validation/)

---

## Citation

```bibtex
@software{robocache2025,
  title={RoboCache: GPU-Accelerated Data Engine for Robot Learning},
  author={Dent, Brandon},
  year={2025},
  url={https://github.com/GOATnote-Inc/robogoat}
}
```

---

## License

Apache 2.0 - See [LICENSE](LICENSE)

---

**Maintained by:** [GOATnote](mailto:b@thegoatnote.com)
**Status:** Archived (August 2026)
