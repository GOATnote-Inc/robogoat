# RoboCache

> **Archived August 2026.** Last GPU validation 2025-11-08 on H100 PCIe at commit
> `0db3726`; numbers below are from that run and have not been re-validated.

**GPU-Accelerated Data Preprocessing for Robot Learning**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.1%2B-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

---

## Overview

RoboCache is a GPU-accelerated data preprocessing library for robot foundation
models: trajectory resampling, multimodal sensor fusion, and point cloud
voxelization as CUDA kernels with a PyTorch fallback.

**Measured results (H100 PCIe, November 2025):**
- Kernel microbenchmarks vs. PyTorch on CPU: ~3.8x to ~110x depending on config
  ([`bench/results/benchmark_h100_20251106_172811.csv`](bench/results/benchmark_h100_20251106_172811.csv),
  5 seeds x 50 repeats per config)
- End-to-end training: **1.30x** faster than the PyTorch-preprocessing baseline
  (14.04 vs 18.28 ms/step, [`profiling/NCU_H100_TRAJECTORY_RESAMPLE.md`](profiling/NCU_H100_TRAJECTORY_RESAMPLE.md))
- Known regression: 0.74x (slower than PyTorch GPU) at 64x4096->1024x32
  ([`benchmarks/results/h100_validated_20251105.json`](benchmarks/results/h100_validated_20251105.json))

**Benchmark inputs were synthetic dataset-shaped tensors (`torch.randn`)** -
shaped like Isaac Gym, TartanAir, nuScenes, and KITTI samples, but no real
dataset files were loaded (see
[`benchmarks/real_world_datasets.py`](benchmarks/real_world_datasets.py)).

---

## Installation

Not published to PyPI. Install from source:

```bash
git clone https://github.com/GOATnote-Inc/robogoat.git
cd robogoat/robocache
pip install -e .
```

Prerequisites: Python 3.10+, PyTorch 2.0+, CUDA 12.1+ toolkit for the CUDA
extensions (a CPU-only install works but uses the fallback - see Known Issues
in the [repository README](../README.md)).

---

## Quick Start

```python
import torch
import robocache

# GPU-accelerated trajectory resampling
source_data = torch.randn(32, 500, 256, device='cuda', dtype=torch.bfloat16)
source_times = torch.linspace(0, 5, 500, device='cuda').unsqueeze(0).expand(32, -1)
target_times = torch.linspace(0, 5, 250, device='cuda').unsqueeze(0).expand(32, -1)

resampled = robocache.resample_trajectories(source_data, source_times, target_times)
# H100 measured: 2.605 ms P50 for this config (32x500->256, dim 256, bf16)
```

---

## Performance

All numbers from the November 2025 H100 PCIe run; see the CSV for raw data.

### Trajectory resampling (CUDA kernel vs. PyTorch CPU)

Source: [`bench/results/benchmark_h100_20251106_172811.csv`](bench/results/benchmark_h100_20251106_172811.csv)

| Config (B x S -> T, D) | CUDA P50 | PyTorch CPU P50 | Speedup |
|---|---|---|---|
| 8 x 250, 128 | 0.184 ms | 20.14 ms | ~110x |
| 32 x 500, 256 | 2.605 ms | 38.39 ms | ~15x |
| 64 x 1000, 512 | 20.05 ms | 75.69 ms | ~3.8x |

### End-to-end training (H100)

| Pipeline | ms/step | Speedup |
|---|---|---|
| Baseline (PyTorch preprocessing) | 18.28 | 1.00x |
| RoboCache preprocessing | 14.04 | **1.30x** |

### Where it loses

| Config (B x S -> T, D) | RoboCache | PyTorch GPU | Result |
|---|---|---|---|
| 64 x 4096 -> 1024, 32 | 0.190 ms | 0.140 ms | 0.74x (slower) |

Long sequences fall out of L1 cache; see
[`../KNOWN_LIMITATIONS.md`](../KNOWN_LIMITATIONS.md) for the crossover analysis.

---

## Architecture

RoboCache implements three memory patterns:

1. **L1-resident (trajectory, fusion):** binary search + linear interpolation;
   effective while timestamp arrays fit in L1.
2. **Bandwidth-bound (voxelization):** atomic scatter operations.
3. **BF16 storage** with FP32 interpolation internally.

**See**: [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)

---

## Documentation

- [Build matrix](docs/BUILD_MATRIX.md) - GPUs and CUDA versions used
- [Examples](examples/) - ROS 2, cuRobo, Isaac Sim
- [NCU profiling notes](profiling/NCU_COMPLETE_ANALYSIS.md)
- [Nsight Systems notes](profiling/NSIGHT_SYSTEMS_H100.md)
- [Known limitations](../KNOWN_LIMITATIONS.md)
- [Contributing](CONTRIBUTING.md) | [Security policy](SECURITY.md)

---

## Citation

```bibtex
@software{robocache2025,
  author = {Dent, Brandon},
  title = {RoboCache: GPU-Accelerated Data Preprocessing for Robot Learning},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/GOATnote-Inc/robogoat}},
  version = {1.0.0}
}
```

---

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
