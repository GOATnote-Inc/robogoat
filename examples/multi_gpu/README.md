# Multi-GPU Performance Testing
**A100 and H100 Scaling Analysis**

## Overview

This directory contains scripts to test RoboCache performance scaling across multiple GPUs. 

**Important:** A100 (NVLink 3.0) and H100 (NVLink 4.0) **cannot** be directly connected via NVLink due to different architectures. However, you can:

1. **Test each architecture separately** - 2x A100 or 2x H100 with NVLink
2. **Use both independently** - Distribute work across heterogeneous GPUs via PCIe/network
3. **Compare performance** - Single vs multi-GPU on each architecture

---

## Hardware Configuration

### Option 1: Homogeneous NVLink Cluster
```
┌─────────┐ NVLink  ┌─────────┐
│ H100 #0 │◄───────►│ H100 #1 │  ← 900 GB/s bidirectional
└─────────┘         └─────────┘

┌─────────┐ NVLink  ┌─────────┐
│ A100 #0 │◄───────►│ A100 #1 │  ← 600 GB/s bidirectional
└─────────┘         └─────────┘
```

### Option 2: Heterogeneous PCIe Cluster  
```
┌─────────┐         ┌─────────┐
│ H100    │   PCIe  │ A100    │  ← 32 GB/s per direction
└─────────┘◄───────►└─────────┘
```

---

## Quick Start

### 1. Single GPU Baseline

```bash
# H100 baseline
python benchmark_single_gpu.py --gpu 0 --arch h100 --output h100_single.json

# A100 baseline  
python benchmark_single_gpu.py --gpu 1 --arch a100 --output a100_single.json
```

### 2. Multi-GPU Scaling (Same Architecture)

```bash
# 2x H100 with NVLink
python benchmark_multi_gpu.py --gpus 0,1 --arch h100 --mode nvlink --output h100_2gpu.json

# 2x A100 with NVLink
python benchmark_multi_gpu.py --gpus 2,3 --arch a100 --mode nvlink --output a100_2gpu.json
```

### 3. Heterogeneous Workload Distribution

```bash
# Use H100 for voxelization, A100 for sensor fusion
python benchmark_heterogeneous.py \
    --h100-gpu 0 \
    --a100-gpu 1 \
    --output hetero_results.json
```

### 4. Compare Results

```bash
python compare_results.py \
    --single h100_single.json a100_single.json \
    --multi h100_2gpu.json a100_2gpu.json \
    --hetero hetero_results.json
```

---

## Expected Results

### H100 NVLink Scaling

| Configuration | Multimodal (ms) | Voxelize (ms) | Speedup |
|---------------|-----------------|---------------|---------|
| 1x H100       | 0.050           | 0.020         | 1.00x   |
| 2x H100 (NVLink) | 0.028        | 0.012         | 1.78x   |
| 2x H100 (DDP) | 0.032           | 0.015         | 1.56x   |

### A100 NVLink Scaling

| Configuration | Multimodal (ms) | Voxelize (ms) | Speedup |
|---------------|-----------------|---------------|---------|
| 1x A100       | 0.057           | 0.032         | 1.00x   |
| 2x A100 (NVLink) | 0.033        | 0.019         | 1.70x   |
| 2x A100 (DDP) | 0.038           | 0.022         | 1.50x   |

### Why Not 2.0x Speedup?

Multi-GPU scaling is limited by:
1. **Communication overhead** - Data transfer between GPUs
2. **Synchronization** - Waiting for all GPUs to finish
3. **Load imbalance** - Work not evenly distributed
4. **Amdahl's Law** - Serial portions don't scale

**1.5-1.8x is excellent** for 2 GPUs in real workloads.

---

## Testing Scripts

### `benchmark_single_gpu.py`
Baseline performance on single GPU:
- Multimodal fusion (various batch sizes)
- Voxelization (various point counts)
- Memory usage
- GPU utilization

### `benchmark_multi_gpu.py`
Multi-GPU scaling with PyTorch DDP:
- Data parallelism
- NVLink vs PCIe comparison
- Scaling efficiency metrics
- Communication overhead analysis

### `benchmark_heterogeneous.py`
Heterogeneous workload distribution:
- H100 handles compute-intensive tasks
- A100 handles memory-intensive tasks
- Load balancing strategies
- Total system throughput

### `compare_results.py`
Comprehensive analysis:
- Speedup curves
- Efficiency plots
- Cost/performance analysis
- Recommendations

---

## Multi-GPU Strategies

### 1. Data Parallel (DDP)

**Best for:** Training with large batches

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl', world_size=2, rank=gpu_id)

# Wrap model
model = DDP(model, device_ids=[gpu_id])

# RoboCache kernels automatically use correct GPU
device = torch.device(f'cuda:{gpu_id}')
out = robocache.fuse_multimodal(vision.to(device), ...)
```

### 2. Pipeline Parallel

**Best for:** Large models that don't fit on one GPU

```python
# Split model across GPUs
model_part1 = ModelPart1().to('cuda:0')  
model_part2 = ModelPart2().to('cuda:1')

# RoboCache preprocessing on each GPU
features_0 = robocache.fuse_multimodal(..., device='cuda:0')
out = model_part1(features_0).to('cuda:1')
final = model_part2(out)
```

### 3. Task Parallel

**Best for:** Independent workloads

```python
import torch.multiprocessing as mp

def process_on_h100(data, gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    return robocache.voxelize_pointcloud(data.to(device), ...)

def process_on_a100(data, gpu_id):
    device = torch.device(f'cuda:{gpu_id}')
    return robocache.fuse_multimodal(data.to(device), ...)

# Run in parallel
with mp.Pool(2) as pool:
    results = pool.starmap(process, [(data1, 0), (data2, 1)])
```

---

## Profiling Multi-GPU

### Nsight Systems Timeline

```bash
# Profile 2-GPU run
nsys profile --trace=cuda,nvtx,nccl --output=multi_gpu.nsys-rep \
    torchrun --nproc_per_node=2 benchmark_multi_gpu.py --gpus 0,1

# View timeline
nsys-ui multi_gpu.nsys-rep
```

**Look for:**
- GPU idle time (indicates synchronization overhead)
- NCCL communication kernels (P2P transfers)
- Overlapping compute and communication

### NCCL Performance

```bash
# Test NVLink bandwidth
nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 2

# Expected bandwidth:
# H100 NVLink 4.0: ~450 GB/s (each direction)
# A100 NVLink 3.0: ~300 GB/s (each direction)
```

---

## Troubleshooting

### NVLink Not Working

```bash
# Check NVLink topology
nvidia-smi topo -m

# Should show "NV12" or "NV18" between GPUs
# "PHB" means PCIe only

# Enable peer access in code
torch.cuda.set_device(0)
torch.cuda.device_enable_peer_access(1)
```

### DDP Initialization Fails

```bash
# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=0  # or 1 for second process

# Use file:// URL for single-node
python -m torch.distributed.launch --nproc_per_node=2 script.py
```

### Imbalanced GPU Usage

```python
# Monitor utilization
import pynvml
pynvml.nvmlInit()
for i in range(torch.cuda.device_count()):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"GPU {i}: {util.gpu}% util, {util.memory}% mem")
```

---

## Advanced: Custom NCCL Collectives

RoboCache kernels work seamlessly with NCCL:

```python
import torch.distributed as dist

# Each GPU processes local data with RoboCache
local_out = robocache.fuse_multimodal(local_vision, ...)

# Aggregate across GPUs with NCCL
dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
local_out /= dist.get_world_size()

# Result: averaged features from all GPUs
```

---

## Cost-Performance Analysis

### H100 vs A100 Multi-GPU

| Metric | 2x H100 | 2x A100 | Winner |
|--------|---------|---------|--------|
| **Performance** | 1.78x | 1.70x | H100 |
| **Cloud Cost** | $6-8/hr | $4-6/hr | A100 |
| **Performance/$** | 0.24x/$ | 0.36x/$ | **A100** |
| **Availability** | Limited | Wide | A100 |

**Recommendation:** Use 2x A100 for cost-effective training, 2x H100 for time-critical workloads.

---

## Next Steps

1. **Run benchmarks** on your hardware
2. **Profile with Nsight** to find bottlenecks  
3. **Tune batch size** to maximize GPU utilization
4. **Consider cost** vs performance tradeoffs

---

## References

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [Multi-GPU Best Practices](https://pytorch.org/docs/stable/notes/cuda.html#best-practices)

---

**Status:** Ready for Testing  
**Hardware Required:** 2+ NVIDIA GPUs (A100, H100, or mixed)  
**Last Updated:** November 7, 2025

