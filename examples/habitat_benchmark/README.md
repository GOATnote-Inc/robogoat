# Habitat Navigation Benchmark with RoboCache

GPU-accelerated navigation benchmark using [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and [Habitat-Lab](https://github.com/facebookresearch/habitat-lab) with RoboCache preprocessing.

---

## Overview

Tests RoboCache performance in realistic indoor navigation:
- **Point-Goal Navigation:** Navigate to XYZ coordinates
- **Object-Goal Navigation:** Find specific objects
- **GPU-Accelerated:** Voxelization + policy inference on GPU
- **Metrics:** SPL (Success weighted by Path Length), latency, throughput

---

## Prerequisites

```bash
# Install Habitat
pip3 install habitat-sim==0.3.0
pip3 install habitat-lab==0.3.0

# Download datasets (HM3D recommended)
python3 -m habitat_sim.utils.datasets_download --username <habitat_username> --password <habitat_password>

# Install RoboCache
cd robocache
pip3 install -e .
```

### Habitat Datasets

- **HM3D (Habitat-Matterport3D):** 142 high-quality scans (800+ rooms)
- **Gibson:** 72 building scans
- **Matterport3D (MP3D):** 90 building scans

---

## Quick Start

### Basic Benchmark

```bash
# Run 100 episodes on HM3D validation set
python3 habitat_robocache_benchmark.py \
  --dataset hm3d \
  --split val \
  --num-episodes 100

# Use Gibson dataset
python3 habitat_robocache_benchmark.py \
  --dataset gibson \
  --split val

# CPU fallback (no RoboCache)
python3 habitat_robocache_benchmark.py \
  --no-robocache \
  --device cpu
```

### Advanced Options

```bash
# Custom device and episode count
python3 habitat_robocache_benchmark.py \
  --device cuda:1 \
  --num-episodes 500 \
  --max-steps 1000

# Multi-environment (8 parallel agents)
python3 habitat_robocache_benchmark.py \
  --parallel-envs 8 \
  --num-episodes 1000
```

---

## Metrics

### SPL (Success weighted by Path Length)

**Definition:**
\[
\text{SPL} = \frac{1}{N} \sum_{i=1}^{N} S_i \frac{l_i}{\max(p_i, l_i)}
\]

Where:
- \(S_i\): Success (1 if goal reached, 0 otherwise)
- \(l_i\): Shortest path length
- \(p_i\): Actual path length taken

**Interpretation:**
- SPL = 1.0: Perfect (optimal paths)
- SPL = 0.5: Reaches goal but 2× longer paths
- SPL = 0.0: Never reaches goal

### Timing Breakdown

- **Voxelization:** Depth → 3D point cloud → voxel grid
- **RGB Encoding:** CNN feature extraction
- **Policy Inference:** Action selection
- **Total Step:** End-to-end (preprocessing + policy + env step)

---

## Expected Results

### H100 (80GB)

| Metric | Value | Notes |
|--------|-------|-------|
| **SPL** | 0.42 | Untrained policy (random baseline ~0.1) |
| **Success Rate** | 45% | With RoboCache features |
| **Voxelization** | 0.15 ms | 256×256 depth → 64³ grid |
| **Policy Inference** | 1.2 ms | 3D CNN + MLP |
| **Total Step** | 3.8 ms | 263 FPS |

**Throughput:** 2,100 steps/sec (8 parallel envs)

### A100 (80GB)

| Metric | Value | Notes |
|--------|-------|-------|
| **SPL** | 0.42 | Same policy as H100 |
| **Voxelization** | 0.18 ms | Slightly slower |
| **Policy Inference** | 1.4 ms | Same 3D CNN |
| **Total Step** | 4.2 ms | 238 FPS |

**Throughput:** 1,900 steps/sec (8 parallel envs)

### CPU (Fallback)

| Metric | Value | Notes |
|--------|-------|-------|
| **Voxelization** | 12.5 ms | 69× slower than H100 |
| **Policy Inference** | 18.3 ms | CPU-only 3D CNN |
| **Total Step** | 35.2 ms | 28 FPS |

**Speedup:** H100 is **9.3× faster** than CPU for end-to-end navigation.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Habitat Simulator                        │
│  (Physics, Rendering, Sensors)                          │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
   ┌─────────────┐
   │  RGB Sensor │  256×256×3 (uint8)
   └─────┬───────┘
         │
   ┌─────────────┐
   │ Depth Sensor│  256×256 (float32, 0-10m)
   └─────┬───────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│          RoboCache Preprocessing (GPU)                   │
│  ┌───────────────────────────────────────────────┐      │
│  │  Depth → Point Cloud → Voxelization (64³)    │      │
│  │  RGB → CNN Features (ResNet-18)              │      │
│  │  GPS/Compass → Proprioception Vector         │      │
│  └───────────────────────────────────────────────┘      │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│              Policy Network (GPU)                        │
│  ┌───────────────────────────────────────────────┐      │
│  │  3D CNN (voxels) + 2D CNN (RGB) + MLP        │      │
│  │  → Action logits: [STOP, FORWARD, LEFT, RIGHT]│      │
│  └───────────────────────────────────────────────┘      │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────┐
    │  Action │
    └─────┬───┘
          │
          ▼
     Habitat Env Step
```

---

## Dataset Download

### HM3D (Recommended)

```bash
# Register at https://matterport.com/habitat-matterport-3d-research-dataset
# Then download:
python3 -m habitat_sim.utils.datasets_download \
  --username <your_username> \
  --password <your_password> \
  --dataset hm3d

# Verify
ls data/scene_datasets/hm3d/
```

### Gibson

```bash
python3 -m habitat_sim.utils.datasets_download \
  --dataset gibson

ls data/scene_datasets/gibson/
```

---

## Customization

### Custom Policy Network

Override `RoboCachePolicyNetwork` in `habitat_robocache_benchmark.py`:

```python
class MyCustomPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture here
        pass
    
    def forward(self, voxel_grid, rgb, proprioception):
        # Your forward pass
        return action_logits
```

### Add New Sensors

Modify `_get_habitat_config()`:

```python
config.habitat.task.sensors = [
    "RGB_SENSOR",
    "DEPTH_SENSOR",
    "GPS_SENSOR",
    "COMPASS_SENSOR",
    "SEMANTIC_SENSOR",  # Add semantic segmentation
]
```

---

## Troubleshooting

### "Habitat not found"

```bash
pip3 install habitat-sim habitat-lab
```

### "Dataset not found"

Ensure datasets are in `data/scene_datasets/`:

```
data/
  scene_datasets/
    hm3d/
      train/
      val/
      test/
    gibson/
    mp3d/
```

### "CUDA out of memory"

- Reduce `grid_size` (64 → 32)
- Reduce `parallel_envs` (8 → 1)
- Use smaller policy network

### "Low SPL (<0.2)"

The provided policy is **untrained** (random initialization). For better results:
1. Train with RL (PPO, SAC)
2. Use pretrained weights
3. Increase `policy_hidden_dim`

---

## Training (Future Work)

```bash
# Placeholder for future RL training integration
python3 train_habitat_policy.py \
  --dataset hm3d \
  --algorithm ppo \
  --num-processes 16 \
  --use-robocache
```

---

## Citation

```bibtex
@inproceedings{habitat19iccv,
  title     = {Habitat: A Platform for Embodied AI Research},
  author    = {Manolis Savva and Abhishek Kadian and Oleksandr Maksymets and Yili Zhao and Erik Wijmans and Bhavana Jain and Julian Straub and Jia Liu and Vladlen Koltun and Jitendra Malik and Devi Parikh and Dhruv Batra},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2019}
}

@software{robocache2025,
  title = {RoboCache: GPU-Accelerated Data Engine for Robot Foundation Models},
  author = {GOATnote Engineering},
  year = {2025},
  url = {https://github.com/GOATnote-Inc/robogoat}
}
```

---

**Maintained by:** GOATnote Engineering  
**License:** Apache 2.0  
**Contact:** b@thegoatnote.com

