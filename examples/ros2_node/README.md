# RoboCache ROS 2 Integration

GPU-accelerated sensor preprocessing for ROS 2 Humble/Iron/Jazzy.

## Overview

`robot_preprocessor.py` is a ROS 2 node that:
- Subscribes to sensor topics (PointCloud2, Image, IMU)
- Voxelizes point clouds with RoboCache on GPU
- Publishes preprocessed data for downstream policy inference

## Prerequisites

```bash
# ROS 2 (Humble or newer)
sudo apt install ros-humble-desktop

# RoboCache
cd /path/to/robogoat/robocache
pip install -e .

# ROS 2 Python dependencies
pip install rclpy sensor_msgs std_msgs
```

## Quick Start

### 1. Source ROS 2
```bash
source /opt/ros/humble/setup.bash
```

### 2. Run Node
```bash
cd /path/to/robogoat/examples/ros2_node
python3 robot_preprocessor.py
```

### 3. Or Use Launch File
```bash
ros2 launch launch/preprocessor.launch.py device:=cuda voxel_size:=0.05 grid_size:=128
```

## Topics

### Subscribed

| Topic | Type | Description |
|-------|------|-------------|
| `/lidar/points` | `sensor_msgs/PointCloud2` | Raw LiDAR point cloud |
| `/camera/image` | `sensor_msgs/Image` | Camera image |
| `/imu/data` | `sensor_msgs/Imu` | IMU measurements |

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `/preprocessed/voxels` | `std_msgs/Float32MultiArray` | Voxelized point cloud |
| `/preprocessed/features` | `std_msgs/Float32MultiArray` | Fused multimodal features |

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | string | `cuda` | PyTorch device (`cuda` or `cpu`) |
| `voxel_size` | float | `0.05` | Voxel size in meters |
| `grid_size` | int | `128` | Voxel grid size (cubic) |
| `use_robocache` | bool | `true` | Enable RoboCache acceleration |

## Performance

**H100 GPU:**
- Point cloud processing: ~10-20μs for 100K points
- Latency: <1ms end-to-end
- Throughput: >100 Hz

**CPU Fallback:**
- Point cloud processing: ~50-100ms for 100K points
- Latency: <150ms end-to-end
- Throughput: ~10 Hz

## Integration with Isaac ROS

```bash
# Install Isaac ROS
sudo apt install ros-humble-isaac-ros-*

# Run with Isaac ROS perception
ros2 launch launch/preprocessor.launch.py
ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam.launch.py
```

## Example: Mobile Manipulator

```python
# In your policy node
import rclpy
from std_msgs.msg import Float32MultiArray

class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')
        self.sub = self.create_subscription(
            Float32MultiArray,
            '/preprocessed/features',
            self.policy_callback,
            10
        )
        self.policy = load_policy()
    
    def policy_callback(self, msg):
        features = torch.tensor(msg.data).reshape(...) # Reshape as needed
        action = self.policy(features)
        # Execute action
```

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce grid size
ros2 param set /robocache_preprocessor grid_size 64
```

### High Latency
```python
# Check GPU utilization
nvidia-smi dmon -s u
```

### CPU Fallback
```python
# Force CPU mode
ros2 launch launch/preprocessor.launch.py device:=cpu
```

## Citation

```bibtex
@software{robocache2025,
  title={RoboCache: GPU-Accelerated Data Engine for Robot Foundation Models},
  author={GOATnote Engineering},
  year={2025},
  url={https://github.com/GOATnote-Inc/robogoat}
}
```

## Support

- Issues: https://github.com/GOATnote-Inc/robogoat/issues
- Docs: https://github.com/GOATnote-Inc/robogoat/tree/main/docs

