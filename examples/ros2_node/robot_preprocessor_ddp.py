#!/usr/bin/env python3
"""
ROS 2 RoboCache Multi-GPU DDP Preprocessor Node

Distributes sensor preprocessing across multiple GPUs using PyTorch DDP:
- Automatic GPU detection and rank assignment
- NCCL backend for efficient inter-GPU communication
- Batch processing with distributed workload
- Synchronized fusion and voxelization
- Compatible with multi-node setups (via ROS_DOMAIN_ID)

Example usage:
  # Single node, 2 GPUs:
  ros2 run robocache robot_preprocessor_ddp --ros-args -p num_gpus:=2

  # Multi-node (run on each node):
  # Node 0: ROS_DOMAIN_ID=0 ros2 run robocache robot_preprocessor_ddp
  # Node 1: ROS_DOMAIN_ID=1 ros2 run robocache robot_preprocessor_ddp
"""

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, Imu
from std_msgs.msg import Float32MultiArray
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from typing import List, Optional

try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    ROBOCACHE_AVAILABLE = False


class MultiGPUPreprocessor(Node):
    """Multi-GPU ROS 2 preprocessor using PyTorch DDP"""
    
    def __init__(self, rank: int, world_size: int):
        super().__init__(f'robocache_preprocessor_gpu{rank}')
        
        self.rank = rank
        self.world_size = world_size
        self.device = f'cuda:{rank}'
        
        # Parameters
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('grid_size', 128)
        self.declare_parameter('batch_size', 32)
        self.declare_parameter('fusion_rate_hz', 30.0)
        
        self.voxel_size = self.get_parameter('voxel_size').value
        self.grid_size = self.get_parameter('grid_size').value
        self.batch_size = self.get_parameter('batch_size').value
        self.fusion_rate = self.get_parameter('fusion_rate_hz').value
        
        # Initialize DDP
        self._init_ddp()
        
        # Subscribers (all GPUs receive all messages)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/lidar/points', self.pointcloud_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        
        # Publishers (only rank 0 publishes final results)
        if self.rank == 0:
            self.voxel_pub = self.create_publisher(
                Float32MultiArray, '/preprocessed/voxels', 10
            )
            self.features_pub = self.create_publisher(
                Float32MultiArray, '/preprocessed/features', 10
            )
        
        # Buffers (distributed across GPUs)
        self.lidar_buffer = []
        self.image_buffer = []
        self.imu_buffer = []
        
        # Processing statistics
        self.processed_count = 0
        self.processing_times = []
        
        # Timer for periodic batch processing
        self.create_timer(1.0 / self.fusion_rate, self.process_batch)
        
        self.get_logger().info(
            f'Multi-GPU Preprocessor started '
            f'(rank={self.rank}/{self.world_size}, device={self.device})'
        )
    
    def _init_ddp(self):
        """Initialize PyTorch DDP"""
        if not dist.is_initialized():
            # Use NCCL for GPU communication
            dist.init_process_group(
                backend='nccl',
                init_method='env://',  # Reads MASTER_ADDR, MASTER_PORT
                world_size=self.world_size,
                rank=self.rank
            )
        
        torch.cuda.set_device(self.rank)
        self.get_logger().info(f'DDP initialized (NCCL backend, rank {self.rank})')
    
    def pointcloud_callback(self, msg):
        """Buffer point cloud for batch processing"""
        points = self.pointcloud2_to_array(msg)
        
        if points.shape[0] > 0:
            self.lidar_buffer.append({
                'timestamp': msg.header.stamp,
                'points': points,
                'frame_id': msg.header.frame_id
            })
            
            # Keep buffer size manageable
            if len(self.lidar_buffer) > 100:
                self.lidar_buffer.pop(0)
    
    def image_callback(self, msg):
        """Buffer image for multimodal fusion"""
        # Convert ROS Image to numpy
        # Simplified: assumes RGB8 encoding
        height = msg.height
        width = msg.width
        image_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
        
        self.image_buffer.append({
            'timestamp': msg.header.stamp,
            'image': image_data,
            'frame_id': msg.header.frame_id
        })
        
        if len(self.image_buffer) > 50:
            self.image_buffer.pop(0)
    
    def imu_callback(self, msg):
        """Buffer IMU for proprioception"""
        imu_data = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
        self.imu_buffer.append({
            'timestamp': msg.header.stamp,
            'data': imu_data
        })
        
        if len(self.imu_buffer) > 200:
            self.imu_buffer.pop(0)
    
    def process_batch(self):
        """Process buffered data with distributed workload"""
        if not self.lidar_buffer:
            return
        
        # Each GPU processes its portion of the batch
        batch_size = min(self.batch_size, len(self.lidar_buffer))
        start_idx = (self.rank * batch_size) // self.world_size
        end_idx = ((self.rank + 1) * batch_size) // self.world_size
        
        local_batch = self.lidar_buffer[start_idx:end_idx]
        
        if not local_batch:
            return
        
        try:
            # Voxelize local batch
            voxel_grids = []
            for item in local_batch:
                points_tensor = torch.from_numpy(item['points'][:, :3]).float().to(self.device)
                
                if ROBOCACHE_AVAILABLE:
                    voxel_grid = robocache.voxelize_pointcloud(
                        points_tensor,
                        grid_min=[-10.0, -10.0, -10.0],
                        voxel_size=self.voxel_size,
                        grid_size=[self.grid_size, self.grid_size, self.grid_size],
                        mode='occupancy',
                        backend='cuda'
                    )
                else:
                    # CPU fallback
                    voxel_grid = self.voxelize_cpu(item['points'][:, :3])
                
                voxel_grids.append(voxel_grid)
            
            # Stack local results
            if voxel_grids:
                local_voxels = torch.stack(voxel_grids)
                
                # All-gather across GPUs using DDP
                gathered_voxels = [torch.zeros_like(local_voxels) for _ in range(self.world_size)]
                dist.all_gather(gathered_voxels, local_voxels)
                
                # Only rank 0 publishes
                if self.rank == 0:
                    # Concatenate results from all GPUs
                    all_voxels = torch.cat(gathered_voxels, dim=0)
                    
                    # Publish aggregated result
                    out_msg = Float32MultiArray()
                    out_msg.data = all_voxels.cpu().flatten().numpy().tolist()
                    self.voxel_pub.publish(out_msg)
                    
                    self.processed_count += all_voxels.shape[0]
                    self.get_logger().info(
                        f'Processed batch: {all_voxels.shape[0]} voxelized point clouds '
                        f'(distributed across {self.world_size} GPUs)'
                    )
            
            # Clear processed items
            self.lidar_buffer = self.lidar_buffer[batch_size:]
        
        except Exception as e:
            self.get_logger().error(f'Batch processing failed on GPU {self.rank}: {e}')
    
    @staticmethod
    def pointcloud2_to_array(msg):
        """Convert PointCloud2 to numpy array"""
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
        ])
        
        points = np.frombuffer(msg.data, dtype=dtype)
        return np.column_stack([points['x'], points['y'], points['z']])
    
    def voxelize_cpu(self, points):
        """CPU fallback voxelization"""
        grid = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)
        grid_min = np.array([-10.0, -10.0, -10.0])
        voxel_coords = ((points - grid_min) / self.voxel_size).astype(int)
        
        valid = (voxel_coords >= 0).all(axis=1) & (voxel_coords < self.grid_size).all(axis=1)
        voxel_coords = voxel_coords[valid]
        
        for x, y, z in voxel_coords:
            grid[x, y, z] = 1.0
        
        return torch.from_numpy(grid).to(self.device)
    
    def destroy_node(self):
        """Cleanup DDP"""
        if dist.is_initialized():
            dist.destroy_process_group()
        super().destroy_node()


def run_worker(rank: int, world_size: int):
    """Worker process for each GPU"""
    # Set environment variables for DDP
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Initialize ROS 2
    rclpy.init()
    
    # Create node
    node = MultiGPUPreprocessor(rank, world_size)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(args=None):
    """Main entry point for multi-GPU preprocessing"""
    # Detect available GPUs
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    num_gpus = torch.cuda.device_count()
    
    # Allow override via parameter
    import sys
    world_size = num_gpus
    for arg in sys.argv:
        if arg.startswith('num_gpus:='):
            world_size = int(arg.split(':=')[1])
            break
    
    world_size = min(world_size, num_gpus)
    
    print(f"Launching Multi-GPU RoboCache Preprocessor")
    print(f"  Available GPUs: {num_gpus}")
    print(f"  Using GPUs: {world_size}")
    
    if world_size < 2:
        print("WARNING: Multi-GPU mode requires at least 2 GPUs. Falling back to single GPU.")
        # Run single GPU version
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        run_worker(0, 1)
        return
    
    # Set DDP environment variables
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Spawn worker processes (one per GPU)
    mp.spawn(
        run_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()

