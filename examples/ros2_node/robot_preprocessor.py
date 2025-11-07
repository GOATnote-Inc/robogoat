#!/usr/bin/env python3
"""
ROS 2 RoboCache Preprocessor Node

Subscribes to sensor topics, fuses/voxelizes with RoboCache, publishes processed data.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, Imu
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np

try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    ROBOCACHE_AVAILABLE = False


class RoboCachePreprocessor(Node):
    """ROS 2 node for GPU-accelerated sensor preprocessing"""
    
    def __init__(self):
        super().__init__('robocache_preprocessor')
        
        # Parameters
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('grid_size', 128)
        self.declare_parameter('use_robocache', True)
        
        self.device = self.get_parameter('device').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.grid_size = self.get_parameter('grid_size').value
        self.use_robocache = self.get_parameter('use_robocache').value and ROBOCACHE_AVAILABLE
        
        if not ROBOCACHE_AVAILABLE:
            self.get_logger().warn('RoboCache not available, using CPU fallback')
        
        # Subscribers
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/lidar/points', self.pointcloud_callback, 10
        )
        self.image_sub = self.create_subscription(
            Image, '/camera/image', self.image_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        
        # Publishers
        self.voxel_pub = self.create_publisher(Float32MultiArray, '/preprocessed/voxels', 10)
        self.features_pub = self.create_publisher(Float32MultiArray, '/preprocessed/features', 10)
        
        # Buffers
        self.lidar_buffer = []
        self.image_buffer = []
        self.imu_buffer = []
        
        self.get_logger().info(f'RoboCache Preprocessor started (device={self.device})')
    
    def pointcloud_callback(self, msg):
        """Process point cloud with voxelization"""
        # Convert PointCloud2 to numpy
        points = self.pointcloud2_to_array(msg)
        
        if points.shape[0] == 0:
            return
        
        # Convert to torch tensor
        points_tensor = torch.from_numpy(points[:, :3]).float().to(self.device)
        
        # Voxelize
        if self.use_robocache:
            voxel_grid = robocache.voxelize_pointcloud(
                points_tensor,
                grid_min=[-10.0, -10.0, -10.0],
                voxel_size=self.voxel_size,
                grid_size=[self.grid_size, self.grid_size, self.grid_size],
                mode='occupancy'
            )
        else:
            # CPU fallback
            voxel_grid = self.voxelize_cpu(points[:, :3])
        
        # Publish
        out_msg = Float32MultiArray()
        out_msg.data = voxel_grid.cpu().flatten().numpy().tolist()
        self.voxel_pub.publish(out_msg)
        
        self.get_logger().debug(f'Processed {points.shape[0]} points -> {voxel_grid.shape} grid')
    
    def image_callback(self, msg):
        """Process image (placeholder)"""
        self.image_buffer.append(msg)
        if len(self.image_buffer) > 100:
            self.image_buffer.pop(0)
    
    def imu_callback(self, msg):
        """Process IMU (placeholder)"""
        self.imu_buffer.append(msg)
        if len(self.imu_buffer) > 200:
            self.imu_buffer.pop(0)
    
    @staticmethod
    def pointcloud2_to_array(msg):
        """Convert PointCloud2 to numpy array"""
        # Simplified conversion (assumes XYZ fields)
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
        
        # Simple occupancy grid
        grid_min = np.array([-10.0, -10.0, -10.0])
        voxel_coords = ((points - grid_min) / self.voxel_size).astype(int)
        
        valid = (voxel_coords >= 0).all(axis=1) & (voxel_coords < self.grid_size).all(axis=1)
        voxel_coords = voxel_coords[valid]
        
        for x, y, z in voxel_coords:
            grid[x, y, z] = 1.0
        
        return torch.from_numpy(grid)


def main(args=None):
    rclpy.init(args=args)
    node = RoboCachePreprocessor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

