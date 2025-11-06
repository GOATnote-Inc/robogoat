#!/usr/bin/env python3
"""
ROS 2 Sensor Fusion Node with RoboCache

Demonstrates GPU-accelerated multimodal sensor alignment for Isaac ROS.
Subscribes to variable-frequency sensor topics, aligns to uniform rate,
publishes fused features for downstream policy network.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, JointState
from std_msgs.msg import Float32MultiArray
import torch
import numpy as np
from collections import deque

try:
    import robocache
    ROBOCACHE_AVAILABLE = True
except ImportError:
    ROBOCACHE_AVAILABLE = False
    print("WARNING: RoboCache not available, using PyTorch fallback")


class SensorFusionNode(Node):
    """
    GPU-accelerated sensor fusion for robot manipulation
    
    Subscribes:
        /camera/color/image_raw (30Hz) - RGB vision
        /joint_states (100Hz) - Robot proprioception
        /imu/data (100Hz) - IMU/force data
    
    Publishes:
        /fused_features (50Hz) - Aligned multimodal features
    """
    
    def __init__(self):
        super().__init__('robocache_fusion_node')
        
        # Parameters
        self.declare_parameter('target_hz', 50.0)
        self.declare_parameter('buffer_size', 500)  # 5 seconds @ 100Hz
        self.declare_parameter('device', 'cuda')
        
        self.target_hz = self.get_parameter('target_hz').value
        self.buffer_size = self.get_parameter('buffer_size').value
        self.device = self.get_parameter('device').value
        
        # Sensor buffers
        self.vision_buffer = deque(maxlen=150)  # 5sec @ 30Hz
        self.proprio_buffer = deque(maxlen=500)  # 5sec @ 100Hz
        self.imu_buffer = deque(maxlen=500)      # 5sec @ 100Hz
        
        # Subscribers
        self.vision_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.vision_callback, 10)
        self.proprio_sub = self.create_subscription(
            JointState, '/joint_states', self.proprio_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        
        # Publisher
        self.fusion_pub = self.create_publisher(
            Float32MultiArray, '/fused_features', 10)
        
        # Fusion timer (50Hz)
        self.fusion_timer = self.create_timer(
            1.0 / self.target_hz, self.fusion_callback)
        
        self.get_logger().info(f'RoboCache Fusion Node started (target: {self.target_hz}Hz)')
        self.get_logger().info(f'Device: {self.device}, RoboCache: {ROBOCACHE_AVAILABLE}')
    
    def vision_callback(self, msg):
        """Store vision data with timestamp"""
        # Convert ROS Image to feature vector (simplified - use actual vision encoder)
        features = np.random.randn(512).astype(np.float32)  # Placeholder
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.vision_buffer.append((timestamp, features))
    
    def proprio_callback(self, msg):
        """Store proprioception data with timestamp"""
        # Robot joint positions/velocities
        features = np.array(msg.position + msg.velocity, dtype=np.float32)
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.proprio_buffer.append((timestamp, features))
    
    def imu_callback(self, msg):
        """Store IMU/force data with timestamp"""
        # Linear acceleration + angular velocity
        features = np.array([
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        ], dtype=np.float32)
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.imu_buffer.append((timestamp, features))
    
    def fusion_callback(self):
        """Fuse sensors at target frequency using RoboCache"""
        # Check if we have enough data
        if len(self.vision_buffer) < 10 or len(self.proprio_buffer) < 50:
            return
        
        try:
            # Extract buffers
            vision_times, vision_data = zip(*list(self.vision_buffer))
            proprio_times, proprio_data = zip(*list(self.proprio_buffer))
            imu_times, imu_data = zip(*list(self.imu_buffer))
            
            # Convert to torch tensors
            v_t = torch.tensor(vision_times, device=self.device).unsqueeze(0)
            v_d = torch.tensor(np.stack(vision_data), device=self.device).unsqueeze(0)
            p_t = torch.tensor(proprio_times, device=self.device).unsqueeze(0)
            p_d = torch.tensor(np.stack(proprio_data), device=self.device).unsqueeze(0)
            i_t = torch.tensor(imu_times, device=self.device).unsqueeze(0)
            i_d = torch.tensor(np.stack(imu_data), device=self.device).unsqueeze(0)
            
            # Target timestamp (now)
            now = self.get_clock().now().seconds_nanoseconds()
            target_time = now[0] + now[1] * 1e-9
            target_times = torch.tensor([[target_time]], device=self.device)
            
            # GPU-accelerated fusion
            if ROBOCACHE_AVAILABLE:
                # RoboCache multimodal fusion (single kernel)
                v_aligned = robocache.resample_trajectories(v_d, v_t, target_times)
                p_aligned = robocache.resample_trajectories(p_d, p_t, target_times)
                i_aligned = robocache.resample_trajectories(i_d, i_t, target_times)
                fused = torch.cat([v_aligned, p_aligned, i_aligned], dim=2)
            else:
                # PyTorch fallback (slower)
                v_aligned = self._interpolate_torch(v_d, v_t, target_times)
                p_aligned = self._interpolate_torch(p_d, p_t, target_times)
                i_aligned = self._interpolate_torch(i_d, i_t, target_times)
                fused = torch.cat([v_aligned, p_aligned, i_aligned], dim=2)
            
            # Publish fused features
            msg = Float32MultiArray()
            msg.data = fused[0, 0].cpu().numpy().tolist()
            self.fusion_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Fusion error: {e}')
    
    def _interpolate_torch(self, data, times, target_times):
        """PyTorch fallback for interpolation"""
        # Simplified linear interpolation
        # Production: use proper searchsorted + lerp
        return torch.nn.functional.interpolate(
            data.transpose(1, 2), size=1, mode='linear'
        ).transpose(1, 2)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

