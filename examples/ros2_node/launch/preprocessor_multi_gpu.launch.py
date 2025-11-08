"""
ROS 2 Launch file for Multi-GPU RoboCache Preprocessor

Automatically detects GPUs and launches distributed preprocessing node.

Usage:
  # Launch with all available GPUs:
  ros2 launch robocache preprocessor_multi_gpu.launch.py

  # Launch with specific number of GPUs:
  ros2 launch robocache preprocessor_multi_gpu.launch.py num_gpus:=2

  # Custom parameters:
  ros2 launch robocache preprocessor_multi_gpu.launch.py num_gpus:=4 voxel_size:=0.1 batch_size:=64
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import torch


def generate_launch_description():
    # Detect available GPUs
    num_available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Launch arguments
    num_gpus_arg = DeclareLaunchArgument(
        'num_gpus',
        default_value=str(num_available_gpus),
        description=f'Number of GPUs to use (max: {num_available_gpus})'
    )
    
    voxel_size_arg = DeclareLaunchArgument(
        'voxel_size',
        default_value='0.05',
        description='Voxel size in meters'
    )
    
    grid_size_arg = DeclareLaunchArgument(
        'grid_size',
        default_value='128',
        description='Grid size (NxNxN)'
    )
    
    batch_size_arg = DeclareLaunchArgument(
        'batch_size',
        default_value='32',
        description='Batch size for processing'
    )
    
    fusion_rate_arg = DeclareLaunchArgument(
        'fusion_rate_hz',
        default_value='30.0',
        description='Fusion rate in Hz'
    )
    
    # Multi-GPU preprocessor node
    preprocessor_node = Node(
        package='robocache',
        executable='robot_preprocessor_ddp',
        name='robocache_preprocessor_ddp',
        output='screen',
        parameters=[{
            'voxel_size': LaunchConfiguration('voxel_size'),
            'grid_size': LaunchConfiguration('grid_size'),
            'batch_size': LaunchConfiguration('batch_size'),
            'fusion_rate_hz': LaunchConfiguration('fusion_rate_hz'),
        }],
        arguments=[
            f'num_gpus:={LaunchConfiguration("num_gpus")}'
        ]
    )
    
    return LaunchDescription([
        num_gpus_arg,
        voxel_size_arg,
        grid_size_arg,
        batch_size_arg,
        fusion_rate_arg,
        preprocessor_node,
    ])

