"""Launch file for RoboCache preprocessor node"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'device',
            default_value='cuda',
            description='Device for RoboCache (cuda or cpu)'
        ),
        DeclareLaunchArgument(
            'voxel_size',
            default_value='0.05',
            description='Voxel size in meters'
        ),
        DeclareLaunchArgument(
            'grid_size',
            default_value='128',
            description='Voxel grid size (cubic)'
        ),
        Node(
            package='robocache_ros',
            executable='robot_preprocessor.py',
            name='robocache_preprocessor',
            parameters=[{
                'device': LaunchConfiguration('device'),
                'voxel_size': LaunchConfiguration('voxel_size'),
                'grid_size': LaunchConfiguration('grid_size'),
                'use_robocache': True,
            }],
            output='screen'
        )
    ])

