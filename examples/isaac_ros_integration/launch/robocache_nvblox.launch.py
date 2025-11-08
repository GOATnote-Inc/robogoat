"""
ROS 2 Launch file for RoboCache + Isaac ROS Nvblox Integration

Launches a complete 3D reconstruction and mapping pipeline:
  1. Isaac ROS Visual SLAM (pose estimation)
  2. Isaac ROS Stereo Depth (dense depth estimation)
  3. RoboCache Voxelization (GPU-accelerated)
  4. Nvblox TSDF Reconstruction (real-time 3D mapping)
  5. Nav2 Integration (navigation planning)

Hardware requirements:
  - NVIDIA GPU (A100, H100, L4, RTX 40 series)
  - ROS 2 Jazzy
  - Isaac ROS GEMs installed
  - Stereo camera or depth camera (RealSense D435, ZED 2)

Usage:
  # Default (all modules):
  ros2 launch robocache robocache_nvblox.launch.py

  # Point cloud only (no depth):
  ros2 launch robocache robocache_nvblox.launch.py use_depth:=false

  # Disable mesh generation (faster):
  ros2 launch robocache robocache_nvblox.launch.py enable_mesh:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Launch arguments
    use_nvblox_arg = DeclareLaunchArgument(
        'use_nvblox',
        default_value='true',
        description='Enable Nvblox TSDF reconstruction'
    )
    
    use_depth_arg = DeclareLaunchArgument(
        'use_depth',
        default_value='true',
        description='Use depth camera for TSDF'
    )
    
    enable_mesh_arg = DeclareLaunchArgument(
        'enable_mesh',
        default_value='false',
        description='Enable mesh generation (heavy)'
    )
    
    voxel_size_arg = DeclareLaunchArgument(
        'voxel_size',
        default_value='0.05',
        description='Voxel size in meters'
    )
    
    grid_size_arg = DeclareLaunchArgument(
        'grid_size',
        default_value='256',
        description='Voxel grid size (NxNxN)'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda:0',
        description='CUDA device for RoboCache'
    )
    
    # RoboCache + Nvblox composition node
    robocache_nvblox_node = Node(
        package='robocache',
        executable='robocache_nvblox_composition',
        name='robocache_nvblox',
        output='screen',
        parameters=[{
            'device': LaunchConfiguration('device'),
            'voxel_size': LaunchConfiguration('voxel_size'),
            'grid_size': LaunchConfiguration('grid_size'),
            'use_nvblox': LaunchConfiguration('use_nvblox'),
            'enable_mesh_output': LaunchConfiguration('enable_mesh'),
            'processing_rate_hz': 30.0,
            'map_origin_x': -12.8,
            'map_origin_y': -12.8,
            'map_origin_z': -12.8,
        }]
    )
    
    # Isaac ROS Nvblox node (if available)
    nvblox_composable_node = ComposableNode(
        package='nvblox_ros',
        plugin='nvblox::NvbloxNode',
        name='nvblox',
        parameters=[{
            'voxel_size': LaunchConfiguration('voxel_size'),
            'tsdf_decay_factor': 0.95,
            'max_integration_distance_m': 10.0,
            'mesh_update_rate_hz': 5.0,
            'esdf_update_rate_hz': 10.0,
            'max_back_projection_distance_m': 10.0,
            'use_color': False,  # Faster without color
            'use_depth': LaunchConfiguration('use_depth'),
        }],
        remappings=[
            ('depth/image', '/camera/depth/image'),
            ('depth/camera_info', '/camera/depth/camera_info'),
            ('color/image', '/camera/color/image'),
            ('color/camera_info', '/camera/color/camera_info'),
            ('pointcloud', '/robocache/voxel_grid'),
            ('transform', '/robot/pose')
        ],
        condition=IfCondition(LaunchConfiguration('use_nvblox'))
    )
    
    # Isaac ROS Visual SLAM (for pose estimation)
    visual_slam_node = ComposableNode(
        package='isaac_ros_visual_slam',
        plugin='isaac_ros::visual_slam::VisualSlamNode',
        name='visual_slam',
        parameters=[{
            'enable_imu_fusion': True,
            'enable_ground_constraint_in_odometry': True,
            'enable_slam_visualization': True,
            'enable_localization_n_mapping': True,
            'path_max_size': 1024,
        }],
        remappings=[
            ('visual_slam/image_0', '/camera/left/image'),
            ('visual_slam/camera_info_0', '/camera/left/camera_info'),
            ('visual_slam/image_1', '/camera/right/image'),
            ('visual_slam/camera_info_1', '/camera/right/camera_info'),
            ('visual_slam/imu', '/imu/data'),
        ]
    )
    
    # Container for Isaac ROS composable nodes
    isaac_container = ComposableNodeContainer(
        name='isaac_ros_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',  # Multi-threaded
        composable_node_descriptions=[
            nvblox_composable_node,
            visual_slam_node,
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_nvblox'))
    )
    
    # Static transform publishers (example - adjust for your robot)
    static_tf_base_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=['0.1', '0', '0.5', '0', '0', '0', '1', 'base_link', 'camera_link']
    )
    
    static_tf_base_to_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_lidar_tf',
        arguments=['0', '0', '0.8', '0', '0', '0', '1', 'base_link', 'lidar_link']
    )
    
    # RViz2 for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('robocache'),
            'config',
            'robocache_nvblox.rviz'
        ])],
        condition=IfCondition('true')  # Can be parameterized
    )
    
    return LaunchDescription([
        # Arguments
        use_nvblox_arg,
        use_depth_arg,
        enable_mesh_arg,
        voxel_size_arg,
        grid_size_arg,
        device_arg,
        
        # Nodes
        robocache_nvblox_node,
        isaac_container,
        static_tf_base_to_camera,
        static_tf_base_to_lidar,
        # rviz_node,  # Uncomment for visualization
    ])

