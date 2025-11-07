Operations
==========

Detailed API reference for all RoboCache operations.

Multimodal Fusion
-----------------

.. autofunction:: robocache.fuse_multimodal

**Description:**

Temporally aligns 3 sensor streams (vision, proprioception, IMU) to a common timestamp grid using linear interpolation.

**Parameters:**

* ``stream1_data`` (torch.Tensor): Vision features, shape (B, T1, D1)
* ``stream1_times`` (torch.Tensor): Vision timestamps, shape (B, T1)
* ``stream2_data`` (torch.Tensor): Proprioception features, shape (B, T2, D2)
* ``stream2_times`` (torch.Tensor): Proprioception timestamps, shape (B, T2)
* ``stream3_data`` (torch.Tensor): IMU features, shape (B, T3, D3)
* ``stream3_times`` (torch.Tensor): IMU timestamps, shape (B, T3)
* ``target_times`` (torch.Tensor): Target timestamps, shape (B, T)

**Returns:**

* ``torch.Tensor``: Fused features, shape (B, T, D1+D2+D3)

**Example:**

.. code-block:: python

   import torch
   import robocache

   batch = 4
   vision = torch.randn(batch, 30, 512, dtype=torch.bfloat16, device='cuda')
   vision_times = torch.linspace(0, 1, 30, device='cuda').expand(batch, -1)
   proprio = torch.randn(batch, 100, 64, dtype=torch.bfloat16, device='cuda')
   proprio_times = torch.linspace(0, 1, 100, device='cuda').expand(batch, -1)
   imu = torch.randn(batch, 200, 12, dtype=torch.bfloat16, device='cuda')
   imu_times = torch.linspace(0, 1, 200, device='cuda').expand(batch, -1)
   target = torch.linspace(0, 1, 50, device='cuda').expand(batch, -1)

   fused = robocache.fuse_multimodal(
       vision, vision_times, proprio, proprio_times, imu, imu_times, target
   )
   # Output: (4, 50, 588)

**Performance:**

* H100: 0.018ms P50 latency
* A100: 0.057ms P50 latency

Point Cloud Voxelization
-------------------------

.. autofunction:: robocache.voxelize_pointcloud

**Description:**

Converts a point cloud to a 3D voxel grid. Supports multiple aggregation modes.

**Parameters:**

* ``points`` (torch.Tensor): Point cloud, shape (N, 3)
* ``features`` (torch.Tensor, optional): Per-point features, shape (N, D)
* ``grid_min`` (tuple): Minimum grid coordinates (x, y, z)
* ``voxel_size`` (float): Voxel size in meters
* ``grid_size`` (tuple): Grid dimensions (nx, ny, nz)
* ``mode`` (str): Aggregation mode ('count', 'occupancy', 'mean', 'max')

**Returns:**

* ``torch.Tensor``: Voxel grid
  
  * ``count``: (nx, ny, nz), dtype=int32
  * ``occupancy``: (nx, ny, nz), dtype=float32, values in [0, 1]
  * ``mean``: (nx, ny, nz, D), dtype=float32
  * ``max``: (nx, ny, nz, D), dtype=float32

**Example:**

.. code-block:: python

   import torch
   import robocache

   # Random point cloud
   points = torch.rand(500000, 3, device='cuda') * 20.0 - 10.0

   # Voxelize (occupancy grid)
   voxel_grid = robocache.voxelize_pointcloud(
       points,
       grid_min=[-10.0, -10.0, -10.0],
       voxel_size=0.05,
       grid_size=[128, 128, 128],
       mode='occupancy'
   )
   # Output: (128, 128, 128)

   # With features
   features = torch.randn(500000, 8, device='cuda')
   voxel_grid = robocache.voxelize_pointcloud(
       points, features,
       grid_min=[-10.0, -10.0, -10.0],
       voxel_size=0.05,
       grid_size=[128, 128, 128],
       mode='mean'
   )
   # Output: (128, 128, 128, 8)

**Performance:**

* H100 (count): 0.014ms P50, 34.5 billion points/sec
* H100 (occupancy): 0.016ms P50, 30.3 billion points/sec
* A100 (occupancy): 0.032ms P50, 15.6 billion points/sec

Trajectory Resampling
---------------------

.. autofunction:: robocache.resample_trajectories

**Description:**

Resamples time-series data to new timestamps using linear interpolation.

**Parameters:**

* ``source_data`` (torch.Tensor): Source data, shape (B, T_src, D)
* ``source_times`` (torch.Tensor): Source timestamps, shape (B, T_src)
* ``target_times`` (torch.Tensor): Target timestamps, shape (B, T_tgt)

**Returns:**

* ``torch.Tensor``: Resampled data, shape (B, T_tgt, D)

**Example:**

.. code-block:: python

   import torch
   import robocache

   batch = 4
   data = torch.randn(batch, 100, 64, dtype=torch.bfloat16, device='cuda')
   source_times = torch.linspace(0, 1, 100, device='cuda').expand(batch, -1)
   target_times = torch.linspace(0, 1, 50, device='cuda').expand(batch, -1)

   resampled = robocache.resample_trajectories(data, source_times, target_times)
   # Output: (4, 50, 64)

**Performance:**

* H100: ~0.005ms P50 latency (part of multimodal fusion)

