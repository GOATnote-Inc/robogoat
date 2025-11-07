Quick Start
===========

This guide will get you up and running with RoboCache in 5 minutes.

Basic Usage
-----------

Multimodal Fusion
~~~~~~~~~~~~~~~~~

Temporally align 3 sensor streams (vision, proprioception, IMU) to a common timestamp grid:

.. code-block:: python

   import torch
   import robocache

   batch = 4

   # Vision stream (30 Hz camera)
   vision = torch.randn(batch, 30, 512, dtype=torch.bfloat16, device='cuda')
   vision_times = torch.linspace(0, 1, 30, device='cuda').expand(batch, -1)

   # Proprioception (100 Hz joint encoders)
   proprio = torch.randn(batch, 100, 64, dtype=torch.bfloat16, device='cuda')
   proprio_times = torch.linspace(0, 1, 100, device='cuda').expand(batch, -1)

   # IMU stream (200 Hz)
   imu = torch.randn(batch, 200, 12, dtype=torch.bfloat16, device='cuda')
   imu_times = torch.linspace(0, 1, 200, device='cuda').expand(batch, -1)

   # Target timestamps (50 Hz policy inference)
   target_times = torch.linspace(0, 1, 50, device='cuda').expand(batch, -1)

   # Fuse all streams
   fused = robocache.fuse_multimodal(
       vision, vision_times,
       proprio, proprio_times,
       imu, imu_times,
       target_times
   )

   print(fused.shape)  # (4, 50, 588) = batch × time × (512+64+12)

Point Cloud Voxelization
~~~~~~~~~~~~~~~~~~~~~~~~~

Convert LiDAR point clouds to 3D voxel grids:

.. code-block:: python

   import torch
   import robocache

   # Point cloud (N × 3)
   points = torch.rand(500000, 3, device='cuda') * 20.0 - 10.0

   # Voxelize (occupancy grid)
   voxel_grid = robocache.voxelize_pointcloud(
       points,
       grid_min=[-10.0, -10.0, -10.0],
       voxel_size=0.05,  # 5cm voxels
       grid_size=[128, 128, 128],
       mode='occupancy'
   )

   print(voxel_grid.shape)  # (128, 128, 128)

CPU Fallback
~~~~~~~~~~~~

RoboCache automatically falls back to CPU if CUDA is unavailable:

.. code-block:: python

   import torch
   import robocache

   # CPU tensors
   points = torch.rand(10000, 3) * 4.0 - 2.0

   # Automatically uses CPU fallback
   voxel_grid = robocache.voxelize_pointcloud(
       points,
       grid_min=[-2.0, -2.0, -2.0],
       voxel_size=0.1,
       grid_size=[64, 64, 64],
       mode='count'
   )

   print(voxel_grid.device)  # cpu

Performance Tips
----------------

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process multiple samples together
   batch = 16  # Increase for higher GPU utilization
   vision = torch.randn(batch, 30, 512, dtype=torch.bfloat16, device='cuda')
   # ...

BF16 Precision
~~~~~~~~~~~~~~

.. code-block:: python

   # Use BF16 for 2× speedup vs FP32
   vision = torch.randn(batch, 30, 512, dtype=torch.bfloat16, device='cuda')
   # Note: Requires Ampere (A100) or newer GPU

Pre-allocation
~~~~~~~~~~~~~~

.. code-block:: python

   # Pre-allocate output tensors for zero-copy
   out = torch.empty(batch, 50, 588, dtype=torch.bfloat16, device='cuda')
   robocache.fuse_multimodal(..., out=out)

CUDA Streams
~~~~~~~~~~~~

.. code-block:: python

   # Use multiple streams for concurrent execution
   stream = torch.cuda.Stream()
   with torch.cuda.stream(stream):
       fused = robocache.fuse_multimodal(...)

Profiling
---------

.. code-block:: python

   import torch
   import robocache

   # Warmup
   for _ in range(50):
       fused = robocache.fuse_multimodal(...)
   torch.cuda.synchronize()

   # Measure latency
   start = torch.cuda.Event(enable_timing=True)
   end = torch.cuda.Event(enable_timing=True)

   start.record()
   for _ in range(200):
       fused = robocache.fuse_multimodal(...)
   end.record()

   torch.cuda.synchronize()
   latency_ms = start.elapsed_time(end) / 200
   print(f"P50 latency: {latency_ms:.3f} ms")

Nsight Systems
~~~~~~~~~~~~~~

.. code-block:: bash

   # Profile with Nsight Systems
   nsys profile -o profile python your_script.py

   # View timeline
   nsys-ui profile.nsys-rep

Next Steps
----------

* :doc:`examples` - Real-world use cases
* :doc:`guides/tuning` - Optimize for your hardware
* :doc:`guides/ros2` - ROS 2 integration
* :doc:`api/core` - Full API reference

