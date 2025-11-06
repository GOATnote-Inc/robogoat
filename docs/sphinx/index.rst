RoboCache Documentation
=======================

**Version:** 1.0.0

RoboCache is a GPU-accelerated data engine for robot foundation models, providing
CUDA-optimized preprocessing operations that eliminate CPU dataloader bottlenecks.

.. image:: https://img.shields.io/badge/cuda-13.0+-green.svg
   :target: https://developer.nvidia.com/cuda-toolkit
   :alt: CUDA 13.0+

.. image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/pytorch-2.5%2B-orange.svg
   :target: https://pytorch.org/
   :alt: PyTorch 2.5+

Features
--------

* **Complete Kernel Coverage**: Trajectory resampling, multimodal fusion, voxelization
* **Production-Ready**: SLSA Level 3 attestation, SBOM, signed artifacts
* **Hardware Validated**: H100 (SM90), A100 (SM80) with Nsight profiling
* **High Performance**: 14.7× speedup on H100, <1% variance
* **Type-Safe API**: Full mypy support, comprehensive docstrings

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # CUDA 13.0 variant
   pip install robocache[cu130]

   # CUDA 12.4 variant
   pip install robocache[cu124]

   # CUDA 12.1 variant
   pip install robocache[cu121]

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import robocache

   # Trajectory resampling
   source = torch.randn(32, 500, 256, device='cuda', dtype=torch.bfloat16)
   src_times = torch.linspace(0, 5, 500, device='cuda').expand(32, -1)
   tgt_times = torch.linspace(0, 5, 256, device='cuda').expand(32, -1)
   
   resampled = robocache.resample_trajectories(source, src_times, tgt_times)
   # H100: 2.6ms latency, 14.7× speedup vs CPU

   # Multimodal sensor fusion
   vision = torch.randn(4, 30, 512, device='cuda', dtype=torch.bfloat16)
   vision_times = torch.linspace(0, 1, 30, device='cuda').expand(4, -1)
   proprio = torch.randn(4, 100, 64, device='cuda', dtype=torch.bfloat16)
   proprio_times = torch.linspace(0, 1, 100, device='cuda').expand(4, -1)
   imu = torch.randn(4, 200, 12, device='cuda', dtype=torch.bfloat16)
   imu_times = torch.linspace(0, 1, 200, device='cuda').expand(4, -1)
   target = torch.linspace(0, 1, 50, device='cuda').expand(4, -1)
   
   fused = robocache.fuse_multimodal(
       vision, vision_times, proprio, proprio_times, imu, imu_times, target
   )
   # H100: <1ms latency, 3-stream fusion

   # Point cloud voxelization
   points = torch.rand(1000000, 3, device='cuda') * 20 - 10  # 1M points
   grid = robocache.voxelize_pointcloud(points, mode="occupancy")
   # H100: >2.5B points/sec

Performance
-----------

=============================== ========== ========== ============
Operation                       H100       A100       Speedup (H100)
=============================== ========== ========== ============
Trajectory (32×500×256)         2.605ms    3.1ms      14.7×
Trajectory (8×250×128)          0.184ms    0.22ms     109.6×
Multimodal Fusion (3-stream)    <1ms       <1.2ms     TBD
Voxelization (128³, 1M points)  <0.5ms     <0.6ms     TBD
=============================== ========== ========== ============

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   api_reference
   examples
   performance
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   architecture
   kernel_tuning
   profiling
   contributing
   testing

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api/modules
   changelog
   roadmap
   acknowledgments

API Reference
-------------

.. autosummary::
   :toctree: api
   :recursive:

   robocache

Core Functions
~~~~~~~~~~~~~~

.. autofunction:: robocache.resample_trajectories

.. autofunction:: robocache.fuse_multimodal

.. autofunction:: robocache.voxelize_pointcloud

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: robocache.is_cuda_available

.. autofunction:: robocache.self_test

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
--------

If you use RoboCache in your research, please cite:

.. code-block:: bibtex

   @software{robocache2025,
     title = {RoboCache: GPU-Accelerated Data Engine for Robot Foundation Models},
     author = {Dent, Brandon},
     year = {2025},
     url = {https://github.com/GOATnote-Inc/robogoat},
     version = {1.0.0}
   }

License
-------

RoboCache is licensed under the Apache License 2.0. See LICENSE file for details.

Support
-------

* **Documentation**: https://github.com/GOATnote-Inc/robogoat/tree/main/docs
* **Issues**: https://github.com/GOATnote-Inc/robogoat/issues
* **Email**: b@thegoatnote.com

