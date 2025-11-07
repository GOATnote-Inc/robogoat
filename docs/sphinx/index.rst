RoboCache Documentation
=======================

.. image:: https://img.shields.io/badge/CUDA-13.0-green.svg
   :target: https://developer.nvidia.com/cuda-toolkit
   :alt: CUDA 13.0

.. image:: https://img.shields.io/badge/PyTorch-2.0+-orange.svg
   :target: https://pytorch.org/
   :alt: PyTorch 2.0+

.. image:: https://img.shields.io/badge/Python-3.8+-blue.svg
   :target: https://www.python.org/
   :alt: Python 3.8+

**GPU-Accelerated Data Engine for Robot Foundation Models**

RoboCache is a high-performance CUDA library for real-time sensor preprocessing in robotics applications. It provides:

* **3-Stream Multimodal Fusion:** Temporal alignment of vision, proprioception, and IMU at 50+ Hz
* **Point Cloud Voxelization:** 25-35 billion points/sec on H100 GPU
* **CPU Fallbacks:** Vectorized PyTorch implementations for development without GPUs
* **Production-Ready:** Validated on A100/H100, ROS 2 integration, 24h burn-in tested

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/ops
   api/logging
   api/metrics

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/tuning
   guides/ros2
   guides/profiling
   guides/multi_gpu

.. toctree::
   :maxdepth: 2
   :caption: Performance

   performance/benchmarks
   performance/validation
   performance/h100
   performance/a100

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   testing
   security

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
