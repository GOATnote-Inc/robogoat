"""
RoboCache test suite.

Test organization:
- test_backends.py: Backend selection and feature parity
- test_trajectory.py: Trajectory resampling (CUDA vs PyTorch)
- test_multimodal.py: Multimodal fusion
- test_voxelization.py: Point cloud voxelization
- test_numerical.py: Numerical accuracy and correctness
- test_performance.py: Performance regression tests
- test_api.py: API stability and error handling
"""

__version__ = "0.2.1"

