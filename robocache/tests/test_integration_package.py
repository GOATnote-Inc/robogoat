"""Integration smoke tests against the wheel-style Python package."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch


def _import_package():
    """Import robocache as if installed from a wheel."""

    package_root = Path(__file__).resolve().parents[1] / "python"
    sys.path.insert(0, str(package_root))
    try:
        importlib.invalidate_caches()
        return importlib.import_module("robocache")
    finally:
        sys.path.pop(0)


@pytest.fixture(scope="module")
def robocache_pkg():
    return _import_package()


def test_resample_cpu_path(robocache_pkg):
    data = torch.tensor([[[0.0], [1.0], [2.0]]], dtype=torch.float32)
    src_times = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    tgt_times = torch.tensor([[0.0, 0.5, 1.5, 2.0]], dtype=torch.float32)

    result = robocache_pkg.resample_trajectories(data, src_times, tgt_times, backend="pytorch")

    expected = torch.tensor([[[0.0], [0.5], [1.5], [2.0]]], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_resample_cuda_request_raises(robocache_pkg):
    data = torch.zeros((1, 2, 1), dtype=torch.float32)
    times = torch.zeros((1, 2), dtype=torch.float32)

    with pytest.raises(NotImplementedError):
        robocache_pkg.resample_trajectories(data, times, times, backend="cuda")


def test_fuse_multimodal_cpu_path(robocache_pkg):
    vision = torch.tensor([[[0.0], [1.0]]], dtype=torch.float32)
    proprio = torch.tensor([[[0.5], [1.5]]], dtype=torch.float32)
    force = torch.tensor([[[1.0], [2.0]]], dtype=torch.float32)
    times = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)

    fused = robocache_pkg.fuse_multimodal(vision, times, proprio, times, force, times, target, backend="pytorch")

    assert fused.shape == (1, 3, 3)


def test_fuse_multimodal_cuda_request_raises(robocache_pkg):
    tensor = torch.zeros((1, 2, 1), dtype=torch.float32)
    times = torch.zeros((1, 2), dtype=torch.float32)
    target = torch.zeros((1, 2), dtype=torch.float32)

    with pytest.raises(NotImplementedError):
        robocache_pkg.fuse_multimodal(tensor, times, tensor, times, tensor, times, target, backend="cuda")


def test_voxelize_cpu_path(robocache_pkg):
    points = torch.tensor([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]], dtype=torch.float32)
    grid = robocache_pkg.voxelize_occupancy(points, (2, 2, 2), voxel_size=1.0)

    assert grid.sum() == pytest.approx(2.0)
    assert grid.shape == (2, 2, 2)


def test_voxelize_cuda_request_raises(robocache_pkg):
    points = torch.zeros((1, 3), dtype=torch.float32)

    with pytest.raises(NotImplementedError):
        robocache_pkg.voxelize_occupancy(points, (1, 1, 1), voxel_size=1.0, backend="cuda")
