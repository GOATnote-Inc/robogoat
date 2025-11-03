"""
Tests for RoboCache installation and environment setup.
"""

import pytest
import sys


def test_import_robocache():
    """Test that robocache can be imported."""
    import robocache

    assert robocache is not None
    assert hasattr(robocache, "__version__")


def test_version():
    """Test that version is properly set."""
    import robocache

    assert isinstance(robocache.__version__, str)
    assert len(robocache.__version__.split(".")) >= 2  # At least major.minor


def test_public_api():
    """Test that public API is properly exported."""
    import robocache

    # Check main functions are exported
    assert hasattr(robocache, "resample_trajectories")
    assert hasattr(robocache, "check_installation")
    assert hasattr(robocache, "print_installation_info")

    # Check __all__ is defined
    assert hasattr(robocache, "__all__")
    assert "resample_trajectories" in robocache.__all__


def test_check_installation():
    """Test the check_installation utility function."""
    import robocache

    info = robocache.check_installation()

    assert isinstance(info, dict)
    assert "robocache_version" in info
    assert "cuda_extension_available" in info
    assert "pytorch_available" in info

    # Version should match
    assert info["robocache_version"] == robocache.__version__


@pytest.mark.gpu
def test_torch_cuda_available():
    """Test that PyTorch CUDA is available."""
    import torch

    assert torch.cuda.is_available(), "CUDA not available for testing"
    assert torch.cuda.device_count() > 0, "No CUDA devices found"


@pytest.mark.gpu
def test_gpu_info():
    """Test that we can query GPU information."""
    import torch

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        assert isinstance(device_name, str)
        assert len(device_name) > 0

        device_props = torch.cuda.get_device_properties(0)
        assert device_props.total_memory > 0


def test_print_installation_info(capsys):
    """Test that print_installation_info produces output."""
    import robocache

    info = robocache.print_installation_info()

    # Should return the same dict as check_installation
    assert isinstance(info, dict)

    # Should produce console output
    captured = capsys.readouterr()
    assert "RoboCache Installation Info" in captured.out
    assert "Version:" in captured.out
