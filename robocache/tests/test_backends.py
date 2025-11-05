"""
Test backend selection and multi-backend feature parity.

Validates:
- Backend detection and selection
- Automatic fallback
- Feature parity between CUDA and PyTorch
- Error handling for unavailable backends
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import robocache
from robocache.backends import BackendType, select_backend, get_backend_status, PyTorchBackend


class TestBackendSelection:
    """Test backend selection logic."""
    
    def test_backend_status(self):
        """Test that backend status is properly detected."""
        status = get_backend_status()
        
        # PyTorch should always be available (it's a requirement)
        assert status.pytorch_available, "PyTorch backend should be available"
        
        # CUDA may or may not be available
        if torch.cuda.is_available():
            print(f"CUDA backend status: {status.cuda_available}")
            if not status.cuda_available:
                print(f"CUDA error: {status.cuda_error}")
    
    def test_auto_backend_selection(self):
        """Test automatic backend selection."""
        backend = select_backend('auto')
        assert backend in [BackendType.CUDA, BackendType.PYTORCH]
        
        backend_none = select_backend(None)
        assert backend_none in [BackendType.CUDA, BackendType.PYTORCH]
    
    def test_explicit_pytorch_backend(self):
        """Test forcing PyTorch backend."""
        backend = select_backend('pytorch')
        assert backend == BackendType.PYTORCH
    
    def test_explicit_cuda_backend(self):
        """Test forcing CUDA backend (if available)."""
        status = get_backend_status()
        
        if status.cuda_available:
            backend = select_backend('cuda')
            assert backend == BackendType.CUDA
        else:
            # Should raise error if CUDA requested but unavailable
            with pytest.raises(RuntimeError, match="CUDA backend"):
                select_backend('cuda')
    
    def test_invalid_backend(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="Invalid backend"):
            select_backend('invalid_backend')
    
    def test_backend_info(self):
        """Test backend info retrieval."""
        info = robocache.get_backend_info()
        
        assert "backends" in info
        assert "cuda" in info["backends"]
        assert "pytorch" in info["backends"]
        assert "default_backend" in info


class TestPyTorchBackend:
    """Test PyTorch fallback implementations."""
    
    def test_trajectory_resampling_pytorch(self):
        """Test PyTorch trajectory resampling implementation."""
        B, S, T, D = 4, 50, 32, 16
        
        source_data = torch.randn(B, S, D, dtype=torch.float32)
        source_times = torch.linspace(0, 1, S).unsqueeze(0).expand(B, -1).contiguous()
        target_times = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1).contiguous()
        
        result = PyTorchBackend.resample_trajectories(source_data, source_times, target_times)
        
        assert result.shape == (B, T, D)
        assert result.dtype == torch.float32
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
    
    def test_multimodal_fusion_pytorch(self):
        """Test PyTorch multimodal fusion implementation."""
        B, T = 4, 32
        
        vision = torch.randn(B, 50, 64, dtype=torch.float32)
        vision_t = torch.linspace(0, 1, 50).unsqueeze(0).expand(B, -1).contiguous()
        
        proprio = torch.randn(B, 80, 16, dtype=torch.float32)
        proprio_t = torch.linspace(0, 1, 80).unsqueeze(0).expand(B, -1).contiguous()
        
        target_t = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1).contiguous()
        
        result = PyTorchBackend.fused_multimodal_alignment(
            vision, vision_t,
            proprio, proprio_t,
            None, None,
            target_t
        )
        
        assert result.shape == (B, T, 64 + 16)
        assert not torch.isnan(result).any()
    
    def test_voxelization_pytorch(self):
        """Test PyTorch voxelization implementation."""
        B, N = 2, 100
        grid_size = [16, 16, 16]
        
        points = torch.randn(B, N, 3, dtype=torch.float32) * 2  # Points in [-2, 2] range
        origin = torch.tensor([-2.0, -2.0, -2.0])
        voxel_size = 0.25  # 16 voxels * 0.25 = 4.0 range
        
        result = PyTorchBackend.voxelize_occupancy(points, grid_size, voxel_size, origin)
        
        assert result.shape == (B, 16, 16, 16)
        assert result.dtype == torch.float32
        assert ((result == 0.0) | (result == 1.0)).all(), "Voxels should be binary (0 or 1)"


class TestBackendParity:
    """Test feature parity between CUDA and PyTorch backends."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_trajectory_parity(self):
        """Test CUDA and PyTorch give same results for trajectory resampling."""
        B, S, T, D = 4, 50, 32, 16
        
        # Generate test data
        source_data = torch.randn(B, S, D, dtype=torch.float32, device='cpu')
        source_times = torch.linspace(0, 1, S).unsqueeze(0).expand(B, -1).contiguous()
        target_times = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1).contiguous()
        
        # PyTorch result (CPU)
        pytorch_result = robocache.resample_trajectories(
            source_data, source_times, target_times, backend='pytorch'
        )
        
        # CUDA result (if available)
        try:
            cuda_data = source_data.cuda()
            cuda_src_t = source_times.cuda()
            cuda_tgt_t = target_times.cuda()
            
            cuda_result = robocache.resample_trajectories(
                cuda_data, cuda_src_t, cuda_tgt_t, backend='cuda'
            ).cpu()
            
            # Check parity
            max_diff = (pytorch_result - cuda_result).abs().max().item()
            print(f"Max difference: {max_diff}")
            
            # Allow some tolerance for floating point differences
            assert max_diff < 1e-5, f"CUDA/PyTorch mismatch: max diff {max_diff}"
        
        except RuntimeError as e:
            if "CUDA backend" in str(e):
                pytest.skip("CUDA extension not available")
            else:
                raise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

