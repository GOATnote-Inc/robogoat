"""
Correctness tests for multimodal sensor fusion.

Tests alignment of multiple sensor streams to common timestamps.
"""

import pytest
import torch


@pytest.fixture
def robocache_module():
    """Import robocache once per module."""
    try:
        import robocache
        return robocache
    except ImportError:
        pytest.skip("robocache not installed")


@pytest.fixture
def device():
    """Get CUDA device if available, else CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class TestMultimodalFusionCorrectness:
    """Correctness tests for multimodal sensor fusion."""
    
    def test_basic_alignment(self, robocache_module, device):
        """Test basic 3-stream alignment."""
        batch_size = 4
        
        # Generate data with different frequencies
        vision_len, proprio_len, force_len = 100, 150, 200
        vision_dim, proprio_dim, force_dim = 256, 16, 6
        
        vision_data = torch.randn(batch_size, vision_len, vision_dim, device=device, dtype=torch.bfloat16)
        vision_times = torch.linspace(0, 5, vision_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        proprio_data = torch.randn(batch_size, proprio_len, proprio_dim, device=device, dtype=torch.bfloat16)
        proprio_times = torch.linspace(0, 5, proprio_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        force_data = torch.randn(batch_size, force_len, force_dim, device=device, dtype=torch.bfloat16)
        force_times = torch.linspace(0, 5, force_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        target_times = torch.linspace(0, 5, 50, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Fuse
        fused = robocache_module.fuse_multimodal(
            vision_data, vision_times,
            proprio_data, proprio_times,
            force_data, force_times,
            target_times
        )
        
        # Verify shape
        expected_shape = (batch_size, 50, vision_dim + proprio_dim + force_dim)
        assert fused.shape == expected_shape, f"Expected {expected_shape}, got {fused.shape}"
        
        # Verify dtype
        assert fused.dtype == torch.bfloat16
        
        # Verify no NaN/Inf
        assert not torch.isnan(fused).any()
        assert not torch.isinf(fused).any()
    
    def test_temporal_consistency(self, robocache_module, device):
        """Test that fusion preserves temporal ordering."""
        batch_size = 2
        
        # Create simple step functions for each modality
        vision_len = 10
        vision_data = torch.zeros(batch_size, vision_len, 8, device=device, dtype=torch.float32)
        vision_data[:, 5:, :] = 1.0  # Step at t=2.5
        vision_times = torch.linspace(0, 5, vision_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        proprio_len = 15
        proprio_data = torch.zeros(batch_size, proprio_len, 4, device=device, dtype=torch.float32)
        proprio_data[:, 7:, :] = 2.0  # Step at t≈2.33
        proprio_times = torch.linspace(0, 5, proprio_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        force_len = 20
        force_data = torch.zeros(batch_size, force_len, 2, device=device, dtype=torch.float32)
        force_data[:, 10:, :] = 3.0  # Step at t=2.5
        force_times = torch.linspace(0, 5, force_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Target at t=2.5 (should see steps)
        target_times = torch.tensor([[2.5]], device=device).expand(batch_size, -1)
        
        fused = robocache_module.fuse_multimodal(
            vision_data, vision_times,
            proprio_data, proprio_times,
            force_data, force_times,
            target_times
        )
        
        # Vision features (first 8 dims) should be near 1.0
        assert fused[:, 0, :8].mean().item() > 0.4, "Vision step not preserved"
        
        # Proprio features (next 4 dims) should be near 2.0
        assert fused[:, 0, 8:12].mean().item() > 0.9, "Proprio step not preserved"
        
        # Force features (last 2 dims) should be near 3.0
        assert fused[:, 0, 12:].mean().item() > 1.4, "Force step not preserved"
    
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    @pytest.mark.parametrize("target_len", [50, 100, 250])
    def test_parametric_shapes(self, robocache_module, device, batch_size, target_len):
        """Test multimodal fusion across various configurations."""
        vision_dim, proprio_dim, force_dim = 128, 16, 6
        
        vision_data = torch.randn(batch_size, 100, vision_dim, device=device, dtype=torch.bfloat16)
        vision_times = torch.linspace(0, 5, 100, device=device).unsqueeze(0).expand(batch_size, -1)
        
        proprio_data = torch.randn(batch_size, 150, proprio_dim, device=device, dtype=torch.bfloat16)
        proprio_times = torch.linspace(0, 5, 150, device=device).unsqueeze(0).expand(batch_size, -1)
        
        force_data = torch.randn(batch_size, 200, force_dim, device=device, dtype=torch.bfloat16)
        force_times = torch.linspace(0, 5, 200, device=device).unsqueeze(0).expand(batch_size, -1)
        
        target_times = torch.linspace(0, 5, target_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        fused = robocache_module.fuse_multimodal(
            vision_data, vision_times,
            proprio_data, proprio_times,
            force_data, force_times,
            target_times
        )
        
        expected_shape = (batch_size, target_len, vision_dim + proprio_dim + force_dim)
        assert fused.shape == expected_shape
        assert not torch.isnan(fused).any()
        assert not torch.isinf(fused).any()
    
    def test_heterogeneous_frequencies(self, robocache_module, device):
        """Test fusion with very different sensor frequencies."""
        batch_size = 4
        
        # Vision: 30 Hz (slow)
        vision_len = 30
        vision_data = torch.randn(batch_size, vision_len, 256, device=device, dtype=torch.bfloat16)
        vision_times = torch.linspace(0, 1, vision_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Proprio: 100 Hz (medium)
        proprio_len = 100
        proprio_data = torch.randn(batch_size, proprio_len, 16, device=device, dtype=torch.bfloat16)
        proprio_times = torch.linspace(0, 1, proprio_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Force: 333 Hz (fast)
        force_len = 333
        force_data = torch.randn(batch_size, force_len, 6, device=device, dtype=torch.bfloat16)
        force_times = torch.linspace(0, 1, force_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Target: 50 Hz (policy frequency)
        target_len = 50
        target_times = torch.linspace(0, 1, target_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        fused = robocache_module.fuse_multimodal(
            vision_data, vision_times,
            proprio_data, proprio_times,
            force_data, force_times,
            target_times
        )
        
        assert fused.shape == (batch_size, target_len, 256 + 16 + 6)
        assert not torch.isnan(fused).any()
        
        # Verify temporal coherence: values should vary smoothly
        # Check that adjacent timesteps have similar values
        diff = (fused[:, 1:, :] - fused[:, :-1, :]).abs().mean()
        # Difference between adjacent steps should be reasonable (not random)
        assert diff < 0.5, "Temporal discontinuity detected"
    
    def test_gradient_flow(self, robocache_module, device):
        """Test that gradients flow through multimodal fusion."""
        if device == "cpu":
            pytest.skip("Gradient test only relevant for GPU")
        
        batch_size = 4
        
        vision_data = torch.randn(batch_size, 50, 128, device=device, dtype=torch.float32, requires_grad=True)
        vision_times = torch.linspace(0, 5, 50, device=device).unsqueeze(0).expand(batch_size, -1)
        
        proprio_data = torch.randn(batch_size, 75, 16, device=device, dtype=torch.float32, requires_grad=True)
        proprio_times = torch.linspace(0, 5, 75, device=device).unsqueeze(0).expand(batch_size, -1)
        
        force_data = torch.randn(batch_size, 100, 6, device=device, dtype=torch.float32, requires_grad=True)
        force_times = torch.linspace(0, 5, 100, device=device).unsqueeze(0).expand(batch_size, -1)
        
        target_times = torch.linspace(0, 5, 25, device=device).unsqueeze(0).expand(batch_size, -1)
        
        fused = robocache_module.fuse_multimodal(
            vision_data, vision_times,
            proprio_data, proprio_times,
            force_data, force_times,
            target_times
        )
        
        # Backward pass
        loss = fused.sum()
        loss.backward()
        
        # Check gradients exist and are reasonable
        for data, name in [(vision_data, "vision"), (proprio_data, "proprio"), (force_data, "force")]:
            assert data.grad is not None, f"{name} gradient is None"
            assert not torch.isnan(data.grad).any(), f"{name} gradient has NaN"
            assert not torch.isinf(data.grad).any(), f"{name} gradient has Inf"
            grad_norm = data.grad.norm().item()
            assert grad_norm > 0, f"{name} gradient is zero"
            assert grad_norm < 10000, f"{name} gradient exploded"
    
    def test_empty_modality_handling(self, robocache_module, device):
        """Test handling of very sparse data (edge case)."""
        batch_size = 2
        
        # Vision: normal
        vision_data = torch.randn(batch_size, 50, 128, device=device, dtype=torch.bfloat16)
        vision_times = torch.linspace(0, 5, 50, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Proprio: only 2 samples (extreme sparsity)
        proprio_data = torch.randn(batch_size, 2, 16, device=device, dtype=torch.bfloat16)
        proprio_times = torch.tensor([[0.0, 5.0]], device=device).expand(batch_size, -1)
        
        # Force: normal
        force_data = torch.randn(batch_size, 100, 6, device=device, dtype=torch.bfloat16)
        force_times = torch.linspace(0, 5, 100, device=device).unsqueeze(0).expand(batch_size, -1)
        
        target_times = torch.linspace(0, 5, 50, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Should not crash
        fused = robocache_module.fuse_multimodal(
            vision_data, vision_times,
            proprio_data, proprio_times,
            force_data, force_times,
            target_times
        )
        
        assert fused.shape == (batch_size, 50, 128 + 16 + 6)
        assert not torch.isnan(fused).any()

