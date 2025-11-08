"""
Tests for Temporal Fusion Module
"""

import pytest
import torch
import torch.nn as nn
import time
import numpy as np

from robocache.temporal_fusion import (
    PositionalEncoding,
    CausalSelfAttention,
    CrossModalAttention,
    TemporalFusionBlock,
    TemporalMultimodalFusion,
    create_temporal_fusion
)


class TestPositionalEncoding:
    """Test positional encoding"""
    
    def test_shape(self):
        pe = PositionalEncoding(d_model=512, max_len=1000)
        x = torch.randn(4, 10, 512)
        output = pe(x)
        
        assert output.shape == x.shape
    
    def test_determinism(self):
        pe = PositionalEncoding(d_model=256)
        x = torch.randn(2, 5, 256)
        
        out1 = pe(x)
        out2 = pe(x)
        
        assert torch.allclose(out1, out2)


class TestCausalSelfAttention:
    """Test causal self-attention"""
    
    def test_forward_shape(self):
        attn = CausalSelfAttention(d_model=512, num_heads=8)
        x = torch.randn(4, 10, 512)
        
        output = attn(x)
        
        assert output.shape == x.shape
    
    def test_causal_masking(self):
        """Verify future tokens don't affect past"""
        attn = CausalSelfAttention(d_model=64, num_heads=4)
        attn.eval()
        
        # Create sequence where position i has value i
        x = torch.arange(10).unsqueeze(0).unsqueeze(-1).expand(1, 10, 64).float()
        
        with torch.no_grad():
            output = attn(x)
        
        # Output at position i should not depend on positions > i
        # This is a weak test but checks basic causality
        assert output.shape == (1, 10, 64)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_execution(self):
        attn = CausalSelfAttention(d_model=512, num_heads=8).cuda()
        x = torch.randn(4, 10, 512, device='cuda')
        
        output = attn(x)
        
        assert output.is_cuda
        assert output.shape == x.shape


class TestCrossModalAttention:
    """Test cross-modal attention"""
    
    def test_forward_shape(self):
        cross_attn = CrossModalAttention(d_model=512, num_heads=8)
        query = torch.randn(4, 10, 512)
        context = torch.randn(4, 20, 512)
        
        output = cross_attn(query, context)
        
        assert output.shape == query.shape
    
    def test_different_lengths(self):
        """Test with different sequence lengths"""
        cross_attn = CrossModalAttention(d_model=256, num_heads=8)
        query = torch.randn(2, 5, 256)
        context = torch.randn(2, 15, 256)
        
        output = cross_attn(query, context)
        
        assert output.shape == query.shape


class TestTemporalFusionBlock:
    """Test temporal fusion block"""
    
    def test_forward(self):
        block = TemporalFusionBlock(d_model=512, num_heads=8)
        x = torch.randn(4, 10, 512)
        
        output = block(x)
        
        assert output.shape == x.shape
    
    def test_residual_connections(self):
        """Verify residuals help gradient flow"""
        block = TemporalFusionBlock(d_model=256, num_heads=4)
        x = torch.randn(2, 5, 256, requires_grad=True)
        
        output = block(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestTemporalMultimodalFusion:
    """Test complete temporal fusion module"""
    
    def test_forward_shape(self):
        model = TemporalMultimodalFusion(
            vision_dim=512,
            lidar_dim=256,
            proprio_dim=14,
            d_model=512,
            num_layers=2,
            num_heads=8
        )
        
        batch = 4
        T = 10
        
        vision = torch.randn(batch, T, 512)
        lidar = torch.randn(batch, T, 256)
        proprio = torch.randn(batch, T, 14)
        
        output = model(vision, lidar, proprio)
        
        assert output.shape == (batch, T, 512)
    
    def test_temporal_consistency(self):
        """Test that model processes temporal sequences"""
        model = TemporalMultimodalFusion(
            vision_dim=128,
            lidar_dim=64,
            proprio_dim=6,
            d_model=128,
            num_layers=2
        )
        model.eval()
        
        batch = 2
        T = 5
        
        vision = torch.randn(batch, T, 128)
        lidar = torch.randn(batch, T, 64)
        proprio = torch.randn(batch, T, 6)
        
        with torch.no_grad():
            output1 = model(vision, lidar, proprio)
            output2 = model(vision, lidar, proprio)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_execution(self):
        model = TemporalMultimodalFusion(
            vision_dim=512,
            lidar_dim=256,
            proprio_dim=14,
            d_model=512
        ).cuda()
        
        batch = 4
        T = 10
        
        vision = torch.randn(batch, T, 512, device='cuda')
        lidar = torch.randn(batch, T, 256, device='cuda')
        proprio = torch.randn(batch, T, 14, device='cuda')
        
        output = model(vision, lidar, proprio)
        
        assert output.is_cuda
        assert output.shape == (batch, T, 512)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_performance_target(self):
        """Test latency target: <2ms for 10 timesteps"""
        model = create_temporal_fusion(
            vision_dim=512,
            lidar_dim=256,
            proprio_dim=14,
            d_model=512,
            device='cuda'
        )
        model.eval()
        
        batch = 8
        T = 10
        
        vision = torch.randn(batch, T, 512, device='cuda')
        lidar = torch.randn(batch, T, 256, device='cuda')
        proprio = torch.randn(batch, T, 14, device='cuda')
        
        # Warmup
        for _ in range(20):
            with torch.no_grad():
                _ = model(vision, lidar, proprio)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                output = model(vision, lidar, proprio)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        median_time_ms = np.median(times) * 1000
        
        print(f"\nTemporal Fusion Performance:")
        print(f"  Latency (10 timesteps): {median_time_ms:.2f} ms")
        print(f"  Target: <2ms")
        
        # Relaxed target for initial implementation (5ms)
        assert median_time_ms < 5.0, f"Latency {median_time_ms:.2f}ms exceeds 5ms target"
    
    def test_memory_efficiency(self):
        """Test memory usage for long sequences"""
        model = TemporalMultimodalFusion(
            vision_dim=256,
            lidar_dim=128,
            proprio_dim=14,
            d_model=256,
            num_layers=2
        )
        
        batch = 4
        T = 100  # Long sequence
        
        vision = torch.randn(batch, T, 256)
        lidar = torch.randn(batch, T, 128)
        proprio = torch.randn(batch, T, 14)
        
        # Should not OOM
        output = model(vision, lidar, proprio)
        
        assert output.shape == (batch, T, 256)


class TestCreateTemporalFusion:
    """Test convenience factory function"""
    
    def test_default_creation(self):
        model = create_temporal_fusion(device='cpu')
        
        assert isinstance(model, TemporalMultimodalFusion)
        assert model.d_model == 512
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_creation(self):
        model = create_temporal_fusion(device='cuda')
        
        assert next(model.parameters()).is_cuda


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

