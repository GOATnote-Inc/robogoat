"""
Performance tests for multimodal sensor fusion with regression gates.
"""

import pytest
import torch

from tests.perf.perf_guard import time_op, perf_guard


@pytest.fixture(scope="module")
def robocache_module():
    """Import robocache once per module."""
    try:
        import robocache
        return robocache
    except ImportError:
        pytest.skip("robocache not installed")


@pytest.fixture
def device():
    """Get CUDA device if available, else skip."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda"


@pytest.mark.perf
class TestMultimodalFusionPerf:
    """Performance tests for multimodal sensor fusion."""
    
    def test_3stream_small_batch_perf(self, robocache_module, device):
        """Test 3-stream fusion with small batch (8 episodes)."""
        batch_size = 8
        
        # Vision: 30 Hz
        vision_data = torch.randn(batch_size, 30, 256, device=device, dtype=torch.bfloat16)
        vision_times = torch.linspace(0, 1, 30, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Proprio: 100 Hz
        proprio_data = torch.randn(batch_size, 100, 16, device=device, dtype=torch.bfloat16)
        proprio_times = torch.linspace(0, 1, 100, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Force: 333 Hz
        force_data = torch.randn(batch_size, 333, 6, device=device, dtype=torch.bfloat16)
        force_times = torch.linspace(0, 1, 333, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Target: 50 Hz
        target_times = torch.linspace(0, 1, 50, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Benchmark
        fn = lambda: robocache_module.fuse_multimodal(
            vision_data, vision_times,
            proprio_data, proprio_times,
            force_data, force_times,
            target_times
        )
        stats = time_op(fn, warmup=10, iters=100, sync_fn=torch.cuda.synchronize)
        
        # Performance gate: < 0.10ms P50, < 0.20ms P99
        perf_guard.require_lt_ms(
            "multimodal_fusion_3stream_small",
            p50=stats.p50,
            p99=stats.p99,
            p50_max=0.10,
            p99_max=0.20
        )
        
        perf_guard.record("multimodal_fusion_3stream_small", stats)
    
    def test_3stream_medium_batch_perf(self, robocache_module, device):
        """Test 3-stream fusion with medium batch (32 episodes)."""
        batch_size = 32
        
        vision_data = torch.randn(batch_size, 30, 256, device=device, dtype=torch.bfloat16)
        vision_times = torch.linspace(0, 1, 30, device=device).unsqueeze(0).expand(batch_size, -1)
        
        proprio_data = torch.randn(batch_size, 100, 16, device=device, dtype=torch.bfloat16)
        proprio_times = torch.linspace(0, 1, 100, device=device).unsqueeze(0).expand(batch_size, -1)
        
        force_data = torch.randn(batch_size, 333, 6, device=device, dtype=torch.bfloat16)
        force_times = torch.linspace(0, 1, 333, device=device).unsqueeze(0).expand(batch_size, -1)
        
        target_times = torch.linspace(0, 1, 50, device=device).unsqueeze(0).expand(batch_size, -1)
        
        fn = lambda: robocache_module.fuse_multimodal(
            vision_data, vision_times,
            proprio_data, proprio_times,
            force_data, force_times,
            target_times
        )
        stats = time_op(fn, warmup=10, iters=100, sync_fn=torch.cuda.synchronize)
        
        # Performance gate: < 0.30ms P50, < 0.60ms P99
        perf_guard.require_lt_ms(
            "multimodal_fusion_3stream_medium",
            p50=stats.p50,
            p99=stats.p99,
            p50_max=0.30,
            p99_max=0.60
        )
        
        perf_guard.record("multimodal_fusion_3stream_medium", stats)
    
    def test_high_frequency_fusion_perf(self, robocache_module, device):
        """Test fusion with high-frequency target (stress test)."""
        batch_size = 16
        
        # Resample to very high frequency (1 kHz)
        vision_data = torch.randn(batch_size, 100, 128, device=device, dtype=torch.bfloat16)
        vision_times = torch.linspace(0, 1, 100, device=device).unsqueeze(0).expand(batch_size, -1)
        
        proprio_data = torch.randn(batch_size, 200, 16, device=device, dtype=torch.bfloat16)
        proprio_times = torch.linspace(0, 1, 200, device=device).unsqueeze(0).expand(batch_size, -1)
        
        force_data = torch.randn(batch_size, 500, 6, device=device, dtype=torch.bfloat16)
        force_times = torch.linspace(0, 1, 500, device=device).unsqueeze(0).expand(batch_size, -1)
        
        target_times = torch.linspace(0, 1, 1000, device=device).unsqueeze(0).expand(batch_size, -1)
        
        fn = lambda: robocache_module.fuse_multimodal(
            vision_data, vision_times,
            proprio_data, proprio_times,
            force_data, force_times,
            target_times
        )
        stats = time_op(fn, warmup=10, iters=100, sync_fn=torch.cuda.synchronize)
        
        # Performance gate: < 2.0ms P50, < 4.0ms P99 (high-frequency is expensive)
        perf_guard.require_lt_ms(
            "multimodal_fusion_high_freq",
            p50=stats.p50,
            p99=stats.p99,
            p50_max=2.0,
            p99_max=4.0
        )
        
        perf_guard.record("multimodal_fusion_high_freq", stats)

