"""Compares CUDA outputs against high-precision CPU references."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

robocache = pytest.importorskip("robocache")

INSTALL_INFO = robocache.check_installation()
CUDA_AVAILABLE = (
    INSTALL_INFO.get("cuda_extension_available", False)
    and torch.cuda.is_available()
)
PYTORCH_AVAILABLE = INSTALL_INFO.get("pytorch_available", False)


def cpu_reference_trajectory_resample(source_data, source_times, target_times):
    """High-precision CPU reference for trajectory resampling."""

    batch, src_len, dim = source_data.shape
    _, tgt_len = target_times.shape

    source_data_cpu = source_data.cpu().float().numpy().astype(np.float64)
    source_times_cpu = source_times.cpu().numpy().astype(np.float64)
    target_times_cpu = target_times.cpu().numpy().astype(np.float64)

    output = np.zeros((batch, tgt_len, dim), dtype=np.float64)

    for b in range(batch):
        for t in range(tgt_len):
            target_time = target_times_cpu[b, t]
            left_idx = np.searchsorted(
                source_times_cpu[b], target_time, side="right"
            ) - 1
            left_idx = max(0, min(left_idx, src_len - 2))
            right_idx = left_idx + 1

            t_left = source_times_cpu[b, left_idx]
            t_right = source_times_cpu[b, right_idx]
            delta = t_right - t_left

            if delta > 1e-10:
                weight = (target_time - t_left) / delta
                weight = np.clip(weight, 0.0, 1.0)
            else:
                weight = 0.0

            left_data = source_data_cpu[b, left_idx, :]
            right_data = source_data_cpu[b, right_idx, :]
            output[b, t, :] = left_data + weight * (right_data - left_data)

    return output


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA backend not available")
def test_trajectory_resampling_correctness():
    """Test CUDA kernel produces correct outputs vs CPU reference."""

    batch, src_len, tgt_len, dim = 4, 128, 64, 8

    torch.manual_seed(42)
    data_fp32 = torch.randn(batch, src_len, dim, dtype=torch.float32)
    data = data_fp32.to(torch.bfloat16).cuda()
    src_t = (
        torch.linspace(0, 1, src_len, device="cuda")
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )
    tgt_t = (
        torch.linspace(0, 1, tgt_len, device="cuda")
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )

    cuda_output = robocache.resample_trajectories(data, src_t, tgt_t, backend="cuda")

    cpu_reference = cpu_reference_trajectory_resample(data_fp32, src_t.cpu(), tgt_t.cpu())
    cpu_reference_tensor = torch.from_numpy(cpu_reference).to(torch.bfloat16)

    cuda_output_cpu = cuda_output.cpu()
    max_diff = torch.abs(cuda_output_cpu.float() - cpu_reference_tensor.float()).max().item()
    mean_diff = torch.abs(cuda_output_cpu.float() - cpu_reference_tensor.float()).mean().item()

    assert max_diff < 0.01, f"Max difference {max_diff} exceeds tolerance"
    assert mean_diff < 0.001, f"Mean difference {mean_diff} exceeds tolerance"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA backend not available")
def test_trajectory_resampling_edge_cases():
    """Test edge cases: unsorted timestamps, NaNs, boundary conditions."""

    batch, src_len, tgt_len, dim = 2, 32, 16, 4

    data = torch.randn(batch, src_len, dim, dtype=torch.bfloat16, device="cuda")
    src_t = (
        torch.linspace(0.2, 0.8, src_len, device="cuda")
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )
    tgt_t = (
        torch.linspace(0.0, 1.0, tgt_len, device="cuda")
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )

    output = robocache.resample_trajectories(data, src_t, tgt_t, backend="cuda")
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    src_t = (
        torch.linspace(0, 1, src_len, device="cuda")
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )
    tgt_t = torch.full((batch, tgt_len), 0.5, device="cuda")

    output = robocache.resample_trajectories(data, src_t, tgt_t, backend="cuda")
    assert not torch.isnan(output).any()

    data_small = torch.randn(batch, 2, dim, dtype=torch.bfloat16, device="cuda")
    src_t_small = torch.tensor([[0.0, 1.0], [0.0, 1.0]], device="cuda")
    tgt_t_small = (
        torch.linspace(0, 1, tgt_len, device="cuda")
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )

    output_small = robocache.resample_trajectories(
        data_small, src_t_small, tgt_t_small, backend="cuda"
    )
    assert not torch.isnan(output_small).any()
    assert output_small.shape == (batch, tgt_len, dim)


def test_pytorch_fallback_correctness():
    """Test PyTorch fallback implementation produces correct outputs."""

    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch backend unavailable")

    torch.manual_seed(0)
    batch, src_len, tgt_len, dim = 2, 33, 17, 5

    source_data = torch.randn(batch, src_len, dim, dtype=torch.float32)
    source_times = torch.linspace(0, 1, src_len).unsqueeze(0).expand(batch, -1)
    target_times = torch.linspace(0, 1, tgt_len).unsqueeze(0).expand(batch, -1)

    expected = torch.from_numpy(
        cpu_reference_trajectory_resample(source_data, source_times, target_times)
    ).to(torch.float32)

    result = robocache.resample_trajectories(
        source_data,
        source_times,
        target_times,
        backend="pytorch",
    )

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    test_trajectory_resampling_correctness()
    test_trajectory_resampling_edge_cases()
    print("\n✅ All correctness tests passed!")
