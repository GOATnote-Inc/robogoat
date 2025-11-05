"""
PyTorch Native Backend Implementation

Provides CPU/GPU fallback implementations using native PyTorch operations.
Performance: 20-70x slower than CUDA, but works without custom extensions.

Use cases:
- Development/testing without CUDA
- CPU-only environments
- Compatibility fallback
"""

import torch
from typing import Optional


class PyTorchBackend:
    """PyTorch native implementation of RoboCache operations"""
    
    @staticmethod
    def resample_trajectories(
        source_data: torch.Tensor,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
        backend: str = "pytorch"
    ) -> torch.Tensor:
        """
        Resample trajectories using PyTorch native operations.
        
        Implementation: Binary search (searchsorted) + linear interpolation (lerp)
        
        Args:
            source_data: [batch, source_len, action_dim] - Input trajectories
            source_times: [batch, source_len] - Source timestamps
            target_times: [batch, target_len] - Target timestamps
            backend: Ignored (for API compatibility)
        
        Returns:
            resampled: [batch, target_len, action_dim] - Resampled trajectories
        
        Performance:
            - CPU: ~20-30x slower than CUDA
            - GPU: ~50-70x slower than CUDA (memory transfer overhead)
        
        Note:
            This is a compatibility fallback. For production use on H100,
            build the CUDA extension for 20-70x speedup.
        """
        # Validate inputs
        if source_data.dim() != 3:
            raise ValueError(
                f"source_data must be 3D [batch, source_len, action_dim], "
                f"got {source_data.dim()}D with shape {source_data.shape}"
            )
        
        if source_times.dim() != 2:
            raise ValueError(
                f"source_times must be 2D [batch, source_len], "
                f"got {source_times.dim()}D with shape {source_times.shape}"
            )
        
        if target_times.dim() != 2:
            raise ValueError(
                f"target_times must be 2D [batch, target_len], "
                f"got {target_times.dim()}D with shape {target_times.shape}"
            )
        
        batch_size, source_len, action_dim = source_data.shape
        target_len = target_times.shape[1]
        
        if source_times.shape[0] != batch_size:
            raise ValueError(
                f"source_times batch size {source_times.shape[0]} "
                f"doesn't match source_data batch size {batch_size}"
            )
        
        if target_times.shape[0] != batch_size:
            raise ValueError(
                f"target_times batch size {target_times.shape[0]} "
                f"doesn't match source_data batch size {batch_size}"
            )
        
        if source_times.shape[1] != source_len:
            raise ValueError(
                f"source_times length {source_times.shape[1]} "
                f"doesn't match source_data length {source_len}"
            )
        
        # Allocate output on same device as input
        device = source_data.device
        dtype = source_data.dtype
        resampled = torch.zeros(
            (batch_size, target_len, action_dim),
            dtype=dtype,
            device=device
        )
        
        # Process each batch independently (PyTorch searchsorted is per-batch)
        for b in range(batch_size):
            src_t = source_times[b]  # [source_len]
            tgt_t = target_times[b]  # [target_len]
            src_d = source_data[b]   # [source_len, action_dim]
            
            # Binary search: Find indices where target_times would be inserted
            # searchsorted returns indices i such that src_t[i-1] <= tgt_t < src_t[i]
            indices = torch.searchsorted(src_t, tgt_t, right=False)  # [target_len]
            
            # Clamp indices to valid range [1, source_len-1]
            # This ensures we can always interpolate between i-1 and i
            indices = torch.clamp(indices, 1, source_len - 1)
            
            # Get left and right indices for interpolation
            left_idx = indices - 1  # [target_len]
            right_idx = indices     # [target_len]
            
            # Get timestamps for interpolation
            left_time = src_t[left_idx]   # [target_len]
            right_time = src_t[right_idx] # [target_len]
            
            # Compute interpolation weights
            # w = (tgt_t - left_time) / (right_time - left_time)
            denom = right_time - left_time
            # Handle division by zero (identical timestamps)
            denom = torch.where(
                torch.abs(denom) < 1e-8,
                torch.ones_like(denom),
                denom
            )
            weight = (tgt_t - left_time) / denom  # [target_len]
            weight = torch.clamp(weight, 0.0, 1.0)
            
            # Get data values for interpolation
            left_data = src_d[left_idx]   # [target_len, action_dim]
            right_data = src_d[right_idx] # [target_len, action_dim]
            
            # Linear interpolation: lerp(left, right, w) = left + w * (right - left)
            weight_expanded = weight.unsqueeze(1)  # [target_len, 1]
            resampled[b] = left_data + weight_expanded * (right_data - left_data)
        
        return resampled
    
    @staticmethod
    def fuse_multimodal(
        primary_data: torch.Tensor,
        primary_times: torch.Tensor,
        secondary_data: torch.Tensor,
        secondary_times: torch.Tensor,
        backend: str = "pytorch"
    ) -> torch.Tensor:
        """
        Fuse multimodal sensor data with temporal alignment.
        
        Implementation: Align secondary data to primary timestamps using interpolation
        
        Args:
            primary_data: [batch, primary_len, primary_dim]
            primary_times: [batch, primary_len]
            secondary_data: [batch, secondary_len, secondary_dim]
            secondary_times: [batch, secondary_len]
            backend: Ignored (for API compatibility)
        
        Returns:
            fused: [batch, primary_len, primary_dim + secondary_dim]
        """
        # Resample secondary data to match primary timestamps
        aligned_secondary = PyTorchBackend.resample_trajectories(
            secondary_data,
            secondary_times,
            primary_times,
            backend=backend
        )
        
        # Concatenate along feature dimension
        fused = torch.cat([primary_data, aligned_secondary], dim=2)
        
        return fused
    
    @staticmethod
    def voxelize_occupancy(
        points: torch.Tensor,
        grid_size: torch.Tensor,
        voxel_size: float,
        origin: torch.Tensor,
        backend: str = "pytorch"
    ) -> torch.Tensor:
        """
        Voxelize point cloud into binary occupancy grid.
        
        Implementation: Naive CPU/GPU voxelization (very slow for large grids)
        
        Args:
            points: [batch, num_points, 3] - XYZ coordinates
            grid_size: [3] - Depth, Height, Width
            voxel_size: Size of each voxel (meters)
            origin: [3] - Origin of voxel grid (XYZ)
            backend: Ignored (for API compatibility)
        
        Returns:
            voxel_grid: [batch, depth, height, width] - Binary occupancy
        
        Warning:
            This is EXTREMELY slow compared to CUDA (500-1000x slower).
            Only use for testing/development on small grids.
        """
        batch_size, num_points, _ = points.shape
        depth, height, width = grid_size[0].item(), grid_size[1].item(), grid_size[2].item()
        
        device = points.device
        voxel_grid = torch.zeros(
            (batch_size, depth, height, width),
            dtype=torch.float32,
            device=device
        )
        
        # Process each batch
        for b in range(batch_size):
            batch_points = points[b]  # [num_points, 3]
            
            # Convert points to voxel indices
            voxel_indices = torch.floor((batch_points - origin) / voxel_size).long()
            
            # Filter out-of-bounds points
            valid_mask = (
                (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < depth) &
                (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < height) &
                (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < width)
            )
            valid_indices = voxel_indices[valid_mask]
            
            # Mark occupied voxels (naive scatter - will be slow)
            if len(valid_indices) > 0:
                voxel_grid[
                    b,
                    valid_indices[:, 0],
                    valid_indices[:, 1],
                    valid_indices[:, 2]
                ] = 1.0
        
        return voxel_grid

