"""
Temporal Attention Fusion Module

GPU-accelerated temporal attention for multi-timestep sensor fusion.
Combines vision, LiDAR, and proprioception across time with causal masking.

Architecture:
  1. Temporal positional encoding
  2. Self-attention over time (within each modality)
  3. Cross-modal attention (across modalities)
  4. Causal masking for online inference
  5. Residual connections + LayerNorm

Performance Target:
  - <2ms latency for 10 timesteps on H100
  - <1GB memory for 1000 timesteps
  - Linear scaling with timesteps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        return x + self.pe[:, :x.size(1), :]


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking for online inference"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # QKV projections
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask buffer
        self.register_buffer('causal_mask', None)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create or retrieve causal mask"""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            # Create causal mask: upper triangle is -inf
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            self.causal_mask = mask
        
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # QKV projections
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        causal_mask = self._get_causal_mask(seq_len, x.device)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Optional additional mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, d_model)
        output = self.out_proj(attn_output)
        
        return output


class CrossModalAttention(nn.Module):
    """Cross-attention between different modalities"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Query from target modality, Key+Value from source modality
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, tgt_len, d_model) - target modality
            context: (batch, src_len, d_model) - source modality
            mask: Optional attention mask
        Returns:
            (batch, tgt_len, d_model)
        """
        batch_size, tgt_len, d_model = query.shape
        src_len = context.size(1)
        
        # Project query
        q = self.q_proj(query).reshape(batch_size, tgt_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch, num_heads, tgt_len, head_dim)
        
        # Project key and value
        kv = self.kv_proj(context).reshape(batch_size, src_len, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, batch, num_heads, src_len, head_dim)
        k, v = kv[0], kv[1]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, tgt_len, d_model)
        output = self.out_proj(attn_output)
        
        return output


class TemporalFusionBlock(nn.Module):
    """Single block of temporal fusion transformer"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # Self-attention (temporal within modality)
        self.self_attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # Self-attention with residual
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + attn_output)
        
        # Feedforward with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class TemporalMultimodalFusion(nn.Module):
    """
    Complete temporal fusion module for robot sensor data.
    
    Processes multiple sensor modalities across time with attention:
      - Vision (RGB/depth): (batch, T, vision_dim)
      - LiDAR (point clouds): (batch, T, lidar_dim)
      - Proprioception (pose, velocity): (batch, T, proprio_dim)
    
    Outputs fused representation: (batch, T, d_model)
    """
    
    def __init__(
        self,
        vision_dim: int,
        lidar_dim: int,
        proprio_dim: int,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projections (modality-specific)
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.lidar_proj = nn.Linear(lidar_dim, d_model)
        self.proprio_proj = nn.Linear(proprio_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Temporal fusion blocks (per modality)
        self.vision_blocks = nn.ModuleList([
            TemporalFusionBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.lidar_blocks = nn.ModuleList([
            TemporalFusionBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.proprio_blocks = nn.ModuleList([
            TemporalFusionBlock(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Cross-modal attention
        self.cross_attn_vision_to_lidar = CrossModalAttention(d_model, num_heads, dropout)
        self.cross_attn_lidar_to_vision = CrossModalAttention(d_model, num_heads, dropout)
        self.cross_attn_proprio = CrossModalAttention(d_model, num_heads, dropout)
        
        # Final fusion
        self.fusion_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        vision: torch.Tensor,
        lidar: torch.Tensor,
        proprio: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        lidar_mask: Optional[torch.Tensor] = None,
        proprio_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            vision: (batch, T, vision_dim)
            lidar: (batch, T, lidar_dim)
            proprio: (batch, T, proprio_dim)
            *_mask: Optional masks for missing data
        
        Returns:
            fused: (batch, T, d_model)
        """
        # Project to common dimension
        vision_emb = self.vision_proj(vision)
        lidar_emb = self.lidar_proj(lidar)
        proprio_emb = self.proprio_proj(proprio)
        
        # Add positional encoding
        vision_emb = self.pos_encoder(vision_emb)
        lidar_emb = self.pos_encoder(lidar_emb)
        proprio_emb = self.pos_encoder(proprio_emb)
        
        # Temporal self-attention per modality
        for vision_block, lidar_block, proprio_block in zip(
            self.vision_blocks, self.lidar_blocks, self.proprio_blocks
        ):
            vision_emb = vision_block(vision_emb, vision_mask)
            lidar_emb = lidar_block(lidar_emb, lidar_mask)
            proprio_emb = proprio_block(proprio_emb, proprio_mask)
        
        # Cross-modal attention
        vision_attended = self.cross_attn_lidar_to_vision(vision_emb, lidar_emb)
        lidar_attended = self.cross_attn_vision_to_lidar(lidar_emb, vision_emb)
        proprio_attended = self.cross_attn_proprio(proprio_emb, 
                                                     torch.cat([vision_emb, lidar_emb], dim=1))
        
        # Combine all modalities
        fused = torch.cat([vision_attended, lidar_attended, proprio_attended], dim=-1)
        fused = self.fusion_proj(fused)
        
        return fused


# Convenience function
def create_temporal_fusion(
    vision_dim: int = 512,
    lidar_dim: int = 256,
    proprio_dim: int = 14,
    d_model: int = 512,
    device: str = 'cuda'
) -> TemporalMultimodalFusion:
    """Create a temporal fusion module with reasonable defaults"""
    model = TemporalMultimodalFusion(
        vision_dim=vision_dim,
        lidar_dim=lidar_dim,
        proprio_dim=proprio_dim,
        d_model=d_model,
        num_layers=4,
        num_heads=8
    )
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    
    return model

