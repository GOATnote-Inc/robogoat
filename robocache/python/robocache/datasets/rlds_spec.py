"""
RLDS (Robotics Language Dataset Specification) data structures

Follows: https://github.com/google-research/rlds
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import torch
import numpy as np


@dataclass
class RLDSStep:
    """Single step in an RLDS episode"""
    
    # Observations (multimodal)
    observation: Dict[str, torch.Tensor]  # e.g., {'rgb': ..., 'proprio': ..., 'depth': ...}
    
    # Action taken
    action: torch.Tensor  # Shape: (action_dim,)
    
    # Reward signal
    reward: float
    
    # Terminal indicator
    is_terminal: bool
    
    # Language instruction (optional)
    language_instruction: Optional[str] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    # Timestamp (milliseconds)
    timestamp_ms: Optional[float] = None


@dataclass
class RLDSEpisode:
    """Complete episode in RLDS format"""
    
    # List of steps
    steps: List[RLDSStep]
    
    # Episode-level metadata
    episode_id: str
    dataset_name: str
    success: bool
    
    # Optional episode-level data
    language_instruction: Optional[str] = None
    scene_description: Optional[str] = None
    
    def __len__(self) -> int:
        return len(self.steps)
    
    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        """
        Convert episode to batched tensors for efficient training.
        
        Returns:
            Dict with keys:
                - observations: Dict[str, Tensor] (T, ...)
                - actions: Tensor (T, action_dim)
                - rewards: Tensor (T,)
                - terminals: Tensor (T,)
        """
        # Stack observations across timesteps
        obs_dict = {}
        first_obs = self.steps[0].observation
        
        for key in first_obs.keys():
            obs_list = [step.observation[key] for step in self.steps]
            obs_dict[key] = torch.stack(obs_list, dim=0)
        
        # Stack actions, rewards, terminals
        actions = torch.stack([step.action for step in self.steps], dim=0)
        rewards = torch.tensor([step.reward for step in self.steps], dtype=torch.float32)
        terminals = torch.tensor([step.is_terminal for step in self.steps], dtype=torch.bool)
        
        return {
            'observations': obs_dict,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals,
            'episode_id': self.episode_id,
            'success': self.success,
        }

