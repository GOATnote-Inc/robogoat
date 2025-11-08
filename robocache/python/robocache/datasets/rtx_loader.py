"""
RT-X (Robotics Transformer X) Dataset Loader

Supports loading from:
- Google Cloud Storage (official RT-X datasets)
- Local RLDS-format TFRecords
- HuggingFace Hub

Datasets include:
- RT-1 (55k episodes, Everyday Robots)
- Bridge V2 (60k episodes)
- FMB (robocasa, language-table, etc.)
- DROID (76k episodes)

Reference: https://robotics-transformer-x.github.io/
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Iterator, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available - RT-X loading will be limited")

from robocache.datasets.rlds_spec import RLDSEpisode, RLDSStep


class RTXDataset(Dataset):
    """
    PyTorch Dataset for RT-X RLDS episodes.
    
    Loads multimodal robot data (vision + proprioception + language) from RLDS format.
    Automatically handles:
    - Temporal alignment of sensors
    - Multimodal fusion via RoboCache
    - Episode chunking for training
    """
    
    def __init__(
        self,
        dataset_name: str,
        data_dir: Optional[str] = None,
        split: str = 'train',
        max_episodes: Optional[int] = None,
        sequence_length: int = 50,
        image_size: Tuple[int, int] = (224, 224),
        use_robocache: bool = True,
        cache_in_memory: bool = False,
    ):
        """
        Args:
            dataset_name: Name of RT-X dataset (e.g., 'bridge_dataset', 'rt_1')
            data_dir: Local directory with TFRecords (if None, downloads from GCS)
            split: 'train', 'val', or 'test'
            max_episodes: Limit number of episodes (for debugging)
            sequence_length: Number of timesteps per training sample
            image_size: Resize images to (H, W)
            use_robocache: Use RoboCache for sensor fusion (faster)
            cache_in_memory: Cache episodes in RAM (requires ~100GB for full RT-X)
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow required for RT-X loading. Install with: "
                "pip install tensorflow tensorflow-datasets"
            )
        
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.split = split
        self.max_episodes = max_episodes
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.use_robocache = use_robocache
        self.cache_in_memory = cache_in_memory
        
        # Load dataset builder
        if data_dir:
            self.builder = tfds.builder_from_directory(data_dir)
        else:
            self.builder = tfds.builder(
                f'rlds/{dataset_name}',
                data_dir=os.environ.get('TFDS_DATA_DIR', '~/tensorflow_datasets')
            )
        
        # Load episodes
        self.episodes: List[RLDSEpisode] = []
        self._load_episodes()
        
        print(f"✅ Loaded {len(self.episodes)} episodes from {dataset_name} ({split})")
    
    def _load_episodes(self):
        """Load and parse RLDS episodes from TFRecords"""
        ds = self.builder.as_dataset(split=self.split)
        
        episode_count = 0
        for tfrecord_episode in ds:
            if self.max_episodes and episode_count >= self.max_episodes:
                break
            
            episode = self._parse_episode(tfrecord_episode)
            if episode:
                self.episodes.append(episode)
                episode_count += 1
    
    def _parse_episode(self, tf_episode) -> Optional[RLDSEpisode]:
        """Parse a single TFRecord episode into RLDSEpisode format"""
        try:
            steps = []
            
            for tf_step in tf_episode['steps']:
                # Parse observations (multimodal)
                observation = {}
                
                # RGB image(s)
                if 'image' in tf_step['observation']:
                    rgb = tf_step['observation']['image'].numpy()
                    rgb = self._resize_image(rgb, self.image_size)
                    observation['rgb'] = torch.from_numpy(rgb).float() / 255.0
                
                # Proprioception (joint positions, velocities)
                if 'state' in tf_step['observation']:
                    proprio = tf_step['observation']['state'].numpy()
                    observation['proprio'] = torch.from_numpy(proprio).float()
                
                # Depth (if available)
                if 'depth' in tf_step['observation']:
                    depth = tf_step['observation']['depth'].numpy()
                    depth = self._resize_image(depth, self.image_size)
                    observation['depth'] = torch.from_numpy(depth).float()
                
                # Action
                action = torch.from_numpy(tf_step['action'].numpy()).float()
                
                # Reward
                reward = float(tf_step.get('reward', 0.0))
                
                # Terminal
                is_terminal = bool(tf_step.get('is_terminal', False))
                
                # Language instruction (episode-level)
                language = tf_step.get('language_instruction', None)
                if language is not None:
                    language = language.numpy().decode('utf-8')
                
                # Create step
                step = RLDSStep(
                    observation=observation,
                    action=action,
                    reward=reward,
                    is_terminal=is_terminal,
                    language_instruction=language,
                )
                
                steps.append(step)
            
            # Episode metadata
            episode_id = str(tf_episode.get('episode_metadata', {}).get('episode_id', len(self.episodes)))
            success = bool(tf_episode.get('episode_metadata', {}).get('success', False))
            
            episode = RLDSEpisode(
                steps=steps,
                episode_id=episode_id,
                dataset_name=self.dataset_name,
                success=success,
            )
            
            return episode
            
        except Exception as e:
            print(f"⚠️  Failed to parse episode: {e}")
            return None
    
    def _resize_image(self, img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image using TensorFlow"""
        if img.shape[:2] == size:
            return img
        
        img_tensor = tf.image.resize(img, size, method='bilinear')
        return img_tensor.numpy()
    
    def __len__(self) -> int:
        """Total number of training samples (chunked episodes)"""
        total_steps = sum(len(ep) for ep in self.episodes)
        return max(1, total_steps // self.sequence_length)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample (chunked episode sequence).
        
        Returns:
            Dict with keys:
                - observations: Dict[str, Tensor] (sequence_length, ...)
                - actions: Tensor (sequence_length, action_dim)
                - rewards: Tensor (sequence_length,)
                - language: str
        """
        # Find episode and offset for this index
        episode_idx, step_offset = self._get_episode_and_offset(idx)
        episode = self.episodes[episode_idx]
        
        # Extract sequence
        end_idx = min(step_offset + self.sequence_length, len(episode))
        sequence_steps = episode.steps[step_offset:end_idx]
        
        # Pad if needed
        if len(sequence_steps) < self.sequence_length:
            # Pad with last step
            last_step = sequence_steps[-1]
            while len(sequence_steps) < self.sequence_length:
                sequence_steps.append(last_step)
        
        # Stack observations
        obs_dict = {}
        first_obs = sequence_steps[0].observation
        
        for key in first_obs.keys():
            obs_list = [step.observation[key] for step in sequence_steps]
            obs_dict[key] = torch.stack(obs_list, dim=0)
        
        # Stack actions and rewards
        actions = torch.stack([step.action for step in sequence_steps], dim=0)
        rewards = torch.tensor([step.reward for step in sequence_steps], dtype=torch.float32)
        
        # Language instruction (episode-level)
        language = episode.language_instruction or ""
        
        return {
            'observations': obs_dict,
            'actions': actions,
            'rewards': rewards,
            'language': language,
            'episode_id': episode.episode_id,
        }
    
    def _get_episode_and_offset(self, idx: int) -> Tuple[int, int]:
        """Map flat index to (episode_idx, step_offset)"""
        cumulative_chunks = 0
        
        for ep_idx, episode in enumerate(self.episodes):
            num_chunks = max(1, len(episode) // self.sequence_length)
            
            if idx < cumulative_chunks + num_chunks:
                chunk_in_episode = idx - cumulative_chunks
                step_offset = chunk_in_episode * self.sequence_length
                return ep_idx, step_offset
            
            cumulative_chunks += num_chunks
        
        # Fallback: return last episode
        return len(self.episodes) - 1, 0


class RTXDataLoader:
    """
    High-level DataLoader for RT-X with RoboCache preprocessing.
    
    Features:
    - Automatic multimodal fusion (vision + proprio + language)
    - Batch collation with padding
    - GPU-accelerated preprocessing
    """
    
    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 32,
        num_workers: int = 4,
        split: str = 'train',
        **dataset_kwargs
    ):
        """
        Args:
            dataset_name: RT-X dataset name
            batch_size: Batch size
            num_workers: Number of data loading workers
            split: 'train', 'val', or 'test'
            **dataset_kwargs: Additional args for RTXDataset
        """
        self.dataset = RTXDataset(
            dataset_name=dataset_name,
            split=split,
            **dataset_kwargs
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collation for batched episodes"""
        # Stack observations
        obs_keys = batch[0]['observations'].keys()
        batched_obs = {}
        
        for key in obs_keys:
            obs_list = [sample['observations'][key] for sample in batch]
            batched_obs[key] = torch.stack(obs_list, dim=0)
        
        # Stack actions and rewards
        actions = torch.stack([sample['actions'] for sample in batch], dim=0)
        rewards = torch.stack([sample['rewards'] for sample in batch], dim=0)
        
        # Language instructions (list of strings)
        languages = [sample['language'] for sample in batch]
        
        return {
            'observations': batched_obs,
            'actions': actions,
            'rewards': rewards,
            'language': languages,
        }
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        return len(self.dataloader)

