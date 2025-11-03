#!/usr/bin/env python3
"""
Download and prepare real robot learning data for benchmarking.

This script downloads or generates synthetic robot learning data compatible with
the Open-X Embodiment format, specifically BridgeData V2.

For demonstration purposes, this generates high-quality synthetic data that
matches real robot dataset characteristics. In production, you would connect
to actual dataset sources.
"""

import os
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import argparse


class RobotDatasetDownloader:
    """Download or generate robot learning datasets."""

    def __init__(self, data_dir='./data/robot_learning'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_or_generate(self, num_trajectories=5000, dataset_type='synthetic'):
        """
        Download real data or generate synthetic data for benchmarking.

        Args:
            num_trajectories: Number of robot trajectories to include
            dataset_type: 'synthetic' or 'bridge' (requires tensorflow_datasets)

        Returns:
            Path to the generated HDF5 file
        """
        if dataset_type == 'bridge':
            try:
                return self._download_bridge_data(num_trajectories)
            except Exception as e:
                print(f"⚠️  Failed to download BridgeData: {e}")
                print("Falling back to synthetic data generation...")
                return self._generate_synthetic_data(num_trajectories)
        else:
            return self._generate_synthetic_data(num_trajectories)

    def _download_bridge_data(self, num_trajectories):
        """
        Download BridgeData V2 from Open-X Embodiment.

        Requires: tensorflow_datasets, rlds
        """
        print("=" * 80)
        print("Downloading BridgeData V2 from Open-X Embodiment")
        print("=" * 80)

        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "tensorflow_datasets is required for downloading real data.\n"
                "Install with: pip install tensorflow-datasets rlds"
            )

        # Load dataset
        builder = tfds.builder('bridge_dataset', data_dir=str(self.data_dir / 'tfds'))
        builder.download_and_prepare()

        ds = builder.as_dataset(split=f'train[:{num_trajectories}]')

        print(f"\nProcessing {num_trajectories} trajectories...")
        return self._process_tensorflow_dataset(ds, num_trajectories, 'bridge_real.h5')

    def _process_tensorflow_dataset(self, dataset, num_trajectories, output_filename):
        """Convert TensorFlow dataset to efficient HDF5 format."""

        output_file = self.data_dir / output_filename

        with h5py.File(output_file, 'w') as f:
            # Pre-allocate arrays
            max_traj_len = 200

            rgb_images = f.create_dataset(
                'rgb_images',
                shape=(num_trajectories, max_traj_len, 128, 128, 3),
                dtype=np.uint8,
                compression='gzip',
                compression_opts=4
            )

            actions = f.create_dataset(
                'actions',
                shape=(num_trajectories, max_traj_len, 7),  # 7-DOF robot
                dtype=np.float32
            )

            proprio = f.create_dataset(
                'proprioception',
                shape=(num_trajectories, max_traj_len, 14),  # pos + vel
                dtype=np.float32
            )

            timestamps = f.create_dataset(
                'timestamps',
                shape=(num_trajectories, max_traj_len),
                dtype=np.float32
            )

            lengths = f.create_dataset(
                'lengths',
                shape=(num_trajectories,),
                dtype=np.int32
            )

            # Process each trajectory
            for idx, episode in enumerate(tqdm(dataset.take(num_trajectories))):
                steps = episode['steps']
                traj_len = min(len(steps['action']), max_traj_len)

                # Extract and store data
                rgb = np.array([s['observation']['image'].numpy() for s in steps[:traj_len]])
                act = np.array([s['action'].numpy() for s in steps[:traj_len]])
                prop = np.array([s['observation']['state'].numpy() for s in steps[:traj_len]])

                rgb_images[idx, :traj_len] = rgb
                actions[idx, :traj_len] = act
                proprio[idx, :traj_len, :7] = prop[:, :7]
                proprio[idx, :traj_len, 7:] = prop[:, 7:14] if prop.shape[1] >= 14 else 0
                timestamps[idx, :traj_len] = np.arange(traj_len) * 0.033  # ~30Hz
                lengths[idx] = traj_len

        print(f"\n✅ Dataset saved to {output_file}")
        print(f"   Size: {output_file.stat().st_size / 1e9:.2f} GB")
        return output_file

    def _generate_synthetic_data(self, num_trajectories):
        """
        Generate high-quality synthetic robot data for benchmarking.

        This data has realistic characteristics:
        - Variable trajectory lengths (like real robots)
        - Variable sampling frequencies (different robot types)
        - Smooth, physically plausible motions
        - Heterogeneous data structure
        """

        print("=" * 80)
        print("Generating Synthetic Robot Learning Data")
        print("=" * 80)
        print(f"\nGenerating {num_trajectories} trajectories with realistic characteristics...")

        output_file = self.data_dir / 'robot_synthetic.h5'

        np.random.seed(42)

        with h5py.File(output_file, 'w') as f:
            max_traj_len = 200

            # Create datasets
            rgb_images = f.create_dataset(
                'rgb_images',
                shape=(num_trajectories, max_traj_len, 128, 128, 3),
                dtype=np.uint8,
                compression='gzip',
                compression_opts=4
            )

            actions = f.create_dataset(
                'actions',
                shape=(num_trajectories, max_traj_len, 7),  # 7-DOF
                dtype=np.float32
            )

            proprio = f.create_dataset(
                'proprioception',
                shape=(num_trajectories, max_traj_len, 14),  # pos + vel
                dtype=np.float32
            )

            timestamps = f.create_dataset(
                'timestamps',
                shape=(num_trajectories, max_traj_len),
                dtype=np.float32
            )

            lengths = f.create_dataset(
                'lengths',
                shape=(num_trajectories,),
                dtype=np.int32
            )

            # Different robot types have different sampling rates
            # Mimics real heterogeneous datasets like Open-X Embodiment
            robot_frequencies = [30, 50, 100, 125, 333]  # Hz
            robot_types = ['franka', 'ur5', 'kuka', 'kinova', 'xarm']

            print(f"\nSimulating {len(robot_types)} different robot types:")
            for rtype, freq in zip(robot_types, robot_frequencies):
                print(f"  - {rtype}: {freq} Hz")

            # Generate trajectories
            for idx in tqdm(range(num_trajectories), desc="Generating trajectories"):
                # Random robot type (different frequencies)
                robot_idx = idx % len(robot_frequencies)
                frequency = robot_frequencies[robot_idx]

                # Variable trajectory length (1-4 seconds)
                duration = np.random.uniform(1.0, 4.0)
                traj_len = min(int(duration * frequency), max_traj_len)

                # Time array
                timestamps[idx, :traj_len] = np.linspace(0, duration, traj_len)

                # Generate smooth, realistic robot actions
                # Use sum of sinusoids for smooth motion
                t = timestamps[idx, :traj_len]
                for dof in range(7):
                    # Multiple frequency components for realistic motion
                    freq1 = np.random.uniform(0.3, 1.0)
                    freq2 = np.random.uniform(1.0, 2.0)
                    phase1 = np.random.uniform(0, 2*np.pi)
                    phase2 = np.random.uniform(0, 2*np.pi)
                    amp1 = np.random.uniform(0.3, 0.8)
                    amp2 = np.random.uniform(0.1, 0.3)

                    actions[idx, :traj_len, dof] = (
                        amp1 * np.sin(2*np.pi*freq1*t + phase1) +
                        amp2 * np.sin(2*np.pi*freq2*t + phase2)
                    )

                # Proprioception follows actions with small noise
                proprio[idx, :traj_len, :7] = actions[idx, :traj_len] + \
                    np.random.randn(traj_len, 7) * 0.01

                # Velocities (numerical derivative)
                proprio[idx, :traj_len, 7:] = np.gradient(
                    proprio[idx, :traj_len, :7], axis=0
                ) * frequency

                # Synthetic images (random for now - in real use case would be actual camera data)
                # For benchmarking data loading, the content doesn't matter
                rgb_images[idx, :traj_len] = np.random.randint(
                    0, 255, (traj_len, 128, 128, 3), dtype=np.uint8
                )

                lengths[idx] = traj_len

            # Store metadata
            f.attrs['num_trajectories'] = num_trajectories
            f.attrs['max_traj_len'] = max_traj_len
            f.attrs['action_dim'] = 7
            f.attrs['robot_types'] = ','.join(robot_types)
            f.attrs['frequencies'] = ','.join(map(str, robot_frequencies))
            f.attrs['description'] = 'Synthetic robot learning data for benchmarking'

        file_size_mb = output_file.stat().st_size / 1e6
        print(f"\n✅ Dataset generated: {output_file}")
        print(f"   Size: {file_size_mb:.1f} MB")
        print(f"   Trajectories: {num_trajectories}")
        print(f"   Heterogeneous frequencies: {robot_frequencies} Hz")

        # Print statistics
        with h5py.File(output_file, 'r') as f:
            lengths_arr = f['lengths'][:]
            print(f"\nTrajectory length statistics:")
            print(f"   Mean: {lengths_arr.mean():.1f} frames")
            print(f"   Min: {lengths_arr.min()} frames")
            print(f"   Max: {lengths_arr.max()} frames")
            print(f"   Std: {lengths_arr.std():.1f} frames")

        return output_file


def main():
    parser = argparse.ArgumentParser(
        description='Download or generate robot learning data for benchmarking'
    )
    parser.add_argument(
        '--num-trajectories',
        type=int,
        default=5000,
        help='Number of trajectories to download/generate (default: 5000)'
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        default='synthetic',
        choices=['synthetic', 'bridge'],
        help='Dataset type: synthetic or bridge (default: synthetic)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/robot_learning',
        help='Directory to store dataset (default: ./data/robot_learning)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("RoboCache Benchmark Data Preparation")
    print("=" * 80 + "\n")

    downloader = RobotDatasetDownloader(data_dir=args.data_dir)
    data_file = downloader.download_or_generate(
        num_trajectories=args.num_trajectories,
        dataset_type=args.dataset_type
    )

    print("\n" + "=" * 80)
    print("✅ Data preparation complete!")
    print("=" * 80)
    print(f"\nDataset location: {data_file}")
    print("\nNext steps:")
    print("  1. Run baseline benchmark:    python baseline_dataloader.py")
    print("  2. Run RoboCache benchmark:   python robocache_dataloader.py")
    print("  3. Run full comparison:       python benchmark_dataloading.py")
    print()


if __name__ == '__main__':
    main()
