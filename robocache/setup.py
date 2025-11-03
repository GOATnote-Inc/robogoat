#!/usr/bin/env python3
"""
RoboCache Python Package Setup

This setup.py provides basic package metadata. The actual CUDA extension
is built using CMake (see CMakeLists.txt).

Build instructions:
    cd robocache
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)

Install:
    pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
version = {}
with open("python/robocache/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "GPU-accelerated data engine for embodied AI foundation models"

setup(
    name="robocache",
    version=version.get("__version__", "0.1.0"),
    author="RoboCache Team",
    author_email="[email protected]",
    description="GPU-accelerated data processing for robot learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/robocache",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/robocache/issues",
        "Source": "https://github.com/yourusername/robocache",
        "Documentation": "https://robocache.readthedocs.io",
    },
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware :: Symmetric Multi-processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: GPU :: NVIDIA CUDA :: 13.0",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cuda",
            "black>=22.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "examples": [
            "matplotlib>=3.5.0",
            "tqdm>=4.60.0",
            "wandb>=0.13.0",
        ],
    },
    keywords=[
        "robot learning",
        "embodied ai",
        "gpu acceleration",
        "cuda",
        "cutlass",
        "h100",
        "data processing",
        "trajectory resampling",
        "deep learning",
    ],
    license="MIT",
    zip_safe=False,
)
