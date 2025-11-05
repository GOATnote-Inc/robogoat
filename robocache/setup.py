"""
RoboCache setup script for building and distributing the package.

This setup.py supports:
- Pure Python wheel (PyTorch backend only)
- Source distribution with CUDA kernels
- Development installation with editable mode
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from _version.py
version_dict = {}
with open(this_directory / "python" / "robocache" / "_version.py") as f:
    exec(f.read(), version_dict)

setup(
    name="robocache",
    version=version_dict["__version__"],
    author="RoboCache Team",
    author_email="team@robocache.ai",
    description="GPU-accelerated data engine for robot learning, optimized for NVIDIA H100",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robocache/robocache",
    project_urls={
        "Bug Tracker": "https://github.com/robocache/robocache/issues",
        "Documentation": "https://robocache.readthedocs.io",
        "Source Code": "https://github.com/robocache/robocache",
    },
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    license="Apache-2.0",
    keywords=[
        "robotics", "machine-learning", "cuda", "gpu", "h100", "data-processing",
        "trajectory", "multimodal", "voxelization", "embodied-ai"
    ],
)
