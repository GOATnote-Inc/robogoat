"""
RoboCache Setup Script
Builds Python package with optional CUDA extensions
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read version
version = {}
with open("python/robocache/_version.py") as f:
    exec(f.read(), version)

# Read README
readme = Path("README.md").read_text(encoding="utf-8")

# CUDA availability
cuda_available = os.environ.get("CUDA_HOME") is not None or os.path.exists("/usr/local/cuda")

setup(
    name="robocache",
    version=version["__version__"],
    author="RoboCache Team",
    author_email="b@thegoatnote.com",
    description="GPU-Accelerated Data Engine for Robot Learning",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/robocache",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/robocache/issues",
        "Documentation": "https://github.com/yourusername/robocache#readme",
        "Source Code": "https://github.com/yourusername/robocache",
    },
    packages=find_packages(where="python") + find_packages(where="robocache"),
    package_dir={
        "": "python",
        "robocache.datasets": "robocache/datasets",
    },
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "benchmark": [
            "matplotlib>=3.5.0",
            "pandas>=1.4.0",
            "tqdm>=4.64.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
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
    keywords="robotics gpu cuda deep-learning transformer foundation-models",
    license="Apache-2.0",
    include_package_data=True,
    zip_safe=False,
)
