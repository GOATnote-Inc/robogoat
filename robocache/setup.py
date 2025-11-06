#!/usr/bin/env python3
"""
RoboCache: GPU-Accelerated Data Engine for Robot Foundation Models
Setup script for building CUDA extensions
"""
import os
import sys
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Read version
version = "1.0.0"

# Read README
long_description = (Path(__file__).parent / "README.md").read_text()

# CUDA extension
cuda_ext = None
if "--no-cuda" not in sys.argv:
    try:
        cuda_ext = CUDAExtension(
            name='robocache._cuda_ops',
            sources=[
                'csrc/cpp/resample_ops.cpp',
                'csrc/cuda/resample_kernel.cu',
            ],
            include_dirs=[
                'csrc/cpp',
                'csrc/cuda',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '-Xcompiler', '-fPIC',
                    # SM architectures
                    '-gencode', 'arch=compute_80,code=sm_80',  # A100
                    '-gencode', 'arch=compute_90,code=sm_90',  # H100
                ]
            }
        )
    except Exception as e:
        print(f"WARNING: CUDA extension build failed: {e}")
        print("Installing without CUDA support. Use --no-cuda to suppress this warning.")
        cuda_ext = None
else:
    sys.argv.remove("--no-cuda")
    print("Building without CUDA support (--no-cuda specified)")

# Extensions list
ext_modules = [cuda_ext] if cuda_ext else []

setup(
    name="robocache",
    version=version,
    author="GOATnote Inc",
    author_email="b@thegoatnote.com",
    description="GPU-Accelerated Data Engine for Robot Foundation Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GOATnote-Inc/robogoat",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    } if ext_modules else {},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-benchmark>=4.0.0',
            'black>=23.0.0',
            'isort>=5.12.0',
            'mypy>=1.0.0',
        ],
        'profile': [
            'nvidia-pyindex',
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
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
    keywords="cuda gpu robotics foundation-models dataloader pytorch h100 a100",
    project_urls={
        "Documentation": "https://github.com/GOATnote-Inc/robogoat/tree/main/robocache/docs",
        "Source": "https://github.com/GOATnote-Inc/robogoat",
        "Tracker": "https://github.com/GOATnote-Inc/robogoat/issues",
    },
)
