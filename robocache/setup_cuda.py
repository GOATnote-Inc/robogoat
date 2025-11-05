#!/usr/bin/env python3
"""
RoboCache CUDA Extension Builder
H100-Optimized for CUDA 13.x + CUTLASS 4.2.1

This builds the PyTorch extension with proper ABI compatibility.
CMake builds create ABI mismatches - use this instead.

Usage:
    python3 setup_cuda.py build_ext --inplace
    python3 setup_cuda.py install
"""

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os
import sys
import subprocess

def validate_environment():
    """Validate CUDA 13.x + H100 environment"""
    print("\n" + "="*70)
    print("RoboCache CUDA Extension Build - H100 Optimized")
    print("="*70)
    
    # Check CUDA version
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if 'release' in line:
                version_str = line.split('release')[1].split(',')[0].strip()
                major = int(version_str.split('.')[0])
                
                if major < 13:
                    print(f"❌ CUDA {version_str} detected")
                    print(f"   RoboCache requires CUDA 13.x for H100 optimizations")
                    print(f"   Note: CUDA 13.0+ dropped sm_70 (Volta) support")
                    sys.exit(1)
                
                print(f"✓ CUDA {version_str}")
                break
    except Exception as e:
        print(f"❌ Could not detect CUDA: {e}")
        sys.exit(1)
    
    # Check PyTorch
    if not torch.cuda.is_available():
        print("⚠️  PyTorch CUDA not available")
    else:
        print(f"✓ PyTorch {torch.__version__} with CUDA")
    
    # Find CUTLASS
    cutlass_dir = find_cutlass()
    print(f"✓ CUTLASS at: {cutlass_dir}")
    
    # Check GPU
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True)
        gpu_name = result.stdout.strip().split('\n')[0]
        if 'H100' not in gpu_name:
            print(f"⚠️  GPU: {gpu_name} (optimized for H100)")
        else:
            print(f"✓ GPU: {gpu_name}")
    except:
        pass
    
    print("="*70 + "\n")
    return cutlass_dir

def find_cutlass():
    """Locate CUTLASS installation"""
    # Priority order:
    # 1. CMake FetchContent build
    # 2. Environment variable
    # 3. System default
    
    locations = [
        'build/_deps/cutlass-src',
        os.environ.get('CUTLASS_DIR', '/usr/local/cutlass'),
        '/usr/local/cutlass',
        '/opt/cutlass',
    ]
    
    for loc in locations:
        if os.path.exists(os.path.join(loc, 'include', 'cutlass', 'cutlass.h')):
            return loc
    
    print("❌ CUTLASS not found in:")
    for loc in locations:
        print(f"   - {loc}")
    print("\nRun cmake first to fetch CUTLASS:")
    print("   cd robocache && mkdir -p build && cd build && cmake ..")
    sys.exit(1)

def get_cuda_flags():
    """Get optimal CUDA compiler flags for H100"""
    return [
        '-O3',
        '-use_fast_math',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '-lineinfo',
        '-Xptxas', '-v',
        '-DNDEBUG',
        # H100 specific
        '-gencode', 'arch=compute_90,code=sm_90',
    ]

# Validate environment
cutlass_dir = validate_environment()

# Define extension
ext_modules = [
    CUDAExtension(
        name='robocache_cuda',  # Will create robocache_cuda.so
        sources=[
            'kernels/cutlass/trajectory_resample.cu',
            'kernels/cutlass/trajectory_resample_torch.cu',
        ],
        include_dirs=[
            'kernels/cutlass',
            os.path.join(cutlass_dir, 'include'),
        ],
        extra_compile_args={
            'cxx': ['-O3', '-DNDEBUG', '-fPIC'],
            'nvcc': get_cuda_flags()
        },
        library_dirs=[],
        libraries=[],
    )
]

setup(
    name='robocache-cuda-extension',
    version='0.1.0',
    description='RoboCache CUDA extension (H100-optimized)',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    zip_safe=False,
)

print("\n" + "="*70)
print("✓ Build complete!")
print("Extension location: python/robocache/robocache_cuda.so")
print("="*70 + "\n")

