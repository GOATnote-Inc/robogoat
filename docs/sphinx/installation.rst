Installation
============

Prerequisites
-------------

**Hardware:**

* NVIDIA GPU with Compute Capability ≥ 8.0 (Ampere or newer)
* Recommended: A100 (sm_80) or H100 (sm_90)
* Optional: CPU-only mode for development

**Software:**

* Python 3.8 or newer
* CUDA 12.1 or 13.0
* PyTorch 2.0 or newer
* NVIDIA driver ≥ 520.xx

Quick Install
-------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone repository
   git clone https://github.com/GOATnote-Inc/robogoat.git
   cd robogoat/robocache

   # Install PyTorch with CUDA
   pip install torch --index-url https://download.pytorch.org/whl/cu121

   # Build and install
   pip install -e .

   # Verify installation
   python -c "import robocache; robocache.self_test()"

From PyPI (Coming Soon)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install robocache

Build Options
-------------

Custom CUDA Architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Build for specific GPU architectures
   export TORCH_CUDA_ARCH_LIST="8.0;9.0"  # A100 + H100
   pip install -e .

Debug Build
~~~~~~~~~~~

.. code-block:: bash

   # Build with debug symbols
   python setup.py build_ext --inplace --debug
   pip install -e .

CMake Build
~~~~~~~~~~~

.. code-block:: bash

   # Use CMake directly
   cd robocache/cpp
   mkdir build && cd build
   cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;90"
   make -j$(nproc)

Verification
------------

.. code-block:: python

   import robocache
   import torch

   # Check version
   print(f"RoboCache: {robocache.__version__}")

   # Check CUDA availability
   print(f"CUDA available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name(0)}")

   # Run self-test
   robocache.self_test()

Environment Scripts
-------------------

.. code-block:: bash

   # Verify environment
   cd robocache
   bash scripts/verify_env.sh

   # Build wheel
   bash scripts/build_wheel.sh

Troubleshooting
---------------

CUDA Not Found
~~~~~~~~~~~~~~

.. code-block:: bash

   # Ensure nvcc is in PATH
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

   # Verify
   nvcc --version

PyTorch CUDA Mismatch
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install matching PyTorch version
   # For CUDA 12.1:
   pip install torch --index-url https://download.pytorch.org/whl/cu121

   # For CUDA 13.0:
   pip install torch --index-url https://download.pytorch.org/whl/nightly/cu130

Build Failures
~~~~~~~~~~~~~~

.. code-block:: bash

   # Clean build artifacts
   cd robocache
   rm -rf build dist *.egg-info
   python setup.py clean --all

   # Rebuild
   pip install -e . -v

CPU-Only Installation
---------------------

.. code-block:: bash

   # Install without CUDA
   pip install torch  # CPU-only PyTorch
   pip install -e .

   # RoboCache will use CPU fallbacks automatically
   python -c "import robocache; print('CPU mode:', not robocache._cuda_available)"

Docker
------

.. code-block:: bash

   # Build Docker image
   docker build -t robocache:latest -f Dockerfile.runtime .

   # Run container
   docker run --gpus all -it robocache:latest

   # Test inside container
   python -c "import robocache; robocache.self_test()"

Next Steps
----------

* :doc:`quickstart` - Run your first example
* :doc:`examples` - See real-world use cases
* :doc:`guides/tuning` - Optimize for your hardware

