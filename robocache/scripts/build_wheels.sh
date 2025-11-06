#!/bin/bash
# Build PyPI wheels for RoboCache across CUDA versions

set -e

CUDA_VERSIONS=("118" "121" "130")
PYTHON_VERSIONS=("3.8" "3.9" "3.10" "3.11")

for CUDA_VER in "${CUDA_VERSIONS[@]}"; do
    for PY_VER in "${PYTHON_VERSIONS[@]}"; do
        echo "Building wheel for CUDA ${CUDA_VER}, Python ${PY_VER}"
        
        docker run --rm \
            -v $(pwd):/workspace \
            -e CUDA_VERSION=${CUDA_VER} \
            -e PYTHON_VERSION=${PY_VER} \
            quay.io/pypa/manylinux2014_x86_64 \
            bash -c "
                cd /workspace
                /opt/python/cp${PY_VER//.}-cp${PY_VER//.}/bin/pip install build wheel
                /opt/python/cp${PY_VER//.}-cp${PY_VER//.}/bin/python -m build --wheel
                auditwheel repair dist/*.whl -w dist/
            "
        
        mv dist/*manylinux*.whl dist/robocache-cu${CUDA_VER}-py${PY_VER//.}-linux_x86_64.whl
    done
done

echo "✓ Wheels built successfully"
ls -lh dist/*.whl

