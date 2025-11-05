#!/usr/bin/env bash
# Execute complete build on GPU
set -e

cd /workspace
rm -rf robocache
tar xzf robocache_final.tar.gz
cd robocache
chmod +x gpu_build_and_test.sh remote_build.sh

echo "Starting RoboCache build with CUTLASS v4.3.0 FetchContent..."
./gpu_build_and_test.sh

