.PHONY: help bench profile test install clean artifacts

# Default target
help:
	@echo "RoboCache Build & Benchmark Commands"
	@echo ""
	@echo "Quick Start:"
	@echo "  make install         Install RoboCache in editable mode"
	@echo "  make bench           Run all benchmarks (5 seeds × 100 repeats)"
	@echo "  make profile         Profile all operations with Nsight"
	@echo "  make test            Run correctness + performance tests"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make bench           Full benchmark suite → bench/results/*.{csv,json,html}"
	@echo "  make bench-quick     Quick benchmark (1 seed × 10 repeats)"
	@echo ""
	@echo "Profiling:"
	@echo "  make profile target=trajectory    Profile trajectory resampling"
	@echo "  make profile target=multimodal     Profile multimodal fusion"
	@echo "  make profile target=voxelize       Profile voxelization"
	@echo "  make profile target=train          Profile training loop"
	@echo "  make profile target=all            Profile all operations"
	@echo ""
	@echo "Testing:"
	@echo "  make test            Run all tests (correctness + perf)"
	@echo "  make test-correctness    Correctness tests only"
	@echo "  make test-perf       Performance tests with gates"
	@echo ""
	@echo "Development:"
	@echo "  make clean           Remove build artifacts"
	@echo "  make artifacts       Package profiling artifacts for CI"
	@echo ""

# Installation
install:
	@echo "Installing RoboCache..."
	pip install -e .
	@echo "✅ Installation complete"

# Benchmarking
bench:
	@echo "Running full benchmark suite..."
	@mkdir -p bench/results
	python bench/benchmark_harness.py --seeds 5 --repeats 100
	@echo "✅ Benchmark complete. Results in bench/results/"

bench-quick:
	@echo "Running quick benchmark..."
	@mkdir -p bench/results
	python bench/benchmark_harness.py --seeds 1 --repeats 10
	@echo "✅ Quick benchmark complete"

# Profiling
profile:
	@if [ -z "$(target)" ]; then \
		echo "Usage: make profile target=<trajectory|multimodal|voxelize|train|all>"; \
		exit 1; \
	fi
	@echo "Profiling $(target)..."
	@chmod +x scripts/profile.sh
	./scripts/profile.sh $(target)
	@echo "✅ Profiling complete. Artifacts in artifacts/"

profile-nsys:
	@if [ -z "$(target)" ]; then \
		echo "Usage: make profile-nsys target=<trajectory|multimodal|voxelize|train|all>"; \
		exit 1; \
	fi
	@echo "Profiling $(target) with Nsight Systems..."
	@chmod +x scripts/profile.sh
	./scripts/profile.sh --tool nsys $(target)

profile-ncu:
	@if [ -z "$(target)" ]; then \
		echo "Usage: make profile-ncu target=<trajectory|multimodal|voxelize|train|all>"; \
		exit 1; \
	fi
	@echo "Profiling $(target) with Nsight Compute..."
	@chmod +x scripts/profile.sh
	./scripts/profile.sh --tool ncu $(target)

# Testing
test:
	@echo "Running all tests..."
	pytest -v tests/
	@echo "✅ Tests complete"

test-correctness:
	@echo "Running correctness tests..."
	pytest -v tests/ -m "not perf"
	@echo "✅ Correctness tests passed"

test-perf:
	@echo "Running performance tests with gates..."
	PERF_GUARD_ENFORCE=1 pytest -v tests/perf/
	@echo "✅ Performance tests passed"

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	rm -rf robocache/__pycache__/ tests/__pycache__/ bench/__pycache__/ scripts/__pycache__/
	rm -rf .pytest_cache/ .coverage htmlcov/
	@echo "✅ Clean complete"

# Package artifacts for CI
artifacts:
	@echo "Packaging profiling artifacts..."
	@mkdir -p artifacts/refs
	tar -czf artifacts/refs/nsight_traces_$(shell date +%Y%m%d).tar.gz artifacts/nsys/ artifacts/ncu/
	@echo "✅ Artifacts packaged: artifacts/refs/nsight_traces_$(shell date +%Y%m%d).tar.gz"

# Docker builds
docker-build:
	@echo "Building Docker runtime image..."
	docker build -f docker/Dockerfile.runtime -t robocache/runtime:latest .
	@echo "✅ Docker build complete"

docker-run:
	@echo "Running RoboCache in Docker..."
	docker run --gpus all -it --rm -v $(PWD):/workspace robocache/runtime:latest bash

# CI simulation
ci-local:
	@echo "Simulating CI pipeline locally..."
	@make install
	@make test-correctness
	@make test-perf
	@make bench-quick
	@echo "✅ Local CI simulation complete"

# Development utilities
format:
	@echo "Formatting code..."
	black robocache/ tests/ bench/ scripts/
	isort robocache/ tests/ bench/ scripts/
	@echo "✅ Formatting complete"

lint:
	@echo "Linting code..."
	flake8 robocache/ tests/ bench/ scripts/
	mypy robocache/
	@echo "✅ Linting complete"

# Quick demo
demo:
	@echo "Running quick demo..."
	python -c "import robocache; robocache.self_test()"
	@echo "✅ Demo complete"

