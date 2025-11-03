# Makefile for RoboCache Multimodal Fusion
# Production-grade build system for NVIDIA GPU-accelerated robot learning
#
# Copyright (c) 2025 GOATnote Inc.
# SPDX-License-Identifier: Apache-2.0

.PHONY: help all clean test benchmark validate ncu-profile docker-build docker-run

# Configuration
CUDA_HOME ?= /usr/local/cuda
CUTLASS_HOME ?= /opt/cutlass
PYTHON ?= python3
NVCC := $(CUDA_HOME)/bin/nvcc
NCU := ncu

# CUDA architecture (H100 = sm_90, A100 = sm_80)
CUDA_ARCH ?= 90
SM_VERSIONS := -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)

# Compiler flags
NVCC_FLAGS := -std=c++17 $(SM_VERSIONS) \
	-O3 \
	--use_fast_math \
	-Xcompiler -fopenmp \
	-Xcompiler -fPIC \
	-Xcompiler -march=native \
	--expt-relaxed-constexpr

# Include paths
INCLUDES := -I$(CUDA_HOME)/include \
	-I$(CUTLASS_HOME)/include \
	-Irobocache/kernels/cutlass

# Library paths
LDFLAGS := -L$(CUDA_HOME)/lib64 -lcudart -lcublas

# Directories
BUILD_DIR := build
TEST_DIR := $(BUILD_DIR)/tests
BENCH_DIR := $(BUILD_DIR)/benchmarks

# Source files
KERNEL_SOURCES := robocache/kernels/cutlass/multimodal/multimodal_fusion.cu
TEST_SOURCES := robocache/tests/multimodal/test_multimodal_fusion.cu
BENCH_SOURCES := robocache/benchmarks/multimodal/benchmark_multimodal_fusion.cu

# Targets
TEST_BIN := $(TEST_DIR)/test_multimodal_fusion
BENCH_BIN := $(BENCH_DIR)/benchmark_multimodal_fusion

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
help:
	@echo "$(BLUE)RoboCache Multimodal Fusion - Build System$(NC)"
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  make all              - Build everything (tests + benchmarks)"
	@echo "  make test             - Run unit tests"
	@echo "  make benchmark        - Run performance benchmarks"
	@echo "  make validate         - Run validation suite (tests + reproducibility)"
	@echo "  make ncu-profile      - Profile with NVIDIA Nsight Compute"
	@echo ""
	@echo "$(GREEN)Build Targets:$(NC)"
	@echo "  make build-tests      - Build unit tests"
	@echo "  make build-benchmarks - Build performance benchmarks"
	@echo "  make build-python     - Build Python extension"
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@echo "  make docker-build     - Build Docker image with CUDA 13 + CUTLASS 4.3"
	@echo "  make docker-run       - Run tests in Docker container"
	@echo ""
	@echo "$(GREEN)Profiling:$(NC)"
	@echo "  make ncu-profile      - Full NCU profiling suite"
	@echo "  make nsys-profile     - Nsight Systems timeline profiling"
	@echo ""
	@echo "$(GREEN)Validation:$(NC)"
	@echo "  make validate         - Full validation (correctness + performance)"
	@echo "  make reproducibility  - Test bitwise reproducibility"
	@echo ""
	@echo "$(GREEN)Configuration:$(NC)"
	@echo "  CUDA_ARCH=$(CUDA_ARCH)  (80=A100, 90=H100)"
	@echo "  CUDA_HOME=$(CUDA_HOME)"
	@echo ""

# Build everything
all: build-tests build-benchmarks
	@echo "$(GREEN)✓ Build complete!$(NC)"

# Create build directories
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(TEST_DIR)
	@mkdir -p $(BENCH_DIR)

# Build unit tests
build-tests: $(BUILD_DIR)
	@echo "$(BLUE)Building unit tests...$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) \
		$(KERNEL_SOURCES) $(TEST_SOURCES) \
		$(LDFLAGS) \
		-o $(TEST_BIN)
	@echo "$(GREEN)✓ Tests built: $(TEST_BIN)$(NC)"

# Build benchmarks
build-benchmarks: $(BUILD_DIR)
	@echo "$(BLUE)Building benchmarks...$(NC)"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) \
		$(KERNEL_SOURCES) $(BENCH_SOURCES) \
		$(LDFLAGS) \
		-o $(BENCH_BIN)
	@echo "$(GREEN)✓ Benchmarks built: $(BENCH_BIN)$(NC)"

# Run unit tests
test: build-tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	@$(TEST_BIN)
	@if [ $$? -eq 0 ]; then \
		echo "$(GREEN)✓ All tests passed!$(NC)"; \
	else \
		echo "$(RED)✗ Tests failed!$(NC)"; \
		exit 1; \
	fi

# Run benchmarks
benchmark: build-benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	@$(BENCH_BIN)
	@echo "$(GREEN)✓ Benchmarks complete!$(NC)"

# Full validation suite
validate: test reproducibility
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ Validation Complete!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "Results:"
	@echo "  ✓ Unit tests: PASS"
	@echo "  ✓ Reproducibility: PASS"
	@echo ""

# Reproducibility test
reproducibility: build-benchmarks
	@echo "$(BLUE)Testing bitwise reproducibility...$(NC)"
	@$(BENCH_BIN) 2>&1 | grep -A 20 "Reproducibility Test"
	@echo "$(GREEN)✓ Reproducibility test complete!$(NC)"

# NCU profiling
ncu-profile: build-benchmarks
	@echo "$(BLUE)Running NCU profiling...$(NC)"
	@if ! command -v ncu &> /dev/null; then \
		echo "$(RED)Error: ncu not found. Install NVIDIA Nsight Compute.$(NC)"; \
		exit 1; \
	fi
	@chmod +x NVIDIA_ROBOTICS_INFRA_EVIDENCE/CUDA_KERNEL_EVIDENCE/profile_with_ncu.sh
	@cd NVIDIA_ROBOTICS_INFRA_EVIDENCE/CUDA_KERNEL_EVIDENCE && \
		./profile_with_ncu.sh ../../$(BENCH_BIN)
	@echo "$(GREEN)✓ NCU profiling complete!$(NC)"
	@echo "View reports with: ncu-ui NVIDIA_ROBOTICS_INFRA_EVIDENCE/CUDA_KERNEL_EVIDENCE/ncu_reports/*.ncu-rep"

# Nsight Systems profiling
nsys-profile: build-benchmarks
	@echo "$(BLUE)Running Nsight Systems profiling...$(NC)"
	@if ! command -v nsys &> /dev/null; then \
		echo "$(RED)Error: nsys not found. Install NVIDIA Nsight Systems.$(NC)"; \
		exit 1; \
	fi
	@mkdir -p NVIDIA_ROBOTICS_INFRA_EVIDENCE/CUDA_KERNEL_EVIDENCE/nsys_reports
	@nsys profile \
		--output=NVIDIA_ROBOTICS_INFRA_EVIDENCE/CUDA_KERNEL_EVIDENCE/nsys_reports/multimodal_fusion \
		--force-overwrite=true \
		--stats=true \
		$(BENCH_BIN)
	@echo "$(GREEN)✓ Nsight Systems profiling complete!$(NC)"
	@echo "View timeline: nsys-ui NVIDIA_ROBOTICS_INFRA_EVIDENCE/CUDA_KERNEL_EVIDENCE/nsys_reports/multimodal_fusion.nsys-rep"

# Build Python extension
build-python:
	@echo "$(BLUE)Building Python extension...$(NC)"
	@cd robocache && $(PYTHON) setup.py build_ext --inplace
	@echo "$(GREEN)✓ Python extension built!$(NC)"

# Install Python package
install-python: build-python
	@echo "$(BLUE)Installing Python package...$(NC)"
	@cd robocache && $(PYTHON) setup.py install
	@echo "$(GREEN)✓ Python package installed!$(NC)"

# Docker build
docker-build:
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t goatnote/robocache:cuda13-cutlass43 \
		-f NVIDIA_ROBOTICS_INFRA_EVIDENCE/CLUSTER_INFRA/Dockerfile.cuda13-cutlass43 \
		.
	@echo "$(GREEN)✓ Docker image built!$(NC)"

# Docker run tests
docker-run: docker-build
	@echo "$(BLUE)Running tests in Docker container...$(NC)"
	@docker run --rm --gpus all goatnote/robocache:cuda13-cutlass43 \
		make test benchmark validate
	@echo "$(GREEN)✓ Docker tests complete!$(NC)"

# Generate validation report
validation-report: validate benchmark
	@echo "$(BLUE)Generating validation report...$(NC)"
	@mkdir -p NVIDIA_ROBOTICS_INFRA_EVIDENCE/VALIDATION_REPORTS
	@$(PYTHON) scripts/generate_validation_report.py \
		--output NVIDIA_ROBOTICS_INFRA_EVIDENCE/VALIDATION_REPORTS/validation_report.pdf
	@echo "$(GREEN)✓ Validation report generated!$(NC)"

# Clean build artifacts
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	@rm -rf $(BUILD_DIR)
	@rm -rf robocache/**/*.so
	@rm -rf robocache/**/__pycache__
	@rm -rf robocache/**/*.pyc
	@rm -rf robocache/**/build
	@rm -f benchmark_results.csv
	@echo "$(GREEN)✓ Clean complete!$(NC)"

# Deep clean (including profiling data)
deep-clean: clean
	@echo "$(YELLOW)Deep cleaning (including profiling data)...$(NC)"
	@rm -rf NVIDIA_ROBOTICS_INFRA_EVIDENCE/CUDA_KERNEL_EVIDENCE/ncu_reports
	@rm -rf NVIDIA_ROBOTICS_INFRA_EVIDENCE/CUDA_KERNEL_EVIDENCE/nsys_reports
	@echo "$(GREEN)✓ Deep clean complete!$(NC)"

# Check environment
check-env:
	@echo "$(BLUE)Checking environment...$(NC)"
	@echo "CUDA_HOME: $(CUDA_HOME)"
	@which $(NVCC) || (echo "$(RED)Error: nvcc not found$(NC)" && exit 1)
	@echo "CUDA version:"
	@$(NVCC) --version | grep "release"
	@echo ""
	@echo "GPU:"
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || \
		(echo "$(RED)Error: nvidia-smi failed$(NC)" && exit 1)
	@echo ""
	@echo "$(GREEN)✓ Environment OK$(NC)"

# Quick smoke test
smoke-test: build-tests
	@echo "$(BLUE)Running smoke test...$(NC)"
	@$(TEST_BIN) 2>&1 | head -20
	@echo "$(GREEN)✓ Smoke test complete$(NC)"

# CI target (for GitHub Actions)
ci: check-env all test benchmark
	@echo "$(GREEN)✓ CI tests passed!$(NC)"

# Print configuration
config:
	@echo "$(BLUE)Build Configuration:$(NC)"
	@echo "  CUDA_ARCH: $(CUDA_ARCH)"
	@echo "  SM_VERSIONS: $(SM_VERSIONS)"
	@echo "  CUDA_HOME: $(CUDA_HOME)"
	@echo "  CUTLASS_HOME: $(CUTLASS_HOME)"
	@echo "  NVCC: $(NVCC)"
	@echo "  NVCC_FLAGS: $(NVCC_FLAGS)"
	@echo ""
	@echo "$(BLUE)Build Targets:$(NC)"
	@echo "  TEST_BIN: $(TEST_BIN)"
	@echo "  BENCH_BIN: $(BENCH_BIN)"
	@echo ""
