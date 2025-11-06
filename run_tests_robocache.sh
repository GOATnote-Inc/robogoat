#!/bin/bash
#
# RoboCache Test Runner
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh --fast       # Run only fast tests
#   ./run_tests.sh --cuda       # Run only CUDA tests
#   ./run_tests.sh --pytorch    # Run only PyTorch tests
#   ./run_tests.sh --coverage   # Run with coverage report
#

set -e

# Ensure local package is importable without installation
export PYTHONPATH="$(pwd)/robocache/python:${PYTHONPATH}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}RoboCache Test Suite${NC}"
echo -e "${GREEN}=================================${NC}"

# Parse arguments
PYTEST_ARGS=""
RUN_COVERAGE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --fast)
      PYTEST_ARGS="$PYTEST_ARGS -m 'not slow'"
      echo -e "${YELLOW}Running fast tests only${NC}"
      shift
      ;;
    --cuda)
      PYTEST_ARGS="$PYTEST_ARGS -m cuda"
      echo -e "${YELLOW}Running CUDA tests only${NC}"
      shift
      ;;
    --pytorch)
      PYTEST_ARGS="$PYTEST_ARGS -m 'not cuda'"
      export ROBOCACHE_BACKEND=pytorch
      echo -e "${YELLOW}Running PyTorch tests only${NC}"
      shift
      ;;
    --integration)
      PYTEST_ARGS="$PYTEST_ARGS -m integration"
      echo -e "${YELLOW}Running integration tests only${NC}"
      shift
      ;;
    --coverage)
      RUN_COVERAGE=true
      echo -e "${YELLOW}Generating coverage report${NC}"
      shift
      ;;
    --verbose|-v)
      PYTEST_ARGS="$PYTEST_ARGS -vv"
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

# Check dependencies
echo ""
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || { echo -e "${RED}PyTorch not installed${NC}"; exit 1; }
python -c "import pytest; print(f'pytest: {pytest.__version__}')" || { echo -e "${RED}pytest not installed${NC}"; exit 1; }

# Check CUDA availability
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "${GREEN}CUDA available${NC}"
    python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
else
    echo -e "${YELLOW}CUDA not available - skipping CUDA tests${NC}"
    PYTEST_ARGS="$PYTEST_ARGS -m 'not cuda'"
fi

# Check RoboCache installation
echo ""
echo "Checking RoboCache installation..."
python -c "import robocache; robocache.print_installation_info()"

# Run tests
echo ""
echo -e "${GREEN}Running tests...${NC}"
TEST_PATH="robocache/tests"
echo "pytest $TEST_PATH $PYTEST_ARGS"
echo ""

if [ "$RUN_COVERAGE" = true ]; then
    pytest "$TEST_PATH" $PYTEST_ARGS --cov=robocache --cov-report=term-missing --cov-report=html
    echo ""
    echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
else
    pytest "$TEST_PATH" $PYTEST_ARGS
fi

# Print summary
echo ""
echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}Tests completed successfully!${NC}"
echo -e "${GREEN}=================================${NC}"

