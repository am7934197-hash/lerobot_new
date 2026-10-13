#!/bin/bash
# Test script for WandB/SwanLab backend functionality
# This script runs comprehensive tests to verify the backend integration

set -e

echo "============================================================"
echo "WandB/SwanLab Backend Integration Test Suite"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_header() {
    echo -e "\n${GREEN}============================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}============================================================${NC}\n"
}

print_info() {
    echo -e "${YELLOW}INFO:${NC} $1"
}

print_error() {
    echo -e "${RED}ERROR:${NC} $1"
}

# Check if running in correct directory
if [ ! -f "test_swanlab_backend.py" ]; then
    print_error "Please run this script from the lerobot directory"
    exit 1
fi

# Test 1: Basic backend functionality
print_header "Test 1: Testing Basic Backend Functionality"
print_info "Testing WandB backend..."
python test_swanlab_backend.py --backend wandb

print_info "Testing SwanLab backend..."
python test_swanlab_backend.py --backend swanlab

# Test 2: Mock training
print_header "Test 2: Testing Mock Training"
print_info "Running mock training with WandB..."
python test_mock_training.py --backend wandb --epochs 3

print_info "Running mock training with SwanLab..."
python test_mock_training.py --backend swanlab --epochs 3

# Test 3: RL-style logging
print_header "Test 3: Testing RL-Style Logging"
print_info "Testing RL-style logging with WandB..."
python test_mock_training.py --backend wandb --epochs 1 --test-rl

print_info "Testing RL-style logging with SwanLab..."
python test_mock_training.py --backend swanlab --epochs 1 --test-rl

# Test 4: Compare both backends
print_header "Test 4: Comparing Both Backends Side-by-Side"
python test_swanlab_backend.py --backend both

print_header "All Tests Completed Successfully!"
echo ""
echo "Summary:"
echo "  ✓ WandB backend: PASSED"
echo "  ✓ SwanLab backend: PASSED"
echo "  ✓ Mock training: PASSED"
echo "  ✓ RL-style logging: PASSED"
echo ""
echo "Check the output directories for logged data:"
echo "  - WandB: wandb/ directories"
echo "  - SwanLab: swanlog/ directories"
echo "============================================================"
